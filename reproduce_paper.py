"""Reproduce every number in the paper, in order, with logs. Finished steps are skipped, so re-running resumes.

    python get_data.py --out <DATA_ROOT>                                   # once: the three benchmark collections
    python reproduce_paper.py --data <DATA_ROOT> --out <OUT_ROOT>
    python reproduce_paper.py --data <DATA_ROOT> --out <OUT_ROOT> --stages holdout --beaver <BEAVER_DIR>   # Section 9.7

Stages, in dependency order (all but holdout by default; pick with --stages):
  main        NIFT-SQL-base with Laya (pre-specified), configs/default.yaml, seeds 1-4 -> results/nift_base_seed<N>/
  curve       training size 10/20/30% with Laya, seeds 1-4                         -> results/nift_train<P>_seed<N>/
  appendix    fine-tuned BGE + join completion (configs/nift_ftj.yaml), seeds 1-2   -> results/<appendix-run>_seed<N>/
  reranker    zero-shot Qwen3-Reranker-8B on the main runs' test questions (GPU with about 20 GB)
  ce          the fine-tuned cross-encoder (the headline scorer) and its variants: zero-shot, no join hints, 10/20/30%
              -> results/nift_ce*_seed<N>/, SUPERVISED_CROSSENCODER_*.txt, CROSSENCODER_*.txt
  additional  retraining replicates, deterministic-mode check, reranker variants, fine-tuned BGE retriever baselines
              (experiments/run_additional_analyses.py, about 2 GPU hours)            -> ADDITIONAL_ANALYSES.txt
  ce_analysis cross-encoder analyses, CPU only                                     -> CE_ANALYSIS.txt
  extensions  NIFT-SQL (fine-tuned retriever) and the high-accuracy mode            -> FUSION_EXTENSIONS.txt
  breakdowns  breakdowns, fusion controls, the worked fusion example, CPU only      -> FINAL_BREAKDOWNS.txt,
                                                                                      FUSION_CONTROLS.txt, FUSION_EXAMPLE.txt
  tables      the Laya-era tables in one file                                      -> PAPER_TABLES.txt
  holdout     frozen NIFT-SQL scored once on the newer BEAVER release (needs --beaver; see get_data.py --beaver)
              -> <OUT_ROOT>/holdout/CONFIRM.txt
Every report lands in <OUT_ROOT>/results/ (the holdout in <OUT_ROOT>/holdout/); the copies in results/paper/ of this
repository are what the paper reports. Each step is its own process (clean GPU memory) and logs to results/logs/.
Exploratory appendix studies (B, C, D) are separate scripts: see experiments/README.md.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
STAGES = ["main", "curve", "appendix", "reranker", "ce", "additional", "ce_analysis", "extensions", "breakdowns", "tables", "holdout"]
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="folder with beaver/, livesqlbench_table/, livesqlbench_large_table/")
ap.add_argument("--out", required=True, help="output root (results go to <out>/results/)")
ap.add_argument("--stages", nargs="+", default=STAGES[:-1], choices=STAGES)
ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
ap.add_argument("--appendix-run", default="nift_ftj", help="run name for the appendix variant (the authors' runs: arjun_v3j)")
ap.add_argument("--appendix-seeds", type=int, nargs="+", default=[1, 2])
ap.add_argument("--reranker", default="Qwen/Qwen3-Reranker-8B")
ap.add_argument("--ce-model", default="BAAI/bge-reranker-large")
ap.add_argument("--beaver", default=None, help="holdout only: folder with table_<db>.parquet and query_<part>.parquet")
args = ap.parse_args()

RES = Path(args.out) / "results"
logs = RES / "logs"
logs.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", PYTHONUNBUFFERED="1")
SEEDS = [str(s) for s in args.seeds]
D_O = ["--data", args.data, "--out", args.out]
CE_TAG = args.ce_model.rstrip("/").split("/")[-1]


def run(cmd, log, done=None):
    """Run one step in its own process; show its output and append it to the log. Stops on failure.
    done: a file that exists only once this step has finished; if it exists, the step is skipped."""
    if done is not None and Path(done).exists():
        print(f"{Path(log).stem}: finished ({Path(done).name} exists), skipping")
        return
    t0 = time.time()
    print(f"\n{'#' * 100}\n# {' '.join(cmd[1:])}\n# log: {log}\n{'#' * 100}", flush=True)
    with open(log, "a") as f:
        p = subprocess.Popen([sys.executable, *cmd[1:]], cwd=REPO, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            print(line, end=""); f.write(line)
        p.wait()
    if p.returncode:
        sys.exit(f"failed (exit {p.returncode}); see {log}. Re-run: finished steps are skipped.")
    print(f"# done in {(time.time() - t0) / 60:.1f} min", flush=True)


def one(name, config, seed):
    run(["python", "main.py", "--seed", str(seed), "--name", name, *D_O, "--config", config],
        logs / f"{name}_seed{seed}.log", done=RES / f"{name}_seed{seed}" / "results.csv")


def ce(run_name, *extra):
    """The cross-encoder script scores every seed's test part once, so it is skipped once its report exists."""
    report = RES / (f"SUPERVISED_CROSSENCODER_{CE_TAG}.txt" if run_name == "nift_ce" else f"CROSSENCODER_{run_name}_{CE_TAG}.txt")
    run(["python", "experiments/supervised_crossencoder.py", "--seeds", *SEEDS, *D_O, "--model", args.ce_model,
         "--run", run_name, *extra], logs / f"{run_name}.log", done=report)


if "main" in args.stages:
    for s in args.seeds:
        one("nift_base", "default", s)
if "curve" in args.stages:
    for p in (10, 20, 30):
        for s in args.seeds:
            one(f"nift_train{p}", f"train{p}", s)
if "appendix" in args.stages:
    for s in args.appendix_seeds:
        one(args.appendix_run, "nift_ftj", s)
if "reranker" in args.stages:
    run(["python", "experiments/reranker_baseline.py", "--run", "nift_base", "--seeds", *SEEDS,
         "--model", args.reranker, *D_O], logs / "reranker_baseline.log")
if "ce" in args.stages:
    ce("nift_ce")                                         # the headline scorer, compared with nift_base
    ce("nift_ce0", "--zero-shot")                         # same model, not fine-tuned
    ce("nift_ce_nojoin", "--no-joins")                    # fine-tuned without join hints
    for p in (10, 20, 30):                                # training-size curve, compared with the Laya curve runs
        ce(f"nift_ce_train{p}", "--config", f"train{p}", "--main", f"nift_train{p}")
if "additional" in args.stages:
    run(["python", "experiments/run_additional_analyses.py", *D_O, "--ce-model", args.ce_model, "--reranker-model", args.reranker],
        logs / "run_additional_analyses.log", done=RES / "ADDITIONAL_ANALYSES.txt")
if "ce_analysis" in args.stages:
    run(["python", "experiments/ce_analysis.py", *D_O, "--seeds", *SEEDS, "--ce", CE_TAG, "--ce-model", args.ce_model],
        logs / "ce_analysis.log", done=RES / "CE_ANALYSIS.txt")
if "extensions" in args.stages:
    run(["python", "experiments/fusion_extensions.py", *D_O, "--seeds", *SEEDS, "--ce", CE_TAG, "--reranker", args.reranker],
        logs / "fusion_extensions.log", done=RES / "FUSION_EXTENSIONS.txt")
if "breakdowns" in args.stages:
    rr_tag = args.reranker.split("/")[-1]
    run(["python", "experiments/final_breakdowns.py", *D_O, "--seeds", *SEEDS, "--ce", CE_TAG, "--reranker", rr_tag],
        logs / "final_breakdowns.log", done=RES / "FINAL_BREAKDOWNS.txt")
    run(["python", "experiments/fusion_controls.py", *D_O, "--seeds", *SEEDS, "--ce", CE_TAG, "--reranker", args.reranker],
        logs / "fusion_controls.log", done=RES / "FUSION_CONTROLS.txt")
    run(["python", "experiments/fusion_example.py", *D_O, "--ce", CE_TAG, "--reranker", rr_tag],
        logs / "fusion_example.log", done=RES / "FUSION_EXAMPLE.txt")
if "tables" in args.stages:
    run(["python", "experiments/paper_tables.py", *D_O, "--seeds", *SEEDS,
         "--appendix", args.appendix_run, "--reranker", args.reranker.split("/")[-1]], logs / "paper_tables.log")
    print(f"\npaper tables: {RES / 'PAPER_TABLES.txt'}")
if "holdout" in args.stages:
    if not args.beaver:
        sys.exit("holdout needs --beaver <folder>; python get_data.py --beaver <folder> prepares it")
    H = Path(args.out) / "holdout"
    keys = H / "schema_keys.json"
    run(["python", "v2/schema_keys.py", "--data", args.beaver, "--out", str(H)], logs / "holdout_schema_keys.log", done=keys)
    run(["python", "experiments/confirm_beaver_v3.py", "--beaver", args.beaver, "--keys", str(keys), "--out", str(H), "--seeds", "1"],
        logs / "holdout.log", done=H / "CONFIRM.txt")
    print(f"\nholdout: {H / 'CONFIRM.txt'}")
