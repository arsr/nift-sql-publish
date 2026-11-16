"""Reproduce every number in the paper, in order, with logs. Finished runs are skipped, so re-running resumes.

    python reproduce_paper.py --data <DATA_ROOT> --out <OUT_ROOT>

Stages (all by default; pick with --stages):
  main      NIFT-SQL, configs/default.yaml, seeds 1-4                     -> results/nift_base_seed<N>/
  curve     training size 10/20/30% (configs/train10/20/30.yaml), seeds 1-4 -> results/nift_train<P>_seed<N>/
  appendix  fine-tuned BGE + join completion (configs/nift_ftj.yaml), seeds 1-2 -> results/<appendix-run>_seed<N>/
  reranker  zero-shot Qwen3-Reranker-8B on the main runs' test questions (needs a GPU with about 20 GB)
  tables    every paper table in one file                                 -> results/PAPER_TABLES.txt
Each run is its own process (clean GPU memory) and logs to results/logs/. A single run is just main.py.
Exploratory appendix studies (B, C, D) are separate scripts: see experiments/README.md.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="folder with beaver/, livesqlbench_table/, livesqlbench_large_table/")
ap.add_argument("--out", required=True, help="output root (results go to <out>/results/)")
ap.add_argument("--stages", nargs="+", default=["main", "curve", "appendix", "reranker", "tables"],
                choices=["main", "curve", "appendix", "reranker", "tables"])
ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
ap.add_argument("--appendix-run", default="nift_ftj", help="run name for the appendix variant (Arjun's runs: arjun_v3j)")
ap.add_argument("--appendix-seeds", type=int, nargs="+", default=[1, 2])
ap.add_argument("--reranker", default="Qwen/Qwen3-Reranker-8B")
args = ap.parse_args()

logs = Path(args.out) / "results" / "logs"
logs.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", PYTHONUNBUFFERED="1")


def run(cmd, log):
    """Run one step in its own process; show its output and append it to the log. Stops on failure."""
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
    if (Path(args.out) / "results" / f"{name}_seed{seed}" / "results.csv").exists():
        print(f"{name} seed {seed}: finished, skipping")
        return
    run(["python", "main.py", "--seed", str(seed), "--name", name, "--data", args.data, "--out", args.out, "--config", config],
        logs / f"{name}_seed{seed}.log")


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
    run(["python", "experiments/reranker_baseline.py", "--run", "nift_base", "--seeds", *map(str, args.seeds),
         "--model", args.reranker, "--data", args.data, "--out", args.out], logs / "reranker_baseline.log")
if "tables" in args.stages:
    run(["python", "experiments/paper_tables.py", "--data", args.data, "--out", args.out, "--seeds", *map(str, args.seeds),
         "--appendix", args.appendix_run, "--reranker", args.reranker.split("/")[-1]], logs / "paper_tables.log")
    print(f"\npaper tables: {Path(args.out) / 'results' / 'PAPER_TABLES.txt'}")
