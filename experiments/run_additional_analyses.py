"""Runs every additional experiment in order, skipping what is already done (safe to rerun after a disconnect).

    python experiments/run_additional_analyses.py --data D --out O

  1  input hashes with and without join hints (CPU)                       -> are BEAVER's inputs really identical?
  2  BEAVER cross-encoder trained twice in deterministic mode, and once
     deterministic without join hints, seeds 1 and 2 (score only)        -> is the BEAVER difference retraining alone?
  3  two retrained replicates of the cross-encoder, all customers,
     seeds 1 to 4 (about 1.5 GPU hours on an A100)                       -> how much does retraining move the results?
  4  Qwen3-Reranker-8B over every table of the known database            -> the reranker without the search shortlist
  4b Qwen3-Reranker-8B over the top 2 databases by search (zero-label routing)
  4c Qwen3-Reranker-8B on validation questions (search top 100, top 2
     databases by search) and over the top 2 databases of its own
     grouped ranking, validation and test (about 1 GPU hour)              -> reranker variant chosen on validation
  5  BGE fine-tuned per customer, seeds 1 to 4 (about 15 GPU-minutes)     -> a supervised-retriever baseline
  5b two more trainings of that BGE (about 30 GPU-minutes)                -> its training variation
  5c BGE fine-tuned on training + validation questions, three trainings
     (about 45 GPU-minutes)                                               -> the same label budget as NIFT-SQL
  6  experiments/additional_analyses.py (CPU)                             -> writes results/ADDITIONAL_ANALYSES.txt
Logs go to <out>/results/logs/additional_*.log.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--replicates", type=int, default=2)
ap.add_argument("--ce-model", default="BAAI/bge-reranker-large")
ap.add_argument("--reranker-model", default="Qwen/Qwen3-Reranker-8B")
ap.add_argument("--k", default="100")
args = ap.parse_args()
CET, RRT = args.ce_model.rstrip("/").split("/")[-1], args.reranker_model.rstrip("/").split("/")[-1]
REPO = Path(__file__).resolve().parents[1]
RES = Path(args.out) / "results"
LOG = RES / "logs"; LOG.mkdir(parents=True, exist_ok=True)
CE = [sys.executable, str(REPO / "experiments" / "supervised_crossencoder.py"), "--data", args.data, "--out", args.out, "--model", args.ce_model]


def run(name, cmd, done=None):
    if done is not None and done():
        print(f"[skip] {name}: already done"); return
    print(f"[run ] {name}", flush=True)
    with open(LOG / f"additional_{name}.log", "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=REPO).returncode
    tail = (LOG / f"additional_{name}.log").read_text().splitlines()[-3:]
    print(f"[{'ok' if rc == 0 else 'FAIL'}] {name}\n      " + "\n      ".join(tail), flush=True)
    if rc:
        sys.exit(f"{name} failed: see {LOG / f'additional_{name}.log'}")


def scores(run_name, seeds, sets=("beaver",)):
    return lambda: all((RES / f"{run_name}_seed{s}" / "scores" / f"{k}_ce_{CET}.npy").exists() for s in seeds for k in sets)


def hashlog(name):
    f = LOG / f"additional_{name}.log"
    return f if f.exists() else LOG / f"review_round2_{name}.log"


def hashed(name):
    return lambda: hashlog(name).exists() and "INPUT HASH seed 4" in hashlog(name).read_text(errors="ignore")


run("1_hash_joins", CE + ["--run", "nift_inputhash", "--input-hash"], hashed("1_hash_joins"))
run("1_hash_nojoins", CE + ["--run", "nift_inputhash", "--input-hash", "--no-joins"], hashed("1_hash_nojoins"))
for n, extra in (("det_a", []), ("det_b", []), ("det_nojoin", ["--no-joins"])):
    run(f"2_{n}", CE + ["--run", f"nift_ce_{n}", "--seeds", "1", "2", "--deterministic", "--customers", "beaver", "--score-only"] + extra,
        scores(f"nift_ce_{n}", (1, 2)))
for R in range(1, args.replicates + 1):
    run(f"3_rep{R}", CE + ["--run", f"nift_ce_rep{R}", "--replicate", str(R)],
        lambda R=R: all((RES / f"nift_ce_rep{R}_seed{s}" / "results.csv").exists() for s in (1, 2, 3, 4)))
run("4_reranker_known_db", [sys.executable, str(REPO / "experiments" / "reranker_baseline.py"), "--data", args.data, "--out", args.out,
                            "--model", args.reranker_model, "--k", args.k, "--known-db", "--batch", "16"],
    lambda: all((RES / ".cache" / "reranker" / f"{k}_{RRT}_db.npy").exists() for k in ("beaver", "livesqlbench", "livesqlbench_large"))
    and any(f.exists() and "known-database scores written" in f.read_text(errors="ignore")
            for f in (LOG / "additional_4_reranker_known_db.log", LOG / "review_round2_4_reranker_known_db.log")))
run("4b_reranker_routed", [sys.executable, str(REPO / "experiments" / "reranker_baseline.py"), "--data", args.data, "--out", args.out,
                           "--model", args.reranker_model, "--k", args.k, "--routed", "2", "--batch", "16"],
    lambda: all((RES / ".cache" / "reranker" / f"{k}_{RRT}_routed2.npy").exists() for k in ("beaver", "livesqlbench", "livesqlbench_large"))
    and (LOG / "additional_4b_reranker_routed.log").exists()
    and "known-database scores written" in (LOG / "additional_4b_reranker_routed.log").read_text(errors="ignore"))
run("5_finetuned_search", [sys.executable, str(REPO / "experiments" / "finetuned_search_baseline.py"), "--data", args.data, "--out", args.out],
    lambda: all((RES / f"nift_ftsearch_seed{s}" / "scores" / f"{k}_hyb_ft.npy").exists() for s in (1, 2, 3, 4)
                for k in ("beaver", "livesqlbench", "livesqlbench_large")))
run("4c_reranker_variants", [sys.executable, str(REPO / "experiments" / "reranker_baseline.py"), "--data", args.data, "--out", args.out,
                             "--model", args.reranker_model, "--k", args.k, "--variants", "--batch", "16"],
    lambda: (LOG / "additional_4c_reranker_variants.log").exists()
    and "known-database scores written" in (LOG / "additional_4c_reranker_variants.log").read_text(errors="ignore"))
FTS = [sys.executable, str(REPO / "experiments" / "finetuned_search_baseline.py"), "--data", args.data, "--out", args.out]


def ft_done(name):
    return lambda: all((RES / f"{name}_seed{s}" / "scores" / f"{k}_hyb_ft.npy").exists() for s in (1, 2, 3, 4)
                       for k in ("beaver", "livesqlbench", "livesqlbench_large"))


for R in (1, 2):
    run(f"5b_finetuned_search_rep{R}", FTS + ["--replicate", str(R)], ft_done(f"nift_ftsearch_rep{R}"))
for R in (0, 1, 2):
    nm = "nift_ftsearch_trval" + (f"_rep{R}" if R else "")
    run(f"5c_finetuned_search_trval{R}", FTS + ["--with-val", "--replicate", str(R)], ft_done(nm))
run("6_additional_analyses", [sys.executable, str(REPO / "experiments" / "additional_analyses.py"), "--data", args.data, "--out", args.out,
                          "--ce", CET, "--reranker", RRT, "--k", args.k])
for name in ("1_hash_joins", "1_hash_nojoins"):
    print("\n".join(l for l in hashlog(name).read_text().splitlines() if l.startswith("INPUT HASH")))
print((RES / "ADDITIONAL_ANALYSES.txt").read_text())
