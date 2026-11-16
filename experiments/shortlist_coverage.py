"""How often does a shortlist contain EVERY gold table? (the ceiling of anything that only reorders that shortlist)

    python experiments/shortlist_coverage.py --run nift_base --seeds 1 2 3 4 --data D --out O

TEST questions of finished runs. Shortlists: search top K (what a reranker sees), Laya top K, and NIFT-SQL's
candidate set (Laya top 50 union search top 50). CPU only; uses the runs' cached embeddings and Laya scores.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nift.config import REPO           # noqa: E402
from nift.pipeline import Run          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="nift_base")
ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
ap.add_argument("--config", default="default")
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

rows = []
for s in args.seeds:
    r = Run(s, args.run, args.data, args.out, config_path=REPO / "configs" / f"{args.config}.yaml")
    r.load(check_descriptions=False); r.split(); r.search(); r.build_rows(show=0); r.train(); r.score()
    top = r.cfg["fusion"]["candidates_top"]
    for k, D in r.sets.items():
        for i in r.SPLIT[k]["test"]:
            g = set(D.Q[i][2])
            ho = np.argsort(-D.extra["hyb"][i], kind="stable")
            co = np.argsort(-D.extra["P"][i], kind="stable")
            row = dict(seed=s, set=k, n_tables=D.n)
            for K in (25, 50, 100, 200):
                row[f"search top {K}"] = float(g <= set(ho[:K].tolist()))
            for K in (50, 100):
                row[f"Laya top {K}"] = float(g <= set(co[:K].tolist()))
            row["NIFT-SQL candidates (Laya 50 + search 50)"] = float(g <= set(co[:top].tolist()) | set(ho[:top].tolist()))
            rows.append(row)
R = pd.DataFrame(rows)
pd.set_option("display.width", 250)
cols = [c for c in R.columns if c not in ("seed", "set", "n_tables")]
print("\n" + "=" * 110 + f"\nSHORTLIST COVERAGE, test, seeds {args.seeds}: % of questions with EVERY gold table in the shortlist\n" + "=" * 110)
T = R.groupby(["set", "seed"])[cols].mean().mul(100).groupby(level=0).agg(["mean", "std"]).round(1)
print(T.T.to_string())
print("\ntables per dataset: " + ", ".join(f"{k} {v}" for k, v in R.groupby("set").n_tables.first().items()))
