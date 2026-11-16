"""Can Laya score only the search top-K tables instead of all of them? Validation only; test is not used.

    python experiments/laya_budget.py --runs arjun_v3j:0 arjun_v3j:1 arjun_v3j:2 --config nift_ftj --data D --out O

Uses each run's cached Laya scores (nothing is retrained or re-scored; see nift/budget.py for why this is exact).
Decision rule, fixed before looking (non-inferiority, seed 0 validation):
    adopt the smallest budget whose mean drop vs "all tables" is at most 1.0 point in BOTH c@8 and c@15
    and no dataset is significantly worse (95% group-bootstrap CI entirely below zero).
Seeds 1 and 2 validation are printed as a check on that choice.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nift import budget, evaluate      # noqa: E402
from nift.config import REPO           # noqa: E402
from nift.pipeline import Run          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True, help="run_name:seed, e.g. arjun_v3j:0")
ap.add_argument("--config", default="nift_ftj")
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

ALL = "all tables (today)"
rows = []
for spec in args.runs:
    name, seed = spec.split(":")
    r = Run(int(seed), name, args.data, args.out, config_path=REPO / "configs" / f"{args.config}.yaml")
    r.load(check_descriptions=False); r.split(); r.search(); r.build_rows(show=0)
    r.train(); r.train_search(); r.score()
    rows += [dict(x, seed=int(seed)) for x in budget.evaluate_budgets(r, "val")]
R = pd.DataFrame(rows)
order = list(budget.budgets(1))
pd.set_option("display.width", 250)

print("\n" + "=" * 110 + "\nLAYA BUDGET, VALIDATION (fusion 2-fold inside validation)\n" + "=" * 110)
sizes = R.groupby(["budget", "set"])["k"].first().unstack("set").reindex(order)
print("tables Laya scores per question:\n" + sizes.to_string())
for seed, S in R.groupby("seed"):
    T = S.groupby(["budget", "set"])[["c@8", "c@15", "gold_in_scored"]].mean().mul(100).round(1).unstack("set").reindex(order)
    print(f"\nseed {seed}  (gold_in_scored = questions whose gold tables are all among the tables Laya scored)\n" + T.to_string())

print("\nchange vs all tables, c@8 and c@15 [95% CI over gold-set groups]")
verdict = {}
for seed, S in R.groupby("seed"):
    for name in order[1:]:
        cells, d8, d15, worse = [], [], [], False
        for k in sorted(S.set.unique()):
            A = S[(S.set == k) & (S.budget == name)].sort_values("qi")
            B = S[(S.set == k) & (S.budget == ALL)].sort_values("qi")
            m8, lo8, hi8 = evaluate.group_boot(A["c@8"].values - B["c@8"].values, A.group.tolist(), 3000)
            m15, lo15, hi15 = evaluate.group_boot(A["c@15"].values - B["c@15"].values, A.group.tolist(), 3000)
            worse |= hi8 < 0 or hi15 < 0
            d8.append(m8); d15.append(m15)
            cells.append(f"{k} {m8:+.1f} [{lo8:+.1f},{hi8:+.1f}] / {m15:+.1f} [{lo15:+.1f},{hi15:+.1f}]")
        ok = np.mean(d8) >= -1.0 and np.mean(d15) >= -1.0 and not worse
        verdict[(seed, name)] = ok
        print(f"seed {seed} {name:14s} mean {np.mean(d8):+.1f} / {np.mean(d15):+.1f}  {'OK' if ok else 'worse'}  | " + " | ".join(cells))

if 0 in R.seed.unique():
    passing = [n for n in order[1:] if verdict[(0, n)]]
    pick = min(passing, key=lambda n: R[(R.seed == 0) & (R.budget == n)].share.mean()) if passing else ALL
    print(f"\nDECISION (seed 0 rule): {pick}")
    for seed in sorted(s for s in R.seed.unique() if s != 0):
        if pick != ALL:
            print(f"  check on seed {seed} validation: {'holds' if verdict[(seed, pick)] else 'does NOT hold'}")
