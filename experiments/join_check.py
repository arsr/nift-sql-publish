"""Seed-0 decision for the join-completion candidate (configs/nift_ftj.yaml). Validation only; test is not used.

    python main.py --seed 0 --name arjun_v3j --config nift_ftj --data D --out O      # candidate run first
    python experiments/join_check.py --cand arjun_v3j --base arjun_v2 --data D --out O

Both runs are rebuilt from their caches (nothing is retrained). The fusion is fitted 2-fold inside validation.
Rule: PASS if mean c@8 gain over v2 >= +2.0 and no dataset significantly worse.
"""
import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nift import evaluate, fusion          # noqa: E402
from nift.config import REPO               # noqa: E402
from nift.pipeline import Run              # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cand", required=True)
ap.add_argument("--base", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()


def rebuild(name, config):
    r = Run(args.seed, name, args.data, args.out, config_path=REPO / "configs" / f"{config}.yaml")
    r.load(check_descriptions=False); r.split(); r.search(); r.build_rows(show=0)
    r.train(); r.train_search(); r.score(); r.fit_fusion()
    return r


def with_features(cfg, names):
    c = copy.deepcopy(cfg); c["fusion"]["features"] = names; return c


def val_rows(run, cfg, tag):
    out = []
    for cust, ks in run.cfg["customers"].items():
        val = [(k, i) for k in ks for i in run.SPLIT[k]["val"]]
        E = {k: {i: fusion.features(run.sets[k], i, cfg) for i in run.SPLIT[k]["val"]} for k in ks}
        perm = np.random.default_rng(1).permutation(len(val)); half = [set(perm[0::2].tolist()), set(perm[1::2].tolist())]
        for a in (0, 1):
            m = fusion.fit([E[k][i] for j, (k, i) in enumerate(val) if j in half[a]], cfg)
            for j, (k, i) in enumerate(val):
                if j in half[1 - a]:
                    out.append(dict(set=k, qi=i, group="|".join(map(str, sorted(E[k][i]["gold"]))), method=tag,
                                    **evaluate.metrics(fusion.rank(m, E[k][i]), E[k][i]["gold"], cfg)))
    return out


base, cand = rebuild(args.base, "nift_ft"), rebuild(args.cand, "nift_ftj")
V = {"0. v2 (core 8)": (base, with_features(base.cfg, fusion.CORE)),
     "1. + join features (declared joins only)": (base, with_features(base.cfg, fusion.CORE + fusion.JOIN)),
     "2. + inferred joins (Laya retrained), core 8": (cand, with_features(cand.cfg, fusion.CORE)),
     "3. + inferred joins + join features": (cand, cand.cfg)}
R = pd.DataFrame(sum((val_rows(r, c, t) for t, (r, c) in V.items()), []))
pd.set_option("display.width", 250)
print("\n" + "=" * 110 + f"\nSEED {args.seed} VALIDATION (fusion 2-fold inside validation)\n" + "=" * 110)
print(R.groupby(["method", "set"])[["c@8", "c@15", "c@25"]].mean().mul(100).round(1).unstack("set").to_string())
for name, r in (("base", base), ("candidate", cand)):
    conn = {}
    for k, D in r.sets.items():
        gs = [D.Q[i][2] for i in r.SPLIT[k]["val"] if len(D.Q[i][2]) > 1]
        conn[k] = round(100 * np.mean([all(D.adj.get(g, set()) & (gold - {g}) for g in gold) for gold in gs]), 1)
    print(f"multi-table validation gold sets where every table joins another gold table, {name}: {conn}")
print()
for tag in list(V)[1:]:
    gains, worse, cells = [], False, []
    for k in base.sets:
        A_ = R[(R.set == k) & (R.method == tag)].sort_values("qi"); B_ = R[(R.set == k) & (R.method == list(V)[0])].sort_values("qi")
        m_, lo, hi = evaluate.group_boot(A_["c@8"].values - B_["c@8"].values, A_.group.tolist(), 3000); gains.append(m_); worse |= hi < 0
        cells.append(f"{k} {m_:+.1f} [{lo:+.1f}, {hi:+.1f}]")
    ok = np.mean(gains) >= 2.0 and not worse
    print(f"{tag:46s} c@8 vs v2: " + " | ".join(cells) + f"  -> mean {np.mean(gains):+.1f}: {'PASS' if ok else 'no'}")
