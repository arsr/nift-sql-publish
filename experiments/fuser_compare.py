"""Implementation study (not the paper): Laya vs fine-tuned pplx as scorers, and stronger fusers than logistic regression.

    python experiments/fuser_compare.py --run arjun_v3j --seed 0 --config nift_ftj --data D --out O [--k 100]

Needs the run's cached Laya scores and the cached fine-tuned pplx scores from experiments/pplx_finetune.py
(<run>/scores/<set>_pplxft_top<k>_val.npy). CPU only: nothing is trained on the GPU or re-scored. Validation only.

Part 1, scorer quality on the same (question, table) pairs (search top-k of each validation question):
  ROC-AUC, average precision, mean P(need) on gold vs other tables, Brier score, calibration error (ECE, 10 bins),
  gold tables ranked first / in the top 5 by the scorer, plus speed and training time from the logs.
Part 2, fusers (fitted 2-fold inside validation, per customer, fixed settings, no tuning):
  logistic regression (today) | MLP neural net | gradient-boosted trees | LambdaRank (LightGBM, if installed)
  each with scorer features from: Laya | pplx | Laya + pplx.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nift import budget, evaluate           # noqa: E402
from nift.config import REPO                # noqa: E402
from nift.pipeline import Run               # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--config", default="nift_ftj")
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--k", type=int, default=100)
args = ap.parse_args()

r = Run(args.seed, args.run, args.data, args.out, config_path=REPO / "configs" / f"{args.config}.yaml")
r.load(check_descriptions=False); r.split(); r.search(); r.build_rows(show=0)
r.train(); r.train_search(); r.score()
cfg = r.cfg
TOP = cfg["fusion"]["candidates_top"]

# ---------------------------------------------------------------- scores, both restricted to the same search top-k
S = {}
for k, D in r.sets.items():
    f = r.rd / "scores" / f"{k}_pplxft_top{args.k}_val.npy"
    assert f.exists(), f"missing {f}: run experiments/pplx_finetune.py for this run and seed first"
    raw = {"Laya": D.extra["P"], "pplx": np.load(f)}
    S[k] = {}
    for name, src in raw.items():
        P = np.full(src.shape, budget.FLOOR)
        for i in r.SPLIT[k]["val"]:
            top = np.argsort(-D.extra["hyb"][i], kind="stable")[:args.k]
            P[i, top] = src[i, top]
        S[k][name] = P

# ---------------------------------------------------------------- part 1: scorer quality on identical pairs
def ece(p, y, bins=10):
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & (p < b) if b < 1 else (p >= a) & (p <= b)
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e


q1 = []
for k, D in r.sets.items():
    for name in ("Laya", "pplx"):
        ps, ys, first, top5 = [], [], [], []
        for i in r.SPLIT[k]["val"]:
            top = np.argsort(-D.extra["hyb"][i], kind="stable")[:args.k]
            g = D.Q[i][2]
            p = S[k][name][i, top]
            ps.append(p); ys.append(np.array([t in g for t in top], float))
            order = top[np.argsort(-p, kind="stable")]
            first.append(float(order[0] in g)); top5.append(len(g & set(order[:5].tolist())) / len(g))
        p, y = np.concatenate(ps), np.concatenate(ys)
        q1.append({"set": k, "scorer": name, "ROC-AUC": roc_auc_score(y, p), "avg precision": average_precision_score(y, p),
                   "mean P gold": p[y == 1].mean(), "mean P other": p[y == 0].mean(), "Brier": brier_score_loss(y, p),
                   "ECE": ece(p, y), "top-1 is gold %": 100 * np.mean(first), "gold in top 5 %": 100 * np.mean(top5)})
Q1 = pd.DataFrame(q1).set_index(["set", "scorer"]).round(3)
pd.set_option("display.width", 250)
print("\n" + "=" * 110 + f"\nPART 1. SCORER QUALITY, seed {args.seed} validation, same search top-{args.k} pairs\n" + "=" * 110)
print(Q1.to_string())
cost = json.loads((r.rd / "costs.json").read_text()) if (r.rd / "costs.json").exists() else {}
lay_ms = {k: v["ms_per_question"] / r.sets[k.split(", ")[1].split(" ")[0]].n
          for k, v in cost.items() if k.startswith("score Laya") and "ms_per_question" in v and k.split(", ")[1].split(" ")[0] in r.sets}
print("\nspeed: Laya " + ", ".join(f"{k.split(', ')[1].split(' ')[0]} {v:.1f} ms/table" for k, v in lay_ms.items())
      + " | pplx: see the timing line of pplx_finetune.py (about 38-48 ms/table)")
print("training: Laya " + ", ".join(f"{k[12:]} {v['gpu_minutes']:.1f} min" for k, v in cost.items() if k.startswith("train Laya"))
      + " | pplx LoRA: see pplx_finetune.py (about 10-16 min per customer)")


# ---------------------------------------------------------------- part 2: features from any set of scorers
def feats(D, i, Ps):
    """Candidates = search top TOP union each scorer's top TOP. Search features + per-scorer features + join features."""
    n, dense, bm, hyb = D.n, D.extra["dense"][i], D.extra["bm25"][i], D.extra["hyb"][i]
    ho = np.argsort(-hyb, kind="stable")
    rh = np.empty(n, int); rh[ho] = np.arange(n)
    dbi = list(D.DBI.values())
    dr = np.zeros(n)
    for ix in dbi:
        dr[ix] = rh[ix].min()
    cand = set(ho[:TOP].tolist())
    per = []
    for P in Ps:
        p = np.clip(P[i].astype(float), 1e-6, 1 - 1e-6); lp = np.log(p / (1 - p))
        co = np.argsort(-p, kind="stable"); rp = np.empty(n, int); rp[co] = np.arange(n)
        dc = np.zeros(n)
        for ix in dbi:
            dc[ix] = rp[ix].min()
        cand |= set(co[:TOP].tolist()); per.append((lp, rp, dc, co))
    cand = np.array(sorted(cand))
    cols = [np.log1p(rh[cand]), dense[cand], bm[cand], np.log1p(dr[cand])]
    top5s = set(ho[:5].tolist())
    for lp, rp, dc, co in per:
        nb = [D.adj.get(int(t), set()) for t in cand]
        t10, t5 = set(co[:10].tolist()), set(co[:5].tolist()) | top5s
        cols += [lp[cand], np.log1p(rp[cand]), np.abs(np.log1p(rp[cand]) - np.log1p(rh[cand])), np.log1p(dc[cand]),
                 np.array([max((lp[u] for u in s), default=-14.0) for s in nb]),
                 np.array([sum(u in t10 for u in s) for s in nb], float),
                 np.array([float(sum(u in t5 for u in s) >= 2) for s in nb])]
    if len(per) == 2:
        cols += [per[0][0][cand] - per[1][0][cand]]           # scorer disagreement
    g = D.Q[i][2]
    return dict(cand=cand, F=np.column_stack(cols), ho=ho, y=np.array([t in g for t in cand], float), gold=set(g))


def make(kind):
    if kind == "logistic regression":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
    if kind == "MLP neural net":
        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-2, max_iter=300,
                                                             random_state=0))
    if kind == "gradient-boosted trees":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
                                              min_samples_leaf=20, random_state=0)
    raise ValueError(kind)


FUSERS = ["logistic regression", "MLP neural net", "gradient-boosted trees"]
try:
    import lightgbm as lgb
    FUSERS.append("LambdaRank (LightGBM)")
except ImportError:
    lgb = None
    print("lightgbm not installed: LambdaRank skipped")


def fit_rank(kind, train_e, test_e):
    if kind.startswith("LambdaRank"):
        X = np.vstack([e["F"] for e in train_e]); y = np.concatenate([e["y"] for e in train_e]).astype(int)
        m = lgb.LGBMRanker(objective="lambdarank", n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20,
                           reg_lambda=1.0, random_state=0, verbose=-1)
        m.fit(X, y, group=[len(e["y"]) for e in train_e])
        score = lambda e: m.predict(e["F"])            # noqa: E731
    else:
        m = make(kind).fit(np.vstack([e["F"] for e in train_e]), np.concatenate([e["y"] for e in train_e]))
        score = lambda e: m.predict_proba(e["F"])[:, 1]  # noqa: E731
    out = []
    for e in test_e:
        rk = np.empty(len(e["ho"]), int); rk[e["ho"]] = np.arange(len(e["ho"]))
        fo = e["cand"][np.lexsort((rk[e["cand"]], -score(e)))].tolist()   # ties broken by search rank
        s = set(fo)
        out.append(fo + [int(t) for t in e["ho"] if int(t) not in s])
    return out


rows, t_fit = [], {}
for scorers in (["Laya"], ["pplx"], ["Laya", "pplx"]):
    tag = " + ".join(scorers)
    for cust, ks in cfg["customers"].items():
        qs = [(k, i) for k in ks for i in r.SPLIT[k]["val"]]
        E = {(k, i): feats(r.sets[k], i, [S[k][s] for s in scorers]) for k, i in qs}
        perm = np.random.default_rng(1).permutation(len(qs))
        half = [set(perm[0::2].tolist()), set(perm[1::2].tolist())]
        for kind in FUSERS:
            t0 = time.time()
            for a in (0, 1):
                tr = [E[q] for j, q in enumerate(qs) if j in half[a]]
                te = [q for j, q in enumerate(qs) if j in half[1 - a]]
                for (k, i), o in zip(te, fit_rank(kind, tr, [E[q] for q in te])):
                    e = E[(k, i)]
                    rows.append(dict(set=k, qi=i, group="|".join(map(str, sorted(e["gold"]))), scorers=tag, fuser=kind,
                                     **evaluate.metrics(o, e["gold"], cfg)))
            t_fit[(tag, kind)] = t_fit.get((tag, kind), 0) + time.time() - t0
R = pd.DataFrame(rows)
R["method"] = R.fuser + " | " + R.scorers
print("\n" + "=" * 110 + f"\nPART 2. FUSERS x SCORERS, seed {args.seed} validation (2-fold inside validation, search top-{args.k})\n" + "=" * 110)
T = R.groupby(["method", "set"])[["c@8", "c@15", "c@25"]].mean().mul(100).round(1).unstack("set")
T = T.assign(**{"mean c@8": T["c@8"].mean(1).round(1), "mean c@15": T["c@15"].mean(1).round(1)})
print(T.sort_values("mean c@15", ascending=False).to_string())
base = "logistic regression | Laya"
print(f"\nchange vs '{base}' (today's setup), c@8 / c@15 [95% CI over gold-set groups]")
for m in sorted(R.method.unique()):
    if m == base:
        continue
    cells, d8, d15 = [], [], []
    for k in sorted(R.set.unique()):
        A_ = R[(R.set == k) & (R.method == m)].sort_values("qi"); B_ = R[(R.set == k) & (R.method == base)].sort_values("qi")
        a8 = evaluate.group_boot(A_["c@8"].values - B_["c@8"].values, A_.group.tolist(), 2000)
        a15 = evaluate.group_boot(A_["c@15"].values - B_["c@15"].values, A_.group.tolist(), 2000)
        d8.append(a8[0]); d15.append(a15[0])
        cells.append(f"{k} {a8[0]:+.1f} [{a8[1]:+.1f},{a8[2]:+.1f}] / {a15[0]:+.1f} [{a15[1]:+.1f},{a15[2]:+.1f}]")
    print(f"{m:45s} mean {np.mean(d8):+5.1f} / {np.mean(d15):+5.1f} | " + " | ".join(cells))
print("\nfit time (both folds, all customers): " + ", ".join(f"{t} / {k} {v:.1f}s" for (t, k), v in sorted(t_fit.items())))
