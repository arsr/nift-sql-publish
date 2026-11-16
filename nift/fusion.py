"""Fusion: logistic regression over the 8 core features, on candidates = Laya top-K union search top-K.

The fused order is: candidates by fused probability, then every other table by search rank.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import Dataset

CORE = ["logit_p", "log_rank_cls", "log_rank_hyb", "dense", "bm25", "disagree", "db_best_rank_hyb", "db_best_rank_cls"]
JOIN = ["join_best_logit", "join_in_laya_top10", "join_bridge"]   # optional: completion signals from the join graph


def features(D: Dataset, i: int, cfg: dict) -> dict:
    """Candidates and their features for question i. Needs D.extra['P'] (Laya) and the search scores."""
    names = cfg["fusion"]["features"]
    assert names[:8] == CORE and set(names[8:]) <= set(JOIN), f"unknown fusion features: {names}"
    p = np.clip(D.extra["P"][i].astype(float), 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p))
    n, dense, bm, hyb = D.n, D.extra["dense"][i], D.extra["bm25"][i], D.extra["hyb"][i]
    co, ho = np.argsort(-p, kind="stable"), np.argsort(-hyb, kind="stable")
    rp, rh = np.empty(n, int), np.empty(n, int)
    rp[co], rh[ho] = np.arange(n), np.arange(n)
    dr, dc = np.zeros(n), np.zeros(n)
    for ix in D.DBI.values():
        dr[ix], dc[ix] = rh[ix].min(), rp[ix].min()
    top = cfg["fusion"]["candidates_top"]
    cand = np.array(sorted(set(co[:top].tolist()) | set(ho[:top].tolist())))
    cols = [lp[cand], np.log1p(rp[cand]), np.log1p(rh[cand]), dense[cand], bm[cand],
            np.abs(np.log1p(rp[cand]) - np.log1p(rh[cand])), np.log1p(dr[cand]), np.log1p(dc[cand])]
    if names[8:]:
        top10, top5 = set(co[:10].tolist()), set(co[:5].tolist()) | set(ho[:5].tolist())
        nb = [D.adj.get(int(t), set()) for t in cand]
        extra = {"join_best_logit": [max((lp[u] for u in s), default=-14.0) for s in nb],   # best-scored joined table
                 "join_in_laya_top10": [sum(u in top10 for u in s) for s in nb],            # joined tables Laya ranks top 10
                 "join_bridge": [float(sum(u in top5 for u in s) >= 2) for s in nb]}         # links two top-5 tables
        cols += [np.asarray(extra[f], float) for f in names[8:]]
    F = np.column_stack(cols)
    g = D.Q[i][2]
    return dict(cand=cand, F=F, co=co, ho=ho, y=np.array([t in g for t in cand], float), gold=set(g))


def fit(examples: list[dict], cfg: dict):
    X = np.vstack([e["F"] for e in examples])
    y = np.concatenate([e["y"] for e in examples])
    assert 0 < y.sum() < len(y), "fusion needs both gold and non-gold candidates"
    return make_pipeline(StandardScaler(), LogisticRegression(C=cfg["fusion"]["C"], max_iter=3000)).fit(X, y)


def rank(model, e: dict) -> list[int]:
    fo = e["cand"][np.argsort(-model.predict_proba(e["F"])[:, 1], kind="stable")].tolist()
    s = set(fo)
    return fo + [int(t) for t in e["ho"] if int(t) not in s]


def weights(model, names: list[str] = CORE) -> dict[str, float]:
    return {f: round(float(w), 3) for f, w in zip(names, model[-1].coef_[0])}


def rank_average(e: dict, k: int = 60) -> list[int]:
    """Ablation: combine Laya and search WITHOUT learning (reciprocal rank fusion, k = 60). Same two inputs as the fusion."""
    n = len(e["co"])
    rc, rh = np.empty(n), np.empty(n)
    rc[e["co"]], rh[e["ho"]] = np.arange(n), np.arange(n)
    return np.argsort(-(1 / (k + rc) + 1 / (k + rh)), kind="stable").tolist()
