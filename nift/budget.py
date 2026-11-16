"""Laya scoring budget: let Laya score only the search top-K tables instead of every table.

Laya scores each (question, table) pair on its own, so the scores it would give the top-K tables are exactly the
scores already cached for them. Scoring only the top K is therefore simulated exactly from a finished run,
with no GPU and no new Laya scoring. Laya time per question falls in proportion to K / (number of tables).
Tables Laya did not score get the lowest score, so they rank below every scored table.
"""
from __future__ import annotations

import copy

import numpy as np

from . import evaluate, fusion

FLOOR = 1e-6    # fusion.features clips probabilities to [1e-6, 1 - 1e-6]; unscored tables sit at the floor


def budgets(n: int) -> dict[str, int]:
    """Named budgets for a schema of n tables: every table (today), half, a quarter, and fixed 100 / 50."""
    out = {"all tables (today)": n, "50% of tables": int(np.ceil(n / 2)), "25% of tables": int(np.ceil(n / 4)),
           "top 100": min(100, n), "top 50": min(50, n)}
    return out


def restrict(D, qidx: list[int], k: int) -> np.ndarray:
    """Copy of D.extra['P'] in which, for each question in qidx, only the search top-k tables keep Laya scores."""
    P = D.extra["P"].copy()
    for i in qidx:
        drop = np.argsort(-D.extra["hyb"][i], kind="stable")[k:]
        P[i, drop] = FLOOR
    return P


def two_fold(run, cfg: dict, part: str = "val") -> list[dict]:
    """Per-question metrics with the fusion fitted 2-fold inside `part` (validation by default), per customer."""
    out = []
    for cust, ks in run.cfg["customers"].items():
        qs = [(k, i) for k in ks for i in run.SPLIT[k][part]]
        E = {k: {i: fusion.features(run.sets[k], i, cfg) for i in run.SPLIT[k][part]} for k in ks}
        perm = np.random.default_rng(1).permutation(len(qs))
        half = [set(perm[0::2].tolist()), set(perm[1::2].tolist())]
        for a in (0, 1):
            m = fusion.fit([E[k][i] for j, (k, i) in enumerate(qs) if j in half[a]], cfg)
            for j, (k, i) in enumerate(qs):
                if j in half[1 - a]:
                    e = E[k][i]
                    out.append(dict(set=k, qi=i, group="|".join(map(str, sorted(e["gold"]))),
                                    **evaluate.metrics(fusion.rank(m, e), e["gold"], cfg)))
    return out


def evaluate_budgets(run, part: str = "val") -> list[dict]:
    """Rows: one per (budget, dataset, question). The run must have scores and search loaded (run.score())."""
    rows = []
    full = {k: D.extra["P"] for k, D in run.sets.items()}
    names = list(budgets(1))
    try:
        for name in names:
            for k, D in run.sets.items():
                kk = budgets(D.n)[name]
                D.extra["P"] = restrict(D, run.SPLIT[k][part], kk) if kk < D.n else full[k]
            for r in two_fold(run, copy.deepcopy(run.cfg), part):
                D = run.sets[r["set"]]
                kk = budgets(D.n)[name]
                top = set(np.argsort(-D.extra["hyb"][r["qi"]], kind="stable")[:kk].tolist())
                rows.append(dict(r, budget=name, k=kk, n=D.n, share=kk / D.n,
                                 gold_in_scored=float(D.Q[r["qi"]][2] <= top)))
    finally:
        for k, D in run.sets.items():
            D.extra["P"] = full[k]
    return rows
