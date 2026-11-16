"""Metrics and group-aware paired bootstrap.

complete@K: 1 if EVERY gold table is in the top K, else 0.  R@N: share of gold tables in the top N.
Confidence intervals resample gold-table-set GROUPS (not single questions), because questions with the same
gold set are not independent.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd


def metrics(order: list[int], gold: set[int], cfg: dict) -> dict[str, float]:
    g = set(gold)
    r = {f"c@{K}": float(g <= set(order[:K])) for K in cfg["evaluate"]["K"]}
    r.update({f"R@{N}": len(g & set(order[:N])) / len(g) for N in cfg["evaluate"]["N"]})
    return r


def group_boot(diff: np.ndarray, groups: list, reps: int = 3000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of a per-question value (x100) with a 95% CI from resampling groups. Returns (mean, lo, hi)."""
    diff = np.asarray(diff, float)
    keys = sorted(set(groups), key=str)
    idx = {k: np.flatnonzero(np.array([g == k for g in groups])) for k in keys}
    sums = np.array([diff[idx[k]].sum() for k in keys])
    cnts = np.array([len(idx[k]) for k in keys])
    rg = np.random.default_rng(seed)
    pick = rg.integers(0, len(keys), (reps, len(keys)))
    bs = sums[pick].sum(1) / cnts[pick].sum(1)
    return 100 * diff.mean(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)


def table(R: pd.DataFrame, methods: list[str], cols: list[str]) -> pd.DataFrame:
    return R.groupby("method")[cols].mean().mul(100).round(1).loc[methods]


def compare(R: pd.DataFrame, a: str, b: str, key: str, reps: int) -> tuple[float, float, float]:
    """a minus b on one dataset's results, paired by question, CI over gold-set groups."""
    A = R[R.method == a].sort_values("qi")
    B = R[R.method == b].sort_values("qi")
    assert A.qi.tolist() == B.qi.tolist()
    return group_boot(A[key].values - B[key].values, A.group.tolist(), reps)


def combine_runs(out_root: str | Path, cfg: dict, seeds=(1, 2, 3, 4)) -> pd.DataFrame:
    """Pool finished runs: mean and std over runs, per dataset and method.

    Only the reported seeds (1-4, never the development seed 0) and only runs made with THIS config
    (same config hash), so variants such as configs/train30.yaml are pooled separately.
    """
    fs = sorted(glob.glob(str(Path(out_root) / "results" / "*_seed*" / "results.csv")))
    keep, skipped = [], []
    for f in fs:
        man = json.loads((Path(f).parent / "manifest.json").read_text())
        ok = man.get("seed") in seeds and man.get("config_sha") == cfg["_config_sha"]
        (keep if ok else skipped).append(Path(f).parent.name)
    if skipped:
        print(f"not pooled (development seed or another config): {skipped}")
    if not keep:
        print("no finished runs for this config and these seeds")
        return pd.DataFrame()
    A = pd.concat([pd.read_csv(Path(out_root) / "results" / r / "results.csv").assign(run=r) for r in keep])
    keys = [f"c@{K}" for K in cfg["evaluate"]["K"]] + ["R@10"]
    per_run = A.groupby(["set", "method", "run"])[keys].mean().mul(100)
    T = per_run.groupby(level=[0, 1]).agg(["mean", "std"]).round(1)
    print(f"pooled runs ({len(keep)}): {keep}")
    return T
