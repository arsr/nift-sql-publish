"""50 / 30 / 20 train / val / test by question count, grouped by gold-table set.

- A gold-table set (the exact combination of tables) lives in exactly one part, so no test question repeats
  a training combination.
- Datasets of one customer are split together: the same question text always goes to the same part
  (LiveSQLBench and LiveSQLBench-Large share questions). Both rules are met exactly by splitting connected
  components: questions are linked if they share a gold set (same dataset) or a question text.
- The split depends only on the seed.
"""
from __future__ import annotations

import collections
import random

import numpy as np

from .data import Dataset

PARTS = ("train", "val", "test")


def _components(sets: dict[str, Dataset], names: list[str]) -> list[list[tuple[str, int]]]:
    """Union-find over one customer's questions: linked if same dataset and same gold set, or same question text."""
    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    first: dict = {}
    for k in names:
        for i, (_, q, g) in enumerate(sets[k].Q):
            node = (k, i)
            find(node)
            for key in (("gold", k, frozenset(g)), ("text", q)):
                if key in first:
                    union(node, first[key])
                else:
                    first[key] = node
    comp: dict = collections.defaultdict(list)
    for node in parent:
        comp[find(node)].append(node)
    return [sorted(v) for v in comp.values()]


def make_split(sets: dict[str, Dataset], cfg: dict, seed: int) -> dict[str, dict[str, list[int]]]:
    frac = {p: cfg["split"][p] for p in PARTS}
    split: dict[str, dict[str, list[int]]] = {}
    for customer, names in cfg["customers"].items():
        comps = sorted(_components(sets, names))
        random.Random(seed).shuffle(comps)
        n = {k: len(sets[k].Q) for k in names}
        part = {k: {p: [] for p in PARTS} for k in names}
        for comp in comps:
            ks = sorted({k for k, _ in comp})
            fill = lambda p: np.mean([len(part[k][p]) / (frac[p] * n[k]) for k in ks])  # noqa: E731
            dest = "test" if fill("test") < 1 else ("val" if fill("val") < 1 else "train")
            for k, i in comp:
                part[k][dest].append(i)
        for k in names:
            split[k] = {p: sorted(v) for p, v in part[k].items()}
    check_split(sets, split, cfg)
    return split


def subsample_train(sets: dict[str, Dataset], split: dict, cfg: dict, seed: int) -> dict:
    """Keep only a share of the TRAIN part (config split.train_use, default 1.0 = all of it).

    Validation and test are untouched, so a smaller-training run is scored on exactly the same test questions.
    Whole gold-set groups are dropped together. Dropping training questions cannot create leakage.
    """
    use = cfg["split"].get("train_use", 1.0)
    if use >= 1.0:
        return split
    for k, D in sets.items():
        groups: dict = collections.defaultdict(list)
        for i in split[k]["train"]:
            groups[frozenset(D.Q[i][2])].append(i)
        keys = sorted(groups, key=lambda x: sorted(x))
        random.Random(seed + 1000).shuffle(keys)
        kept, target = [], use * len(split[k]["train"])
        for g in keys:
            if len(kept) >= target:
                break
            kept += groups[g]
        split[k]["train"] = sorted(kept)
    return split


def check_split(sets: dict[str, Dataset], split: dict, cfg: dict) -> None:
    """Hard checks: every question in exactly one part, no gold set in two parts, shared question texts on one side."""
    for k, D in sets.items():
        allq = sum((split[k][p] for p in PARTS), [])
        assert sorted(allq) == list(range(len(D.Q))), f"{k}: questions missing or duplicated across parts"
        owner: dict[frozenset, str] = {}
        for p in PARTS:
            for i in split[k][p]:
                g = frozenset(D.Q[i][2])
                assert owner.setdefault(g, p) == p, f"{k}: gold set {sorted(g)} is in both {owner[g]} and {p}"
    for names in cfg["customers"].values():
        side: dict[str, str] = {}
        for k in names:
            for p in PARTS:
                for i in split[k][p]:
                    q = sets[k].Q[i][1]
                    assert side.setdefault(q, p) == p, f"question text in two parts: {q[:80]!r}"


def split_report(sets: dict[str, Dataset], split: dict) -> None:
    for k, D in sets.items():
        s = split[k]
        trt = set().union(*[D.Q[i][2] for i in s["train"]]) if s["train"] else set()
        unseen = np.mean([len(D.Q[i][2] - trt) > 0 for i in s["test"]]) if s["test"] else 0.0
        ngrp = {p: len({frozenset(D.Q[i][2]) for i in s[p]}) for p in PARTS}
        big = max(collections.Counter(frozenset(D.Q[i][2]) for i in s["test"]).values(), default=0)
        print(f"{k:20s} " + " | ".join(f"{p} {len(s[p]):4d} q / {ngrp[p]:3d} sets" for p in PARTS)
              + f" | test needing a table never gold in train: {100 * unseen:.1f}% | largest test set group: {big} q")
