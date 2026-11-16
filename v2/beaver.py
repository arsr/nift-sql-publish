"""BEAVER (current public release) for the v2 experiments: tables, questions, labels, splits, join graph.

Input: the parquet files written by colab/beaver_v3_* cells into one folder:
  table_dw.parquet, table_nova.parquet, table_neutron.parquet     (one row per table: db, table_name, column_names, ...)
  query_dw.parquet, query_nova.parquet, query_neutron.parquet     (synthetic, expert-verified questions)
  query_dw_real.parquet                                           (121 real questions from query logs; test only)
Everything here is deterministic and CPU only.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DBS = ("dw", "nova", "neutron")
JSON_FIELDS = ("tables", "join_keys", "column_mapping", "domain_knowledge")


@dataclass
class Catalog:
    """Every table of the three databases, pooled (the main setting gives no database identifier)."""
    names: list[str]                     # "db.TABLE"
    db: list[str]                        # database of each table
    columns: list[list[str]]             # column names of each table
    index: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        self.index = {n: i for i, n in enumerate(self.names)}

    @property
    def n(self) -> int:
        return len(self.names)

    def card(self, t: int, neighbours: list[str] | None = None, max_chars: int = 700) -> str:
        """Names and keys only: 'TABLE: col, col, ...' (+ optional join neighbours for the second pass)."""
        s = f"{self.names[t].split('.', 1)[1]}: " + ", ".join(self.columns[t])
        s = s[:max_chars]
        if neighbours:
            s += "\nJoins: " + ", ".join(neighbours)
        return s


@dataclass
class Question:
    qid: str
    db: str
    text: str
    sql: str
    gold: list[int]                      # catalog indices
    role: dict[int, str]                 # gold table -> "content" (a mapped column) or "bridge" (join only)
    joins: list[tuple[int, int]]         # gold join edges (catalog indices)
    real: bool
    category: str


def _parse(v):
    if isinstance(v, str) and v[:1] in "[{":
        return json.loads(v)
    return v


def load(folder: str | Path) -> tuple[Catalog, list[Question], dict]:
    folder = Path(folder)
    names, dbs, cols = [], [], []
    for db in DBS:
        T = pd.read_parquet(folder / f"table_{db}.parquet")
        for _, r in T.iterrows():
            names.append(f"{db}.{str(r['table_name']).upper()}")
            dbs.append(db)
            cols.append([str(c) for c in _parse(r["column_names"])])
    cat = Catalog(names, dbs, cols)
    qs, missing = [], 0
    for split in DBS + ("dw_real",):
        Q = pd.read_parquet(folder / f"query_{split}.parquet")
        for _, r in Q.iterrows():
            db = str(r["db"])
            tabs = [f"{db}.{t.upper()}" for t in _parse(r["tables"]) or []]
            if any(t not in cat.index for t in tabs) or not tabs:
                missing += 1
                continue
            gold = [cat.index[t] for t in tabs]
            cm = _parse(r["column_mapping"]) or {}
            mapped = {f"{db}.{c.split('.', 1)[0].upper()}" for cs in cm.values() for c in (cs or []) if "." in c}
            role = {cat.index[t]: ("content" if t in mapped else "bridge") for t in tabs}
            jk = _parse(r["join_keys"]) or []
            joins = []
            for pair in jk:
                a, b = (f"{db}.{x.split('.', 1)[0].upper()}" for x in pair[:2])
                if a in cat.index and b in cat.index and a != b:
                    joins.append((cat.index[a], cat.index[b]))
            qs.append(Question(str(r["id"]), db, str(r["question"]), str(r["sql"]), gold, role, joins,
                               split == "dw_real", str(r.get("category"))))
    return cat, qs, {"questions dropped (gold table not in the public schema)": missing}


def _norm_sql(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def split(qs: list[Question], seed: int, frac=(0.5, 0.3, 0.2)) -> dict[str, list[int]]:
    """Per database, grouped: questions with the same gold-table set or the same SQL never cross parts.
    The real questions (dw_real) are test only, a separate 'real' part."""
    parent = list(range(len(qs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    first: dict = {}
    for i, q in enumerate(qs):
        if q.real:
            continue
        for key in (("set", q.db, frozenset(q.gold)), ("sql", q.db, _norm_sql(q.sql))):
            if key in first:
                parent[find(i)] = find(first[key])
            else:
                first[key] = i
    out = {"train": [], "val": [], "test": [], "real": [i for i, q in enumerate(qs) if q.real]}
    rng = np.random.default_rng(seed)
    for db in DBS:
        groups: dict = {}
        for i, q in enumerate(qs):
            if q.db == db and not q.real:
                groups.setdefault(find(i), []).append(i)
        keys = sorted(groups, key=lambda g: min(groups[g]))
        rng.shuffle(keys)
        n = sum(len(groups[g]) for g in keys)
        acc = 0
        for g in keys:                                       # fill train, then val, then test by question count
            part = "train" if acc < frac[0] * n else "val" if acc < (frac[0] + frac[1]) * n else "test"
            out[part] += groups[g]; acc += len(groups[g])
    return out


def join_graph(qs: list[Question], idx: list[int], n: int) -> dict[int, set[int]]:
    """Tables this customer has joined in the given (training) questions."""
    adj: dict[int, set[int]] = {t: set() for t in range(n)}
    for i in idx:
        for a, b in qs[i].joins:
            adj[a].add(b); adj[b].add(a)
    return adj


def report(cat: Catalog, qs: list[Question], S: dict[str, list[int]], extra: dict) -> str:
    L = [f"catalog: {cat.n} tables | " + ", ".join(f"{db} {cat.db.count(db)}" for db in DBS), json.dumps(extra)]
    train_gold = {t for i in S["train"] for t in qs[i].gold}
    G = join_graph(qs, S["train"], cat.n)
    for part in ("train", "val", "test", "real"):
        ix = S[part]
        if not ix:
            continue
        nt = np.array([len(qs[i].gold) for i in ix])
        br = np.mean([r == "bridge" for i in ix for r in qs[i].role.values()])
        unseen = np.mean([any(t not in train_gold for t in qs[i].gold) for i in ix])
        jk = [(a, b) for i in ix for a, b in qs[i].joins]
        seen_j = np.mean([b in G[a] for a, b in jk]) if jk else float("nan")
        per_db = ", ".join(f"{db} {sum(qs[i].db == db for i in ix)}" for db in DBS)
        L.append(f"{part:5s} {len(ix):5d} q ({per_db}) | tables/q {nt.mean():.2f}, 5+ {100 * (nt >= 5).mean():.1f}% | "
                 f"bridge share of gold {100 * br:.1f}% | needs a table never gold in train {100 * unseen:.1f}% | "
                 f"gold joins already in the train graph {100 * seen_j:.1f}%")
    for db in DBS:
        sets = pd.Series([frozenset(q.gold) for q in qs if q.db == db and not q.real]).value_counts()
        L.append(f"{db}: {len(sets)} distinct gold-table sets; largest holds {sets.iloc[0] if len(sets) else 0} questions "
                 f"({100 * sets.iloc[0] / max(1, sets.sum()):.1f}%)")
    dup = len(qs) - len({(q.db, _norm_sql(q.sql)) for q in qs})
    L.append(f"questions sharing their exact SQL with another question: {dup}")
    return "\n".join(L)
