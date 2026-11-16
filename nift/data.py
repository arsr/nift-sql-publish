"""Load the Snowflake schema-retrieval suite (BEAVER, LiveSQLBench, LiveSQLBench-Large) into one format.

Expected layout (the public release):
    <data_root>/<subset>/corpus.jsonl       one document per table: {"_id": "<db>__<table>", "text": "CREATE TABLE ..."}
    <data_root>/<subset>/queries.jsonl      {"_id": ..., "text": <question>}
    <data_root>/<subset>/qrels/test.tsv     query-id <tab> corpus-id <tab> score (header line first)

Cards are NAMES ONLY: "TABLE: col, col, ...". Column types, comments and sample rows in the document are ignored.
Joins come from FOREIGN KEY ... REFERENCES clauses inside the same database.
"""
from __future__ import annotations

import collections
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

SKIP = ("primary key", "foreign key", "constraint", "unique", "check", ")")
FK_RE = re.compile(r'FOREIGN KEY \(([^)]*)\) REFERENCES "?([\w]+)"?\s*\(([^)]*)\)')
# Some LiveSQLBench documents quote SQL keywords ('"CREATE" TABLE', '"PRIMARY" KEY', '"FOREIGN" KEY').
# Without this fix about 30% of LiveSQLBench tables got a card with no columns and no joins.
QUOTED_KW = [(re.compile(r'"CREATE"\s+TABLE', re.I), "CREATE TABLE"), (re.compile(r'"(PRIMARY|FOREIGN)"\s+KEY', re.I), r"\1 KEY")]


@dataclass
class Dataset:
    name: str
    cards: list[str]                       # "TABLE: col, col, ..."
    joins: dict[int, list[str]]            # table index -> ["a.x = b.y", ...]
    dbo: np.ndarray                        # database name of each table
    Q: list[tuple[str, str, set[int]]]     # (question id, question text, gold table indices)
    ids: list[str] = field(default_factory=list)
    DBI: dict[str, np.ndarray] = field(default_factory=dict)
    extra: dict = field(default_factory=dict)   # filled by later steps (search scores, Laya scores, features)
    adj: dict = field(default_factory=dict)     # join graph: table index -> set of joined table indices

    @property
    def n(self) -> int:
        return len(self.cards)


def parse_doc(doc: dict) -> tuple[str, str, list[str], list[tuple[str, str, str]]]:
    """Return (database, table, column names, foreign keys) from one corpus document."""
    db, table = doc["_id"].split("__", 1)
    txt = doc["text"]
    for rx, rep in QUOTED_KW:
        txt = rx.sub(rep, txt)
    cols, fks = [], []
    if "CREATE TABLE" in txt:
        body = txt.split("CREATE TABLE", 1)[1].split("(", 1)[1]
        depth, buf = 1, ""
        for ch in body:
            depth += (ch == "(") - (ch == ")")
            if depth == 0:
                break
            buf += ch
        for line in buf.split("\n"):
            s = line.strip().rstrip(",")
            if not s or s.lower().startswith(SKIP) or s.startswith("--"):
                continue
            cols.append(s.split()[0].strip('"'))
        for a, rt, rc in FK_RE.findall(txt):
            fks.append((a.strip().strip('"'), rt, rc.strip().strip('"')))
    else:  # pipe-separated header fallback
        cols = [c.strip() for c in txt.split("\n", 1)[0].strip().strip("|").split("|") if c.strip()]
    return db, table, cols, fks


def _read_jsonl(p: Path) -> list[dict]:
    with open(p) as f:
        return [json.loads(line) for line in tqdm(f, desc=f"Loading {p.name}", unit="line", leave=False) if line.strip()]


def load_subset(data_root: str | Path, name: str, subset: str) -> Dataset:
    root = Path(data_root) / subset
    corpus, queries = _read_jsonl(root / "corpus.jsonl"), _read_jsonl(root / "queries.jsonl")
    qrels: dict[str, set[str]] = collections.defaultdict(set)
    with open(root / "qrels" / "test.tsv") as f:
        next(f, None)
        for line in tqdm(f, desc=f"{name}: loading qrels", unit="line", leave=False):
            q, c, s = line.rstrip("\n").split("\t")
            if float(s) > 0:
                qrels[q].add(c)
    ids = [d["_id"] for d in corpus]
    pos = {x: k for k, x in enumerate(ids)}
    parsed = [parse_doc(d) for d in tqdm(corpus, desc=f"{name}: parsing tables", unit="table", leave=False)]
    cards = [f"{t}: " + ", ".join(c) for _, t, c, _ in parsed]
    by_name = {(db, t.lower()): k for k, (db, t, _, _) in enumerate(parsed)}   # REFERENCES may use another case
    joins: dict[int, list[str]] = collections.defaultdict(list)
    for k, (db, t, _, fks) in enumerate(parsed):
        for a, rt, rc in fks:
            j = by_name.get((db, rt.lower()))
            if j is not None:
                joins[k].append(f"{t}.{a} = {rt}.{rc}")
                joins[j].append(f"{rt}.{rc} = {t}.{a}")
    Q = [(q["_id"], q["text"], {pos[c] for c in qrels[q["_id"]] if c in pos}) for q in queries if qrels.get(q["_id"])]
    Q = [x for x in Q if x[2]]
    dbo = np.array([db for db, _, _, _ in parsed])
    ds = Dataset(name=name, cards=cards, joins=dict(joins), dbo=dbo, Q=Q, ids=ids)
    ds.DBI = {d: np.flatnonzero(dbo == d) for d in sorted(set(dbo))}
    return ds


def load_all(data_root: str | Path, cfg: dict) -> dict[str, Dataset]:
    sets = {name: load_subset(data_root, name, sub) for name, sub in cfg["datasets"].items()}
    for D in sets.values():
        if cfg["card"].get("infer_joins"):
            infer_joins(D, cfg["card"].get("infer_max_tables", 8))
        D.adj = adjacency(D)
    return sets


def adjacency(D: Dataset) -> dict[int, set[int]]:
    """Join graph from the join strings 'a.x = b.y' (declared or inferred), inside one database."""
    idx = {(D.dbo[t], D.cards[t].split(":")[0].lower()): t for t in range(D.n)}
    adj: dict[int, set[int]] = collections.defaultdict(set)
    for t, js in D.joins.items():
        for j in js:
            u = idx.get((D.dbo[t], j.split("=", 1)[1].strip().split(".")[0].lower()))
            if u is not None and u != t:
                adj[t].add(u)
                adj[u].add(t)
    return dict(adj)


GENERIC_COLS = {"id", "key", "name", "code", "type", "status", "version", "url", "deleted", "description", "notes",
                "created_at", "updated_at", "deleted_at", "warehouse_load_date"}
KEYLIKE = re.compile(r"(key|id|code|no|num|nbr|ref|registry)$", re.I)


def infer_joins(D: Dataset, max_tables: int = 8) -> int:
    """Names-only join inference for schemas without declared keys (BEAVER has none).

    Two tables in the same database are linked when they share a key-like column name (ends in key, id, code, no,
    num, nbr, ref, registry), the name is not generic, and at most `max_tables` tables carry it. Marked '(inferred)'.
    Only pairs that are not already joined are added. Returns the number of new links.
    """
    names = [c.split(":", 1)[0] for c in D.cards]
    cols = [{x.strip() for x in c.split(":", 1)[1].split(",") if x.strip()} for c in D.cards]
    linked = {(a, b) for a, bs in adjacency(D).items() for b in bs}
    added = 0
    for ix in D.DBI.values():
        by: dict[str, list] = collections.defaultdict(list)
        for t in ix:
            for c in sorted(cols[t]):
                if c.lower() not in GENERIC_COLS and KEYLIKE.search(c):
                    by[c.lower()].append((int(t), c))
        for _, ts in sorted(by.items()):
            if not 2 <= len(ts) <= max_tables:
                continue
            for x, (a, ca) in enumerate(ts):
                for b, cb in ts[x + 1:]:
                    if (a, b) in linked:
                        continue
                    D.joins.setdefault(a, []).append(f"{names[a]}.{ca} = {names[b]}.{cb} (inferred)")
                    D.joins.setdefault(b, []).append(f"{names[b]}.{cb} = {names[a]}.{ca} (inferred)")
                    linked |= {(a, b), (b, a)}
                    added += 1
    return added


def summary(sets: dict[str, Dataset]) -> None:
    for k, D in sets.items():
        empty = sum(1 for c in D.cards if not c.split(":", 1)[1].strip())
        nj = sum(1 for t in range(D.n) if D.joins.get(t))
        ninf = sum(j.endswith("(inferred)") for js in D.joins.values() for j in js) // 2
        print(f"{k:20s} cards with NO columns: {empty} of {D.n} | tables with at least one join: {nj} | inferred links: {ninf}")
        ng = np.array([len(g) for _, _, g in D.Q])
        print(f"{k:20s} {D.n:5d} tables, {len(D.DBI):3d} databases, {len(D.Q):4d} questions | tables per SQL: mean {ng.mean():.2f}, "
              f"3+ {100 * np.mean(ng >= 3):.0f}%, 5+ {100 * np.mean(ng >= 5):.0f}% | {len({frozenset(g) for _, _, g in D.Q})} distinct gold sets")


DESC_PATTERNS = {
    "COMMENT clause": re.compile(r"\bCOMMENT\b", re.I),
    "-- comment line": re.compile(r"^\s*--", re.M),
    "/* block comment */": re.compile(r"/\*"),
    "'description' word": re.compile(r"\bdescription\b", re.I),
    "sample rows (INSERT / pipe table)": re.compile(r"\bINSERT INTO\b|^\s*\|.*\|\s*$", re.I | re.M),
}


def description_report(data_root: str | Path, cfg: dict, show: int = 1) -> dict:
    """Answer 'is there a description for each table?' by scanning the raw corpus documents.

    Counts documents that contain anything beyond names: comments, descriptions, sample rows, or extra JSON fields.
    """
    rep = {}
    for name, sub in cfg["datasets"].items():
        docs = _read_jsonl(Path(data_root) / sub / "corpus.jsonl")
        keys = collections.Counter(k for d in docs for k in d if k not in ("_id", "text"))
        hits = {lab: sum(bool(rx.search(d["text"])) for d in docs) for lab, rx in DESC_PATTERNS.items()}
        rep[name] = dict(docs=len(docs), extra_json_fields=dict(keys), **hits)
        print(f"\n{name}: {len(docs)} table documents | extra JSON fields: {dict(keys) or 'none'}")
        for lab, n in hits.items():
            print(f"   {lab:36s} {n:5d} docs ({100 * n / max(len(docs), 1):.0f}%)")
        for d in docs[:show]:
            print("   example document (first 600 chars):\n      " + d["text"][:600].replace("\n", "\n      "))
    return rep
