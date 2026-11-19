"""Read the declared keys of every BEAVER table from the database dumps (CPU). Read top to bottom.

    python v2/schema_keys.py --data <BEAVER_DIR> --out <KEYS_DIR>

The dumps (beaver_db.zip: dw.sql, nova.sql, neutron.sql, or the .sql files themselves) are searched for under --data.
Only the CREATE TABLE and ALTER TABLE statements are read; the dumps are scanned in 64 MB chunks, so the 2 GB nova dump
with its long INSERT lines never sits in memory. Writes <out>/schema_keys.json:
    {db: {TABLE: {"cols": [...], "pk": [...], "fk": [[[cols], REF_TABLE, [ref cols]], ...]}}}
and prints how many catalog tables were found, and how many declare a primary key or a foreign key.
No labels are used: this is what any customer's database already states about itself.
"""
import argparse
import glob
import json
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v2 import beaver     # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

STMT = re.compile(rb"(CREATE\s+TABLE|ALTER\s+TABLE)\b[^;]*;", re.I | re.S)
NAME = re.compile(r"""^(?:CREATE|ALTER)\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+|ONLY\s+)?([`"\[\]\w.$#]+)""", re.I)
PK = re.compile(r"PRIMARY\s+KEY\s*\(([^)]*)\)", re.I)
FK = re.compile(r"FOREIGN\s+KEY\s*\(([^)]*)\)\s*REFERENCES\s+([`\"\[\]\w.$#]+)\s*\(([^)]*)\)", re.I)
INLINE_PK = re.compile(r"""^\s*([`"\[]?[\w$#]+[`"\]]?)\s+[^,]*\bPRIMARY\s+KEY\b""", re.I)
SKIP_COL = re.compile(r"^\s*(PRIMARY|KEY|UNIQUE|INDEX|CONSTRAINT|FOREIGN|FULLTEXT|CHECK|SPATIAL)\b", re.I)


def clean(x):
    return re.sub(r"""[`"\[\]]""", "", x).split(".")[-1].strip().upper()


def cols_of(x):
    return [clean(c.split("(")[0]) for c in x.split(",") if c.strip()]


def statements(stream, chunk=64 << 20, keep=4 << 20):
    """Every CREATE/ALTER TABLE statement of a byte stream, scanning fixed-size chunks with an overlap."""
    tail, seen = b"", set()
    while True:
        block = stream.read(chunk)
        buf = tail + block
        for m in STMT.finditer(buf):
            s = m.group(0)
            if len(s) < 2_000_000 and s not in seen:
                seen.add(s); yield s.decode("utf-8", "replace")
        if not block:
            return
        tail = buf[-keep:]


def parse(stmts):
    T = {}
    for s in stmts:
        m = NAME.match(s.strip())
        if not m:
            continue
        t = T.setdefault(clean(m.group(1)), {"cols": [], "pk": [], "fk": []})
        if s.strip().upper().startswith("CREATE"):
            body = s[s.find("(") + 1: s.rfind(")")]
            depth, cur, parts = 0, "", []
            for ch in body:                                   # split the body on top-level commas
                depth += (ch == "(") - (ch == ")")
                if ch == "," and depth == 0:
                    parts.append(cur); cur = ""
                else:
                    cur += ch
            parts.append(cur)
            for p in parts:
                if not SKIP_COL.match(p) and p.strip():
                    t["cols"].append(clean(p.strip().split()[0]))
                    if INLINE_PK.match(p):
                        t["pk"].append(clean(p.strip().split()[0]))
        for pk in PK.findall(s):
            t["pk"] = list(dict.fromkeys(t["pk"] + cols_of(pk)))
        for a, ref, b in FK.findall(s):
            t["fk"].append([cols_of(a), clean(ref), cols_of(b)])
    return T


dumps = {}
for f in sorted(glob.glob(f"{args.data}/**/*.zip", recursive=True)):
    with zipfile.ZipFile(f) as z:
        for n in z.namelist():
            db = Path(n).stem.lower()
            if n.endswith(".sql") and db in beaver.DBS:
                with z.open(n) as fh:
                    dumps[db] = parse(statements(fh)); print(f"{f} : {n} -> {len(dumps[db])} tables", flush=True)
for f in sorted(glob.glob(f"{args.data}/**/*.sql", recursive=True)):
    db = Path(f).stem.lower()
    if db in beaver.DBS and db not in dumps:
        with open(f, "rb") as fh:
            dumps[db] = parse(statements(fh)); print(f"{f} -> {len(dumps[db])} tables", flush=True)
if not dumps:
    sys.exit(f"no dumps (dw.sql, nova.sql, neutron.sql or a zip of them) under {args.data}")

cat, _, _ = beaver.load(args.data)
for db, T in dumps.items():
    names = {n.split(".", 1)[1] for n in cat.names if cat.db[cat.index[n]] == db}
    found = names & set(T)
    print(f"{db}: catalog {len(names)} tables, found in the dump {len(found)} | with a primary key "
          f"{sum(bool(T[t]['pk']) for t in found)} | with a foreign key {sum(bool(T[t]['fk']) for t in found)} | "
          f"foreign keys {sum(len(T[t]['fk']) for t in found)}")
Path(args.out).mkdir(parents=True, exist_ok=True)
(Path(args.out) / "schema_keys.json").write_text(json.dumps(dumps))
print("wrote", Path(args.out) / "schema_keys.json")
