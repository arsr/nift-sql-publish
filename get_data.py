"""Download the data the paper uses, into the layout the code expects. Read top to bottom. CPU only.

    python get_data.py --out <DATA_ROOT>                          # the three collections of the main experiments
    python get_data.py --beaver <BEAVER_DIR>                      # the newer BEAVER release (holdout, Section 9.7)

1  Main experiments. The table-retrieval versions of BEAVER, LiveSQLBench and LiveSQLBench-Large from the SQL schema
   retrieval benchmark of Zeng et al. (2026), Hugging Face dataset Snowflake/SQL-Schema-Retrieval, pinned to a fixed
   revision. Writes <DATA_ROOT>/{beaver, livesqlbench_table, livesqlbench_large_table}/ with corpus.jsonl,
   queries.jsonl and qrels/test.tsv, then checks the table and question counts against the paper (463/209, 244/410,
   971/332); a different count stops the script, because the splits and every number depend on these files.
2  Holdout. The current public BEAVER release, Hugging Face datasets beaverbench/beaver-query and beaverbench/beaver-table.
   Both are gated: accept their terms on huggingface.co with your account and set HF_TOKEN (read access). The parquet
   files are copied flat into <BEAVER_DIR> (table_dw.parquet, ..., query_dw_real.parquet), as
   experiments/confirm_beaver_v3.py expects; the database dumps stay in their subfolder for v2/schema_keys.py.
Each benchmark has its own licence; follow it. Nothing downloaded here is committed (see .gitignore).
"""
import argparse
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

SCHEMA_RETRIEVAL = "Snowflake/SQL-Schema-Retrieval"
REVISION = "e60c65cf9aeab052f0747902d31db1807099fa23"
EXPECTED = {"beaver": (463, 209), "livesqlbench_table": (244, 410), "livesqlbench_large_table": (971, 332)}
BEAVER_FILES = ["table_dw.parquet", "table_nova.parquet", "table_neutron.parquet",
                "query_dw.parquet", "query_nova.parquet", "query_neutron.parquet", "query_dw_real.parquet"]

ap = argparse.ArgumentParser()
ap.add_argument("--out", help="DATA_ROOT for the main experiments")
ap.add_argument("--beaver", help="BEAVER_DIR for the holdout (needs HF_TOKEN and the accepted terms)")
args = ap.parse_args()
if not (args.out or args.beaver):
    ap.error("give --out, --beaver, or both")


def count_lines(f: Path) -> int:
    with open(f) as h:
        return sum(1 for line in h if line.strip())


# ---------------------------------------------------------------- 1  main experiments
if args.out:
    root = Path(args.out)
    snapshot_download(SCHEMA_RETRIEVAL, repo_type="dataset", revision=REVISION, local_dir=str(root),
                      allow_patterns=[f"{name}/*" for name in EXPECTED] + [f"{name}/qrels/*" for name in EXPECTED])
    for name, (n_tables, n_questions) in EXPECTED.items():
        d = root / name
        missing = [f for f in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv") if not (d / f).exists()]
        if missing:
            raise SystemExit(f"{d}: missing {missing}")
        got = (count_lines(d / "corpus.jsonl"), count_lines(d / "queries.jsonl"))
        if got != (n_tables, n_questions):
            raise SystemExit(f"{name}: {got[0]} tables and {got[1]} questions, the paper used {n_tables} and {n_questions}")
        print(f"{name:26s} {got[0]:4d} tables {got[1]:4d} questions  ok")
    print(f"data ready: python reproduce_paper.py --data {root} --out <OUT_ROOT>")

# ---------------------------------------------------------------- 2  holdout: the newer BEAVER release
if args.beaver:
    dst = Path(args.beaver)
    token = os.environ.get("HF_TOKEN")
    for repo in ("beaverbench/beaver-query", "beaverbench/beaver-table"):
        snapshot_download(repo, repo_type="dataset", token=token, local_dir=str(dst / repo.split("/")[1]))
    found = {p.name: p for p in sorted(dst.rglob("*.parquet")) if p.parent != dst}
    missing = [f for f in BEAVER_FILES if f not in found]
    if missing:
        raise SystemExit(f"not found in the download: {missing}; parquet files present: {sorted(found)}")
    for f in BEAVER_FILES:
        shutil.copy2(found[f], dst / f)
    import pandas as pd
    sizes = {f: len(pd.read_parquet(dst / f)) for f in BEAVER_FILES}
    print(json.dumps(sizes, indent=1))
    n_tables = sum(v for f, v in sizes.items() if f.startswith("table_"))
    if n_tables != 381:
        raise SystemExit(f"{n_tables} tables; the paper's holdout catalog has 381")
    print("381 tables  ok. Question counts: confirm_beaver_v3.py prints how many it keeps (questions whose gold tables are "
          "not in the public schema are dropped); the paper's holdout has 7,857 synthetic and 121 real questions.")
    print(f"holdout data ready: python reproduce_paper.py --data <DATA_ROOT> --out <OUT_ROOT> --stages holdout --beaver {dst}")
