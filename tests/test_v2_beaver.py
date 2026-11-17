"""v2/beaver.py on a tiny synthetic BEAVER folder: loading, labels, and the grouped split."""
import json

import pandas as pd

from v2 import beaver


def _folder(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    for db in beaver.DBS:
        pd.DataFrame([dict(db=db, table_name=f"T{i}", column_names=json.dumps(["ID", f"C{i}"]), column_types="[]",
                           example_rows="[]", example_columns="[]") for i in range(6)]).to_parquet(tmp_path / f"table_{db}.parquet")
        rows = []
        for i in range(40):
            tabs = [f"T{i % 5}", f"T{(i + 1) % 5}"]
            rows.append(dict(id=f"{db}_{i}", category="complex query", db=db, question=f"q {i}", sql=f"select {i % 7}",
                             tables=json.dumps(tabs), join_keys=json.dumps([[f"{tabs[0]}.ID", f"{tabs[1]}.ID"]]),
                             column_mapping=json.dumps({"x": [f"{tabs[0]}.C1"]}), domain_knowledge="[]"))
        pd.DataFrame(rows).to_parquet(tmp_path / f"query_{db}.parquet")
    pd.DataFrame([dict(id="r0", category="real", db="dw", question="real q", sql="select r", tables=json.dumps(["T5", "NOPE"]),
                       join_keys="[]", column_mapping="{}", domain_knowledge="[]"),
                  dict(id="r1", category="real", db="dw", question="real q2", sql="select r2", tables=json.dumps(["T5"]),
                       join_keys="[]", column_mapping="{}", domain_knowledge="[]")]).to_parquet(tmp_path / "query_dw_real.parquet")
    return tmp_path


def test_load_split_and_labels(tmp_path):
    cat, qs, extra = beaver.load(_folder(tmp_path))
    assert cat.n == 18 and extra["questions dropped (gold table not in the public schema)"] == 1
    q = qs[0]
    assert {cat.names[t] for t in q.gold} == {"dw.T0", "dw.T1"} and q.joins
    assert sorted(q.role.values()) == ["bridge", "content"]          # only T0 has a mapped column
    S = beaver.split(qs, seed=1)
    assert [qs[i].qid for i in S["real"]] == ["r1"] and not set(S["real"]) & set(S["train"] + S["val"] + S["test"])
    side = {}
    for part in ("train", "val", "test"):
        for i in S[part]:
            for key in (frozenset(qs[i].gold), qs[i].sql):
                assert side.setdefault((qs[i].db, key), part) == part, "a gold set or an SQL crosses parts"
    assert len(S["train"]) + len(S["val"]) + len(S["test"]) == 120
