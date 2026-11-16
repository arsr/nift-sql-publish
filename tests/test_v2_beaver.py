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


def test_step1_and_step2_end_to_end(tmp_path):
    """Both scripts on a tiny catalog with a hash embedder and a tiny random cross-encoder (wiring only, no accuracy)."""
    import random
    import subprocess
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    rng = random.Random(0)
    data = tmp_path / "data"; data.mkdir()
    vocab = "student room building course term port subnet tenant instance host".split()
    for db, n in (("dw", 12), ("nova", 10), ("neutron", 10)):
        T = [dict(db=db, table_name=f"{rng.choice(vocab).upper()}_{t}", column_names=json.dumps(["ID", "NAME"]),
                  column_types="[]", example_rows="[]", example_columns="[]") for t in range(n)]
        pd.DataFrame(T).to_parquet(data / f"table_{db}.parquet")
        for split, m in ((db, 40),) + ((("dw_real", 6),) if db == "dw" else ()):
            Q = []
            for i in range(m):
                names = [x["table_name"] for x in rng.sample(T, rng.randint(1, 4))]
                Q.append(dict(id=f"{split}_{i}", category="c", db=db, question=" ".join(x.split("_")[0].lower() for x in names),
                              sql=f"select {split} {i}", tables=json.dumps(names),
                              join_keys=json.dumps([[f"{a}.ID", f"{b}.ID"] for a, b in zip(names, names[1:])]),
                              column_mapping=json.dumps({"x": [f"{names[0]}.NAME"]}), domain_knowledge="[]"))
            pd.DataFrame(Q).to_parquet(data / f"query_{split}.parquet")
    out = tmp_path / "out"
    run = lambda *a: subprocess.run([sys.executable, *a], cwd=repo, capture_output=True, text=True)   # noqa: E731
    p = run("v2/main.py", "--data", str(data), "--out", str(out), "--dense", "fake")
    assert p.returncode == 0, p.stderr[-2000:]
    p = run("v2/step2.py", "--data", str(data), "--out", str(out), "--model", "tiny", "--epochs", "1", "--top-n", "10", "--p2-scope", "0")
    assert p.returncode == 0, p.stderr[-3000:]
    txt = (out / "step2_seed1" / "STEP2.txt").read_text()
    assert "pass 2, all fields, N=all minus pass 2, all fields, N=10" in txt and "fuser: search + pass 1 + pass 2" in txt
    import numpy as np                                     # pass 2 is scored over every table, so any N is a truncation
    P2 = np.load(out / "step2_seed1" / "p2_all.npy")
    assert np.isfinite(P2[~np.isnan(P2).all(axis=1)]).all()
    p = run("v2/step2.py", "--data", str(data), "--out", str(out), "--model", "tiny", "--epochs", "1", "--top-n", "10",
            "--ns", "5", "--p2-scope", "20")                       # staged run: reuses the model, top-20 scope served from cache
    assert p.returncode == 0, p.stderr[-3000:]
    txt = (out / "step2_seed1" / "STEP2.txt").read_text()
    assert "pass-1 top 20" in txt and "N=all" not in txt and "N=5 " in txt


def test_directional_check(tmp_path):
    import subprocess, sys
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    data, out = _folder(tmp_path / "d"), tmp_path / "out"
    run = lambda *a: subprocess.run([sys.executable, *a], cwd=repo, capture_output=True, text=True)   # noqa: E731
    assert run("v2/main.py", "--data", str(data), "--out", str(out), "--dense", "fake").returncode == 0
    import zipfile
    with zipfile.ZipFile(data / "beaver_db.zip", "w") as z:
        for db in ("dw", "nova", "neutron"):
            ddl = "".join(f"CREATE TABLE `T{i}` (\n  `ID` int NOT NULL,\n  `C{i}` varchar(10) DEFAULT NULL,\n  PRIMARY KEY (`ID`)"
                          + (f",\n  CONSTRAINT `f{i}` FOREIGN KEY (`ID`) REFERENCES `T{i - 1}` (`ID`)" if i else "")
                          + "\n) ENGINE=InnoDB;\n" for i in range(6))
            ddl += "".join(f"INSERT INTO `T{i}` VALUES " + ",".join(f"({k},'v{i}_{k};(x)')" for k in range(60)) + ";\n" for i in range(6))
            z.writestr(f"{db}.sql", ddl)
    p = run("v2/schema_keys.py", "--data", str(data), "--out", str(out))
    assert p.returncode == 0, p.stderr[-3000:]
    assert "found in the dump 6 | with a primary key 6 | with a foreign key 5" in p.stdout, p.stdout
    p = run("v2/value_joins.py", "--data", str(data), "--out", str(out))
    assert p.returncode == 0, p.stderr[-3000:]
    assert "dw: 6 tables read (360 rows)" in p.stdout and "15 table pairs share values" in p.stdout, p.stdout
    p = run("v2/check.py", "--data", str(data), "--out", str(out), "--seeds", "1")
    assert p.returncode == 0, p.stderr[-3000:]
    txt = (out / "CHECK.txt").read_text()
    assert "c. complete join graph (upper bound)" in txt and "d. declared foreign keys (dumps): 15 edges" in txt, txt[:1500]
    assert "f. value overlap rule (dumps, no labels): 45 edges" in txt and "g. learned join decider" in txt, txt[:2500]
