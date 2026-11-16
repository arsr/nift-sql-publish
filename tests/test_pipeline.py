import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import fake_data
from nift import config as C
from nift import data, evaluate, fusion, scorer, search, split
from nift.data import Dataset
from nift.pipeline import METHODS, Run

REPO = Path(__file__).resolve().parents[1]


def test_parse_names_only():
    doc = {"_id": "db__ORD", "text": 'CREATE TABLE "ORD" (\n "ID" NUMBER(10),\n "CUST" VARCHAR(5) COMMENT \'x\',\n -- note\n'
                                     ' PRIMARY KEY (ID),\n FOREIGN KEY (CUST) REFERENCES "CUS" (ID)\n);'}
    db, t, cols, fks = data.parse_doc(doc)
    assert (db, t, cols, fks) == ("db", "ORD", ["ID", "CUST"], [("CUST", "CUS", "ID")])


def test_parse_pipe_separated_header():
    doc = {"_id": "db__ORD", "text": "| ORD_ID | CUST_ID | AMOUNT |\n| 1 | 2 | 3 |"}
    assert data.parse_doc(doc) == ("db", "ORD", ["ORD_ID", "CUST_ID", "AMOUNT"], [])


def test_load_subset_builds_database_indexes_and_joins(data_root):
    D = data.load_subset(data_root, "beaver", "beaver")

    assert D.n == 36
    assert len(D.Q) > 0
    assert D.cards[0].startswith("T00") and ": " in D.cards[0]
    assert set(D.DBI) == set(D.dbo)
    assert sum(map(len, D.DBI.values())) == D.n
    assert D.joins
    assert all(" = " in join for joins in D.joins.values() for join in joins)


def test_split_rules(data_root):
    cfg = C.load_config()
    sets = data.load_all(data_root, cfg)
    s1, s1b, s2 = (split.make_split(sets, cfg, s) for s in (1, 1, 2))
    assert s1 == s1b, "same seed must give the same split"
    assert s1 != s2, "different seeds must give different splits"
    a, b = sets["livesqlbench"], sets["livesqlbench_large"]
    side = {a.Q[i][1]: p for p in split.PARTS for i in s1["livesqlbench"][p]}
    for p in split.PARTS:
        for i in s1["livesqlbench_large"][p]:
            q = b.Q[i][1]
            if q in side:
                assert side[q] == p, "a question shared by the two LiveSQLBench sets landed on different sides"
    for k, D in sets.items():
        n = len(D.Q)
        assert abs(len(s1[k]["test"]) / n - 0.2) < 0.15 and len(s1[k]["train"]) > 0


def test_hybrid_scores_and_embedding_cache(tmp_path):
    D = Dataset(
        name="mini",
        cards=["CUSTOMER: CUST_ID, REGION", "ORDERS: ORDER_ID, CUST_ID", "PRODUCT: PRODUCT_ID, DESCRIPTION"],
        joins={},
        dbo=np.array(["db", "db", "other"]),
        Q=[("q1", "find customer region", {0}), ("q2", "order customer", {1})],
    )
    cfg = C.load_config()
    cache = tmp_path / "embeddings.npz"
    calls = []

    def embed(texts):
        calls.append(list(texts))
        return fake_data.hash_embedder(texts)

    search.hybrid_scores(D, embed, cfg, cache)
    first = {key: value.copy() for key, value in D.extra.items()}
    cards = fake_data.hash_embedder(D.cards)
    questions = fake_data.hash_embedder([q for _, q, _ in D.Q])
    dense = questions @ cards.T
    expected_dense = np.stack([search.minmax(row) for row in dense])
    expected_hybrid = cfg["search"]["w_dense"] * expected_dense + cfg["search"]["w_bm25"] * first["bm25"]
    np.testing.assert_allclose(first["dense"], dense)
    np.testing.assert_allclose(first["hyb"], expected_hybrid)
    assert cache.exists()

    def fail_if_called(_texts):
        pytest.fail("embedding cache miss unexpectedly")

    search.hybrid_scores(D, fail_if_called, cfg, cache)
    for key in first:
        np.testing.assert_array_equal(D.extra[key], first[key])

    cache.write_text("corrupt cache")
    search.hybrid_scores(D, embed, cfg, cache)
    assert len(calls) == 4
    np.testing.assert_array_equal(D.extra["hyb"], first["hyb"])


def test_embedding_cache_key_tracks_model_and_content(tmp_path):
    D = Dataset("mini", ["TABLE: ID"], {}, np.array(["db"]), [("q1", "question", {0})])
    first = search.embedding_cache_path(tmp_path, "model-a", D)
    assert first != search.embedding_cache_path(tmp_path, "model-b", D)
    D.cards[0] = "TABLE: ID, NAME"
    assert first != search.embedding_cache_path(tmp_path, "model-a", D)


def test_embedding_cache_reused_across_runs(data_root, tmp_path, monkeypatch):
    calls = []

    def embed(texts):
        calls.append(len(texts))
        return fake_data.hash_embedder(texts)

    monkeypatch.delenv("NIFT_FAKE_EMBED", raising=False)
    monkeypatch.setattr("nift.pipeline.search.bge_embedder", lambda model, device: embed)
    first = Run(1, "cachetest", str(data_root), str(tmp_path), device="cpu")
    first.load(check_descriptions=False)
    first.search()
    call_count = len(calls)
    expected = {k: D.extra["hyb"].copy() for k, D in first.sets.items()}

    second = Run(2, "cachetest2", str(data_root), str(tmp_path), device="cpu")
    second.load(check_descriptions=False)
    second.search()

    assert len(calls) == call_count
    for k, D in second.sets.items():
        assert np.array_equal(D.extra["hyb"], expected[k])


def test_scorer_batches_preserve_question_table_order(data_root, tmp_path):
    D = data.load_subset(data_root, "beaver", "beaver")
    ck = tmp_path / "checkpoint"
    ck.mkdir()
    (ck / "rl_agent_config.json").write_text("{}")
    cfg = C.load_config()
    qidx = [3, 0]

    cfg["scorer"]["batch_size"] = 7
    chunked = scorer.score(ck, D, qidx, cfg, "cpu")
    cfg["scorer"]["batch_size"] = 10_000
    unchunked = scorer.score(ck, D, qidx, cfg, "cpu")

    assert chunked.shape == (len(qidx), D.n)
    np.testing.assert_array_equal(chunked, unchunked)


def test_fusion_features_and_rank(data_root):
    D = data.load_subset(data_root, "beaver", "beaver")
    rng = np.random.default_rng(0)
    D.extra.update(
        dense=rng.random((len(D.Q), D.n)),
        bm25=rng.random((len(D.Q), D.n)),
        hyb=rng.random((len(D.Q), D.n)),
        P=rng.uniform(0.01, 0.99, (len(D.Q), D.n)),
    )
    cfg = C.load_config()
    examples = [fusion.features(D, i, cfg) for i in range(len(D.Q))]
    model = fusion.fit(examples, cfg)

    assert examples[0]["F"].shape == (D.n, 8)
    assert set(fusion.rank(model, examples[0])) == set(range(D.n))


def test_group_boot():
    d = np.array([1, 1, 0, 0, 1.0])
    m, lo, hi = evaluate.group_boot(d, ["a", "a", "b", "b", "c"], reps=500)
    assert m == pytest.approx(60.0) and 0 <= lo <= m <= hi <= 100
    m, lo, hi = evaluate.group_boot(np.zeros(4), list("abcd"), reps=100)
    assert (m, lo, hi) == (0, 0, 0)


@pytest.mark.parametrize("seed", [1, 2])
def test_end_to_end(data_root, tmp_path, seed):
    r = Run(seed, "tester", str(data_root), str(tmp_path), device="cpu")
    r.load(); r.split(); r.search(embed=fake_data.hash_embedder); r.build_rows(); r.train(); r.score(); r.fit_fusion()
    R = r.test()
    rd = tmp_path / "results" / f"tester_seed{seed}"
    for f in ("results.csv", "summary.txt", "manifest.json", "costs.json", "split.json", "fusion_weights.json", "config_used.yaml"):
        assert (rd / f).exists(), f
    assert set(R.method) == set(METHODS)
    assert set(R.qi[R.set == "beaver"]) == set(r.SPLIT["beaver"]["test"])
    rows = [json.loads(l) for l in open(rd / "rows" / "beaver.jsonl")]
    trq = {r.sets["beaver"].Q[i][1] for i in r.SPLIT["beaver"]["train"]}
    assert {x["state"]["question"] for x in rows} <= trq, "training rows must come from TRAIN questions only"
    with pytest.raises(FileExistsError):
        r.test()
    r2 = Run(seed, "tester", str(data_root), str(tmp_path), device="cpu")   # rerun uses the caches
    r2.load(check_descriptions=False); r2.split(); r2.search(embed=fake_data.hash_embedder); r2.build_rows(show=0); r2.train(); r2.score()
    assert np.allclose(np.nan_to_num(r2.sets["beaver"].extra["P"]), np.nan_to_num(r.sets["beaver"].extra["P"]))


def test_combine(data_root, tmp_path):
    for name, seed in (("arjun", 1), ("collab", 3)):
        r = Run(seed, name, str(data_root), str(tmp_path), device="cpu")
        r.load(check_descriptions=False); r.split(); r.search(embed=fake_data.hash_embedder); r.build_rows(show=0); r.train(); r.score(); r.fit_fusion(); r.test()
    T = evaluate.combine_runs(tmp_path, C.load_config())
    assert ("c@8", "mean") in T.columns and len(T) == 3 * len(METHODS)


def test_run_name_rule(tmp_path):
    with pytest.raises(ValueError):
        C.run_dir(tmp_path, "Arjun Rao", 1)


def test_no_secrets_or_data_in_repo():
    files = subprocess.run(["git", "-C", str(REPO), "ls-files"], capture_output=True, text=True).stdout.split() or \
        [str(p.relative_to(REPO)) for p in REPO.rglob("*") if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts]
    bad = re.compile(r"(api[_-]?key\s*[=:]\s*['\"][A-Za-z0-9]{16,}|azure\.com|openai\.azure|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,})", re.I)
    for f in files:
        p = REPO / f
        assert p.suffix not in (".jsonl", ".npy", ".pkl", ".pt", ".safetensors"), f"data or weights must not be committed: {f}"
        if p.suffix in (".py", ".md", ".yaml", ".ipynb", ".txt", ".toml", ".cfg") and p.name != "test_pipeline.py":
            assert not bad.search(p.read_text(errors="ignore")), f"possible secret in {f}"


def test_changed_config_needs_new_run_name(data_root, tmp_path):
    Run(1, "cfgtest", str(data_root), str(tmp_path), device="cpu")
    alt = tmp_path / "alt.yaml"
    alt.write_text((REPO / "configs" / "default.yaml").read_text().replace("epochs: 4", "epochs: 6"))
    with pytest.raises(RuntimeError):
        Run(1, "cfgtest", str(data_root), str(tmp_path), device="cpu", config_path=str(alt))
    Run(1, "cfgtest_ep6", str(data_root), str(tmp_path), device="cpu", config_path=str(alt))


def test_default_config_hash_frozen():
    assert C.load_config()["_config_sha"] == "0b1ff5b42ddf", "configs/default.yaml changed: the headline runs would no longer pool"


def test_train_subsample_keeps_val_test(data_root):
    full, small = C.load_config(), C.load_config(REPO / "configs" / "train30.yaml")
    sets = data.load_all(data_root, full)
    a = split.make_split(sets, full, 1)
    b = split.subsample_train(sets, split.make_split(sets, small, 1), small, 1)
    for k in sets:
        assert a[k]["val"] == b[k]["val"] and a[k]["test"] == b[k]["test"], "val/test must not change with training size"
        assert set(b[k]["train"]) < set(a[k]["train"])
        assert 0.4 < len(b[k]["train"]) / len(a[k]["train"]) < 0.85


def test_combine_skips_seed0_and_other_configs(data_root, tmp_path):
    for seed in (0, 1):
        r = Run(seed, "arjun", str(data_root), str(tmp_path), device="cpu")
        r.load(check_descriptions=False); r.split(); r.search(embed=fake_data.hash_embedder); r.build_rows(show=0); r.train(); r.score(); r.fit_fusion(); r.test()
    T = evaluate.combine_runs(tmp_path, C.load_config())
    assert len(T) == 3 * len(METHODS)
    assert evaluate.combine_runs(tmp_path, C.load_config(REPO / "configs" / "train30.yaml")).empty


def test_finetuned_search_config(data_root, tmp_path, tiny_st):
    """configs/nift_ft.yaml: BGE fine-tuned on Laya's rows, base search kept as a baseline, cached on rerun."""
    import yaml
    cfg = yaml.safe_load((REPO / "configs" / "nift_ft.yaml").read_text())
    cfg["search"]["model"] = str(tiny_st)
    cfg["search"]["finetune"]["batch"] = 8
    cp = tmp_path / "ft.yaml"; cp.write_text(yaml.safe_dump(cfg))
    r = Run(1, "tester_ft", str(data_root), str(tmp_path), device="cpu", config_path=str(cp))
    r.load(check_descriptions=False); r.split(); r.search(embed=fake_data.hash_embedder); r.build_rows(show=0); r.train()
    rows_before = (r.rd / "rows" / "beaver.jsonl").read_text()
    r.train_search()
    assert (r.rd / "checkpoints" / "bge_beaver" / "modules.json").exists()
    assert "hyb_base" in r.sets["beaver"].extra and not np.allclose(r.sets["beaver"].extra["hyb_base"], r.sets["beaver"].extra["hyb"])
    assert (r.rd / "rows" / "beaver.jsonl").read_text() == rows_before, "Laya's rows must not change"
    r.score(); r.fit_fusion(); R = r.test()
    assert set(R.method) == set(METHODS) | {"search alone, base BGE"}
    trip = __import__("nift.search", fromlist=["triplets"]).triplets(r.rows["beaver"])
    trq = {r.sets["beaver"].Q[i][1] for i in r.SPLIT["beaver"]["train"]}
    assert trip and {t[0] for t in trip} <= trq, "BGE triplets must come from TRAIN questions only"


def test_default_has_no_finetune():
    assert not C.load_config()["search"].get("finetune")


def test_parse_quoted_keywords_and_join_case():
    """LiveSQLBench quirk: '"CREATE" TABLE', '"PRIMARY" KEY', '"FOREIGN" KEY', and REFERENCES in another case."""
    doc = {"_id": "cross_border__riskmanagement", "text": 'Database: cross_border\nTable: RiskManagement\n\n"CREATE" TABLE "RiskManagement" (\n'
           '"RISKTRACE" bigint NOT NULL,\nflow_link text NOT NULL,\n    "PRIMARY" KEY (RISKTRACE),\n'
           '    "FOREIGN" KEY (flow_link) REFERENCES DataFlow(RecordRegistry)\n);\n\n"First" 3 rows:\nRISKTRACE  flow_link'}
    db, t, cols, fks = data.parse_doc(doc)
    assert cols == ["RISKTRACE", "flow_link"], cols
    assert fks == [("flow_link", "DataFlow", "RecordRegistry")]


def test_infer_joins_names_only():
    D = data.Dataset(name="x", cards=["rooms: ROOM_KEY, BUILDING_KEY, FLOOR", "buildings: BUILDING_KEY, NAME, CITY", "people: PERSON_ID, ROOM_KEY",
                                      "log: id, created_at"], joins={}, dbo=np.array(["db"] * 4), Q=[])
    D.DBI = {"db": np.arange(4)}
    assert data.infer_joins(D) == 2                                  # rooms-buildings via BUILDING_KEY, rooms-people via ROOM_KEY
    adj = data.adjacency(D)
    assert adj[0] == {1, 2} and adj[1] == {0} and 3 not in adj        # generic 'id' and 'created_at' never link


def test_join_features_change_only_with_config(data_root, tmp_path):
    import copy
    r = Run(0, "joinfeat", str(data_root), str(tmp_path), device="cpu")
    r.load(check_descriptions=False); r.split(); r.search(embed=fake_data.hash_embedder); r.build_rows(show=0); r.train(); r.score()
    D, i = r.sets["livesqlbench"], r.SPLIT["livesqlbench"]["val"][0]
    from nift import fusion
    cfg2 = copy.deepcopy(r.cfg); cfg2["fusion"]["features"] = fusion.CORE + fusion.JOIN
    a, b = fusion.features(D, i, r.cfg), fusion.features(D, i, cfg2)
    assert a["F"].shape[1] == 8 and b["F"].shape[1] == 11 and np.allclose(a["F"], b["F"][:, :8])


def test_laya_budget_is_exact_and_restores_scores(data_root, tmp_path):
    from nift import budget
    r = Run(0, "budget", str(data_root), str(tmp_path), device="cpu")
    r.load(check_descriptions=False); r.split(); r.search(embed=fake_data.hash_embedder); r.build_rows(show=0); r.train(); r.score()
    D, val = r.sets["livesqlbench"], r.SPLIT["livesqlbench"]["val"]
    before = D.extra["P"].copy()
    P = budget.restrict(D, val, 3)
    i = val[0]
    top = np.argsort(-D.extra["hyb"][i], kind="stable")[:3]
    assert np.allclose(P[i, top], before[i, top])                         # scored tables keep their exact scores
    rest = np.setdiff1d(np.arange(D.n), top)
    assert np.all(P[i, rest] == budget.FLOOR)
    R = budget.evaluate_budgets(r, "val")
    assert {x["budget"] for x in R} == set(budget.budgets(D.n))
    assert np.allclose(np.nan_to_num(D.extra["P"]), np.nan_to_num(before)), "scores must be restored afterwards"
    full = [x for x in R if x["budget"] == "all tables (today)"]
    assert all(x["gold_in_scored"] == 1.0 for x in full)


def test_finished_run_keeps_its_manifest(data_root, tmp_path):
    r = Run(1, "keep", str(data_root), str(tmp_path), device="cpu")
    r.load(check_descriptions=False); r.split(); r.search(embed=fake_data.hash_embedder); r.build_rows(show=0); r.train(); r.score(); r.fit_fusion(); r.test()
    f = tmp_path / "results" / "keep_seed1" / "manifest.json"
    m = json.loads(f.read_text()); m["git_commit"] = "scored-here"; f.write_text(json.dumps(m))
    Run(1, "keep", str(data_root), str(tmp_path), device="cpu")
    assert json.loads(f.read_text())["git_commit"] == "scored-here"
