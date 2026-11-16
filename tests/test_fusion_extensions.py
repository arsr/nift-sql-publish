"""experiments/fusion_extensions.py end to end on fake data: fake cross-encoder and fine-tuned-search scores, hash reranker."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import fake_data
from nift.pipeline import Run

REPO = Path(__file__).resolve().parents[1]


def _mm(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-9)


def _fake_runs(data_root, out, seeds, reps):
    """Score files the script reads: cross-encoder P per training run, fine-tuned hybrid scores and their raw cosine."""
    rng = np.random.default_rng(0)
    rr = Path(out) / "results" / ".cache" / "reranker"
    rr.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        r = Run(seed, "nift_base", str(data_root), str(out), device="cpu")
        r.load(check_descriptions=False); r.split(); r.search(embed=fake_data.hash_embedder)
        for R in reps:
            ce_dir = Path(out) / "results" / f"{'nift_ce' if R == 0 else f'nift_ce_rep{R}'}_seed{seed}" / "scores"
            ft_run = "nift_ftsearch" if R == 0 else f"nift_ftsearch_rep{R}"
            ft_dir = Path(out) / "results" / f"{ft_run}_seed{seed}" / "scores"
            ce_dir.mkdir(parents=True, exist_ok=True); ft_dir.mkdir(parents=True, exist_ok=True)
            for k, D in r.sets.items():
                G = np.zeros((len(D.Q), D.n))
                for i, (_, _, g) in enumerate(D.Q):
                    G[i, list(g)] = 1
                P = np.clip(0.35 * G + 0.65 * rng.random(G.shape), 0, 1)       # informative but noisy
                np.save(ce_dir / f"{k}_ce_bge-reranker-large.npy", P)
                dense = 0.3 * G + rng.random(G.shape)
                hyb = 0.6 * np.stack([_mm(x) for x in dense]) + 0.4 * D.extra["bm25"]
                np.save(ft_dir / f"{k}_hyb_ft.npy", hyb)
                cache = Path(out) / "results" / "fusion_ext" / ".cache"
                cache.mkdir(parents=True, exist_ok=True)
                np.save(cache / f"dense_{ft_run}_seed{seed}_{k}.npy", dense)
                tv = Path(out) / "results" / f"nift_ftsearch_trval_seed{seed}" / "scores"
                tv.mkdir(parents=True, exist_ok=True)
                np.save(tv / f"{k}_hyb_ft.npy", hyb)
                for tag in ("all", "db"):                       # the earlier reranker caches, used only for the baselines
                    np.save(rr / f"{k}_fake_{tag}.npy", rng.random((len(D.Q), D.n)))


def _run(data_root, out, *extra):
    env = dict(os.environ, NIFT_FAKE_EMBED="1",
               PYTHONPATH=os.pathsep.join([str(REPO / "tests" / "fake_laya"), str(REPO / "tests"), str(REPO)]))
    return subprocess.run([sys.executable, str(REPO / "experiments" / "fusion_extensions.py"), "--data", str(data_root),
                           "--out", str(out), "--seeds", "1", "2", "--reps", "0", "1", "--reranker", "fake", *extra],
                          capture_output=True, text=True, env=env)


def test_fusion_extensions_end_to_end(data_root, tmp_path):
    _fake_runs(data_root, tmp_path, (1, 2), (0, 1))

    v = _run(data_root, tmp_path, "--validation-only")
    assert v.returncode != 0 and "stopped before the test part" in v.stderr, v.stderr[-2000:]
    assert (tmp_path / "results" / "fusion_ext" / "validation.txt").exists()
    assert not (tmp_path / "results" / "fusion_ext" / "test_rows.csv").exists(), "--validation-only must not touch test"

    p = _run(data_root, tmp_path)
    assert p.returncode == 0, p.stderr[-3000:]
    T = pd.read_csv(tmp_path / "results" / "fusion_ext" / "test_rows.csv")
    for m in ("NIFT-SQL", "A: NIFT-SQL + fine-tuned search", "B: NIFT-SQL + reranker features", "C: A + reranker features"):
        assert set(T[T.method == m].rep) == {0, 1}, m
    rep = (tmp_path / "results" / "FUSION_EXTENSIONS.txt").read_text()
    assert "Against NIFT-SQL" in rep and "reranker_logit" in rep
    assert "minus reranker, variant chosen on validation" in rep and "minus reranker over the whole database" in rep
    assert {"reranker (search top 100)", "reranker, variant chosen on validation", "fine-tuned search, train + validation"} <= set(T.method)

    # the reranker cache covers exactly the candidates, the same way for every question (no gold-dependent coverage)
    R = np.load(tmp_path / "results" / ".cache" / "reranker" / "beaver_fake_fusionext.npy")
    scored = ~np.isnan(R)
    assert scored.any() and not scored.all()

    again = _run(data_root, tmp_path)                    # the test part is scored once
    assert again.returncode != 0 and "already scored" in again.stderr


def test_fusion_extensions_rejects_wrong_model_but_not_gpu_drift(data_root, tmp_path):
    _fake_runs(data_root, tmp_path, (1, 2), (0, 1))
    cache = tmp_path / "results" / "fusion_ext" / ".cache"
    f = cache / "dense_nift_ftsearch_seed1_beaver.npy"
    good = np.load(f)
    rng = np.random.default_rng(1)

    np.save(f, good + rng.normal(0, 3e-4, good.shape))           # GPU-precision drift: must pass the check
    ok = _run(data_root, tmp_path, "--validation-only")
    assert "stopped before the test part" in ok.stderr, ok.stderr[-2000:]
    assert "top-50 agreement" in ok.stdout

    np.save(f, rng.random(good.shape))                           # a different model: must stop
    bad = _run(data_root, tmp_path, "--validation-only")
    assert bad.returncode != 0 and "wrong model?" in bad.stderr


def test_fusion_example_prints_validation_rows(data_root, tmp_path):
    _fake_runs(data_root, tmp_path, (1,), (0,))
    env = dict(os.environ, NIFT_FAKE_EMBED="1",
               PYTHONPATH=os.pathsep.join([str(REPO / "tests" / "fake_laya"), str(REPO / "tests"), str(REPO)]))
    p = subprocess.run([sys.executable, str(REPO / "experiments" / "fusion_example.py"), "--data", str(data_root), "--out", str(tmp_path),
                        "--reranker", "fake"], capture_output=True, text=True, env=env)
    assert p.returncode == 0, p.stderr[-3000:]
    txt = (tmp_path / "results" / "FUSION_EXAMPLE.txt").read_text()
    assert "validation questions" in txt and "| table | label |" in txt and "Standardized fusion weights" in txt


def test_final_breakdowns_reproduces_reported_rows(data_root, tmp_path):
    _fake_runs(data_root, tmp_path, (1, 2), (0, 1))
    assert _run(data_root, tmp_path).returncode == 0
    ref = (tmp_path / "results" / "fusion_ext" / "test_rows.csv").read_bytes()
    env = dict(os.environ, NIFT_FAKE_EMBED="1",
               PYTHONPATH=os.pathsep.join([str(REPO / "tests" / "fake_laya"), str(REPO / "tests"), str(REPO)]))
    p = subprocess.run([sys.executable, str(REPO / "experiments" / "final_breakdowns.py"), "--data", str(data_root), "--out", str(tmp_path),
                        "--seeds", "1", "2", "--reps", "0", "1", "--reranker", "fake"], capture_output=True, text=True, env=env)
    assert p.returncode == 0, p.stderr[-3000:] + p.stdout[-3000:]
    txt = (tmp_path / "results" / "FINAL_BREAKDOWNS.txt").read_text()
    assert " 0 differ." in txt and "1. Reranker routed" in txt and "5 or more" in txt and "shares no word" in txt
    assert (tmp_path / "results" / "fusion_ext" / "test_rows.csv").read_bytes() == ref     # never rewritten


def test_fusion_controls_reads_cached_rankings_only(data_root, tmp_path):
    _fake_runs(data_root, tmp_path, (1, 2), (0, 1))
    for seed in (1, 2):                                   # cross-encoder trained on train + validation (run 0 only)
        src = tmp_path / "results" / f"nift_ce_seed{seed}" / "scores"
        dst = tmp_path / "results" / f"nift_ce_trval_seed{seed}" / "scores"; dst.mkdir(parents=True, exist_ok=True)
        for f in src.glob("*.npy"):
            np.save(dst / f.name, np.load(f) ** 0.9)
    assert _run(data_root, tmp_path).returncode == 0
    ref = (tmp_path / "results" / "fusion_ext" / "test_rows.csv").read_bytes()
    env = dict(os.environ, NIFT_FAKE_EMBED="1",
               PYTHONPATH=os.pathsep.join([str(REPO / "tests" / "fake_laya"), str(REPO / "tests"), str(REPO)]))
    p = subprocess.run([sys.executable, str(REPO / "experiments" / "fusion_controls.py"), "--data", str(data_root), "--out", str(tmp_path),
                        "--seeds", "1", "2", "--reps", "0", "1", "--reranker", "fake"], capture_output=True, text=True, env=env)
    assert p.returncode == 0, p.stderr[-3000:] + p.stdout[-3000:]
    txt = (tmp_path / "results" / "FUSION_CONTROLS.txt").read_text()
    assert "minus rank average, cross-encoder + fine-tuned search" in txt
    assert "C: A + reranker features minus reranker alone on NIFT-SQL's candidates" in txt
    C = pd.read_csv(tmp_path / "results" / "fusion_ext" / "control_rows.csv")
    assert set(C.rep) == {0, 1} and C.method.nunique() == 8
    assert "C: A + reranker features minus rank average of all three signals | grouped" in txt
    assert "3. The same labels spent on training" in txt and "minus rank average, both trained on train + validation | grouped" in txt
    assert (tmp_path / "results" / "fusion_ext" / "test_rows.csv").read_bytes() == ref     # never rewritten
