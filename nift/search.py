"""Hybrid search over every table card: w_dense * dense (BGE) + w_bm25 * BM25, each min-max normalized per question."""
from __future__ import annotations

import hashlib
import json
import os
import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Callable

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm.auto import tqdm

from .data import Dataset

STOP = set("a an the of in on at to and or for with by from is are was were be it this that which what how who list show give find name names many".split())
Embedder = Callable[[list[str]], np.ndarray]


def words(s: str) -> list[str]:
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(s)).replace("_", " ").lower()
    return [w for w in re.findall(r"[a-z0-9]+", s) if w not in STOP and len(w) > 1]


def minmax(v: np.ndarray) -> np.ndarray:
    return (v - v.min()) / (v.max() - v.min() + 1e-9)


def bge_embedder(model: str, device: str) -> Embedder:
    loaded = None

    def encode(xs: list[str]) -> np.ndarray:
        nonlocal loaded
        if loaded is None:
            from sentence_transformers import SentenceTransformer
            loaded = SentenceTransformer(model, device=device)
        return loaded.encode(list(xs), normalize_embeddings=True, show_progress_bar=True, batch_size=128)

    return encode


def embedding_cache_path(cache_dir: str | Path, model: str, D: Dataset) -> Path:
    try:
        st_version = version("sentence-transformers")
    except PackageNotFoundError:
        st_version = "unavailable"
    identity = json.dumps(
        ("bge-embeddings-v1", model, st_version, D.cards, [q for _, q, _ in D.Q]),
        ensure_ascii=False, separators=(",", ":"),
    ).encode()
    key = hashlib.sha256(identity).hexdigest()
    return Path(cache_dir) / f"{key}.npz"


def _read_embeddings(path: Path, n_cards: int, n_questions: int) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        with np.load(path, allow_pickle=False) as cache:
            V, QV = cache["cards"], cache["questions"]
        if V.ndim == 2 and QV.ndim == 2 and V.shape[0] == n_cards and QV.shape[0] == n_questions and V.shape[1] == QV.shape[1]:
            return V, QV
    except Exception as exc:
        print(f"ignoring unreadable embedding cache {path.name}: {exc}")
    return None


def _write_embeddings(path: Path, V: np.ndarray, QV: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp.open("wb") as f:
            np.savez(f, cards=V, questions=QV)
        os.replace(temp, path)
    except OSError as exc:
        print(f"could not write embedding cache {path.name}: {exc}")
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def hybrid_scores(D: Dataset, embed: Embedder, cfg: dict, cache_path: str | Path | None = None) -> None:
    """Adds D.extra['dense'], ['bm25'], ['hyb'], each (questions x tables)."""
    cached = _read_embeddings(Path(cache_path), D.n, len(D.Q)) if cache_path and Path(cache_path).exists() else None
    if cached is None:
        V = np.asarray(embed(D.cards))
        QV = np.asarray(embed([q for _, q, _ in D.Q]))
        if cache_path:
            _write_embeddings(Path(cache_path), V, QV)
    else:
        V, QV = cached
        print(f"{D.name}: loaded cached BGE embeddings")
    bm = BM25Okapi([words(c) or ["_"] for c in D.cards])
    dense = QV @ V.T
    bm25 = np.stack([
        minmax(np.asarray(bm.get_scores(words(q)), float))
        for _, q, _ in tqdm(D.Q, desc=f"{D.name}: BM25", unit="question", leave=False)
    ])
    hyb = cfg["search"]["w_dense"] * np.stack([minmax(x) for x in dense]) + cfg["search"]["w_bm25"] * bm25
    D.extra.update(dense=dense, bm25=bm25, hyb=hyb)


# ---------------------------------------------------------------- fine-tuning (config: search.finetune)
def triplets(rows: list[dict]) -> list[tuple[str, str, str]]:
    """(question, gold table, hard negative) built directly from Laya's training rows: no new data, no new mining."""
    by_q: dict = {}
    for r in rows:
        by_q.setdefault((r["_meta"]["dataset"], r["_meta"]["qid"]), []).append(r)
    out = []
    for rs in by_q.values():
        q = rs[0]["state"]["question"]
        gold = [r["state"]["table"] for r in rs if r["_meta"]["kind"] == "gold"]
        hard = [r["state"]["table"] for r in rs if r["_meta"]["kind"] == "hard negative"]
        if hard:
            out += [(q, g, hard[j % len(hard)]) for j, g in enumerate(gold)]
    return out


def finetune_bge(rows: list[dict], cfg: dict, path, seed: int, device: str):
    """Contrastive fine-tuning of the dense model (in-batch + one hard negative). Cached at `path`. Returns (model, seconds)."""
    import time
    from pathlib import Path

    import torch
    from datasets import Dataset as HFDataset
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments, losses)
    from sentence_transformers.training_args import BatchSamplers

    path, ft = Path(path), cfg["search"]["finetune"]
    if (path / "modules.json").exists():
        print(f"{path.name}: fine-tuned model exists, loading")
        return SentenceTransformer(str(path), device=device), 0.0
    trip = triplets(rows)
    q, pos, neg = trip[0]
    print(f"{path.name}: {len(trip)} triplets | example:\n   Q: {q[:110]}\n   +: {pos[:110]}\n   -: {neg[:110]}")
    model = SentenceTransformer(cfg["search"]["model"], device=device)
    model[0].auto_model.gradient_checkpointing_enable()             # same result, less GPU memory
    ds = HFDataset.from_dict({"anchor": [a for a, _, _ in trip], "positive": [b for _, b, _ in trip], "negative": [c for _, _, c in trip]})
    args = SentenceTransformerTrainingArguments(
        output_dir=str(path) + "_tmp", num_train_epochs=ft["epochs"], per_device_train_batch_size=ft["batch"],
        learning_rate=ft["lr"], warmup_ratio=0.1, bf16=torch.cuda.is_available(),
        batch_sampler=BatchSamplers.NO_DUPLICATES,                    # a question's other gold tables never act as its negatives
        save_strategy="no", logging_steps=50, report_to="none", seed=seed)
    t0 = time.time()
    SentenceTransformerTrainer(model=model, args=args, train_dataset=ds, loss=losses.MultipleNegativesRankingLoss(model)).train()
    sec = time.time() - t0
    model.save(str(path))
    return model, sec


def st_embedder(model) -> Embedder:
    return lambda xs: model.encode(list(xs), normalize_embeddings=True, batch_size=128, show_progress_bar=False)
