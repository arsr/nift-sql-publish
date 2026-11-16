"""Laya necessity scorer: training rows, training from the base checkpoint, scoring.

Training rows per training question: every gold table (yes), the 4 highest-ranked non-gold tables by search
(hard no), 1 random non-gold table (no). Search scores must exist (search.hybrid_scores) before building rows.
"""
from __future__ import annotations

import gc
import json
import random
import time
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from .data import Dataset

NEED_Q = {"need": {"type": "noul", "instructions":
          "Is this table required to write the SQL query that answers the question? Yes if it holds answer columns, "
          "filter columns or values, or is needed to join such tables."}}


def state(D: Dataset, q: str, t: int, cfg: dict) -> dict:
    return {"question": q, "joins": D.joins.get(t, [])[:cfg["card"]["joins_max"]], "table": D.cards[t][:cfg["card"]["card_max"]]}


def training_rows(sets: dict[str, Dataset], qs: list[tuple[str, int]], cfg: dict, seed: int) -> list[dict]:
    """qs: (dataset name, question index) pairs, all from the TRAIN part."""
    rng, sc, rows = random.Random(seed), cfg["scorer"], []
    for k, i in tqdm(sorted(qs), desc="Building training rows", unit="question", leave=False):
        D = sets[k]
        qid, q, g = D.Q[i]
        negs = [int(t) for t in np.argsort(-D.extra["hyb"][i], kind="stable") if int(t) not in g]
        pick = sorted(g) + negs[:sc["hard_neg"]]
        if len(negs) > sc["hard_neg"]:
            pick += rng.sample(negs[sc["hard_neg"]:], min(sc["rand_neg"], len(negs) - sc["hard_neg"]))
        for t in pick:
            kind = "gold" if t in g else ("hard negative" if t in negs[:sc["hard_neg"]] else "random negative")
            rows.append({"state": state(D, q, t, cfg), "questions": NEED_Q, "expected": {"need": bool(t in g)},
                         "_meta": {"dataset": k, "qid": qid, "table": D.ids[t] if D.ids else t, "kind": kind}})
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    """Laya reads state / questions / expected; _meta is kept for inspection only."""
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ("state", "questions", "expected")}) + "\n")


def show_rows(rows: list[dict], n_questions: int = 2) -> None:
    """Print what the scorer is trained on: a few questions, each with all of its rows."""
    import collections
    kinds = collections.Counter(r["_meta"]["kind"] for r in rows)
    nq = len({(r["_meta"]["dataset"], r["_meta"]["qid"]) for r in rows})
    print(f"{len(rows)} training rows from {nq} questions | " + ", ".join(f"{k}: {v}" for k, v in kinds.items()))
    seen = []
    for r in rows:
        key = (r["_meta"]["dataset"], r["_meta"]["qid"])
        if key not in seen:
            if len(seen) == n_questions:
                break
            seen.append(key)
            print("\n" + "-" * 100 + f"\n[{key[0]}] question {key[1]}: {r['state']['question']}")
        st = r["state"]
        print(f"  expected need={str(r['expected']['need']):5s} ({r['_meta']['kind']:15s}) table: {st['table'][:110]}")
        if st["joins"]:
            print(f"  {'':40s} joins: {'; '.join(st['joins'][:3])}{' ...' if len(st['joins']) > 3 else ''}")
    print("\nOne raw row exactly as Laya receives it:")
    print(json.dumps({k: rows[0][k] for k in ("state", "questions", "expected")}, indent=1)[:1500])


def _free():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def train(rows_path: Path, ck: Path, cfg: dict, seed: int, device: str) -> tuple[Path, float]:
    """Fine-tune Laya from the base checkpoint. Cached: returns at once if the checkpoint exists. Returns (ck, seconds)."""
    import laya
    from laya.train import TrainConfig, finetune
    assert laya.__version__ == cfg["scorer"]["laya_version"], f"need laya=={cfg['scorer']['laya_version']}, have {laya.__version__}"
    if (ck / "rl_agent_config.json").exists():
        print(f"{ck.name}: checkpoint exists, skipping training")
        meta = ck.parent / f"{ck.name}_summary.json"
        return ck, json.loads(meta.read_text()).get("_seconds", 0.0) if meta.exists() else 0.0
    sc = cfg["scorer"]
    t0 = time.time()
    summ = finetune(str(rows_path), sc["base"], str(ck), TrainConfig(epochs=sc["epochs"], loss=sc["loss"], micro_batch=sc["micro_batch"],
                    grad_accum=sc["grad_accum"], calib_max=sc["calib_max"], seed=seed), device=device)
    sec = time.time() - t0
    out = dict(summ) if isinstance(summ, dict) else {"summary": str(summ)}
    out["_seconds"] = sec
    (ck.parent / f"{ck.name}_summary.json").write_text(json.dumps(out, indent=1, default=str))
    _free()
    return ck, sec


def score(ck: Path, D: Dataset, qidx: list[int], cfg: dict, device: str) -> np.ndarray:
    """P(need) for every (question in qidx) x (every table). Returns (len(qidx), D.n)."""
    import laya
    if not qidx:
        return np.zeros((0, D.n))
    ag = laya.load(str(ck), device=device)
    S = []
    for i in tqdm(qidx, desc=f"{D.name}: preparing scores", unit="question", leave=False):
        S.extend(state(D, D.Q[i][1], t, cfg) for t in range(D.n))
    order = np.argsort([len(json.dumps(s)) for s in S], kind="stable")          # length-sorted batches are faster
    batch_size = cfg["scorer"]["batch_size"]
    res = []
    for start in tqdm(range(0, len(order), batch_size), desc=f"{D.name}: scoring", unit="batch", leave=False):
        batch = [S[j] for j in order[start:start + batch_size]]
        res.extend(ag.predict_batch(batch, NEED_Q, batch_size=batch_size))
    ps = np.array([float(r["answers"]["need"]["noul"]) for r in res])
    assert len(ps) == len(S) and np.all((ps >= 0) & (ps <= 1)), "Laya returned a wrong number of scores or values outside [0, 1]"
    flat = np.empty(len(S))
    flat[order] = ps
    del ag
    _free()
    return flat.reshape(len(qidx), D.n)
