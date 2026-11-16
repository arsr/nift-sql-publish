"""Baseline: hybrid search, then an off-the-shelf cross-encoder reranker (no training on our data) on its top-k.

    python experiments/reranker_baseline.py --run nift_base --seeds 1 2 3 4 --data D --out O [--model Qwen/Qwen3-Reranker-0.6B]

Answers the reviewer question: is NIFT-SQL's gain from per-customer training, or just from reading the question and
the table together? The reranker reads exactly what Laya reads (table card + joins) with the question, using the
model card's recommended prompt and a task instruction, and reorders the search top-k; the rest keeps search order.
TEST questions of the given finished runs (search is the run's own; no fitting of anything on test).
Scores depend only on (question, table), so they are cached once for all seeds in
<out>/results/.cache/reranker/<set>_<model>_top<k>.npy. Reports per seed and pooled against the run's NIFT-SQL.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nift import evaluate, scorer      # noqa: E402
from nift.config import REPO           # noqa: E402
from nift.pipeline import Run          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="nift_base")
ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
ap.add_argument("--config", default="default")
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="Qwen/Qwen3-Reranker-0.6B")
ap.add_argument("--k", type=int, default=100)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--known-db", action="store_true",
                help="score every table of the question's own database (questions whose gold tables sit in one database) "
                     "into <set>_<model>_db.npy, reusing the top-k scores; then stop (experiments/review_analysis.py reads it)")
ap.add_argument("--routed", type=int, default=0,
                help="zero-label routing: score every table of the top N databases by search (databases ordered by their "
                     "best-ranked table) into <set>_<model>_routed<N>.npy, reusing earlier scores; then stop")
ap.add_argument("--variants", action="store_true",
                help="scores for choosing the reranker variant on VALIDATION questions and for the reranker-routed variant: "
                     "validation questions get the search top-k and the top 2 databases by search; validation and test "
                     "questions get every table of the top 2 databases of the reranker's own grouped ranking. Merged with "
                     "all earlier scores into <set>_<model>_all.npy; then stop")
args = ap.parse_args()
INSTRUCT = ("Given a question about a database, judge whether this table is required to write the SQL query that answers "
            "the question: it holds answer columns, filter columns, or is needed to join such tables.")
tag = args.model.split("/")[-1]

runs = {}
for s in args.seeds:
    r = Run(s, args.run, args.data, args.out, config_path=REPO / "configs" / f"{args.config}.yaml")
    assert (r.rd / "results.csv").exists(), f"{r.rd} is not finished: run main.py for seed {s} first"
    r.load(check_descriptions=False); r.split(); r.search()
    runs[s] = r
first = runs[args.seeds[0]]
cfg = first.cfg

# ---------------------------------------------------------------- reranker
if args.model == "fake":                                          # tests only
    import hashlib

    def rerank(pairs):
        return [int(hashlib.md5((q + d).encode()).hexdigest()[:6], 16) / 16**6 for q, d in pairs]
else:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    mdl = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev).eval()
    yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
    prefix = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
              "provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    pre_ids, suf_ids = tok.encode(prefix, add_special_tokens=False), tok.encode(suffix, add_special_tokens=False)

    def rerank(pairs):
        texts = [f"<Instruct>: {INSTRUCT}\n<Query>: {q}\n<Document>: {d}" for q, d in pairs]
        enc = tok(texts, padding=False, truncation="longest_first", return_attention_mask=False,
                  max_length=2048 - len(pre_ids) - len(suf_ids))
        enc["input_ids"] = [pre_ids + ids + suf_ids for ids in enc["input_ids"]]
        batch = tok.pad(enc, padding=True, return_tensors="pt").to(dev)
        with torch.inference_mode():
            last = mdl(**batch).logits[:, -1, :]
            two = torch.stack([last[:, no], last[:, yes]], 1).float().log_softmax(1)
        return two[:, 1].exp().tolist()


def doc(D, t):
    st = scorer.state(D, "", t, cfg)
    return st["table"] + (("\nJoins: " + "; ".join(st["joins"])) if st["joins"] else "")


def top_dbs(order, D, n):
    """The first n databases of a ranking, in order of their best-ranked table."""
    seen = []
    for t in order:
        if D.dbo[int(t)] not in seen:
            seen.append(D.dbo[int(t)])
        if len(seen) == n:
            break
    return seen


def score(todo, P, f, k, D):
    todo.sort(key=lambda it: len(D.Q[it[0]][1]) + len(doc(D, it[1])))
    t0 = time.time()
    for s0 in range(0, len(todo), args.batch):
        chunk = todo[s0:s0 + args.batch]
        for (i, t), p in zip(chunk, rerank([(D.Q[i][1], doc(D, t)) for i, t in chunk])):
            P[i, t] = p
        if s0 % (args.batch * 100) == 0:
            np.save(f, P); print(f"{k}: {s0 + len(chunk)}/{len(todo)} pairs", flush=True)
    np.save(f, P)
    if todo:
        timing[k].append(1000 * (time.time() - t0) / len(todo))


cache = Path(args.out) / "results" / ".cache" / "reranker"; cache.mkdir(parents=True, exist_ok=True)
timing = {}
for k, D in first.sets.items():
    f = cache / f"{k}_{tag}_top{args.k}.npy"
    P = np.load(f) if f.exists() else np.full((len(D.Q), D.n), np.nan)
    qs = sorted({i for r in runs.values() for i in r.SPLIT[k]["test"]})
    if args.variants:
        f = cache / f"{k}_{tag}_all.npy"
        if f.exists():
            P = np.load(f)
        else:
            P = P.copy()
            for extra in ("db", "routed2"):
                if (fx := cache / f"{k}_{tag}_{extra}.npy").exists():
                    Px = np.load(fx); P = np.where(np.isnan(P), Px, P)
        val = sorted({i for r in runs.values() for i in r.SPLIT[k]["val"]})
        timing[k] = []
        todo = [(i, int(t)) for i in val for t in np.argsort(-D.extra["hyb"][i], kind="stable")[:args.k]]
        todo += [(i, t) for i in val for t in range(D.n) if D.dbo[t] in top_dbs(np.argsort(-D.extra["hyb"][i], kind="stable"), D, 2)]
        score(sorted({it for it in todo if np.isnan(P[it])}), P, f, k, D)          # phase 1: what the routing below needs
        todo = []
        for i in sorted(set(val) | set(qs)):
            ho = np.argsort(-D.extra["hyb"][i], kind="stable"); top = ho[:args.k]
            rr = top[np.argsort(-P[i, top], kind="stable")].tolist() + ho[args.k:].tolist()
            todo += [(i, t) for t in range(D.n) if D.dbo[t] in top_dbs(rr, D, 2)]
        score(sorted({it for it in todo if np.isnan(P[it])}), P, f, k, D)          # phase 2: the reranker-routed variant
        timing[k] = float(np.mean(timing[k])) if timing[k] else 0.0
        continue
    if args.routed:                                       # zero-label routing by search, then the reranker over those databases
        f = cache / f"{k}_{tag}_routed{args.routed}.npy"
        if f.exists():
            P = np.load(f)
        else:
            P = P.copy()
            fdb = cache / f"{k}_{tag}_db.npy"
            if fdb.exists():
                Pdb = np.load(fdb); P = np.where(np.isnan(P), Pdb, P)
        todo = []
        for i in qs:
            seen = []
            for t in np.argsort(-D.extra["hyb"][i], kind="stable"):
                if D.dbo[int(t)] not in seen:
                    seen.append(D.dbo[int(t)])
                if len(seen) == args.routed:
                    break
            todo += [(i, t) for t in range(D.n) if D.dbo[t] in seen and np.isnan(P[i, t])]
    elif args.known_db:                                   # a (question, table) score does not depend on the shortlist
        f = cache / f"{k}_{tag}_db.npy"
        P = np.load(f) if f.exists() else P.copy()
        todo = []
        for i in qs:
            dbs = {D.dbo[t] for t in D.Q[i][2]}
            if len(dbs) == 1:
                todo += [(i, t) for t in range(D.n) if D.dbo[t] in dbs and np.isnan(P[i, t])]
    else:
        todo = [(i, int(t)) for i in qs for t in np.argsort(-D.extra["hyb"][i], kind="stable")[:args.k] if np.isnan(P[i, t])]
    todo.sort(key=lambda it: len(D.Q[it[0]][1]) + len(doc(D, it[1])))
    t0 = time.time()
    for s0 in range(0, len(todo), args.batch):
        chunk = todo[s0:s0 + args.batch]
        for (i, t), p in zip(chunk, rerank([(D.Q[i][1], doc(D, t)) for i, t in chunk])):
            P[i, t] = p
        if s0 % (args.batch * 100) == 0:
            np.save(f, P); print(f"{k}: {s0 + len(chunk)}/{len(todo)} pairs")
    np.save(f, P)
    if todo:
        timing[k] = 1000 * (time.time() - t0) / len(todo)
    for r in runs.values():
        r.sets[k].extra["RR"] = P

if args.known_db or args.routed or args.variants:
    print("known-database scores written: " + ", ".join(f"{k} {v:.1f} ms per table" for k, v in timing.items()))
    sys.exit(0)

# ---------------------------------------------------------------- evaluate on each run's TEST part
rows = []
for s, r in runs.items():
    main = pd.read_csv(r.rd / "results.csv")
    for k, D in r.sets.items():
        for i in r.SPLIT[k]["test"]:
            ho = np.argsort(-D.extra["hyb"][i], kind="stable")
            top = ho[:args.k]
            order = top[np.argsort(-D.extra["RR"][i, top], kind="stable")].tolist() + ho[args.k:].tolist()
            g = set(D.Q[i][2])
            rows.append(dict(seed=s, set=k, qi=i, group="|".join(map(str, sorted(g))), method=f"search + {tag} (zero-shot)",
                             **evaluate.metrics(order, g, cfg)))
    keep = main[main.method.isin(["search alone", "NIFT-SQL (fusion)"])]
    rows += keep.assign(seed=s)[["seed", "set", "qi", "group", "method"] + [c for c in keep.columns if c.startswith(("c@", "R@"))]].to_dict("records")
A = pd.DataFrame(rows)
cols = [c for c in A.columns if c.startswith(("c@", "R@"))]
pd.set_option("display.width", 250)
print("\n" + "=" * 110 + f"\nTEST, seeds {args.seeds}: mean over seeds (std)\n" + "=" * 110)
per = A.groupby(["set", "method", "seed"])[cols].mean().mul(100)
print(per.groupby(level=[0, 1]).agg(["mean", "std"]).round(1).to_string())
print(f"\nNIFT-SQL minus search + {tag}, pooled over seeds, c@8 / c@15 [95% CI, groups = seed x gold set]")
for k in sorted(A.set.unique()):
    a = A[(A.set == k) & (A.method == "NIFT-SQL (fusion)")].sort_values(["seed", "qi"])
    b = A[(A.set == k) & (A.method == f"search + {tag} (zero-shot)")].sort_values(["seed", "qi"])
    cells = []
    for key in ("c@8", "c@15"):
        m_, lo, hi = evaluate.group_boot(a[key].values - b[key].values, (a.seed.astype(str) + "|" + a.group).tolist(), 3000)
        cells.append(f"{key} {m_:+.1f} [{lo:+.1f}, {hi:+.1f}]")
    print(f"  {k:20s} " + "   ".join(cells))
if timing:
    print(f"\n{tag} ms per table: " + ", ".join(f"{k} {v:.1f}" for k, v in timing.items()))
