"""Supervised baseline: an off-the-shelf cross-encoder reranker fine-tuned on exactly Laya's training rows, used in place
of Laya (alone, and inside the same fusion). Answers: is NIFT-SQL's gain just from having supervised customer data?

    python experiments/supervised_crossencoder.py --seeds 1 2 3 4 --data D --out O [--model BAAI/bge-reranker-large]

Variants (each its own run name, same splits and fusion; settings fixed before running):
  --zero-shot --run nift_ce0                      the same cross-encoder, NOT fine-tuned (isolates the effect of training)
  --no-joins --run nift_ce_nojoin                 fine-tuned, but the input has no join hints (tests the join mechanism)
  --config train10 --main nift_train10 --run nift_ce_train10   training-size curve (also train20, train30)

Same protocol as the main runs (configs/default.yaml), per seed and per customer:
  rows   built exactly as for Laya (same questions, same gold / 4 hard / 1 random negatives, same seed)
  train  binary cross-entropy on (question, table card + joins) -> needed, 4 epochs (as Laya), lr 2e-5, batch 16
  score  every table for validation and test questions (as Laya)
  fusion the same 8 features and logistic regression, fitted on validation; test scored once
Writes results/<run>_seed<N>/results.csv (run name default nift_ce) and compares, on the same test questions, with the
main runs (nift_base): cross-encoder alone vs Laya alone, and NIFT-SQL with each scorer, pooled over seeds with 95%
intervals clustered by gold-table set.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nift import evaluate, fusion, scorer      # noqa: E402
from nift.pipeline import Run                  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="BAAI/bge-reranker-large")
ap.add_argument("--run", default="nift_ce")
ap.add_argument("--main", default="nift_base", help="finished main runs to compare with (same config, same seeds)")
ap.add_argument("--epochs", type=int, default=4)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--max-len", type=int, default=512)
ap.add_argument("--config", default="default", help="configs/<name>.yaml; train10/20/30 for the training-size curve")
ap.add_argument("--zero-shot", action="store_true", help="score with the public checkpoint, no fine-tuning")
ap.add_argument("--no-joins", action="store_true", help="leave the join hints out of the input (train and score)")
ap.add_argument("--replicate", type=int, default=0, help="retraining replicate: 0 = the original run; R > 0 reseeds dropout and data order")
ap.add_argument("--deterministic", action="store_true", help="deterministic GPU kernels (to test whether two runs match exactly)")
ap.add_argument("--customers", nargs="+", default=None, help="train and score only these customers (with --score-only)")
ap.add_argument("--score-only", action="store_true", help="stop after writing the scores (no fusion, no results.csv)")
ap.add_argument("--input-hash", action="store_true", help="CPU: print a hash of every training and scoring input, then exit")
ap.add_argument("--with-val", action="store_true",
                help="train on the training AND validation questions (needs --score-only: nothing is left to fit a fusion on); "
                     "for label-free fusions such as grouped rank averaging, experiments/fusion_controls.py")
args = ap.parse_args()
assert not args.customers or args.score_only, "--customers needs --score-only: the fusion needs every customer's scores"
assert not args.with_val or args.score_only, "--with-val needs --score-only: the validation questions are used for training"
if args.deterministic:
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")       # must be set before CUDA starts

import torch                                                                   # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer    # noqa: E402

if args.deterministic:
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
DEV = "cuda" if torch.cuda.is_available() else "cpu"
AMP = dict(device_type="cuda", dtype=torch.bfloat16) if DEV == "cuda" else None
TAG = args.model.rstrip("/").split("/")[-1]
KIND = "zero-shot" if args.zero_shot else ("fine-tuned, no joins" if args.no_joins else "fine-tuned")


def doc(st):
    """The same input Laya reads: the table card plus its joins."""
    return st["table"] + (("\nJoins: " + "; ".join(st["joins"])) if st["joins"] and not args.no_joins else "")


def logits(model, tok, pairs):
    enc = tok([q for q, _ in pairs], [d for _, d in pairs], padding=True, truncation="longest_first",
              max_length=args.max_len, return_tensors="pt").to(DEV)
    if AMP:
        with torch.autocast(**AMP):
            return model(**enc).logits.float().view(-1)
    return model(**enc).logits.float().view(-1)


def train(rows, ck: Path):
    tok = AutoTokenizer.from_pretrained(args.model)
    if args.zero_shot:
        print(f"{ck.name}: zero-shot, public checkpoint {args.model}, no training")
        return AutoModelForSequenceClassification.from_pretrained(args.model).to(DEV).eval(), tok, 0.0
    if (ck / "config.json").exists():
        print(f"{ck.name}: fine-tuned cross-encoder exists, loading")
        return AutoModelForSequenceClassification.from_pretrained(str(ck)).to(DEV).eval(), tok, 0.0
    if args.replicate:                                                   # replicate 0 keeps the original run's random state
        torch.manual_seed(1000 * args.seed_now + args.replicate)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1).to(DEV)
    data = [(r["state"]["question"], doc(r["state"]), float(r["expected"]["need"])) for r in rows]
    steps = args.epochs * math.ceil(len(data) / args.batch); warm = max(1, int(0.1 * steps))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (steps - s) / max(1, steps - warm)))
    lossf, rng, t0, step = torch.nn.BCEWithLogitsLoss(), random.Random(args.replicate), time.time(), 0
    model.train()
    for ep in range(args.epochs):
        rng.shuffle(data); hist = []
        for s in range(0, len(data), args.batch):
            b = data[s:s + args.batch]
            loss = lossf(logits(model, tok, [(q, d) for q, d, _ in b]), torch.tensor([y for _, _, y in b], device=DEV))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1; hist.append(loss.item())
        print(f"{ck.name}: epoch {ep + 1}/{args.epochs} | loss {np.mean(hist):.4f} | {(time.time() - t0) / 60:.1f} min")
    sec = time.time() - t0
    model.eval(); ck.mkdir(parents=True, exist_ok=True); model.save_pretrained(str(ck)); tok.save_pretrained(str(ck))
    return model, tok, sec


@torch.inference_mode()
def score_all(model, tok, D, qidx, cfg):
    """P(needed) for every (question, table), questions in qidx; same states as Laya's scorer."""
    pairs = [(i, t) for i in qidx for t in range(D.n)]
    docs = [doc(scorer.state(D, "", t, cfg)) for t in range(D.n)]
    order = sorted(range(len(pairs)), key=lambda j: len(docs[pairs[j][1]]))       # similar lengths per batch
    out = np.empty(len(pairs))
    for s in range(0, len(order), 256):
        js = order[s:s + 256]
        out[js] = torch.sigmoid(logits(model, tok, [(D.Q[pairs[j][0]][1], docs[pairs[j][1]]) for j in js])).cpu().numpy()
    P = np.full((len(D.Q), D.n), np.nan)
    for (i, t), p in zip(pairs, out):
        P[i, t] = p
    return P


rows_all = []
for seed in args.seeds:
    from nift.config import REPO
    r = Run(seed, args.run, args.data, args.out, config_path=REPO / "configs" / f"{args.config}.yaml")
    main_csv = Path(args.out) / "results" / f"{args.main}_seed{seed}" / "results.csv"
    assert main_csv.exists(), f"{main_csv} missing: finish the main run for seed {seed} first"
    args.seed_now = seed
    r.load(check_descriptions=False); r.split()
    if args.with_val:                                 # every labeled question goes to the scorer; test stays held out
        for k in r.SPLIT:
            r.SPLIT[k] = dict(r.SPLIT[k], train=sorted(set(r.SPLIT[k]["train"]) | set(r.SPLIT[k]["val"])))
            assert not set(r.SPLIT[k]["test"]) & set(r.SPLIT[k]["train"])
    if args.input_hash:                               # same text the model would read, train rows and every table card
        import hashlib
        r.search(); r.build_rows(show=0)          # hard negatives come from search
        for cust, names in r.cfg["customers"].items():
            h = hashlib.sha256()
            for row in r.rows[cust]:
                h.update((row["state"]["question"] + "\x1f" + doc(row["state"]) + "\x1e").encode())
            for k in names:
                for t in range(r.sets[k].n):
                    h.update((doc(scorer.state(r.sets[k], "", t, r.cfg)) + "\x1e").encode())
            print(f"INPUT HASH seed {seed} {cust:14s} no_joins={args.no_joins} {h.hexdigest()[:16]} ({len(r.rows[cust])} training rows)")
        continue
    r.search(); r.build_rows(show=0)
    timing = {}
    for cust, names in r.cfg["customers"].items():
        if args.customers and cust not in args.customers:
            continue
        model, tok, sec = train(r.rows[cust], r.rd / "checkpoints" / f"ce_{TAG}_{cust}")
        if sec:
            timing[f"train {cust} (min)"] = round(sec / 60, 2)
        for k in names:
            D, f = r.sets[k], r.rd / "scores" / f"{k}_ce_{TAG}.npy"
            qs = r.SPLIT[k]["val"] + r.SPLIT[k]["test"]
            if f.exists():
                D.extra["P"] = np.load(f)
            else:
                t0 = time.time(); D.extra["P"] = score_all(model, tok, D, qs, r.cfg)
                timing[f"score {k} (ms per table)"] = round(1000 * (time.time() - t0) / (len(qs) * D.n), 2)
                np.save(f, D.extra["P"])
        del model
        if DEV == "cuda":
            torch.cuda.empty_cache()
    if args.score_only:
        print(f"seed {seed}: scores written, --score-only so no fusion | timing {timing}")
        continue
    r.fit_fusion()                                    # same 8 features, fitted on validation
    for cust, names in r.cfg["customers"].items():
        for k in names:
            for i in r.SPLIT[k]["test"]:
                e = r.E[k][i]
                for m, o in ((f"{TAG} {KIND}, alone", e["co"].tolist()),
                             (f"NIFT-SQL with {KIND} {TAG}", fusion.rank(r.FUSION[cust], e))):
                    rows_all.append(dict(seed=seed, set=k, qi=i, group="|".join(map(str, sorted(e["gold"]))), method=m,
                                         **evaluate.metrics(o, e["gold"], r.cfg)))
    pd.DataFrame([x for x in rows_all if x["seed"] == seed]).to_csv(r.rd / "results.csv", index=False)
    (r.rd / "timing.json").write_text(json.dumps(timing, indent=1))
    print(f"seed {seed} timing: {timing}")
    base = pd.read_csv(main_csv)
    rows_all += base[base.method.isin(["search alone", "Laya alone", "NIFT-SQL (fusion)"])].assign(seed=seed)[
        ["seed", "set", "qi", "group", "method"] + [c for c in base.columns if c.startswith(("c@", "R@"))]].to_dict("records")

if args.input_hash or args.score_only:
    sys.exit(0)

import contextlib
import io

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    A = pd.DataFrame(rows_all)
    pd.set_option("display.width", 250)
    cols = ["c@5", "c@8", "c@15", "c@25", "R@10"]
    print("\n" + "=" * 110 + f"\nTEST, seeds {args.seeds}: mean over seeds (std)\n" + "=" * 110)
    print(A.groupby(["set", "method", "seed"])[cols].mean().mul(100).groupby(level=[0, 1]).agg(["mean", "std"]).round(1).to_string())
    pairs = [(f"{TAG} {KIND}, alone", "Laya alone"), (f"NIFT-SQL with {KIND} {TAG}", "NIFT-SQL (fusion)"),
             (f"NIFT-SQL with {KIND} {TAG}", "search alone")]
    print("\nA minus B, pooled over seeds, c@8 / c@15 / R@10 [95% CI, clustered by gold-table set across seeds]")
    for a, b in pairs:
        for k in sorted(A.set.unique()):
            X = A[(A.set == k) & (A.method == a)].sort_values(["seed", "qi"]); Y = A[(A.set == k) & (A.method == b)].sort_values(["seed", "qi"])
            assert (X.qi.values == Y.qi.values).all() and (X.seed.values == Y.seed.values).all(), "test questions differ"
            cells = []
            for key in ("c@8", "c@15", "R@10"):
                m_, lo, hi = evaluate.group_boot(X[key].values - Y[key].values, X.group.tolist(), 3000)
                cells.append(f"{key} {m_:+5.1f} [{lo:+.1f}, {hi:+.1f}]")
            print(f"  {a} minus {b} | {k:20s} " + "   ".join(cells))
print(buf.getvalue())
out = Path(args.out) / "results" / (f"SUPERVISED_CROSSENCODER_{TAG}.txt" if args.run == "nift_ce" else f"CROSSENCODER_{args.run}_{TAG}.txt")
out.write_text(buf.getvalue())
print(f"saved: {out} | per-seed results in results/{args.run}_seed<N>/results.csv")
