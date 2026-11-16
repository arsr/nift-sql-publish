"""Every number the paper needs, in one text file. CPU only: reads finished runs and caches, trains and scores nothing.

    python experiments/paper_tables.py --data D --out O

Tables: T1 datasets | T2 main result (test, 4 seeds) | T3 NIFT-SQL vs every baseline, pooled CIs | T4 complete@15 by
number of gold tables | T5 shortlist coverage | T6 cost and speed | A1 appendix: fine-tuned BGE + join completion vs main,
paired on the seeds both have. Writes <out>/results/PAPER_TABLES.txt.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nift import evaluate              # noqa: E402
from nift.config import REPO           # noqa: E402
from nift.pipeline import Run          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--main", default="nift_base")
ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
ap.add_argument("--reranker", default="Qwen3-Reranker-8B")
ap.add_argument("--appendix", default="arjun_v3j", help="fine-tuned BGE + join completion runs (config nift_ftj)")
ap.add_argument("--k", type=int, default=100)
args = ap.parse_args()
res = Path(args.out) / "results"
out, MET = [], ["c@5", "c@8", "c@15", "c@25", "R@5", "R@10", "R@15"]
RR = f"search + {args.reranker} (zero-shot)"
pd.set_option("display.width", 250)


def section(title, body):
    out.append("\n" + "=" * 110 + f"\n{title}\n" + "=" * 110 + "\n" + body)
    print(out[-1])


# ---------------------------------------------------------------- rebuild main runs (cached), add reranker orders
rows, cov, cost, stats, runs = [], [], [], None, {}
for s in args.seeds:
    r = Run(s, args.main, args.data, args.out)
    r.load(check_descriptions=False); r.split(); r.search(); r.build_rows(show=0); r.train(); r.score()
    runs[s] = r
    if stats is None:
        stats = []
        for k, D in r.sets.items():
            ng = np.array([len(g) for _, _, g in D.Q])
            ncol = np.array([len(c.split(":", 1)[1].split(",")) if ":" in c else 0 for c in D.cards])
            stats.append(dict(dataset=k, tables=D.n, databases=len(D.DBI), questions=len(D.Q), columns_per_table=round(ncol.mean(), 1),
                              tables_per_sql=round(ng.mean(), 2), **{"3+ tables %": round(100 * np.mean(ng >= 3)), "5+ tables %": round(100 * np.mean(ng >= 5))},
                              distinct_gold_sets=len({frozenset(g) for _, _, g in D.Q}),
                              train_val_test=f"{len(r.SPLIT[k]['train'])}/{len(r.SPLIT[k]['val'])}/{len(r.SPLIT[k]['test'])} (seed {s})"))
    A = pd.read_csv(r.rd / "results.csv").assign(seed=s)
    rows.append(A)
    top = r.cfg["fusion"]["candidates_top"]
    for k, D in r.sets.items():
        f = res / ".cache" / "reranker" / f"{k}_{args.reranker}_top{args.k}.npy"
        P = np.load(f) if f.exists() else None
        for i in r.SPLIT[k]["test"]:
            g = set(D.Q[i][2]); ho = np.argsort(-D.extra["hyb"][i], kind="stable"); co = np.argsort(-D.extra["P"][i], kind="stable")
            if P is not None and not np.isnan(P[i, ho[:args.k]]).any():
                t = ho[:args.k]
                order = t[np.argsort(-P[i, t], kind="stable")].tolist() + ho[args.k:].tolist()
                rows.append(pd.DataFrame([dict(set=k, qi=i, n_gold=len(g), group="|".join(map(str, sorted(g))), method=RR, seed=s,
                                               **evaluate.metrics(order, g, r.cfg))]))
            c = dict(seed=s, set=k)
            for K in (50, 100, 200):
                c[f"search top {K}"] = float(g <= set(ho[:K].tolist()))
            c["Laya top 100"] = float(g <= set(co[:100].tolist()))
            c["NIFT-SQL candidates"] = float(g <= set(co[:top].tolist()) | set(ho[:top].tolist()))
            cov.append(c)
    cj = json.loads((r.rd / "costs.json").read_text())
    for name, v in cj.items():
        cost.append(dict(seed=s, stage=name, minutes=v["gpu_minutes"], usd=v["usd"], ms_per_question=v.get("ms_per_question")))
A = pd.concat(rows, ignore_index=True)
methods = ["search alone", RR, "Laya alone", "rank average (no learning)", "NIFT-SQL (fusion)"]
methods = [m for m in methods if m in set(A.method)]

section("T1. DATASETS (names-only cards)", pd.DataFrame(stats).set_index("dataset").T.to_string())

per = A.groupby(["set", "method", "seed"])[MET].mean().mul(100)
T2 = per.groupby(level=[0, 1]).agg(["mean", "std"]).round(1)
T2 = T2.reindex([(k, m) for k in sorted(A.set.unique()) for m in methods])
section(f"T2. MAIN RESULT, test, seeds {args.seeds} ({args.main}), mean and std over seeds", T2.to_string())

lines = []
for k in sorted(A.set.unique()):
    a = A[(A.set == k) & (A.method == "NIFT-SQL (fusion)")].sort_values(["seed", "qi"])
    for b in [m for m in methods if m != "NIFT-SQL (fusion)"]:
        c = A[(A.set == k) & (A.method == b)].sort_values(["seed", "qi"])
        cells = []
        for key in ("c@8", "c@15", "R@10"):
            m_, lo, hi = evaluate.group_boot(a[key].values - c[key].values, a.group.tolist(), 3000)
            cells.append(f"{key} {m_:+5.1f} [{lo:+.1f}, {hi:+.1f}]")
        lines.append(f"{k:20s} vs {b:42s} " + "   ".join(cells))
section("T3. NIFT-SQL MINUS EACH BASELINE, 4 seeds pooled [95% CI, clustered by gold-table set across seeds]", "\n".join(lines))

A["size"] = pd.cut(A.n_gold, [0, 1, 2, 4, 100], labels=["1", "2", "3-4", "5+"])
T4 = A.groupby(["set", "method", "size"], observed=True)["c@15"].mean().mul(100).round(1).unstack("size")
n4 = A[A.method == "NIFT-SQL (fusion)"].groupby(["set", "size"], observed=True).size().unstack("size")
section("T4. COMPLETE@15 BY NUMBER OF GOLD TABLES, all test questions of the 4 seeds",
        T4.reindex([(k, m) for k in sorted(A.set.unique()) for m in methods]).to_string() + "\n\nquestions per cell (summed over seeds):\n" + n4.to_string())

C = pd.DataFrame(cov)
T5 = C.groupby(["set", "seed"]).mean().mul(100).groupby(level=0).agg(["mean", "std"]).round(1)
section("T5. SHORTLIST COVERAGE, test: % of questions with every gold table in the shortlist", T5.T.to_string())

K = pd.DataFrame(cost)
T6 = K.groupby("stage")[["minutes", "usd", "ms_per_question"]].mean().round(3)
tables = {k: n for k, n in zip([s_["dataset"] for s_ in stats], [s_["tables"] for s_ in stats])}
extra = []
for k, n in tables.items():
    row = K[K.stage.str.startswith(f"score Laya, {k} ")]
    if len(row):
        extra.append(f"Laya: {k} {row.ms_per_question.mean() / n:.1f} ms per table ({n} tables, {row.ms_per_question.mean() / 1000:.2f} s per question)")
rrlog = res / "logs"
section("T6. COST AND SPEED (mean over seeds; GPU price as in configs)", T6.to_string() + "\n\n" + "\n".join(extra) +
        f"\n{args.reranker}: see the reranker run's 'ms per table' line (about 25-34 ms per table on the top {args.k})")

# ---------------------------------------------------------------- appendix: fine-tuned BGE + join completion, paired seeds
app = []
for s in args.seeds:
    f = res / f"{args.appendix}_seed{s}" / "results.csv"
    if f.exists():
        app.append(pd.read_csv(f).assign(seed=s))
if app:
    B = pd.concat(app)
    seeds_b = [int(x) for x in sorted(B.seed.unique())]
    M = A[A.seed.isin(seeds_b) & (A.method == "NIFT-SQL (fusion)")]
    X = B[B.method == "NIFT-SQL (fusion)"]
    lines = [f"seeds with both runs: {seeds_b}", ""]
    t = pd.concat([M.assign(variant="main (base BGE)"), X.assign(variant="fine-tuned BGE + join completion")])
    lines.append(t.groupby(["set", "variant"])[["c@8", "c@15", "c@25", "R@10"]].mean().mul(100).round(1).to_string())
    lines.append("\nfine-tuned BGE + join completion minus main, paired [95% CI]")
    for k in sorted(X.set.unique()):
        a = X[X.set == k].sort_values(["seed", "qi"]); b = M[M.set == k].sort_values(["seed", "qi"])
        if len(a) != len(b):
            continue
        cells = []
        for key in ("c@8", "c@15"):
            m_, lo, hi = evaluate.group_boot(a[key].values - b[key].values, a.group.tolist(), 3000)
            cells.append(f"{key} {m_:+.1f} [{lo:+.1f}, {hi:+.1f}]")
        lines.append(f"  {k:20s} " + "   ".join(cells))
    section("A1. APPENDIX: FINE-TUNED BGE + JOIN COMPLETION vs MAIN (test)", "\n".join(lines))

# ---------------------------------------------------------------- T7: training-size curve, clustered CIs
curve = [("10%", "nift_train10"), ("20%", "nift_train20"), ("30%", "nift_train30"), ("50% (main)", args.main)]
cr = []
for lab, name in curve:
    for s in args.seeds:
        f = res / f"{name}_seed{s}" / "results.csv"
        if f.exists():
            cr.append(pd.read_csv(f).assign(size=lab, seed=s))
if cr:
    Cc = pd.concat(cr); order = [c for c, _ in curve if c in set(Cc["size"])]
    lines = []
    for key in ("c@8", "c@15"):
        T = Cc.groupby(["set", "method", "size", "seed"])[key].mean().mul(100).groupby(level=[0, 1, 2]).mean().round(1).unstack("size")[order]
        lines += [f"{key}, mean over seeds:", T.to_string(), ""]
    lines.append("NIFT-SQL at each size minus at 50% [95% CI, clustered by gold-table set across seeds]")
    for k in sorted(Cc.set.unique()):
        full = Cc[(Cc.set == k) & (Cc.method == "NIFT-SQL (fusion)") & (Cc["size"] == "50% (main)")].sort_values(["seed", "qi"])
        for lab in order[:-1]:
            a = Cc[(Cc.set == k) & (Cc.method == "NIFT-SQL (fusion)") & (Cc["size"] == lab)].sort_values(["seed", "qi"])
            if len(a) != len(full) or (a.qi.values != full.qi.values).any():
                continue
            cells = []
            for key in ("c@8", "c@15"):
                m_, lo, hi = evaluate.group_boot(a[key].values - full[key].values, a.group.tolist(), 3000)
                cells.append(f"{key} {m_:+.1f} [{lo:+.1f}, {hi:+.1f}]")
            lines.append(f"  {k:20s} {lab:4s} " + "   ".join(cells))
    section("T7. TRAINING-SIZE CURVE (test)", "\n".join(lines))

# ---------------------------------------------------------------- T0: data and protocol checks (raw files)
first = runs[args.seeds[0]]
import re
from collections import Counter
from nift import config as Cfg
cfg0 = Cfg.load_config()
chk = []
KW = re.compile(r"^(default|not|null|references|constraint|primary|foreign|unique|check|on|using|key|index)$", re.I)
for name, sub in cfg0["datasets"].items():
    root = Path(args.data) / sub
    ids = [json.loads(l)["_id"] for l in open(root / "corpus.jsonl") if l.strip()]
    qrel = [l.rstrip("\n").split("\t") for l in list(open(root / "qrels" / "test.tsv"))[1:] if l.strip()]
    pos = [x for x in qrel if float(x[2]) > 0]
    missing = sum(x[1] not in set(ids) for x in pos)
    D = first.sets[name]
    bad_cols = Counter(c.strip() for card in D.cards if ":" in card for c in card.split(":", 1)[1].split(",") if KW.match(c.strip()))
    norm = lambda t: re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()                   # noqa: E731
    chk.append(f"{name}: corpus ids {len(ids)} (duplicates {len(ids) - len(set(ids))}) | positive qrels {len(pos)}, pointing to a "
               f"missing table {missing} | columns that look like SQL keywords: {dict(bad_cols) or 'none'}")
    cust = [c for c, ks in cfg0["customers"].items() if name in ks][0]
    for s, r in runs.items():
        side = {}
        for k in cfg0["customers"][cust]:                # whole customer: LiveSQLBench and -Large share questions
            for p in ("train", "val", "test"):
                for i in r.SPLIT[k][p]:
                    side.setdefault(norm(r.sets[k].Q[i][1]), set()).add(p)
        cross = sum(len(v) > 1 for v in side.values())
        chk.append(f"   seed {s}: question texts (customer {cust}) equal up to case/punctuation but in two parts: {cross}")
    seen = Counter(i for r in runs.values() for i in r.SPLIT[name]["test"])
    chk.append(f"   test questions appearing in more than one seed's test set: {sum(v > 1 for v in seen.values())} of {len(seen)} "
               f"(so pooled CIs are clustered by gold set across seeds, not per seed)")
section("T0. DATA AND PROTOCOL CHECKS", "\n".join(chk))

try:                                                   # real search cost (the runs reuse cached embeddings)
    import time
    import torch
    if torch.cuda.is_available():
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(cfg0["search"]["model"], device="cuda")
        lines = []
        for k, D in first.sets.items():
            t0 = time.time(); m.encode(D.cards, batch_size=128, normalize_embeddings=True); tc = time.time() - t0
            t0 = time.time(); m.encode([q for _, q, _ in D.Q], batch_size=128, normalize_embeddings=True); tq = time.time() - t0
            lines.append(f"{k}: embed {D.n} cards {tc:.1f} s (once per schema) | {len(D.Q)} questions {1000 * tq / len(D.Q):.1f} ms per question")
        section("T6b. SEARCH COST WITHOUT CACHE (BGE-large on this GPU)", "\n".join(lines))
except Exception as e:
    print("search timing skipped:", e)

(res / "PAPER_TABLES.txt").write_text("\n".join(out))
print(f"\nsaved: {res / 'PAPER_TABLES.txt'}")
