"""Two extensions of NIFT-SQL with the fine-tuned cross-encoder, fixed before any test scoring (paper Section 6.8).

    python experiments/fusion_extensions.py --data D --out O [--seeds 1 2 3 4] [--validation-only]

  A  NIFT-SQL + fine-tuned search. The same fusion (8 features, C = 1, candidates = scorer top 50 + search top 50), with the
     base hybrid search replaced by hybrid search on the per-customer fine-tuned BGE of experiments/finetuned_search_baseline.py.
     That BGE was trained on the TRAINING questions only, so its validation scores come from a model that never saw them, as
     the fusion requires. The dense feature is the fine-tuned model's raw cosine, recomputed from its saved checkpoint and
     checked against the cached hybrid scores. Tables outside the candidates follow in fine-tuned search order.
  B  High-accuracy mode: NIFT-SQL (base search) plus two reranker features for every candidate, the Qwen3-Reranker-8B
     logit and the log rank of that score among the candidates. Zero-shot reranker, prompt as in Appendix E.
  C  Both: A plus the two reranker features.

Protocol, fixed before the test part is scored:
  * The recomputed fine-tuned cosine must reproduce the cached hybrid scores up to GPU precision: max difference below
    0.02 and at least 98% agreement on each question's top 50 (a different model fails both by a wide margin).
  * The three variants are reported whatever they show; none is dropped or renamed after the test results.
  * Every cross-encoder training run R (nift_ce, nift_ce_rep1, nift_ce_rep2) is paired with fine-tuned-search run R
    (nift_ftsearch, nift_ftsearch_rep1, nift_ftsearch_rep2). Headline differences are means over training runs with 95%
    intervals from a two-level bootstrap over training runs and gold-table-set clusters, as in Table G5.
  * A variant "improves" on a baseline only if that interval excludes zero.
  * Reranker scores are computed FRESH for exactly the candidate pairs of every validation and test question (the union of
    the candidates of every variant and training run), into their own cache file. The earlier reranker caches are not used
    for features: their coverage depended on the gold database (Section 6.5), so whether a pair had a score could leak it.
    Every candidate must have a score; this is asserted.
  * The fusion is fitted on validation questions only; test questions are scored once. A second test scoring is refused.

Outputs: <out>/results/FUSION_EXTENSIONS.txt (report), <out>/results/fusion_ext/test_rows.csv (per question),
<out>/results/fusion_ext/validation.txt (the validation-only check). GPU: about 20 GB for the reranker, roughly 40 minutes on an
A100 for all four seeds; the dense recompute is a few minutes. Use --reranker fake for the offline tests only.
"""
import argparse
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nift import evaluate, fusion, scorer  # noqa: E402
from nift.config import REPO               # noqa: E402
from nift.pipeline import Run              # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--ce", default="bge-reranker-large", help="tag of the cross-encoder score files")
ap.add_argument("--reranker", default="Qwen/Qwen3-Reranker-8B", help="'fake' = deterministic hash scores, tests only")
ap.add_argument("--reranker-k", type=int, default=100, help="the reranker baseline's search shortlist (Section 4)")
ap.add_argument("--reps", type=int, nargs="+", default=[0, 1, 2], help="training runs: 0 = main run, R = retrain R")
ap.add_argument("--batch", type=int, default=16, help="reranker pairs per batch; halved automatically on out-of-memory")
ap.add_argument("--validation-only", action="store_true", help="stop after the validation check; nothing on test")
ap.add_argument("--overwrite", action="store_true", help="only if an earlier test scoring was broken")
args = ap.parse_args()

RES = Path(args.out) / "results"
OUT = RES / "fusion_ext"; OUT.mkdir(parents=True, exist_ok=True)
CACHE = RES / ".cache" / "reranker"; CACHE.mkdir(parents=True, exist_ok=True)
SETS = ["beaver", "livesqlbench", "livesqlbench_large"]
RTAG = args.reranker.rstrip("/").split("/")[-1]
TEST_ROWS = OUT / "test_rows.csv"
if TEST_ROWS.exists() and not (args.overwrite or args.validation_only):
    sys.exit(f"{TEST_ROWS} exists: the test part was already scored. Pass --overwrite only if that run was broken.")

CE_RUN = lambda R: "nift_ce" if R == 0 else f"nift_ce_rep{R}"                 # noqa: E731
FT_RUN = lambda R: "nift_ftsearch" if R == 0 else f"nift_ftsearch_rep{R}"     # noqa: E731
NIFT, VA, VB, VC = "NIFT-SQL", "A: NIFT-SQL + fine-tuned search", "B: NIFT-SQL + reranker features", "C: A + reranker features"
VARS = [NIFT, VA, VB, VC]


# ------------------------------------------------------------------ loading
def ce_scores(seed, R, k):
    f = RES / f"{CE_RUN(R)}_seed{seed}" / "scores" / f"{k}_ce_{args.ce}.npy"
    assert f.exists(), f"missing {f}: run experiments/supervised_crossencoder.py (replicate {R}) for seed {seed}"
    return np.load(f)


def ft_search(seed, R, r):
    """Fine-tuned hybrid scores and the fine-tuned raw cosine for each set, for training run R."""
    rd = RES / f"{FT_RUN(R)}_seed{seed}"
    out = {}
    for cust, names in r.cfg["customers"].items():
        ck = rd / "checkpoints" / f"bge_{cust}"
        dense_f = {k: OUT / ".cache" / f"dense_{FT_RUN(R)}_seed{seed}_{k}.npy" for k in names}
        need = [k for k in names if not dense_f[k].exists()]
        if need:
            assert (ck / "modules.json").exists(), (f"missing fine-tuned BGE checkpoint {ck}: the dense feature needs it. "
                                                    "Rerun experiments/finetuned_search_baseline.py for this seed and replicate.")
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(str(ck), device=r.device)
            enc = lambda xs: model.encode(list(xs), normalize_embeddings=True, batch_size=128, show_progress_bar=False)  # noqa: E731
            dense_f[need[0]].parent.mkdir(parents=True, exist_ok=True)
            for k in need:
                D = r.sets[k]
                np.save(dense_f[k], enc([q for _, q, _ in D.Q]) @ enc(D.cards).T)
            del model
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()             # free BGE before the 8B reranker loads
            except ImportError:
                pass
        for k in names:
            D = r.sets[k]
            hyb = np.load(rd / "scores" / f"{k}_hyb_ft.npy")
            dense = np.load(dense_f[k])
            mm = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)         # noqa: E731  same normalization as nift/search.py
            rebuilt = r.cfg["search"]["w_dense"] * np.stack([mm(x) for x in dense]) + r.cfg["search"]["w_bm25"] * D.extra["bm25"]
            # Same model, recomputed later: GPU matmul precision (TF32) moves cosines by ~1e-3, which min-max scaling enlarges.
            # A wrong checkpoint (base BGE, another seed) moves scores by 0.1 or more and reorders the ranking. Check both.
            err = float(np.abs(rebuilt - hyb).max())
            qs = r.SPLIT[k]["val"] + r.SPLIT[k]["test"]
            top = lambda h: set(np.argsort(-h, kind="stable")[:50].tolist())  # noqa: E731
            agree = float(np.mean([len(top(rebuilt[i]) & top(hyb[i])) / min(50, D.n) for i in qs]))
            print(f"{FT_RUN(R)} seed {seed} {k}: recomputed vs cached fine-tuned hybrid scores: max diff {err:.2e}, "
                  f"top-50 agreement {100 * agree:.1f}%")
            assert err < 0.02 and agree >= 0.98, (f"{FT_RUN(R)} seed {seed} {k}: the checkpoint does not reproduce the cached "
                                                   f"hybrid scores (max diff {err:.2e}, top-50 agreement {100 * agree:.1f}%): wrong model?")
            out[k] = dict(hyb=hyb, dense=dense)
    return out


def with_inputs(D, P, search=None):
    """A copy of D whose scorer scores are P and, optionally, whose search inputs are the fine-tuned ones."""
    ex = dict(D.extra, P=P)
    if search is not None:
        ex.update(dense=search["dense"], hyb=search["hyb"])
    return dataclasses.replace(D, extra=ex)


# ------------------------------------------------------------------ reranker
if args.reranker == "fake":
    import hashlib

    def rerank(pairs):
        return [int(hashlib.md5((q + d).encode()).hexdigest()[:6], 16) / 16**6 for q, d in pairs]
else:
    _rr = {}

    def rerank(pairs):
        if not _rr:                                                   # loaded on first use, so the cached path needs no GPU
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            tok = AutoTokenizer.from_pretrained(args.reranker, padding_side="left")
            mdl = AutoModelForCausalLM.from_pretrained(args.reranker, torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev).eval()
            pre = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
                   "provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
            suf = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            _rr.update(torch=torch, dev=dev, tok=tok, mdl=mdl, yes=tok.convert_tokens_to_ids("yes"), no=tok.convert_tokens_to_ids("no"),
                       pre=tok.encode(pre, add_special_tokens=False), suf=tok.encode(suf, add_special_tokens=False))
        z, torch = _rr, _rr["torch"]
        instruct = ("Given a question about a database, judge whether this table is required to write the SQL query that answers "
                    "the question: it holds answer columns, filter columns, or is needed to join such tables.")
        enc = z["tok"]([f"<Instruct>: {instruct}\n<Query>: {q}\n<Document>: {d}" for q, d in pairs], padding=False,
                       truncation="longest_first", return_attention_mask=False, max_length=2048 - len(z["pre"]) - len(z["suf"]))
        enc["input_ids"] = [z["pre"] + ids + z["suf"] for ids in enc["input_ids"]]
        batch = z["tok"].pad(enc, padding=True, return_tensors="pt").to(z["dev"])
        with torch.inference_mode():
            # Only the last position's "no" and "yes" logits are needed. Calling the full model would build logits over the
            # whole 151k vocabulary at every position (about 9 GB for a batch of long inputs). The decoder's last hidden
            # state times those two rows of the output layer gives the same two logits. Left padding: position -1 is real.
            h = z["mdl"].model(**batch).last_hidden_state[:, -1, :]
            head = z["mdl"].get_output_embeddings()
            rows = torch.tensor([z["no"], z["yes"]], device=h.device)
            two = h @ head.weight[rows].T
            if getattr(head, "bias", None) is not None:
                two = two + head.bias[rows]
            two = two.float().log_softmax(1)
        return two[:, 1].exp().tolist()


def rerank_safe(pairs):
    """rerank() with the batch halved on CUDA out-of-memory (long inputs come last, because batches are sorted by length)."""
    try:
        return rerank(pairs)
    except Exception as exc:                                          # torch.OutOfMemoryError without importing torch here
        if "out of memory" not in str(exc).lower() or len(pairs) == 1:
            raise
        import torch
        torch.cuda.empty_cache()
        half = len(pairs) // 2
        print(f"  out of GPU memory on {len(pairs)} pairs: retrying in two halves", flush=True)
        return rerank_safe(pairs[:half]) + rerank_safe(pairs[half:])


def doc(D, t, cfg):
    st = scorer.state(D, "", t, cfg)
    return st["table"] + (("\nJoins: " + "; ".join(st["joins"])) if st["joins"] else "")


def score_pairs(k, D, cfg, pairs):
    """Fill this experiment's own reranker cache for the given (question, table) pairs. Returns ms per pair (0 if cached)."""
    f = CACHE / f"{k}_{RTAG}_fusionext.npy"
    P = np.load(f) if f.exists() else np.full((len(D.Q), D.n), np.nan)
    todo = sorted({p for p in pairs if np.isnan(P[p])}, key=lambda it: len(D.Q[it[0]][1]) + len(doc(D, it[1], cfg)))
    t0 = time.time()
    for s0 in range(0, len(todo), args.batch):
        chunk = todo[s0:s0 + args.batch]
        for (i, t), p in zip(chunk, rerank_safe([(D.Q[i][1], doc(D, t, cfg)) for i, t in chunk])):
            P[i, t] = p
        if s0 % (args.batch * 100) == 0:
            np.save(f, P); print(f"{k}: reranker {s0 + len(chunk)}/{len(todo)} pairs", flush=True)
    np.save(f, P)
    return P, (1000 * (time.time() - t0) / len(todo) if todo else 0.0)


def add_reranker(e, RRk, i):
    """Two features from the reranker for every candidate: its logit and the log rank of its score among the candidates."""
    p = RRk[i, e["cand"]]
    assert not np.isnan(p).any(), "a candidate has no reranker score"
    p = np.clip(p, 1e-6, 1 - 1e-6)
    rank = np.empty(len(p)); rank[np.argsort(-p, kind="stable")] = np.arange(len(p))
    return dict(e, F=np.column_stack([e["F"], np.log(p / (1 - p)), np.log1p(rank)]))


# ------------------------------------------------------------------ build every variant's candidates and features
def build(seed):
    r = Run(seed, "nift_base", args.data, args.out, config_path=REPO / "configs" / "default.yaml")
    r.load(check_descriptions=False); r.split(); r.search()
    cfg, ex = r.cfg, {}                                        # ex[(R, variant, k, i)] = fusion example
    for R in args.reps:
        ft = ft_search(seed, R, r)
        for k, D in r.sets.items():
            P = ce_scores(seed, R, k)
            Db, Df = with_inputs(D, P), with_inputs(D, P, ft[k])
            for i in r.SPLIT[k]["val"] + r.SPLIT[k]["test"]:
                ex[(R, NIFT, k, i)] = fusion.features(Db, i, cfg)
                ex[(R, VA, k, i)] = fusion.features(Df, i, cfg)
    return r, ex


runs, EX, RR, ms_pair = {}, {}, {}, {}
for seed in args.seeds:
    runs[seed], EX[seed] = build(seed)
first = runs[args.seeds[0]]
for k in SETS:                                                  # reranker: every candidate of every variant, fresh
    pairs = {(i, int(t)) for seed in args.seeds for (R, v, kk, i), e in EX[seed].items() if kk == k for t in e["cand"]}
    RR[k], ms_pair[k] = score_pairs(k, first.sets[k], first.cfg, pairs)
for seed in args.seeds:
    for (R, v, k, i), e in list(EX[seed].items()):
        if v == NIFT:
            EX[seed][(R, VB, k, i)] = add_reranker(e, RR[k], i)
        elif v == VA:
            EX[seed][(R, VC, k, i)] = add_reranker(e, RR[k], i)


def fit_all(seed, R, parts):
    """One fusion per customer and variant, fitted on the given question indices of each set (validation only)."""
    r, models = runs[seed], {}
    for cust, names in r.cfg["customers"].items():
        for v in VARS:
            exs = [EX[seed][(R, v, k, i)] for k in names for i in parts[k]]
            models[(cust, v)] = fusion.fit(exs, r.cfg)
    return models


# ------------------------------------------------------------------ validation-only check (two folds inside validation)
vrows = []
for seed in args.seeds:
    r = runs[seed]
    rng = np.random.default_rng(seed)
    for R in args.reps:
        folds = {k: np.array_split(rng.permutation(r.SPLIT[k]["val"]), 2) for k in r.sets}
        for f_ in (0, 1):
            fit_on = {k: folds[k][1 - f_].tolist() for k in r.sets}
            models = fit_all(seed, R, fit_on)
            for cust, names in r.cfg["customers"].items():
                for k in names:
                    for i in folds[k][f_].tolist():
                        for v in VARS:
                            e = EX[seed][(R, v, k, i)]
                            vrows.append(dict(seed=seed, rep=R, set=k, qi=i, method=v,
                                              **evaluate.metrics(fusion.rank(models[(cust, v)], e), e["gold"], r.cfg)))
V = pd.DataFrame(vrows)
vtab = V.groupby(["set", "method", "seed", "rep"])[["c@8", "c@15"]].mean().mul(100).groupby(level=[0, 1]).mean().round(1).unstack("set")
vtxt = ("VALIDATION ONLY (two folds inside validation, fusion fitted on one half and scored on the other), mean over seeds and "
        f"training runs, %\n{vtab.to_string()}\n")
(OUT / "validation.txt").write_text(vtxt)
print(vtxt)
if args.validation_only:
    sys.exit("--validation-only: stopped before the test part")


# ------------------------------------------------------------------ test, once
rows, wts, ncand = [], [], {v: [] for v in VARS}
for seed in args.seeds:
    r = runs[seed]
    for R in args.reps:
        models = fit_all(seed, R, {k: r.SPLIT[k]["val"] for k in r.sets})
        for (cust, v), m in models.items():
            if v in (VB, VC):
                w = m[-1].coef_[0]
                wts.append(dict(seed=seed, rep=R, customer=cust, method=v, reranker_logit=w[-2], reranker_log_rank=w[-1],
                                scorer_logit=w[0], search_log_rank=w[2]))
        for cust, names in r.cfg["customers"].items():
            for k in names:
                D = r.sets[k]
                for i in r.SPLIT[k]["test"]:
                    g = set(D.Q[i][2]); dbs = {D.dbo[t] for t in g}
                    base = dict(seed=seed, rep=R, set=k, qi=i, group="|".join(map(str, sorted(g))))
                    for v in VARS:
                        e = EX[seed][(R, v, k, i)]
                        o = fusion.rank(models[(cust, v)], e)
                        rows.append(dict(base, method=v, **evaluate.metrics(o, g, r.cfg)))
                        if len(dbs) == 1:
                            db = next(iter(dbs))
                            rows.append(dict(base, method=f"{v} | database given", **evaluate.metrics([t for t in o if D.dbo[t] == db], g, r.cfg)))
                        if R == args.reps[0]:
                            ncand[v].append(len(e["cand"]))


# baselines without training runs, rebuilt exactly as in experiments/additional_analyses.py
def regroup(order, D):
    pos = {}
    for p, t in enumerate(order):
        pos.setdefault(D.dbo[t], p)
    return sorted(order, key=lambda t: pos[D.dbo[t]])


def top2_tables(base, D):
    dbs = []
    for t in base:
        if D.dbo[int(t)] not in dbs:
            dbs.append(D.dbo[int(t)])
        if len(dbs) == 2:
            break
    return [t for t in range(D.n) if D.dbo[t] in dbs], dbs


RRN = "reranker (search top 100)"
RRV = "reranker, variant chosen on validation"
RRW = "reranker over the whole database | database given"
VARIANTS = ["plain", "grouped by database", "routed by search", "routed by its own ranking"]
for seed in args.seeds:
    r = runs[seed]
    PA = {k: (np.load(f) if (f := CACHE / f"{k}_{RTAG}_all.npy").exists() else None) for k in r.sets}
    PD = {k: (np.load(f) if (f := CACHE / f"{k}_{RTAG}_db.npy").exists() else None) for k in r.sets}
    for k, D in r.sets.items():
        def variants(i):
            P_ = PA[k]
            ho = np.argsort(-D.extra["hyb"][i], kind="stable"); top = ho[:args.reranker_k]
            if P_ is None or np.isnan(P_[i, top]).any():
                return None
            plain = top[np.argsort(-P_[i, top], kind="stable")].tolist() + ho[args.reranker_k:].tolist()
            out = {"plain": plain, "grouped by database": regroup(plain, D)}
            for v, b in (("routed by search", ho.tolist()), ("routed by its own ranking", plain)):
                ix, dbs = top2_tables(b, D)
                if np.isnan(P_[i, ix]).any():
                    return None
                out[v] = sorted(ix, key=lambda t: -P_[i, t]) + [int(t) for t in regroup(b, D) if D.dbo[int(t)] not in dbs]
            return out
        vs = {v: [] for v in VARIANTS}
        for i in r.SPLIT[k]["val"]:
            if (o := variants(i)) is not None:
                for v in VARIANTS:
                    vs[v].append(evaluate.metrics(o[v], set(D.Q[i][2]), r.cfg)["c@15"])
        pick = max(VARIANTS, key=lambda v: (np.mean(vs[v]) if vs[v] else -1.0, -VARIANTS.index(v)))
        for i in r.SPLIT[k]["test"]:
            g = set(D.Q[i][2]); base = dict(seed=seed, rep=-1, set=k, qi=i, group="|".join(map(str, sorted(g))))
            ho = np.argsort(-D.extra["hyb"][i], kind="stable").tolist()
            rows.append(dict(base, method="search alone", **evaluate.metrics(ho, g, r.cfg)))
            if (o := variants(i)) is not None:
                rows.append(dict(base, method=RRN, **evaluate.metrics(o["plain"], g, r.cfg)))
                rows.append(dict(base, method=RRV, **evaluate.metrics(o[pick], g, r.cfg)))
            dbs = {D.dbo[t] for t in g}
            if PD[k] is not None and len(dbs) == 1:
                ix = [t for t in range(D.n) if D.dbo[t] in dbs]
                if not np.isnan(PD[k][i, ix]).any():
                    rows.append(dict(base, method=RRW, **evaluate.metrics(sorted(ix, key=lambda t: -PD[k][i, t]), g, r.cfg)))
    for R in args.reps:                                            # fine-tuned retrievers on their own (paired run R)
        for k, D in r.sets.items():
            f = RES / f"{FT_RUN(R)}_seed{seed}" / "scores" / f"{k}_hyb_ft.npy"
            H = np.load(f)
            for i in r.SPLIT[k]["test"]:
                g = set(D.Q[i][2])
                rows.append(dict(seed=seed, rep=R, set=k, qi=i, group="|".join(map(str, sorted(g))), method="fine-tuned search alone",
                                 **evaluate.metrics(np.argsort(-H[i], kind="stable").tolist(), g, r.cfg)))
    for R in range(0, 6):                                          # matched label budget: trained on train + validation
        run_ = "nift_ftsearch_trval" + (f"_rep{R}" if R else "")
        for k, D in r.sets.items():
            if (f := RES / f"{run_}_seed{seed}" / "scores" / f"{k}_hyb_ft.npy").exists():
                H = np.load(f)
                for i in r.SPLIT[k]["test"]:
                    g = set(D.Q[i][2])
                    rows.append(dict(seed=seed, rep=100 + R, set=k, qi=i, group="|".join(map(str, sorted(g))),
                                     method="fine-tuned search, train + validation", **evaluate.metrics(np.argsort(-H[i], kind="stable").tolist(), g, r.cfg)))

T = pd.DataFrame(rows)
T.to_csv(TEST_ROWS, index=False)


# ------------------------------------------------------------------ report
def per_run(m, k, key):
    """{rep: per-question series indexed by (seed, qi)} for method m on set k."""
    X = T[(T.method == m) & (T.set == k)]
    return {R: g.set_index(["seed", "qi"])[key] for R, g in X.groupby("rep")}, X.drop_duplicates(["seed", "qi"]).set_index(["seed", "qi"]).group


def boot(a, b, k, key, paired):
    """Mean over training runs of (a - b), with a two-level bootstrap over runs and gold-table-set clusters.
    paired: b has the same training runs as a (difference taken run by run). Otherwise b's runs (or its single ranking)
    are resampled independently."""
    ra, grp = per_run(a, k, key)
    rb, _ = per_run(b, k, key)
    if not ra or not rb:
        return None
    idx = sorted(set.intersection(*[set(s.index) for s in list(ra.values()) + list(rb.values())]))
    if not idx:
        return None
    A_ = np.vstack([ra[R].loc[idx].values for R in sorted(ra)])
    B_ = np.vstack([rb[R].loc[idx].values for R in sorted(rb)])
    gl = (k + "#" + grp.loc[idx]).tolist(); ids = {g: j for j, g in enumerate(dict.fromkeys(gl))}
    gi = np.array([ids[g] for g in gl]); N = np.bincount(gi, minlength=len(ids))
    S = lambda M: np.stack([np.bincount(gi, weights=x, minlength=len(ids)) for x in M])   # noqa: E731
    rng, st = np.random.default_rng(0), np.empty(3000)
    if paired:
        assert sorted(ra) == sorted(rb)
        Sd = S(A_ - B_)
        for b_ in range(3000):
            rr, cc = rng.integers(0, len(Sd), len(Sd)), rng.integers(0, len(ids), len(ids))
            st[b_] = Sd[rr][:, cc].sum() / len(Sd) / N[cc].sum()
    else:
        Sa, Sb = S(A_), S(B_)
        for b_ in range(3000):
            cc = rng.integers(0, len(ids), len(ids))
            st[b_] = (Sa[rng.integers(0, len(Sa), len(Sa))][:, cc].sum() / len(Sa)
                      - Sb[rng.integers(0, len(Sb), len(Sb))][:, cc].sum() / len(Sb)) / N[cc].sum()
    return 100 * (A_.mean() - B_.mean()), 100 * np.percentile(st, 2.5), 100 * np.percentile(st, 97.5)


def line(a, b, paired):
    out = []
    for k in SETS:
        cells = []
        for key in ("c@8", "c@15"):
            res = boot(a, b, k, key, paired)
            if res is None:
                break
            flag = " *" if res[1] > 0 or res[2] < 0 else "  "
            cells.append(f"{key} {res[0]:+5.1f} [{res[1]:+.1f}, {res[2]:+.1f}]{flag}")
        if len(cells) == 2:
            out.append(f"  {a} minus {b} | {k:20s} " + "   ".join(cells))
    return out


means = (T.groupby(["set", "method", "rep", "seed"])[["c@8", "c@15"]].mean().mul(100)
         .groupby(level=[0, 1, 2]).mean().groupby(level=[0, 1]).mean().round(1).unstack("set"))
D_ = lambda m: f"{m} | database given"   # noqa: E731
FTS, FTV = "fine-tuned search alone", "fine-tuned search, train + validation"
L = [f"FUSION EXTENSIONS | test, seeds {' '.join(map(str, args.seeds))} | training runs {args.reps} | fusion fitted on validation",
     "Means: over seeds of each seed's test mean, then over training runs, %. Differences: mean over training runs, 95% interval "
     "from a two-level bootstrap over training runs and gold-table-set clusters; * = interval excludes zero.", "",
     means.to_string(), "", "Against NIFT-SQL (paired run by run):"]
for v in (VA, VB, VC):
    L += line(v, NIFT, True)
L += ["", "Between variants (paired):"] + line(VC, VA, True) + line(VC, VB, True)
L += ["", "Against the supervised retriever:"] + line(VA, FTS, True) + line(VA, FTV, False) + line(VC, FTV, False) + line(NIFT, FTV, False)
L += ["", "Against the zero-shot reranker (no training runs; resampled as a single ranking):"]
for v in VARS:
    L += line(v, RRV, False)
L += line(VB, RRN, False)
L += ["", "Database given:"]
for v in (VA, VB, VC):
    L += line(D_(v), D_(NIFT), True) + line(D_(v), RRW, False)
W = pd.DataFrame(wts)
L += ["", "Fusion weights of the reranker features (standardized, mean over seeds and training runs):",
      W.groupby(["method", "customer"])[["scorer_logit", "search_log_rank", "reranker_logit", "reranker_log_rank"]].mean().round(3).to_string(),
      "", "Cost: candidates per test question (the pairs the reranker scores in B and C) and reranker time per pair"]
for v in VARS:
    L.append(f"  {v}: {np.mean(ncand[v]):.1f} candidates per question")
L += [f"  reranker ms per pair (this run, batched): " + ", ".join(f"{k} {ms_pair[k]:.1f}" for k in SETS), "", vtxt]
(RES / "FUSION_EXTENSIONS.txt").write_text("\n".join(L) + "\n")
print("\n".join(L))
