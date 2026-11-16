"""Additional analyses (Sections 6.5 to 7, Appendix G). CPU only: cached scores of the finished runs, nothing fitted on test.

    python experiments/additional_analyses.py --data D --out O [--seeds 1 2 3 4]

Reads the same runs as experiments/ce_analysis.py (nift_base, nift_ce, nift_train10/20/30, nift_ce_train10/20/30, the cached
reranker scores) and writes <out>/results/ADDITIONAL_ANALYSES.txt:

  R  Database grouping for the baselines. NIFT-SQL's fusion has two database features; search, rank averaging and the
     reranker have none. Here every ranking is regrouped by database without training: databases are ordered by their
     best-ranked table (rule "best") or by the summed reciprocal rank of their top 3 tables (rule "top3"), and tables keep
     their order inside each database. NIFT-SQL is regrouped the same way, so every method gets the same routing step.
  V  Database given: intervals for NIFT-SQL against the cross-encoder alone, Laya, search and the reranker, each ranking
     restricted to the question's database.
  S  Fusion fitted on fewer validation questions: 10, 20 and 40 questions per customer, 10 random draws per seed,
     against the fusion fitted on all of them. Rank averaging needs none.
  T  NIFT-SQL with the cross-encoder minus NIFT-SQL with Laya at each training size, with intervals.
  W  Inputs cut by the caps applied before tokenization: cards longer than card_max characters, tables with more than
     joins_max joins, and how many columns the cut removes, for all tables and for gold tables.
  E  Retrained replicates of the cross-encoder (nift_ce_rep<R>) and the deterministic-mode check.
  X  Replicate-aware comparisons: NIFT-SQL with the cross-encoder averaged over its training runs (the main run and each
     retrain), with 95% intervals from a two-level bootstrap that resamples training runs and gold-table-set clusters.
  P  Supervised-retriever baseline: hybrid search with BGE fine-tuned per customer (experiments/finetuned_search_baseline.py).
  Q  Zero-label routed reranker (top 2 databases by search). Q2: routed by the reranker's own ranking, and one reranker
     variant chosen per collection and seed on validation questions.
  P2 The fine-tuned retriever retrained (training variation) and trained on training plus validation questions (matched
     label budget), with intervals that resample both methods' training runs.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nift import evaluate, fusion      # noqa: E402
from nift.config import REPO           # noqa: E402
from nift.pipeline import Run          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--ce", default="bge-reranker-large")
ap.add_argument("--reranker", default="Qwen3-Reranker-8B")
ap.add_argument("--k", type=int, default=100)
ap.add_argument("--draws", type=int, default=10)
args = ap.parse_args()
RES = Path(args.out) / "results"
SETS = ["beaver", "livesqlbench", "livesqlbench_large"]
RRN = f"search + {args.reranker}"
rows = []


def regroup(order, D, rule):
    """Training-free routing: order databases by their best-ranked table ('best') or top-3 reciprocal ranks ('top3')."""
    pos = {}
    for p, t in enumerate(order):
        pos.setdefault(D.dbo[t], []).append(p)
    key = {db: (ps[0] if rule == "best" else -sum(1 / (1 + x) for x in ps[:3])) for db, ps in pos.items()}
    return sorted(order, key=lambda t: key[D.dbo[t]])          # stable: keeps the order inside each database


def record(seed, size, k, D, i, method, order, cfg, extra_given=False, routes=False):
    g = set(D.Q[i][2]); dbs = {D.dbo[t] for t in g}
    base = dict(seed=seed, size=size, set=k, qi=i, group="|".join(map(str, sorted(g))), qtext=D.Q[i][1].strip().lower())
    rows.append(dict(base, method=method, **evaluate.metrics(order, g, cfg)))
    if routes:
        for rule in ("best", "top3"):
            rows.append(dict(base, method=f"{method} | grouped by database ({rule})", **evaluate.metrics(regroup(order, D, rule), g, cfg)))
    if extra_given and len(dbs) == 1:
        db = next(iter(dbs))
        rows.append(dict(base, method=f"{method} | database given", **evaluate.metrics([t for t in order if D.dbo[t] == db], g, cfg)))


def rep_m(m, R):
    """Method name of training run R (0 = the main run)."""
    if not R:
        return m
    head, _, tail = m.partition(" | ")
    return f"{head}, retrain {R}" + (f" | {tail}" if tail else "")


def load_scores(r, run, fname):
    d = RES / f"{run}_seed{r.seed}" / "scores"
    fs = {k: d / fname.format(k=k) for k in r.sets}
    if not all(f.exists() for f in fs.values()):
        return False
    for k, f in fs.items():
        r.sets[k].extra["P"] = np.load(f)
    return True


def lr(cfg):
    return make_pipeline(StandardScaler(), LogisticRegression(C=cfg["fusion"]["C"], max_iter=3000))


def small_fits(r, seed, sizes):
    """S: fusions fitted on n randomly drawn validation questions per customer, args.draws draws each."""
    rng, small = np.random.default_rng(seed), {}
    for cust, names in r.cfg["customers"].items():
        val = [r.E[k][i] for k in names for i in r.SPLIT[k]["val"]]
        for n in sizes:
            fits, tries = [], 0
            while len(fits) < args.draws and n < len(val) and tries < 10 * args.draws:
                tries += 1
                pick = [val[j] for j in rng.choice(len(val), n, replace=False)]
                y = np.concatenate([e["y"] for e in pick])
                if 0 < y.sum() < len(y):
                    fits.append(lr(r.cfg).fit(np.vstack([e["F"] for e in pick]), y))
            small[(cust, n)] = fits
    return small


def record_small(seed, size, k, D, i, name, e, fits, n, cfg):
    if fits:                                                    # metrics averaged over the draws
        ms = pd.DataFrame([evaluate.metrics(fusion.rank(m, e), set(D.Q[i][2]), cfg) for m in fits]).mean()
        rows.append(dict(seed=seed, size=size, set=k, qi=i, group="|".join(map(str, sorted(D.Q[i][2]))),
                         qtext=D.Q[i][1].strip().lower(), method=f"NIFT-SQL with {name}, fusion on {n} validation questions", **ms.to_dict()))


caps, dets, choices, choice = [], [], [], {}
FTN, FTV = "fine-tuned search alone", "fine-tuned search alone, train + validation"
RRO, RRV = f"{RRN}, routed by its own ranking to the top 2 databases", f"{RRN}, variant chosen on validation"
VARIANTS = ["plain", "grouped by database", "routed by search", "routed by its own ranking"]


def rr_variants(i, D, P_):
    """The four zero-label reranker variants for question i, or None if a needed score is missing."""
    ho = np.argsort(-D.extra["hyb"][i], kind="stable"); top = ho[:args.k]
    if np.isnan(P_[i, top]).any():
        return None
    plain = top[np.argsort(-P_[i, top], kind="stable")].tolist() + ho[args.k:].tolist()
    out = {"plain": plain, "grouped by database": regroup(plain, D, "best")}
    for v, base in (("routed by search", ho.tolist()), ("routed by its own ranking", plain)):
        dbs2 = []
        for t in base:
            if D.dbo[int(t)] not in dbs2:
                dbs2.append(D.dbo[int(t)])
            if len(dbs2) == 2:
                break
        ix = [t for t in range(D.n) if D.dbo[t] in dbs2]
        if np.isnan(P_[i, ix]).any():
            return None
        out[v] = sorted(ix, key=lambda t: -P_[i, t]) + [int(t) for t in regroup(base, D, "best") if D.dbo[int(t)] not in dbs2]
    return out


NC_ = "NIFT-SQL with cross-encoder"
for seed in args.seeds:
    r = Run(seed, "nift_base", args.data, args.out, config_path=REPO / "configs" / "default.yaml")
    r.load(check_descriptions=False); r.split(); r.search()
    if seed == args.seeds[0]:                                   # W: what the caps cut (the same for every seed)
        cm, jm = r.cfg["card"]["card_max"], r.cfg["card"]["joins_max"]
        for k, D in r.sets.items():
            gold = set().union(*[set(q[2]) for q in D.Q])
            for t in range(D.n):
                c = D.cards[t]; cols = c.split(":", 1)[1].split(",") if ":" in c else []
                kept = c[:cm].split(":", 1)[1].split(",") if ":" in c[:cm] else []
                lost = 0 if len(c) <= cm else len(cols) - (len(kept) - 1)        # the last kept column may be cut mid-name
                caps.append(dict(set=k, gold=t in gold, cut=len(c) > cm, cols=len(cols), lost=max(lost, 0),
                                 joins=len(D.joins.get(t, [])), joins_cut=len(D.joins.get(t, [])) > jm))
    RRRT = {k: (np.load(f) if (f := RES / ".cache" / "reranker" / f"{k}_{args.reranker}_routed2.npy").exists() else None) for k in r.sets}
    RRDB = {k: (np.load(f) if (f := RES / ".cache" / "reranker" / f"{k}_{args.reranker}_db.npy").exists() else None) for k in r.sets}
    RR = {k: (np.load(f) if (f := RES / ".cache" / "reranker" / f"{k}_{args.reranker}_top{args.k}.npy").exists() else None) for k in r.sets}
    FT = {k: (np.load(f) if (f := RES / f"nift_ftsearch_seed{seed}" / "scores" / f"{k}_hyb_ft.npy").exists() else None) for k in r.sets}
    RRA = {k: (np.load(f) if (f := RES / ".cache" / "reranker" / f"{k}_{args.reranker}_all.npy").exists() else None) for k in r.sets}
    for k, D in r.sets.items():                                 # Q2: reranker routed by its own ranking; variant chosen on validation
        if RRA[k] is None:
            continue
        P_ = RRA[k]
        val_scores = {v: [] for v in VARIANTS}
        for part in ("val", "test"):
            for i in r.SPLIT[k][part]:
                o = rr_variants(i, D, P_)
                if o is None:
                    continue
                g = set(D.Q[i][2])
                if part == "val":
                    for v in VARIANTS:
                        val_scores[v].append(evaluate.metrics(o[v], g, r.cfg)["c@15"])
                else:
                    pick = choice[(seed, k)]
                    record(seed, "full", k, D, i, RRO, o["routed by its own ranking"], r.cfg)
                    record(seed, "full", k, D, i, RRV, o[pick], r.cfg)
            if part == "val":
                vm = {v: float(np.mean(x)) if x else -1.0 for v, x in val_scores.items()}
                choice[(seed, k)] = max(VARIANTS, key=lambda v: (vm[v], -VARIANTS.index(v)))   # ties: earlier in VARIANTS
                choices.append(dict(seed=seed, set=k, chosen=choice[(seed, k)], n_val=len(val_scores[VARIANTS[0]]),
                                    **{f"val c@15 {v}": round(100 * vm[v], 1) for v in VARIANTS}))
    for k, D in r.sets.items():                                 # P2: retrained and train + validation fine-tuned retrievers
        for R in range(0, 6):
            for run_, m in ((f"nift_ftsearch_rep{R}", FTN), (f"nift_ftsearch_trval{f'_rep{R}' if R else ''}", FTV)):
                if run_ == "nift_ftsearch_rep0":
                    continue
                if (f := RES / f"{run_}_seed{seed}" / "scores" / f"{k}_hyb_ft.npy").exists():
                    H = np.load(f)
                    for i in r.SPLIT[k]["test"]:
                        record(seed, "full", k, D, i, rep_m(m, R), np.argsort(-H[i], kind="stable").tolist(), r.cfg, extra_given=True)
    for k, D in r.sets.items():
        for i in r.SPLIT[k]["test"]:
            ho = np.argsort(-D.extra["hyb"][i], kind="stable")
            record(seed, "full", k, D, i, "search alone", ho.tolist(), r.cfg, extra_given=True, routes=True)
            dbs = {D.dbo[t] for t in D.Q[i][2]}
            if RRDB[k] is not None and len(dbs) == 1:          # F: the reranker over every table of the known database
                ix = [t for t in range(D.n) if D.dbo[t] in dbs]
                if not np.isnan(RRDB[k][i, ix]).any():
                    o = sorted(ix, key=lambda t: -RRDB[k][i, t])
                    g = set(D.Q[i][2])
                    rows.append(dict(seed=seed, size="full", set=k, qi=i, group="|".join(map(str, sorted(g))), qtext=D.Q[i][1].strip().lower(),
                                     method=f"{RRN} over the whole database | database given", **evaluate.metrics(o, g, r.cfg)))
            if RRRT[k] is not None:                             # Q: zero-label routing by search, reranker over the top 2 databases
                dbs2 = []
                for t in ho:
                    if D.dbo[int(t)] not in dbs2:
                        dbs2.append(D.dbo[int(t)])
                    if len(dbs2) == 2:
                        break
                ix = [t for t in range(D.n) if D.dbo[t] in dbs2]
                if not np.isnan(RRRT[k][i, ix]).any():
                    o = sorted(ix, key=lambda t: -RRRT[k][i, t])
                    rest = [int(t) for t in regroup(ho.tolist(), D, "best") if D.dbo[int(t)] not in dbs2]
                    record(seed, "full", k, D, i, f"{RRN}, routed to the top 2 databases", o + rest, r.cfg)
            if FT[k] is not None:                               # P: supervised retriever (fine-tuned BGE hybrid search)
                record(seed, "full", k, D, i, "fine-tuned search alone", np.argsort(-FT[k][i], kind="stable").tolist(), r.cfg,
                       extra_given=True, routes=True)
            if RR[k] is not None and not np.isnan(RR[k][i, ho[:args.k]]).any():
                top = ho[:args.k]
                record(seed, "full", k, D, i, RRN, top[np.argsort(-RR[k][i, top], kind="stable")].tolist() + ho[args.k:].tolist(),
                       r.cfg, extra_given=True, routes=True)
    for name, run, fname in (("Laya", "nift_base", "{k}_laya.npy"), ("cross-encoder", "nift_ce", "{k}_ce_" + args.ce + ".npy")):
        if not load_scores(r, run, fname):
            print(f"seed {seed}: no scores for {run}, skipped"); continue
        r.fit_fusion()
        small = small_fits(r, seed, (10, 20, 40))
        for cust, names in r.cfg["customers"].items():
            for k in names:
                D = r.sets[k]
                for i in r.SPLIT[k]["test"]:
                    e = r.E[k][i]
                    record(seed, "full", k, D, i, f"NIFT-SQL with {name}", fusion.rank(r.FUSION[cust], e), r.cfg, extra_given=True, routes=True)
                    record(seed, "full", k, D, i, f"{name} alone", e["co"].tolist(), r.cfg, extra_given=True, routes=True)
                    record(seed, "full", k, D, i, f"rank average, {name} + search", fusion.rank_average(e), r.cfg, routes=True)
                    for n in (10, 20, 40):
                        record_small(seed, "full", k, D, i, name, e, small[(cust, n)], n, r.cfg)
    fce = "{k}_ce_" + args.ce + ".npy"
    for R in range(1, 6):                                      # E: retrained replicates of the cross-encoder (same inputs)
        if load_scores(r, f"nift_ce_rep{R}", fce):
            r.fit_fusion()
            for cust, names in r.cfg["customers"].items():
                for k in names:
                    for i in r.SPLIT[k]["test"]:
                        e = r.E[k][i]
                        record(seed, "full", k, r.sets[k], i, f"{NC_}, retrain {R}", fusion.rank(r.FUSION[cust], e), r.cfg,
                               extra_given=True, routes=True)
                        record(seed, "full", k, r.sets[k], i, f"cross-encoder alone, retrain {R}", e["co"].tolist(), r.cfg,
                               extra_given=True, routes=True)
                        record(seed, "full", k, r.sets[k], i, f"rank average, cross-encoder + search, retrain {R}",
                               fusion.rank_average(e), r.cfg, routes=True)
    det = {n: RES / f"{n}_seed{seed}" / "scores" / f"beaver_ce_{args.ce}.npy" for n in ("nift_ce_det_a", "nift_ce_det_b", "nift_ce_det_nojoin")}
    if all(f.exists() for f in det.values()):
        a, b, j = (np.load(f) for f in det.values())
        dets.append(f"  seed {seed} BEAVER: two deterministic runs identical: {np.array_equal(a, b, equal_nan=True)} | "
                    f"deterministic run without join hints identical to them: {np.array_equal(a, j, equal_nan=True)} | "
                    f"max |difference| {np.nanmax(np.abs(a - b)):.2e} and {np.nanmax(np.abs(a - j)):.2e}")
    for cfg_name in ("train10", "train20", "train30"):          # T: both scorers at each training size
        rt = Run(seed, f"nift_{cfg_name}", args.data, args.out, config_path=REPO / "configs" / f"{cfg_name}.yaml")
        if not (rt.rd / "results.csv").exists():
            continue
        rt.load(check_descriptions=False); rt.split(); rt.search()
        for name, run, fname in (("Laya", f"nift_{cfg_name}", "{k}_laya.npy"), ("cross-encoder", f"nift_ce_{cfg_name}", "{k}_ce_" + args.ce + ".npy")):
            if not load_scores(rt, run, fname):
                continue
            rt.fit_fusion()
            small = small_fits(rt, seed, (20,))                 # the smallest total budget: scorer at this size, fusion on 20
            for cust, names in rt.cfg["customers"].items():
                for k in names:
                    for i in rt.SPLIT[k]["test"]:
                        record(seed, cfg_name, k, rt.sets[k], i, f"NIFT-SQL with {name}", fusion.rank(rt.FUSION[cust], rt.E[k][i]), rt.cfg)
                        record_small(seed, cfg_name, k, rt.sets[k], i, name, rt.E[k][i], small[(cust, 20)], 20, rt.cfg)
                        if name == "Laya":
                            ho = np.argsort(-rt.sets[k].extra["hyb"][i], kind="stable")
                            record(seed, cfg_name, k, rt.sets[k], i, "search alone", ho.tolist(), rt.cfg)

A = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)


def diff(a, b, X, keys=("c@8", "c@15")):
    out = []
    for k in SETS:
        Xa = X[(X.method == a) & (X.set == k)].sort_values(["seed", "qi"])
        Xb = X[(X.method == b) & (X.set == k)].sort_values(["seed", "qi"])
        if Xa.empty or len(Xa) != len(Xb):
            continue
        assert (Xa[["seed", "qi"]].values == Xb[["seed", "qi"]].values).all()
        grp = (Xa.set + "#" + Xa.group).tolist()
        cells = []
        for key in keys:
            m_, lo, hi = evaluate.group_boot(Xa[key].values - Xb[key].values, grp, 3000)
            cells.append(f"{key} {m_:+5.1f} [{lo:+.1f}, {hi:+.1f}]")
        out.append(f"  {a} minus {b} | {k:20s} " + "   ".join(cells))
    return out


def means(X, methods):
    X = X[X.method.isin(methods)]
    return X.groupby(["set", "method", "seed"])[["c@8", "c@15"]].mean().mul(100).groupby(level=[0, 1]).mean().round(1).unstack("set").to_string()


F = A[A["size"] == "full"]
NC, NL = "NIFT-SQL with cross-encoder", "NIFT-SQL with Laya"
G = lambda m, rule="best": f"{m} | grouped by database ({rule})"   # noqa: E731
L = ["ADDITIONAL ANALYSES | test, seeds " + " ".join(map(str, args.seeds)) + " | cached scores, fusion refitted on validation", ""]

base = ["search alone", RRN, "rank average, cross-encoder + search", "rank average, Laya + search", "cross-encoder alone", NL, NC]
L += ["R. Training-free database grouping for every method, complete@8 / complete@15, mean over seeds of each seed's test mean, %",
      means(F, base + [G(m, r_) for m in base for r_ in ("best", "top3")]), ""]
L += ["   NIFT-SQL against each baseline with the same grouping (rule 'best'):"]
for m in ["search alone", RRN, "rank average, cross-encoder + search", "cross-encoder alone"]:
    L += diff(NC, G(m), F) + diff(G(NC), G(m), F)
L += ["   grouping NIFT-SQL itself:"] + diff(G(NC), NC, F) + diff(G(NL), NL, F)
L += ["   learned fusion against rank averaging when both are grouped:"] + diff(G(NC), G("rank average, cross-encoder + search"), F) + diff(G(NL), G("rank average, Laya + search"), F) + [""]

D_ = lambda m: f"{m} | database given"   # noqa: E731
L += ["V. Database given (questions whose gold tables sit in one database), paired intervals"]
for b in ["cross-encoder alone", NL, "search alone", RRN]:
    L += diff(D_(NC), D_(b), F)
L += diff(D_(NL), D_("Laya alone"), F)
RW = f"{RRN} over the whole database | database given"
if (F.method == RW).any():
    L += ["   the reranker scoring every table of the known database (no search shortlist):",
          means(F, [RW, D_(RRN), D_(NC), D_("cross-encoder alone"), D_("search alone")])] + diff(D_(NC), RW, F) + diff(D_("cross-encoder alone"), RW, F)
L += [""]

sm = [f"{NC}, fusion on {n} validation questions" for n in (10, 20, 40)] + [f"{NL}, fusion on {n} validation questions" for n in (10, 20, 40)]
L += [f"S. Fusion fitted on fewer validation questions per customer ({args.draws} random draws per seed, metrics averaged over draws)",
      means(F, sm + [NC, NL, "rank average, cross-encoder + search", "rank average, Laya + search", "search alone"])]
for n in (10, 20, 40):
    L += diff(f"{NC}, fusion on {n} validation questions", NC, F) + diff(f"{NC}, fusion on {n} validation questions", "rank average, cross-encoder + search", F) + diff(f"{NC}, fusion on {n} validation questions", "search alone", F)
L += [""]

T = A[A["size"] != "full"]
if not T.empty:
    L += ["T. NIFT-SQL with the cross-encoder minus NIFT-SQL with Laya at each training size (same splits, same validation questions)"]
    for sz in ("train10", "train20", "train30"):
        if (T["size"] == sz).any():
            X = T[T["size"] == sz]
            L += [f"  {sz}:"] + diff(NC, NL, X) + diff(f"{NC}, fusion on 20 validation questions", "search alone", X)
    L += ["  means, with the fusion on all validation questions or on 20:",
          T.groupby(["size", "set", "method", "seed"])[["c@8", "c@15"]].mean().mul(100).groupby(level=[0, 1, 2]).mean().round(1).unstack("set").to_string()]
    L += ["  50% (main runs):"] + diff(NC, NL, F) + [""]

REP = F[F.method.str.match(r"^(NIFT-SQL with cross-encoder|cross-encoder alone), retrain \d$")]
if not REP.empty or dets:
    L += ["E. Retraining the cross-encoder on identical inputs"] + dets
if not REP.empty:
    X = pd.concat([F[F.method.isin([NC, "cross-encoder alone"])].assign(rep=0),
                   REP.assign(rep=REP.method.str.extract(r"retrain (\d)")[0].astype(int),
                              method=REP.method.str.replace(r", retrain \d", "", regex=True))])
    per = X.groupby(["set", "method", "seed", "rep"])[["c@8", "c@15"]].mean().mul(100)
    L += ["   per-seed test means for the original run (rep 0) and each retrain, %:", per.round(1).unstack("rep").to_string(),
          "   standard deviation across retrains of a seed's test mean, averaged over seeds (the retraining noise), points:",
          per.groupby(level=[0, 1, 2]).std().groupby(level=[0, 1]).mean().round(2).to_string(),
          "   the same for the four-seed mean (each retrain's mean over seeds), points:",
          per.groupby(level=[0, 1, 3]).mean().groupby(level=[0, 1]).agg(["mean", "std", "min", "max"]).round(2).to_string()]
    lay = F[F.method == NL].groupby(["set", "seed"])[["c@8", "c@15"]].mean().mul(100)
    m_ = per.xs(NC, level=1).reset_index().merge(lay.reset_index(), on=["set", "seed"], suffixes=("", "_laya"))
    gap = m_.assign(**{c: m_[c] - m_[c + "_laya"] for c in ("c@8", "c@15")}).groupby(["set", "rep"])[["c@8", "c@15"]].mean()
    L += ["   NIFT-SQL with the cross-encoder minus NIFT-SQL with Laya (four-seed mean), for each retrain of the cross-encoder:",
          gap.groupby(level=0).agg(["min", "max"]).round(1).to_string(), ""]

REPS = [R for R in range(0, 6) if (F.method == rep_m(NC, R)).any()]


def two_level(a, b, k, key, b_reps):
    """Mean over training runs of (a - b), 95% interval resampling training runs and gold-table-set clusters."""
    parts = []
    for R in REPS:
        Xa = F[(F.method == rep_m(a, R)) & (F.set == k)][["seed", "qi", "group", key]]
        Xb = F[(F.method == (rep_m(b, R) if b_reps else b)) & (F.set == k)][["seed", "qi", key]]
        if Xa.empty or Xb.empty:
            return None
        M = Xa.merge(Xb, on=["seed", "qi"], suffixes=("", "_b"))
        parts.append(M.assign(d=M[key] - M[key + "_b"])[["seed", "qi", "group", "d"]].rename(columns={"d": f"d{R}"}))
    J = parts[0]
    for P_ in parts[1:]:
        J = J.merge(P_.drop(columns="group"), on=["seed", "qi"])     # questions present in every training run
    if J.empty:
        return None
    J = J.sort_values(["seed", "qi"])
    cols = [J[f"d{R}"].values for R in REPS]
    grp = (k + "#" + J.group).tolist()
    Dm = np.vstack(cols)                                        # training runs x questions
    ids = {g: j for j, g in enumerate(dict.fromkeys(grp))}
    gi = np.array([ids[g] for g in grp])
    S = np.stack([np.bincount(gi, weights=d, minlength=len(ids)) for d in Dm]); N = np.bincount(gi, minlength=len(ids))
    rng = np.random.default_rng(0); nR, nC = S.shape; st = np.empty(3000)
    for b_ in range(3000):
        rr, cc = rng.integers(0, nR, nR), rng.integers(0, nC, nC)
        st[b_] = S[rr][:, cc].sum() / nR / N[cc].sum()
    return 100 * Dm.mean(), 100 * np.percentile(st, 2.5), 100 * np.percentile(st, 97.5)


if len(REPS) > 1:
    RWN = f"{RRN} over the whole database | database given"
    L += [f"X. Replicate-aware comparisons: NIFT-SQL with the cross-encoder over {len(REPS)} training runs (main + retrains), "
          "two-level bootstrap over training runs and gold-table-set clusters"]
    mm = []
    for R in REPS:
        for m in (NC, G(NC), D_(NC), "cross-encoder alone", D_("cross-encoder alone"), G("rank average, cross-encoder + search")):
            X_ = F[F.method == rep_m(m, R)]
            if not X_.empty:
                mm.append(X_.groupby(["set", "seed"])[["c@8", "c@15"]].mean().mul(100).groupby(level=0).mean().assign(method=m, rep=R))
    MM = pd.concat(mm).reset_index()
    L += ["   four-seed means per training run and their mean, %:",
          MM.pivot_table(index=["method", "rep"], columns="set", values="c@15").round(1).to_string(),
          MM.groupby(["method", "set"])[["c@8", "c@15"]].mean().round(1).unstack("set").to_string()]
    comps = [(NC, "search alone", False), (NC, G("search alone"), False), (NC, RRN, False), (NC, G(RRN), False), (NC, NL, False),
             (NC, G("rank average, cross-encoder + search"), True), (NC, "fine-tuned search alone", False),
             (NC, G("fine-tuned search alone"), False), ("cross-encoder alone", "fine-tuned search alone", False),
             (D_(NC), RWN, False), (D_(NC), D_("search alone"), False), (D_(NC), D_("cross-encoder alone"), True),
             (D_("cross-encoder alone"), RWN, False), (D_(NC), D_("fine-tuned search alone"), False)]
    for a, b, br in comps:
        for k in SETS:
            cells = []
            for key in ("c@8", "c@15"):
                res = two_level(a, b, k, key, br)
                if res is None:
                    break
                cells.append(f"{key} {res[0]:+5.1f} [{res[1]:+.1f}, {res[2]:+.1f}]")
            if len(cells) == 2:
                L.append(f"  {a} (mean of runs) minus {b}{' (same run)' if br else ''} | {k:20s} " + "   ".join(cells))
    L += [""]

RTN = f"{RRN}, routed to the top 2 databases"
if (F.method == RTN).any():
    L += ["Q. Zero-label routed reranker: databases ordered by their best search-ranked table, the reranker scores every table of the "
          "top 2 databases, the rest follow in grouped search order", means(F, [RTN, RRN, G(RRN), NC, "search alone"])]
    L += diff(NC, RTN, F)
    for k in SETS:
        cells = []
        for key in ("c@8", "c@15"):
            res = two_level(NC, RTN, k, key, False) if len(REPS) > 1 else None
            if res:
                cells.append(f"{key} {res[0]:+5.1f} [{res[1]:+.1f}, {res[2]:+.1f}]")
        if cells:
            L.append(f"  {NC} (mean of runs) minus {RTN} | {k:20s} " + "   ".join(cells))
    L += [""]

if (F.method == "fine-tuned search alone").any():
    P_ = ["search alone", "fine-tuned search alone", G("fine-tuned search alone"), D_("fine-tuned search alone"), D_("search alone"),
          "cross-encoder alone", NC]
    L += ["P. Supervised-retriever baseline: BGE fine-tuned per customer on the scorer's training rows, hybrid search",
          means(F, P_)] + diff("fine-tuned search alone", "search alone", F) + diff(NC, "fine-tuned search alone", F) + \
        diff("cross-encoder alone", "fine-tuned search alone", F) + diff(D_(NC), D_("fine-tuned search alone"), F) + [""]

def two_level_ind(a, b, k, key, b_reps):
    """Mean over a's training runs minus mean over b's training runs, resampling a's runs, b's runs (independently) and
    gold-table-set clusters. Use when a and b were trained separately (not paired run by run)."""
    def runs(m, reps, tag):
        out = []
        for R in reps:
            X_ = F[(F.method == rep_m(m, R)) & (F.set == k)][["seed", "qi", "group", key]].rename(columns={key: f"{tag}{R}"})
            if X_.empty:
                return None
            out.append(X_)
        return out
    ra, rb = runs(a, REPS, "a"), runs(b, b_reps, "b")
    if ra is None or rb is None:
        return None
    J = ra[0]
    for X_ in ra[1:] + rb:
        J = J.merge(X_.drop(columns="group"), on=["seed", "qi"])     # questions present in every run of both methods
    if J.empty:
        return None
    J = J.sort_values(["seed", "qi"])
    va = np.vstack([J[f"a{R}"].values for R in REPS])
    vb = np.vstack([J[f"b{R}"].values for R in b_reps])
    grp = (k + "#" + J.group).tolist(); ids = {g: j for j, g in enumerate(dict.fromkeys(grp))}
    gi = np.array([ids[g] for g in grp]); N = np.bincount(gi, minlength=len(ids))
    Sa = np.stack([np.bincount(gi, weights=x, minlength=len(ids)) for x in va])
    Sb = np.stack([np.bincount(gi, weights=x, minlength=len(ids)) for x in vb])
    rng = np.random.default_rng(0); st = np.empty(3000)
    for b_ in range(3000):
        cc = rng.integers(0, len(ids), len(ids))
        a_ = Sa[rng.integers(0, len(Sa), len(Sa))][:, cc].sum() / len(Sa)
        b2 = Sb[rng.integers(0, len(Sb), len(Sb))][:, cc].sum() / len(Sb)
        st[b_] = (a_ - b2) / N[cc].sum()
    return 100 * (va.mean() - vb.mean()), 100 * np.percentile(st, 2.5), 100 * np.percentile(st, 97.5)


def cells_for(fn, a, b, k, *extra):
    cells = []
    for key in ("c@8", "c@15"):
        res = fn(a, b, k, key, *extra)
        if res is None:
            return None
        cells.append(f"{key} {res[0]:+5.1f} [{res[1]:+.1f}, {res[2]:+.1f}]")
    return "   ".join(cells)


if (F.method == RRV).any():
    L += ["Q2. Zero-label reranker variants: routed by its own ranking (every table of the top 2 databases of the reranker's grouped "
          "ranking, the rest in that grouped order), and one variant per collection and seed chosen on VALIDATION complete@15 "
          "among plain, grouped by database, routed by search and routed by its own ranking (ties go to the earlier one)",
          means(F, [NC, RRN, G(RRN), RTN, RRO, RRV])]
    CH = pd.DataFrame(choices)
    L += ["   variant chosen on validation, per collection and seed (validation complete@15 of each variant, %):", CH.to_string(index=False)]
    L += diff(NC, RRO, F) + diff(NC, RRV, F)
    for b in (RRO, RRV):
        for k in SETS:
            c = cells_for(two_level, NC, b, k, False) if len(REPS) > 1 else None
            if c:
                L.append(f"  {NC} (mean of runs) minus {b} | {k:20s} {c}")
    L += [""]

FT_REPS = [R for R in range(0, 6) if (F.method == rep_m(FTN, R)).any()]
FV_REPS = [R for R in range(0, 6) if (F.method == rep_m(FTV, R)).any()]
if len(FT_REPS) > 1 or FV_REPS:
    mm = []
    for m, reps in ((FTN, FT_REPS), (FTV, FV_REPS)):
        for R in reps:
            for mx in (m, D_(m)):
                X_ = F[F.method == rep_m(mx, R)]
                if not X_.empty:
                    mm.append(X_.groupby(["set", "seed"])[["c@8", "c@15"]].mean().mul(100).groupby(level=0).mean().assign(method=mx, rep=R))
    MM = pd.concat(mm).reset_index()
    L += ["P2. Supervised retriever with training variation and a matched label budget: BGE fine-tuned on the training questions "
          f"(runs {FT_REPS}) and on the training plus validation questions, every labeled question NIFT-SQL uses (runs {FV_REPS}); "
          "intervals resample NIFT-SQL's runs, the retriever's runs and gold-table-set clusters",
          "   four-seed means per training run, %:", MM.pivot_table(index=["method", "rep"], columns="set", values=["c@8", "c@15"]).round(1).to_string(),
          "   mean over runs, %:", MM.groupby(["method", "set"])[["c@8", "c@15"]].mean().round(1).unstack("set").to_string(),
          "   standard deviation across the retriever's runs of the four-seed mean, points:",
          MM.groupby(["method", "set"])[["c@8", "c@15"]].std().round(2).unstack("set").to_string()]
    for a, b, reps in ((NC, FTN, FT_REPS), (NC, FTV, FV_REPS), (D_(NC), D_(FTN), FT_REPS), (D_(NC), D_(FTV), FV_REPS)):
        for k in SETS:
            c = cells_for(two_level_ind, a, b, k, reps) if len(REPS) > 1 and reps else None
            if c:
                L.append(f"  {a} (mean of runs) minus {b} (mean of {len(reps)} runs) | {k:20s} {c}")
    L += [""]

W = pd.DataFrame(caps)
if not W.empty:
    agg = W.groupby(["set", "gold"]).agg(tables=("cut", "size"), cards_cut_pct=("cut", "mean"), cols_mean=("cols", "mean"),
                                         cols_lost_when_cut=("lost", lambda s: s[s > 0].mean() if (s > 0).any() else 0.0),
                                         joins_cut_pct=("joins_cut", "mean"))
    agg["cards_cut_pct"] *= 100; agg["joins_cut_pct"] *= 100
    L += [f"W. Inputs cut by the caps before tokenization (card_max {cm} characters, joins_max {jm}); gold = table is gold for some question",
          agg.round(1).to_string(), ""]

(RES / "ADDITIONAL_ANALYSES.txt").write_text("\n".join(L) + "\n")
print("\n".join(L))
