"""Checks and breakdowns for the final NIFT-SQL, from cached scores only (CPU; no model is loaded, nothing is re-scored).

    python experiments/final_breakdowns.py --data D --out O [--seeds 1 2 3 4] [--reps 0 1 2]

Run after experiments/fusion_extensions.py. It rebuilds that experiment's rankings from its caches (cross-encoder scores,
fine-tuned dense scores in results/fusion_ext/.cache/, the reranker cache *_fusionext.npy), refits the fusions on
validation questions exactly as there, and first checks that every test question's complete@8 and complete@15 match
results/fusion_ext/test_rows.csv; it stops if any differs. It never writes test_rows.csv and adds no new test scoring:
the rankings are the ones already reported, only broken down differently.

  1. Routing check (paper note to Table G5). The reranker "routed by its own ranking" scores every table of the top two
     databases of its own grouped ranking. On BEAVER it matched the plain reranker exactly. Per question: how often the two
     orders are identical, how often their top 15 differ, how often complete@15 differs, how many tables routing scores,
     how often the gold database is among the two, and how many questions are dropped for a missing score.
  2. Table 7 against the strongest baselines: complete@15 by number of gold tables (1, 2, 3 to 4, 5 or more) for the
     reranker with its variant chosen on validation, the retriever fine-tuned on training + validation questions,
     NIFT-SQL-base, NIFT-SQL, and the high-accuracy mode, with differences for the multi-table buckets.
  3. Table 8 against the same methods: gold tables whose name shares no word with the question, share ranked in the top 15.

Means of trained methods are over training runs (each pooled over the test questions of all seeds); intervals as in Table
G7: two-level bootstrap over training runs and gold-table-set clusters, paired when both methods have the same runs.
Writes <out>/results/FINAL_BREAKDOWNS.txt.
"""
import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nift import evaluate, fusion, search  # noqa: E402
from nift.config import REPO               # noqa: E402
from nift.pipeline import Run              # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--ce", default="bge-reranker-large")
ap.add_argument("--reranker", default="Qwen3-Reranker-8B", help="tag of the reranker caches")
ap.add_argument("--reranker-k", type=int, default=100)
ap.add_argument("--reps", type=int, nargs="+", default=[0, 1, 2])
args = ap.parse_args()

RES = Path(args.out) / "results"
EXT = RES / "fusion_ext"
CACHE = RES / ".cache" / "reranker"
SETS = ["beaver", "livesqlbench", "livesqlbench_large"]
CE_RUN = lambda R: "nift_ce" if R == 0 else f"nift_ce_rep{R}"                 # noqa: E731
FT_RUN = lambda R: "nift_ftsearch" if R == 0 else f"nift_ftsearch_rep{R}"     # noqa: E731
# method names exactly as in fusion_extensions.py, so rows can be matched to its test_rows.csv
NIFT, VA, VB, VC = "NIFT-SQL", "A: NIFT-SQL + fine-tuned search", "B: NIFT-SQL + reranker features", "C: A + reranker features"
RRN, RRV, RRO = "reranker (search top 100)", "reranker, variant chosen on validation", "reranker, routed by its own ranking"
FTS, FTV = "fine-tuned search alone", "fine-tuned search, train + validation"
LABEL = {NIFT: "NIFT-SQL-base", VA: "NIFT-SQL", VB: "NIFT-SQL-base + reranker features", VC: "NIFT-SQL, high-accuracy mode",
         RRN: "reranker (search top 100)", RRV: "reranker, variant chosen on validation", RRO: "reranker, routed by its own ranking",
         FTS: "fine-tuned retriever (train)", FTV: "fine-tuned retriever (train + validation)", "search alone": "search alone"}
assert (EXT / "test_rows.csv").exists(), "run experiments/fusion_extensions.py first: its test_rows.csv is the reference"
REF = pd.read_csv(EXT / "test_rows.csv")


def need(f):
    assert Path(f).exists(), f"missing cache {f}: this script only reads caches written by fusion_extensions.py"
    return np.load(f)


def lexmiss(D, i):
    """Gold tables whose name shares no word with the question (as in experiments/ce_analysis.py)."""
    q = set(search.words(D.Q[i][1]))
    return {t for t in D.Q[i][2] if not (set(search.words(D.cards[t].split(":")[0])) & q)}


def regroup(order, D):
    pos = {}
    for p, t in enumerate(order):
        pos.setdefault(D.dbo[t], p)
    return sorted(order, key=lambda t: pos[D.dbo[t]])


def top2(base, D):
    dbs = []
    for t in base:
        if D.dbo[int(t)] not in dbs:
            dbs.append(D.dbo[int(t)])
        if len(dbs) == 2:
            break
    return [t for t in range(D.n) if D.dbo[t] in dbs], dbs


VARIANTS = ["plain", "grouped by database", "routed by search", "routed by its own ranking"]


def rr_variants(D, P_, i):
    ho = np.argsort(-D.extra["hyb"][i], kind="stable"); top = ho[:args.reranker_k]
    if np.isnan(P_[i, top]).any():
        return None, "top 100 unscored"
    plain = top[np.argsort(-P_[i, top], kind="stable")].tolist() + ho[args.reranker_k:].tolist()
    out = {"plain": plain, "grouped by database": regroup(plain, D)}
    for v, b in (("routed by search", ho.tolist()), ("routed by its own ranking", plain)):
        ix, dbs = top2(b, D)
        if np.isnan(P_[i, ix]).any():
            return None, f"{v}: {int(np.isnan(P_[i, ix]).sum())} of {len(ix)} tables unscored"
        out[v] = sorted(ix, key=lambda t: -P_[i, t]) + [int(t) for t in regroup(b, D) if D.dbo[int(t)] not in dbs]
        out[v + " dbs"] = dbs
        out[v + " n"] = len(ix)
    return out, ""


def add_rr(e, RRk, i):
    p = RRk[i, e["cand"]]
    assert not np.isnan(p).any(), "a candidate has no reranker score"
    p = np.clip(p, 1e-6, 1 - 1e-6)
    rank = np.empty(len(p)); rank[np.argsort(-p, kind="stable")] = np.arange(len(p))
    return dict(e, F=np.column_stack([e["F"], np.log(p / (1 - p)), np.log1p(rank)]))


rows, route = [], []


def rec(seed, R, k, D, i, method, order, cfg):
    g = set(D.Q[i][2]); lm = lexmiss(D, i); top15 = set(order[:15])
    rows.append(dict(seed=seed, rep=R, set=k, qi=i, group="|".join(map(str, sorted(g))), n_gold=len(g), method=method,
                     lm_n=len(lm), lm_found=len(lm & top15), **evaluate.metrics(order, g, cfg)))


for seed in args.seeds:
    r = Run(seed, "nift_base", args.data, args.out, config_path=REPO / "configs" / "default.yaml")
    r.load(check_descriptions=False); r.split(); r.search()
    cfg = r.cfg
    RRF = {k: need(CACHE / f"{k}_{args.reranker}_fusionext.npy") for k in r.sets}
    PA = {k: (np.load(f) if (f := CACHE / f"{k}_{args.reranker}_all.npy").exists() else None) for k in r.sets}
    for R in args.reps:
        ex = {}
        for k, D in r.sets.items():
            P = need(RES / f"{CE_RUN(R)}_seed{seed}" / "scores" / f"{k}_ce_{args.ce}.npy")
            hyb = need(RES / f"{FT_RUN(R)}_seed{seed}" / "scores" / f"{k}_hyb_ft.npy")
            dense = need(EXT / ".cache" / f"dense_{FT_RUN(R)}_seed{seed}_{k}.npy")
            Db = dataclasses.replace(D, extra=dict(D.extra, P=P))
            Df = dataclasses.replace(D, extra=dict(D.extra, P=P, dense=dense, hyb=hyb))
            for i in r.SPLIT[k]["val"] + r.SPLIT[k]["test"]:
                ex[(NIFT, k, i)] = fusion.features(Db, i, cfg)
                ex[(VA, k, i)] = fusion.features(Df, i, cfg)
                ex[(VB, k, i)] = add_rr(ex[(NIFT, k, i)], RRF[k], i)
                ex[(VC, k, i)] = add_rr(ex[(VA, k, i)], RRF[k], i)
            for i in r.SPLIT[k]["test"]:
                rec(seed, R, k, D, i, FTS, np.argsort(-hyb[i], kind="stable").tolist(), cfg)
        for cust, names in cfg["customers"].items():
            for v in (NIFT, VA, VB, VC):
                m = fusion.fit([ex[(v, k, i)] for k in names for i in r.SPLIT[k]["val"]], cfg)
                for k in names:
                    for i in r.SPLIT[k]["test"]:
                        rec(seed, R, k, r.sets[k], i, v, fusion.rank(m, ex[(v, k, i)]), cfg)
    for k, D in r.sets.items():
        for i in r.SPLIT[k]["test"]:
            rec(seed, -1, k, D, i, "search alone", np.argsort(-D.extra["hyb"][i], kind="stable").tolist(), cfg)
        for R in range(0, 6):
            run_ = "nift_ftsearch_trval" + (f"_rep{R}" if R else "")
            if (f := RES / f"{run_}_seed{seed}" / "scores" / f"{k}_hyb_ft.npy").exists():
                H = np.load(f)
                for i in r.SPLIT[k]["test"]:
                    rec(seed, 100 + R, k, D, i, FTV, np.argsort(-H[i], kind="stable").tolist(), cfg)
        if PA[k] is None:
            continue
        vs = {v: [] for v in VARIANTS}
        for i in r.SPLIT[k]["val"]:
            o, _ = rr_variants(D, PA[k], i)
            if o is not None:
                for v in VARIANTS:
                    vs[v].append(evaluate.metrics(o[v], set(D.Q[i][2]), cfg)["c@15"])
        pick = max(VARIANTS, key=lambda v: (np.mean(vs[v]) if vs[v] else -1.0, -VARIANTS.index(v)))
        for i in r.SPLIT[k]["test"]:
            o, why = rr_variants(D, PA[k], i)
            g = set(D.Q[i][2]); gdb = {D.dbo[t] for t in g}
            if o is None:
                route.append(dict(seed=seed, set=k, qi=i, dropped=why))
                continue
            rec(seed, -1, k, D, i, RRN, o["plain"], cfg)
            rec(seed, -1, k, D, i, RRV, o[pick], cfg)
            rec(seed, -1, k, D, i, RRO, o["routed by its own ranking"], cfg)
            pl, ro = o["plain"], o["routed by its own ranking"]
            m_pl, m_ro = evaluate.metrics(pl, g, cfg), evaluate.metrics(ro, g, cfg)
            outside = [t for t in g if t not in set(np.argsort(-D.extra["hyb"][i], kind="stable")[:args.reranker_k].tolist())]
            route.append(dict(seed=seed, set=k, qi=i, dropped="", n_routed=o["routed by its own ranking n"],
                              n_db=len(set(D.dbo)), same_order=pl == ro, same_top15=set(pl[:15]) == set(ro[:15]),
                              same_top15_order=pl[:15] == ro[:15], c15_plain=m_pl["c@15"], c15_routed=m_ro["c@15"],
                              c8_plain=m_pl["c@8"], c8_routed=m_ro["c@8"],
                              gold_db_in_top2=bool(gdb & set(o["routed by its own ranking dbs"])),
                              gold_db_first=D.dbo[pl[0]] in gdb, gold_outside_top100=len(outside) > 0,
                              outside_in_routed_dbs=any(D.dbo[t] in o["routed by its own ranking dbs"] for t in outside),
                              pick=pick))

T = pd.DataFrame(rows)

# ------------------------------------------------------------------ reproduction check against the reported test rows
key = ["seed", "rep", "set", "qi", "method"]
J = T.merge(REF[key + ["c@8", "c@15"]], on=key, how="inner", suffixes=("", "_ref"))
bad = J[(J["c@8"] != J["c@8_ref"]) | (J["c@15"] != J["c@15_ref"])]
covered = sorted(set(J.method))
L = [f"FINAL BREAKDOWNS | test, seeds {' '.join(map(str, args.seeds))} | training runs {args.reps} | from cached scores", "",
     f"Reproduction check against fusion_ext/test_rows.csv: {len(J)} question rows matched over {len(covered)} methods "
     f"({', '.join(LABEL.get(m, m) for m in covered)}); {len(bad)} differ."]
if len(bad):
    print("\n".join(L)); print(bad.head(20).to_string())
    sys.exit("the rebuilt rankings do not reproduce the reported test rows: stopping before any breakdown")
missing = sorted({NIFT, VA, VC, RRV, FTV} - set(covered))
if missing:
    L.append(f"  not in test_rows.csv, so not cross-checked: {missing}")
L.append("")

# ------------------------------------------------------------------ 1. routing check
Rt = pd.DataFrame(route)
L += ["1. Reranker routed by its own ranking (top 2 databases of its grouped ranking) against the plain reranker, test questions "
      "of all seeds"]
for k in SETS:
    X = Rt[Rt.set == k]
    if X.empty:
        L.append(f"  {k}: no reranker cache (*_all.npy)"); continue
    ok = X[X.dropped == ""]
    dr = X[X.dropped != ""]
    if ok.empty:
        L.append(f"  {k}: every question dropped ({dr.dropped.value_counts().to_dict()})"); continue
    L.append(f"  {k}: {len(ok)} questions kept, {len(dr)} dropped for missing scores"
             + (f" ({dr.dropped.value_counts().head(3).to_dict()})" if len(dr) else ""))
    L.append(f"     databases in the collection {int(ok.n_db.iloc[0])}; tables scored when routed: mean {ok.n_routed.mean():.0f}, "
             f"min {ok.n_routed.min()}, max {ok.n_routed.max()}")
    L.append(f"     identical full order {100 * ok.same_order.mean():.1f}%; identical top-15 set {100 * ok.same_top15.mean():.1f}% "
             f"(same order {100 * ok.same_top15_order.mean():.1f}%)")
    L.append(f"     complete@15: plain {100 * ok.c15_plain.mean():.1f}, routed {100 * ok.c15_routed.mean():.1f}; routed gains on "
             f"{int(((ok.c15_routed > ok.c15_plain)).sum())} questions, loses on {int((ok.c15_routed < ok.c15_plain).sum())}; "
             f"complete@8 gains {int((ok.c8_routed > ok.c8_plain).sum())}, losses {int((ok.c8_routed < ok.c8_plain).sum())}")
    L.append(f"     gold database first in the plain ranking {100 * ok.gold_db_first.mean():.1f}%, among the two routed databases "
             f"{100 * ok.gold_db_in_top2.mean():.1f}%; some gold table outside the search top 100 {100 * ok.gold_outside_top100.mean():.1f}% "
             f"(of those, in a routed database {100 * ok[ok.gold_outside_top100].outside_in_routed_dbs.mean() if ok.gold_outside_top100.any() else float('nan'):.1f}%)")
    L.append(f"     variant chosen on validation, by seed: {ok.drop_duplicates('seed').set_index('seed').pick.to_dict()}")
    if int(ok.n_db.iloc[0]) <= 2:
        L.append("     -> the collection has at most two databases, so routing scores every table")
    elif ok.same_order.all():
        L.append("     -> routing never changes the order: check top2() and the cache coverage, this is not plausible")
    elif ok.same_top15.all():
        L.append("     -> routing changes the order only below position 15, so the tie at complete@8 and @15 is real")
L.append("")


# ------------------------------------------------------------------ bootstrap (ratio of sums, so it also serves Table 8)
def boot(X, a, b, k, num, den, reps=3000):
    def runs(m):
        Y = X[(X.method == m) & (X.set == k)]
        return {R: g.set_index(["seed", "qi"]) for R, g in Y.groupby("rep")}, Y.drop_duplicates(["seed", "qi"]).set_index(["seed", "qi"]).group
    ra, grp = runs(a); rb, _ = runs(b)
    if not ra or not rb:
        return None
    idx = sorted(set.intersection(*[set(s.index) for s in list(ra.values()) + list(rb.values())]))
    if not idx:
        return None
    M = lambda rr, c: np.vstack([rr[R].loc[idx, c].values.astype(float) for R in sorted(rr)])   # noqa: E731
    gl = grp.loc[idx].tolist(); ids = {g: j for j, g in enumerate(dict.fromkeys(gl))}
    gi = np.array([ids[g] for g in gl])
    S = lambda Mx: np.stack([np.bincount(gi, weights=x, minlength=len(ids)) for x in Mx])        # noqa: E731
    An, Ad, Bn, Bd = S(M(ra, num)), S(M(ra, den)), S(M(rb, num)), S(M(rb, den))
    ratio = lambda n, d: n.sum(1) / np.maximum(d.sum(1), 1e-9)                                    # noqa: E731
    point = 100 * (ratio(An, Ad).mean() - ratio(Bn, Bd).mean())
    paired = sorted(ra) == sorted(rb) and len(ra) > 1
    rng, st = np.random.default_rng(0), np.empty(reps)
    for b_ in range(reps):
        cc = rng.integers(0, len(ids), len(ids))
        ia = rng.integers(0, len(An), len(An))
        ib = ia if paired else rng.integers(0, len(Bn), len(Bn))
        st[b_] = ratio(An[ia][:, cc], Ad[ia][:, cc]).mean() - ratio(Bn[ib][:, cc], Bd[ib][:, cc]).mean()
    return point, 100 * np.percentile(st, 2.5), 100 * np.percentile(st, 97.5)


def fmt(res):
    if res is None:
        return "n/a"
    star = " *" if res[1] > 0 or res[2] < 0 else ""
    return f"{res[0]:+.1f} [{res[1]:+.1f}, {res[2]:+.1f}]{star}"


T["one"] = 1.0
T["tables"] = pd.cut(T.n_gold, [0, 1, 2, 4, 100], labels=["1", "2", "3 to 4", "5 or more"]).astype(str)
SHOW = [m for m in ("search alone", RRN, RRV, FTV, NIFT, VA, VC) if m in set(T.method)]

# ------------------------------------------------------------------ 2. Table 7 against the strongest baselines
L += ["2. Complete@15 by number of gold tables, %: test questions of all seeds pooled, mean over training runs (question counts in "
      "brackets)"]
for k in SETS:
    X = T[T.set == k]
    tab = (X[X.method.isin(SHOW)].groupby(["method", "rep", "tables"])["c@15"].mean().mul(100)
           .groupby(level=[0, 2]).mean().unstack("tables").reindex(SHOW).rename(index=LABEL))
    n = X[X.method == "search alone"].groupby("tables").size()
    tab.columns = [f"{c} ({n.get(c, 0)})" for c in tab.columns]
    L += [f"  {k}", tab.round(1).to_string(), ""]
L += ["  Differences in complete@15 by bucket (* = interval excludes zero):"]
for k in SETS:
    for bucket in ("3 to 4", "5 or more"):
        X = T[(T.set == k) & (T.tables == bucket)]
        cells = [f"{LABEL[a]} - {LABEL[b]} {fmt(boot(X, a, b, k, 'c@15', 'one'))}"
                 for a, b in ((VA, RRV), (VA, FTV), (VC, RRV), (VA, NIFT)) if a in SHOW and b in SHOW]
        L.append(f"  {k}, {bucket}: " + "; ".join(cells))
L.append("")

# ------------------------------------------------------------------ 3. Table 8 against the same methods
L += ["3. Gold tables whose name shares no word with the question: share ranked in the top 15, %, pooled over the test questions "
      "of all seeds, mean over training runs (count of such gold tables in brackets)"]
E = (T[T.method.isin(SHOW)].groupby(["set", "method", "rep"])[["lm_found", "lm_n"]].sum()
     .assign(share=lambda d: 100 * d.lm_found / d.lm_n).groupby(level=[0, 1]).share.mean().unstack("set")
     .reindex(index=SHOW, columns=SETS).rename(index=LABEL))
cnt = T[T.method == "search alone"].groupby("set").lm_n.sum()
E.columns = [f"{c} ({int(cnt.get(c, 0))})" for c in E.columns]
L += [E.round(1).to_string(), "", "  Differences in that share:"]
for k in SETS:
    X = T[T.set == k]
    cells = [f"{LABEL[a]} - {LABEL[b]} {fmt(boot(X, a, b, k, 'lm_found', 'lm_n'))}"
             for a, b in ((VA, RRV), (VA, FTV), (VC, RRV), (VA, NIFT)) if a in SHOW and b in SHOW]
    L.append(f"  {k}: " + "; ".join(cells))

(RES / "FINAL_BREAKDOWNS.txt").write_text("\n".join(L) + "\n")
print("\n".join(L))
