"""Two controls for NIFT-SQL's fusion (paper Section 6.8), on the rankings that experiments/fusion_extensions.py already made.

    python experiments/fusion_controls.py --data D --out O [--seeds 1 2 3 4]

CPU only, nothing is fitted and nothing new is scored by a model. Reads, for every seed and training run R:
the cross-encoder scores (nift_ce[_repR]), the fine-tuned search scores (nift_ftsearch[_repR]), the reranker cache that
fusion_extensions.py filled for exactly the fusion candidates (<set>_<reranker>_fusionext.npy), and its test rows
(results/fusion_ext/test_rows.csv), which it never rewrites.

  1  Training-free fusion of the same two trained parts. Reciprocal rank fusion (k = 60) of the fine-tuned cross-encoder and
     fine-tuned hybrid search, over all tables, and the same grouped by database (databases ordered by their best-ranked
     table). Against NIFT-SQL (variant A: the same two parts, learned fusion), paired run by run. This isolates what the
     logistic regression and its validation labels add.
  2  The reranker alone on NIFT-SQL's own candidates. The same 80 to 90 candidates the high-accuracy mode sees (scorer top 50
     plus fine-tuned search top 50), ordered by the zero-shot reranker's score, the rest in fine-tuned search order; and the
     same grouped by database. Against the high-accuracy mode (variant C), paired run by run, and against NIFT-SQL.
     This separates a better candidate pool from learned three-way fusion.
  3  The same labels spent differently: grouped rank averaging of a cross-encoder and a fine-tuned search that were BOTH
     trained on the training and validation questions (nift_ce_trval[_repR], nift_ftsearch_trval[_repR]; a label-free fusion
     needs no validation questions, so they can go to training). Against NIFT-SQL (variant A), intervals resampling both
     methods' training runs independently. Skipped if those runs are missing.

  4  Training-free fusion of all THREE signals the high-accuracy mode uses: reciprocal rank fusion (k = 60) of the
     fine-tuned cross-encoder, fine-tuned hybrid search, and the zero-shot reranker (its rank among NIFT-SQL's candidates;
     tables outside the candidates get no reranker term), plain and grouped by database. Against the high-accuracy mode
     (variant C), paired run by run. This tests whether the learned weights earn their place once a third signal is added.

Writes <out>/results/FUSION_CONTROLS.txt and <out>/results/fusion_ext/control_rows.csv.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nift import evaluate, fusion       # noqa: E402
from nift.config import REPO            # noqa: E402
from nift.pipeline import Run           # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--ce", default="bge-reranker-large")
ap.add_argument("--reranker", default="Qwen/Qwen3-Reranker-8B")
ap.add_argument("--reps", type=int, nargs="+", default=[0, 1, 2])
args = ap.parse_args()

RES = Path(args.out) / "results"
OUT = RES / "fusion_ext"
RTAG = args.reranker.rstrip("/").split("/")[-1]
SETS = ["beaver", "livesqlbench", "livesqlbench_large"]
CE_RUN = lambda R: "nift_ce" if R == 0 else f"nift_ce_rep{R}"                 # noqa: E731
FT_RUN = lambda R: "nift_ftsearch" if R == 0 else f"nift_ftsearch_rep{R}"     # noqa: E731
NIFT, VA, VC = "NIFT-SQL", "A: NIFT-SQL + fine-tuned search", "C: A + reranker features"
RRV = "reranker, variant chosen on validation"
RRF, RRFG = "rank average, cross-encoder + fine-tuned search", "rank average, cross-encoder + fine-tuned search | grouped"
QC, QCG = "reranker alone on NIFT-SQL's candidates", "reranker alone on NIFT-SQL's candidates | grouped"
R3, R3G = "rank average of all three signals", "rank average of all three signals | grouped"
TV, TVG = "rank average, both trained on train + validation", "rank average, both trained on train + validation | grouped"
CE_TV = lambda R: "nift_ce_trval" if R == 0 else f"nift_ce_trval_rep{R}"            # noqa: E731
FT_TV = lambda R: "nift_ftsearch_trval" if R == 0 else f"nift_ftsearch_trval_rep{R}"  # noqa: E731

T0 = pd.read_csv(OUT / "test_rows.csv")
CACHE = RES / ".cache" / "reranker"


def regroup(order, D):
    pos = {}
    for p, t in enumerate(order):
        pos.setdefault(D.dbo[t], p)
    return sorted(order, key=lambda t: pos[D.dbo[t]])


rows = []
for seed in args.seeds:
    r = Run(seed, "nift_base", args.data, args.out, config_path=REPO / "configs" / "default.yaml")
    r.load(check_descriptions=False); r.split()
    top = r.cfg["fusion"]["candidates_top"]
    RR = {k: np.load(CACHE / f"{k}_{RTAG}_fusionext.npy") for k in r.sets}
    for R in args.reps:
        for k, D in r.sets.items():
            P = np.load(RES / f"{CE_RUN(R)}_seed{seed}" / "scores" / f"{k}_ce_{args.ce}.npy")
            H = np.load(RES / f"{FT_RUN(R)}_seed{seed}" / "scores" / f"{k}_hyb_ft.npy")
            for i in r.SPLIT[k]["test"]:
                g = set(D.Q[i][2])
                co, ho = np.argsort(-P[i], kind="stable"), np.argsort(-H[i], kind="stable")
                base = dict(seed=seed, rep=R, set=k, qi=i, group="|".join(map(str, sorted(g))))
                ra = fusion.rank_average(dict(co=co, ho=ho))
                cand = sorted(set(co[:top].tolist()) | set(ho[:top].tolist()))
                s = RR[k][i, cand]
                assert not np.isnan(s).any(), f"seed {seed} run {R} {k} q{i}: a candidate has no reranker score"
                q = [cand[j] for j in np.argsort(-s, kind="stable")]
                qo = q + [int(t) for t in ho if int(t) not in set(q)]
                n_ = len(co); rc_, rh_, rr_ = np.empty(n_), np.empty(n_), np.zeros(n_)
                rc_[co], rh_[ho] = np.arange(n_), np.arange(n_)
                rr_[q] = 1 / (60 + np.arange(len(q)))                          # 4: reranker term, candidates only
                r3 = np.argsort(-(1 / (60 + rc_) + 1 / (60 + rh_) + rr_), kind="stable").tolist()
                for m, o in ((RRF, ra), (RRFG, regroup(ra, D)), (QC, qo), (QCG, regroup(qo, D)), (R3, r3), (R3G, regroup(r3, D))):
                    rows.append(dict(base, method=m, **evaluate.metrics(o, g, r.cfg)))
    for R in args.reps:                                         # 3: the same labels, all spent on training
        for k, D in r.sets.items():
            fc = RES / f"{CE_TV(R)}_seed{seed}" / "scores" / f"{k}_ce_{args.ce}.npy"
            ff = RES / f"{FT_TV(R)}_seed{seed}" / "scores" / f"{k}_hyb_ft.npy"
            if not (fc.exists() and ff.exists()):
                continue
            P, H = np.load(fc), np.load(ff)
            for i in r.SPLIT[k]["test"]:
                g = set(D.Q[i][2])
                ra = fusion.rank_average(dict(co=np.argsort(-P[i], kind="stable"), ho=np.argsort(-H[i], kind="stable")))
                base = dict(seed=seed, rep=R, set=k, qi=i, group="|".join(map(str, sorted(g))))
                rows.append(dict(base, method=TV, **evaluate.metrics(ra, g, r.cfg)))
                rows.append(dict(base, method=TVG, **evaluate.metrics(regroup(ra, D), g, r.cfg)))
C = pd.DataFrame(rows)
C.to_csv(OUT / "control_rows.csv", index=False)
T = pd.concat([T0, C], ignore_index=True)


def per_run(m, k, key):
    X = T[(T.method == m) & (T.set == k)]
    return {R: g.set_index(["seed", "qi"])[key] for R, g in X.groupby("rep")}, X.drop_duplicates(["seed", "qi"]).set_index(["seed", "qi"]).group


def boot(a, b, k, key, paired):
    """As in fusion_extensions.py: mean over training runs of (a - b), two-level bootstrap over runs and gold-set clusters."""
    ra, grp = per_run(a, k, key)
    rb, _ = per_run(b, k, key)
    if not ra or not rb:
        return None
    idx = sorted(set.intersection(*[set(s.index) for s in list(ra.values()) + list(rb.values())]))
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


show = [NIFT, VA, VC, RRF, RRFG, QC, QCG, R3, R3G, RRV, TV, TVG]
means = (T[T.method.isin(show)].groupby(["set", "method", "rep", "seed"])[["c@8", "c@15"]].mean().mul(100)
         .groupby(level=[0, 1, 2]).mean().groupby(level=[0, 1]).mean().round(1).unstack("set"))
L = [f"FUSION CONTROLS | test, seeds {' '.join(map(str, args.seeds))} | training runs {args.reps} | nothing fitted, no new model scoring",
     "Means over seeds, then over training runs, %. Differences: mean over training runs, 95% interval from a two-level bootstrap "
     "over training runs and gold-table-set clusters; * = interval excludes zero.", "", means.to_string(), "",
     "1. Learned fusion against training-free fusion of the same two trained parts (paired run by run):"]
L += line(VA, RRF, True) + line(VA, RRFG, True)
L += ["", "2. The reranker alone on NIFT-SQL's candidates (paired run by run):"]
L += line(VC, QC, True) + line(VC, QCG, True) + line(VA, QC, True) + line(VA, QCG, True)
L += ["", "4. The high-accuracy mode against training-free rank fusion of the same three signals (paired run by run):"]
L += line(VC, R3, True) + line(VC, R3G, True)
L += ["", "   the same candidates against the reranker's best zero-label variant chosen on validation (unpaired):"]
L += line(QC, RRV, False) + line(QCG, RRV, False)
if (T.method == TVG).any():
    L += ["", "3. The same labels spent on training: rank averaging of a cross-encoder and a fine-tuned search both trained on "
          "training + validation questions (separate training runs, resampled independently):"]
    L += line(VA, TVG, False) + line(VA, TV, False) + line(TVG, RRFG, False)
(RES / "FUSION_CONTROLS.txt").write_text("\n".join(L) + "\n")
print("\n".join(L))
