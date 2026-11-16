"""Analyses of the cross-encoder runs that the paper (and its reviewers) need. CPU only: it reads cached scores, no model runs.

    python experiments/ce_analysis.py --data D --out O [--seeds 1 2 3 4]

Reads, per seed, the finished runs and their cached scores:
  nift_base_seed<N>      Laya scores (scores/<set>_laya.npy), the main runs
  nift_ce_seed<N>        fine-tuned bge-reranker-large (experiments/supervised_crossencoder.py)
  nift_ce0_seed<N>       the same cross-encoder zero-shot             (--zero-shot --run nift_ce0)          optional
  nift_ce_nojoin_seed<N> fine-tuned without join hints in the input   (--no-joins --run nift_ce_nojoin)     optional
  nift_train10/20/30, nift_ce_train10/20/30   training-size runs                                           optional
  .cache/reranker/<set>_Qwen3-Reranker-8B_top100.npy   the zero-shot reranker baseline
Each ranking is rebuilt exactly as in the runs (same splits, same fusion fitted on validation), and the rebuilt
NIFT-SQL numbers are checked against the runs' own results.csv. Nothing is fitted on test.

Writes <out>/results/CE_ANALYSIS.txt:
  A  means for every method (test, seeds pooled)
  B  paired differences with 95% intervals clustered by gold-table set across seeds, including the reranker row,
     learned fusion vs rank average with the cross-encoder, zero-shot vs fine-tuned (same model), no-joins,
     and NIFT-SQL (cross-encoder) vs NIFT-SQL (Laya) pooled over the three datasets
  C  shortlist coverage            D  complete@15 by number of gold tables
  E  gold tables that share no word with the question: share found in the top 15 (a proxy for bridge / join tables)
  F  training-size curve with the cross-encoder      G  labelled questions per setting, unique texts
  H  GPU recorded in the runs' manifests
"""
import argparse
import json
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
ap.add_argument("--ce-model", default="BAAI/bge-reranker-large", help="tokenizer for the input-length check (section K)")
ap.add_argument("--reranker", default="Qwen3-Reranker-8B")
ap.add_argument("--k", type=int, default=100)
args = ap.parse_args()
RES = Path(args.out) / "results"
SETS = ["beaver", "livesqlbench", "livesqlbench_large"]
CE_VARIANTS = {"nift_ce": "fine-tuned cross-encoder", "nift_ce0": "zero-shot cross-encoder",
               "nift_ce_nojoin": "fine-tuned cross-encoder, no joins"}


def words(s):
    return set(search.words(s))


def lexmiss(D, i):
    """Gold tables whose name shares no word with the question."""
    q = words(D.Q[i][1])
    return {t for t in D.Q[i][2] if not (words(D.cards[t].split(":")[0]) & q)}


def record(rows, seed, k, D, i, method, order, cfg, size="full", known_db=True):
    g = set(D.Q[i][2]); lm = lexmiss(D, i); top15 = set(order[:15])
    dbs = {D.dbo[t] for t in g}
    base = dict(seed=seed, size=size, set=k, qi=i, n_gold=len(g), group="|".join(map(str, sorted(g))),
                qtext=D.Q[i][1].strip().lower(), one_db=len(dbs) == 1, db_hit=float(D.dbo[order[0]] in dbs))
    rows.append(dict(base, method=method, lm_n=len(lm), lm_found=len(lm & top15), cand_ok=np.nan, **evaluate.metrics(order, g, cfg)))
    if known_db and size == "full" and len(dbs) == 1:          # the same ranking, restricted to the gold database
        db = next(iter(dbs))
        rows.append(dict(base, method=f"{method} | database given", lm_n=0, lm_found=0, cand_ok=np.nan,
                         **evaluate.metrics([t for t in order if D.dbo[t] == db], g, cfg)))


def load_scores(r, run, fname):
    d = RES / f"{run}_seed{r.seed}" / "scores"
    fs = {k: d / fname.format(k=k) for k in r.sets}
    if not all(f.exists() for f in fs.values()):
        return False
    for k, f in fs.items():
        r.sets[k].extra["P"] = np.load(f)
    return True


rows, cover, checks, gpus = [], [], [], {}
rr_cache = RES / ".cache" / "reranker"


def one_run(seed, base_run, cfg_name, ce_runs, size):
    r = Run(seed, base_run, args.data, args.out, config_path=REPO / "configs" / f"{cfg_name}.yaml")
    assert (r.rd / "results.csv").exists(), f"{r.rd} is not finished"
    r.load(check_descriptions=False); r.split(); r.search()
    gpus[f"{base_run}_seed{seed}"] = r.manifest.get("gpu", "?")
    RR = {}
    for k, D in r.sets.items():
        f = rr_cache / f"{k}_{args.reranker}_top{args.k}.npy"
        RR[k] = np.load(f) if f.exists() else None
    scorers = [("Laya", base_run, "{k}_laya.npy")] + [(CE_VARIANTS[c.split("_train")[0]], c, "{k}_ce_" + args.ce + ".npy") for c in ce_runs]
    for name, run, fname in scorers:
        if not load_scores(r, run, fname):
            continue
        r.fit_fusion()
        nodb = {}
        if size == "full":                                      # the same fusion without the two database features
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            for cust, names in r.cfg["customers"].items():
                val = [r.E[k][i] for k in names for i in r.SPLIT[k]["val"]]
                nodb[cust] = make_pipeline(StandardScaler(), LogisticRegression(C=r.cfg["fusion"]["C"], max_iter=3000)).fit(
                    np.vstack([e["F"][:, :6] for e in val]), np.concatenate([e["y"] for e in val]))
        main = pd.read_csv(RES / f"{run}_seed{seed}" / "results.csv") if (RES / f"{run}_seed{seed}" / "results.csv").exists() else None
        for cust, names in r.cfg["customers"].items():
            for k in names:
                D = r.sets[k]
                for i in r.SPLIT[k]["test"]:
                    e = r.E[k][i]
                    nift = fusion.rank(r.FUSION[cust], e)
                    record(rows, seed, k, D, i, f"NIFT-SQL with {name}", nift, r.cfg, size)
                    record(rows, seed, k, D, i, f"{name} alone", e["co"].tolist(), r.cfg, size)
                    record(rows, seed, k, D, i, f"rank average, {name} + search", fusion.rank_average(e), r.cfg, size)
                    if nodb:
                        fo = e["cand"][np.argsort(-nodb[cust].predict_proba(e["F"][:, :6])[:, 1], kind="stable")].tolist()
                        record(rows, seed, k, D, i, f"NIFT-SQL with {name}, no database features",
                               fo + [int(t) for t in e["ho"] if int(t) not in set(fo)], r.cfg, size, known_db=False)
                    if size == "full":
                        g = set(D.Q[i][2])
                        cover.append(dict(seed=seed, set=k, scorer=name, top100=float(g <= set(e["co"][:100].tolist())),
                                          cand=float(g <= set(e["cand"].tolist())), search100=float(g <= set(e["ho"][:100].tolist()))))
        if main is not None:                                    # the rebuilt ranking must reproduce the run's own numbers
            mine = pd.DataFrame([x for x in rows if x["seed"] == seed and x["size"] == size and x["method"] == f"NIFT-SQL with {name}"])
            theirs = main[main.method.str.startswith("NIFT-SQL")]
            for k in SETS:
                a, b = mine[mine.set == k]["c@15"].mean(), theirs[theirs.set == k]["c@15"].mean()
                checks.append(f"{run}_seed{seed} {k:20s} rebuilt c@15 {100 * a:5.1f} | run's results.csv {100 * b:5.1f} | "
                              + ("ok" if abs(a - b) < 1e-9 else "DIFFERENT"))
    for k, D in r.sets.items():                                # search and the zero-shot reranker (no scorer needed)
        for i in r.SPLIT[k]["test"]:
            ho = np.argsort(-D.extra["hyb"][i], kind="stable")
            record(rows, seed, k, D, i, "search alone", ho.tolist(), r.cfg, size)
            if RR[k] is not None and not np.isnan(RR[k][i, ho[:args.k]]).any():
                top = ho[:args.k]
                record(rows, seed, k, D, i, f"search + {args.reranker} (zero-shot)",
                       top[np.argsort(-RR[k][i, top], kind="stable")].tolist() + ho[args.k:].tolist(), r.cfg, size)
    return r


labels = []
for seed in args.seeds:
    r = one_run(seed, "nift_base", "default", [c for c in CE_VARIANTS if (RES / f"{c}_seed{seed}" / "scores").exists()], "full")
    for size, cfg_name in (("10%", "train10"), ("20%", "train20"), ("30%", "train30"), ("50% (main)", "default")):
        rr = r if cfg_name == "default" else Run(seed, f"nift_{cfg_name}", args.data, args.out, config_path=REPO / "configs" / f"{cfg_name}.yaml")
        if cfg_name != "default":
            if not (rr.rd / "results.csv").exists():
                continue
            rr.load(check_descriptions=False); rr.split()
        for cust, names in rr.cfg["customers"].items():
            tr = {rr.sets[k].Q[i][1].strip().lower() for k in names for i in rr.SPLIT[k]["train"]}
            va = {rr.sets[k].Q[i][1].strip().lower() for k in names for i in rr.SPLIT[k]["val"]}
            labels.append(dict(seed=seed, size=size, customer=cust, train=len(tr), val=len(va), total=len(tr | va),
                               train_rows=sum(len(rr.SPLIT[k]["train"]) for k in names)))
        if cfg_name != "default" and (RES / f"nift_ce_{cfg_name}_seed{seed}" / "scores").exists():
            one_run(seed, f"nift_{cfg_name}", cfg_name, [f"nift_ce_{cfg_name}"], size)

A = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
L = ["CE ANALYSIS | test, seeds " + " ".join(map(str, args.seeds)) + " | rankings rebuilt from cached scores, fusion refitted on validation", ""]
L += ["Check: rebuilt NIFT-SQL vs each run's own results.csv"] + ["  " + c for c in checks] + [""]
F = A[(A["size"] == "full") & ~A.method.str.contains("database given|no database features")]
FX = A[A["size"] == "full"]                                  # including the known-database and no-database-feature rows


def linked_groups(X):
    """Cluster ids for pooled tests: gold-table sets, linked across LiveSQLBench and LiveSQLBench-Large by question text."""
    par = {}

    def find(a):
        par.setdefault(a, a)
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    for st, g, q in zip(X.set, X.group, X.qtext):
        cust = "beaver" if st == "beaver" else "lsb"
        par[find(f"{st}#{g}")] = find(f"{cust}#text#{q}")
    return [find(f"{st}#{g}") for st, g in zip(X.set, X.group)]
cols = ["c@5", "c@8", "c@15", "c@25", "R@10"]
L += ["A. Mean (std) over seeds of each seed's test mean, %, as in the paper's tables",
      F.groupby(["set", "method", "seed"])[cols].mean().mul(100).groupby(level=[0, 1]).agg(["mean", "std"]).round(1).to_string(), ""]


def diff(a, b, X=F, sets=SETS, pooled=False):
    out = []
    for k in (["all three"] if pooled else sets):
        Xa = X[(X.method == a) & ((X.set.isin(SETS)) if pooled else (X.set == k))].sort_values(["seed", "set", "qi"])
        Xb = X[(X.method == b) & ((X.set.isin(SETS)) if pooled else (X.set == k))].sort_values(["seed", "set", "qi"])
        if Xa.empty or Xb.empty or len(Xa) != len(Xb):
            continue
        assert (Xa[["seed", "set", "qi"]].values == Xb[["seed", "set", "qi"]].values).all()
        grp = linked_groups(Xa) if pooled else (Xa.set + "#" + Xa.group).tolist()
        cells = []
        for key in ("c@8", "c@15", "R@10"):
            m_, lo, hi = evaluate.group_boot(Xa[key].values - Xb[key].values, grp, 3000)
            cells.append(f"{key} {m_:+5.1f} [{lo:+.1f}, {hi:+.1f}]")
        out.append(f"  {a} minus {b} | {k:20s} " + "   ".join(cells))
    return out


RRN = f"search + {args.reranker} (zero-shot)"
NCE, NCE0, NCEJ, NL = ("NIFT-SQL with fine-tuned cross-encoder", "NIFT-SQL with zero-shot cross-encoder",
                       "NIFT-SQL with fine-tuned cross-encoder, no joins", "NIFT-SQL with Laya")
pairs = [(NCE, RRN), ("fine-tuned cross-encoder alone", RRN), (NCE, "rank average, fine-tuned cross-encoder + search"),
         (NCE, NL), ("fine-tuned cross-encoder alone", "zero-shot cross-encoder alone"), (NCE, NCE0), (NCE0, "search alone"),
         ("zero-shot cross-encoder alone", RRN), (NCEJ, NCE), ("fine-tuned cross-encoder, no joins alone", "fine-tuned cross-encoder alone"),
         (NCE, "fine-tuned cross-encoder alone"), ("fine-tuned cross-encoder alone", "Laya alone"), (NCE, "search alone")]
L += ["B. Paired differences, pooled over seeds, 95% CI clustered by gold-table set across seeds"]
for a, b in pairs:
    L += diff(a, b)
L += ["  pooled over all three datasets (clusters: gold-table sets, linked across the two LiveSQLBench sets by question text):"] + diff(NCE, NL, pooled=True) + diff(NCE, RRN, pooled=True) + diff(NCE, NCE0, pooled=True) + [""]

if cover:
    C = pd.DataFrame(cover)
    L += ["C. Shortlist coverage: share of test questions whose list holds every gold table, %, mean over seeds",
          C.groupby(["set", "scorer", "seed"])[["search100", "top100", "cand"]].mean().mul(100).groupby(level=[0, 1]).mean().round(1)
           .rename(columns={"search100": "search top 100", "top100": "scorer top 100", "cand": "fusion candidates"}).to_string(), ""]

F2 = F.assign(tables=pd.cut(F.n_gold, [0, 1, 2, 4, 100], labels=["1", "2", "3-4", "5+"]))
keep = [m for m in (RRN, NL, NCE, "fine-tuned cross-encoder alone", NCE0) if m in set(F.method)]
L += ["D. complete@15 by number of gold tables, %, (questions)",
      F2[F2.method.isin(keep)].groupby(["set", "method", "tables"], observed=True)["c@15"].mean().mul(100).round(1).unstack().to_string(),
      F2[F2.method == keep[0]].groupby(["set", "tables"], observed=True).size().unstack().to_string(), ""]

E = F[F.method.isin(["search alone", RRN, "Laya alone", "zero-shot cross-encoder alone", "fine-tuned cross-encoder alone",
                     "fine-tuned cross-encoder, no joins alone", NL, NCE0, NCE, NCEJ])]
E = E.groupby(["set", "method"])[["lm_found", "lm_n"]].sum()
E["found in top 15, %"] = (100 * E.lm_found / E.lm_n).round(1)
L += ["E. Gold tables whose name shares no word with the question (lm_n, pooled over seeds): share in the top 15",
      E.to_string(), ""]

S = A[A.method.isin([NCE, NL, "search alone", RRN])]
if S["size"].nunique() > 1:
    L += ["F. Training-size curve, complete@8 / complete@15, mean over seeds of each seed's test mean, %",
          S.groupby(["set", "method", "size", "seed"])[["c@8", "c@15"]].mean().mul(100).groupby(level=[0, 1, 2]).mean().round(1).unstack("size").to_string()]
    for sz in sorted(set(S["size"]) - {"full"}):
        X = A[A["size"] == sz]
        L += [f"  training size {sz}:"] + diff(NCE, "search alone", X) + diff(NCE, RRN, X)
        Y = pd.concat([X[X.method == NCE], F[F.method == NCE].assign(method="full")])
        L += [x.replace(f"{NCE} minus full", f"{NCE} at {sz} minus at 50%") for x in diff(NCE, "full", Y)]
    L += [""]

if labels:
    G = pd.DataFrame(labels).groupby(["customer", "size"])[["train", "val", "total"]].mean().round(0)
    L += ["G. Labelled questions per customer (unique question texts; LiveSQLBench sets share texts), mean over seeds",
          G.to_string(), ""]
DBQ = FX[FX.one_db]
I_methods = ["search alone", RRN, "fine-tuned cross-encoder alone", NL, NCE]
I = DBQ[DBQ.method.isin([m + " | database given" for m in I_methods] + I_methods)]
L += ["I. Database given: questions whose gold tables all sit in one database, each ranking restricted to that database",
      "   share of test questions with a single gold database: " + ", ".join(
          f"{k} {100 * F[(F.set == k) & (F.method == 'search alone')].one_db.mean():.1f}%" for k in SETS),
      "   (the reranker's list is restricted after reranking the global search top 100, so it is not a reranker of the whole database)",
      I.groupby(["set", "method", "seed"])[["c@8", "c@15"]].mean().mul(100).groupby(level=[0, 1]).mean().round(1).to_string(),
      "   top-ranked table in the gold database (database routing), %, all test questions:",
      F[F.method.isin(I_methods)].groupby(["set", "method", "seed"])["db_hit"].mean().mul(100).groupby(level=[0, 1]).mean().round(1).to_string()]
L += diff(NCE + " | database given", "search alone | database given", DBQ) + diff(NCE + " | database given", RRN + " | database given", DBQ) \
     + diff(NCE + " | database given", NL + " | database given", DBQ) + [""]
J = FX[FX.method.str.contains("no database features") | FX.method.isin([NL, NCE])]
L += ["J. Fusion without the two database features (refitted on validation without them)",
      J.groupby(["set", "method", "seed"])[["c@8", "c@15"]].mean().mul(100).groupby(level=[0, 1]).mean().round(1).to_string()]
L += diff(NCE, NCE + ", no database features", FX) + diff(NL, NL + ", no database features", FX) + [""]

try:                                                          # K. how often the 512-token limit cuts the join hints
    from transformers import AutoTokenizer
    from nift import scorer as _sc
    tok = AutoTokenizer.from_pretrained(args.ce_model)
    r1 = Run(args.seeds[0], "nift_base", args.data, args.out, config_path=REPO / "configs" / "default.yaml")
    r1.load(check_descriptions=False); r1.split()
    lines = ["K. Input length for the cross-encoder (max 512 tokens, longest_first truncation cuts the end of the table text first)"]
    for k, D in r1.sets.items():
        qs = r1.SPLIT[k]["test"]
        ql = np.array([len(tok(D.Q[i][1], add_special_tokens=False)["input_ids"]) for i in qs])
        card, full = [], []
        for t in range(D.n):
            st = _sc.state(D, "", t, r1.cfg)
            card.append(len(tok(st["table"], add_special_tokens=False)["input_ids"]))
            full.append(len(tok(st["table"] + (("\nJoins: " + "; ".join(st["joins"])) if st["joins"] else ""), add_special_tokens=False)["input_ids"]))
        card, full, room = np.array(card), np.array(full), 512 - 4 - np.median(ql)
        lines.append(f"   {k:20s} tables over the limit with joins: {100 * np.mean(full > room):.1f}% | joins cut (card fits, card+joins does not): "
                     f"{100 * np.mean((card <= room) & (full > room)):.1f}% | card alone over the limit: {100 * np.mean(card > room):.1f}% "
                     f"(median question {np.median(ql):.0f} tokens)")
    L += lines + [""]
except Exception as exc:
    L += [f"K. Input length: not computed ({exc})", ""]

L += ["", "H. GPU in manifests: " + ", ".join(f"{k} {v}" for k, v in sorted(gpus.items()))]
for c in CE_VARIANTS:
    for s in args.seeds:
        m = RES / f"{c}_seed{s}" / "manifest.json"
        if m.exists():
            L[-1] += f", {c}_seed{s} {json.loads(m.read_text()).get('gpu', '?')}"

txt = "\n".join(L)
print(txt)
(RES / "CE_ANALYSIS.txt").write_text(txt)
print(f"\nsaved: {RES / 'CE_ANALYSIS.txt'}")
