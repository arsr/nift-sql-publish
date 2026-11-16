"""What the fusion's training data looks like: one real validation question, row by row (paper Section 3.3, Table 2b).

    python experiments/fusion_example.py --data D --out O [--seed 1] [--set beaver]

CPU only, from cached scores, VALIDATION questions only (nothing on test). Uses the main cross-encoder run (nift_ce) and,
when present, this experiment's reranker cache from experiments/fusion_extensions.py for the two reranker features.

Prints and writes <out>/results/FUSION_EXAMPLE.txt:
  1. the size of each customer's fusion training set (validation questions, candidate rows, positive rows);
  2. one validation question of --set, chosen as the first (by index) whose gold tables all lie in the candidates, with
     three or more gold tables, and with one gold table that search ranks below 20 but the scorer ranks in its top 10
     (a table search misses and the scorer finds). The choice is for illustration and is stated in the caption.
     Shown: every gold candidate, the 6 non-gold candidates the fitted fusion scores highest, and the 3 it scores lowest;
  3. the fitted fusion's standardized weights.
Ranks are shown as ranks (0 = first); the fusion itself uses log(1 + rank). Logits are log(p / (1 - p)).
"""
import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nift import fusion                # noqa: E402
from nift.config import REPO           # noqa: E402
from nift.pipeline import Run          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--set", default="beaver")
ap.add_argument("--ce", default="bge-reranker-large")
ap.add_argument("--reranker", default="Qwen3-Reranker-8B", help="tag of the reranker cache written by fusion_extensions.py")
args = ap.parse_args()
RES = Path(args.out) / "results"

r = Run(args.seed, "nift_base", args.data, args.out, config_path=REPO / "configs" / "default.yaml")
r.load(check_descriptions=False); r.split(); r.search()
cfg = r.cfg
RR = {}
for k, D in r.sets.items():
    D.extra["P"] = np.load(RES / f"nift_ce_seed{args.seed}" / "scores" / f"{k}_ce_{args.ce}.npy")
    f = RES / ".cache" / "reranker" / f"{k}_{args.reranker}_fusionext.npy"
    RR[k] = np.load(f) if f.exists() else None


def with_rr(e, k, i):
    if RR[k] is None or np.isnan(RR[k][i, e["cand"]]).any():
        return None
    p = np.clip(RR[k][i, e["cand"]], 1e-6, 1 - 1e-6)
    rank = np.empty(len(p)); rank[np.argsort(-p, kind="stable")] = np.arange(len(p))
    return dict(e, F=np.column_stack([e["F"], np.log(p / (1 - p)), np.log1p(rank)]))


L = [f"FUSION TRAINING DATA | seed {args.seed} | cross-encoder run nift_ce | validation questions only", ""]
models = {}
for cust, names in cfg["customers"].items():
    keys = [(k, i) for k in names for i in r.SPLIT[k]["val"]]
    ex = [fusion.features(r.sets[k], i, cfg) for k, i in keys]
    y = np.concatenate([e["y"] for e in ex])
    models[cust] = fusion.fit(ex, cfg)
    exr = [with_rr(e, k, i) for (k, i), e in zip(keys, ex)]
    rr_ok = all(x is not None for x in exr)
    L.append(f"customer {cust} ({' + '.join(names)}): {len(ex)} validation questions, {len(y)} candidate rows "
             f"({len(y) / len(ex):.1f} per question), {int(y.sum())} positive ({100 * y.mean():.1f}%), 8 feature columns"
             + (", 10 with the reranker features" if rr_ok else ", reranker features not cached"))
    if rr_ok:
        models[(cust, "rr")] = fusion.fit(exr, cfg)
L.append("")

k = args.set
D = r.sets[k]
cust = next(c for c, ns in cfg["customers"].items() if k in ns)
pick = None
for i in r.SPLIT[k]["val"]:
    e = fusion.features(D, i, cfg)
    g = e["gold"]
    if len(g) < 3 or not g <= set(e["cand"].tolist()):
        continue
    rp = np.empty(D.n, int); rp[e["co"]] = np.arange(D.n)
    rh = np.empty(D.n, int); rh[e["ho"]] = np.arange(D.n)
    if any(rh[t] >= 20 and rp[t] < 10 for t in g):
        pick = i
        break
if pick is None:                                               # fall back: first with every gold table in the candidates
    pick = next(i for i in r.SPLIT[k]["val"] if (e := fusion.features(D, i, cfg))["gold"] <= set(e["cand"].tolist()))
i = pick
e = fusion.features(D, i, cfg)
er = with_rr(e, k, i)
prob = models[cust].predict_proba(e["F"])[:, 1]
order = np.argsort(-prob, kind="stable")
gold_rows = [j for j in order if e["y"][j] == 1]
neg = [j for j in order if e["y"][j] == 0]
show = gold_rows + neg[:6] + neg[-3:]

hdr = ["table", "label", "scorer logit", "scorer rank", "search rank", "dense cosine", "BM25", "rank gap",
       "DB best search rank", "DB best scorer rank"] + (["reranker logit", "reranker rank"] if er is not None else []) + ["fused P"]
L += [f"Example: {k} validation question {i}: \"{D.Q[i][1].strip()}\"",
      f"gold tables: {', '.join(D.cards[t].split(':')[0] for t in sorted(e['gold']))} | {len(e['cand'])} candidate rows, "
      f"{int(e['y'].sum())} positive; shown: every gold row, the 6 highest- and 3 lowest-scored non-gold rows", "",
      "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
for n_, j in enumerate(show):
    if n_ == len(gold_rows) + 6:
        L.append("| ... | | | | | | | | | |" + (" | |" if er is not None else "") + " |")
    F = e["F"][j]
    cells = [D.cards[e["cand"][j]].split(":")[0], str(int(e["y"][j])), f"{F[0]:.2f}", f"{np.expm1(F[1]):.0f}", f"{np.expm1(F[2]):.0f}",
             f"{F[3]:.3f}", f"{F[4]:.2f}", f"{F[5]:.2f}", f"{np.expm1(F[6]):.0f}", f"{np.expm1(F[7]):.0f}"]
    if er is not None:
        cells += [f"{er['F'][j][8]:.2f}", f"{np.expm1(er['F'][j][9]):.0f}"]
    cells.append(f"{prob[j]:.3f}")
    L.append("| " + " | ".join(cells) + " |")
L += ["", "Fused P: the 8-feature fusion fitted on this customer's validation questions (this question included, as a training row).",
      "", f"Standardized fusion weights, customer {cust}, 8 features: {fusion.weights(models[cust], cfg['fusion']['features'])}"]
if (cust, "rr") in models:
    L.append(f"with the two reranker features: {fusion.weights(models[(cust, 'rr')], cfg['fusion']['features'] + ['reranker_logit', 'reranker_log_rank'])}")
(RES / "FUSION_EXAMPLE.txt").write_text("\n".join(L) + "\n")
print("\n".join(L))
