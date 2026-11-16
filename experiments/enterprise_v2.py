"""Enterprise implementation study (personal, not the paper): push recall with (1) a stronger pplx fine-tune and
(2) richer fusion features. Validation only; the fusion is fitted 2-fold inside validation.

    python experiments/enterprise_v2.py --run arjun_v3j --seed 0 --config nift_ftj --data D --out O

Lever 1, pplx v2 (vs v1 from pplx_finetune.py):
  * hard negatives mined from the FINE-TUNED search (the same search whose top-k pplx scores at inference; v1 used
    Laya's rows, mined from the base search), more of them (--hard 8, --rand 2 per question instead of 4 + 1)
  * LoRA rank 32 on attention, MLP and the linear-attention projections (v1: rank 16, attention + MLP)
Lever 2, fusion features (26 instead of 11), all per (question, candidate table):
  * within-question context: gap to the best score, z-score among the scored tables, share of probability mass,
    rank relative to the expected number of needed tables (sum of P), dense-score gap to the best
  * database coherence: same database as the scorer's top table, share of the scorer's top 10 in this database
  * join graph: hops to the scorer's top table, number of joins
  * table: number of columns, name looks like history / backup / staging
  * question vs table, lexical: question words found in the table name, question words found in its columns
  * question vs table, semantic: cosine(question, table name), best cosine(question, any column name)
  Note: a feature that is the same for every table of a question (e.g. the raw question embedding) cannot change
  the ranking inside that question under a linear model, so the question enters only through relations to tables.
Fusers: logistic regression on the core 11 or the 26 features, and logistic regression with spline terms (non-linear).
"""
import argparse
import collections
import importlib
import importlib.metadata as _md
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

try:                                                 # Colab's torchao 0.10 breaks new peft; unused here
    if tuple(int(x) for x in _md.version("torchao").split(".")[:2]) < (0, 16):
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=True)
        importlib.invalidate_caches()
except _md.PackageNotFoundError:
    pass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nift import budget, evaluate, scorer, search   # noqa: E402
from nift.config import REPO                        # noqa: E402
from nift.pipeline import Run                       # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--config", default="nift_ftj")
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--k", type=int, default=100)
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--hard", type=int, default=8)
ap.add_argument("--rand", type=int, default=2)
ap.add_argument("--rank", type=int, default=32)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--model", default="perplexity-ai/pplx-decider-flash-v1.2")
args = ap.parse_args()

r = Run(args.seed, args.run, args.data, args.out, config_path=REPO / "configs" / f"{args.config}.yaml")
r.load(check_descriptions=False); r.split(); r.search(); r.build_rows(show=0)
r.train(); r.train_search(); r.score()
cfg, QUESTION, TOP = r.cfg, scorer.NEED_Q["need"], r.cfg["fusion"]["candidates_top"]

# ================================================================ lever 1: pplx v2
import torch                                         # noqa: E402
from huggingface_hub import snapshot_download        # noqa: E402
scorer._free()
local = Path(args.model).exists()
path = Path(args.model) if local else Path(snapshot_download(args.model))
req = path / "requirements.txt"
if req.exists() and not local:
    keep = [l for l in req.read_text().splitlines() if l.strip() and not l.strip().lower().startswith(("torch", "#"))]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft>=0.13", *keep], check=True)
sys.path.insert(0, str(path / "source" / "src"))
from pplx_decider import Decider                     # noqa: E402
from pplx_decider.protocol import left_pad, options  # noqa: E402
from peft import LoraConfig, get_peft_model_state_dict, inject_adapter_in_model, set_peft_model_state_dict  # noqa: E402

TRUE = [str(x).lower() for x in options(QUESTION)[0]].index("true")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LORA = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                  target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|in_proj\w*|out_proj)")
TAG = f"pplx2_h{args.hard}r{args.rand}_e{args.epochs}_r{args.rank}"


def v2_rows(cust):
    """Gold + hard negatives from the fine-tuned search (what pplx sees at inference) + random negatives."""
    rng, rows = random.Random(args.seed), []
    for k in cfg["customers"][cust]:
        D = r.sets[k]
        for i in r.SPLIT[k]["train"]:
            g = D.Q[i][2]
            negs = [int(t) for t in np.argsort(-D.extra["hyb"][i], kind="stable") if int(t) not in g]
            pick = sorted(g) + negs[:args.hard] + rng.sample(negs[args.hard:], min(args.rand, max(0, len(negs) - args.hard)))
            rows += [{"state": scorer.state(D, D.Q[i][1], t, cfg), "need": t in g} for t in pick]
    return rows


def forward(d, requests):
    chunks = [d.encoder.plan(q, d.max_length)[0] for q in requests]
    inputs = {k: v.to(d.device) for k, v in left_pad(chunks, d.encoder.pad_id).items()}
    hidden = d.backbone(**inputs, use_cache=False).last_hidden_state
    width = inputs["input_ids"].shape[1]
    cols = torch.stack([c.markers[0] + (width - c.length) for c in chunks]).to(d.device)
    return d.readout(hidden[torch.arange(len(chunks), device=d.device), cols]).float()[:, :2]


def train_pplx(cust):
    d = Decider.from_pretrained(path, device=DEVICE)
    inject_adapter_in_model(LORA, d.backbone); d.backbone.to(DEVICE)
    wrapped = collections.Counter(n.split(".")[-2] for n, _ in d.backbone.named_parameters() if "lora_A" in n)
    ck = r.rd / "checkpoints" / f"{TAG}_{cust}"
    if (ck / "adapter.pt").exists():
        sd = torch.load(ck / "adapter.pt", map_location="cpu")
        set_peft_model_state_dict(d.backbone, sd["lora"]); d.readout.load_state_dict(sd["readout"])
        print(f"{ck.name}: adapter exists, loaded")
        return d
    rows = v2_rows(cust)
    print(f"{ck.name}: LoRA on {dict(wrapped)} | {len(rows)} rows ({sum(x['need'] for x in rows)} gold)")
    for n, p in d.backbone.named_parameters():
        p.requires_grad_("lora_" in n)
    d.readout.requires_grad_(True)
    d.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    d.backbone.enable_input_require_grads(); d.backbone.train(); d.readout.train()
    params = [p for p in d.backbone.parameters() if p.requires_grad] + list(d.readout.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr)
    steps = args.epochs * math.ceil(len(rows) / args.batch); warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    rng, t0, step, hist = random.Random(args.seed), time.time(), 0, []
    for ep in range(args.epochs):
        order = rows[:]; rng.shuffle(order)
        for s in range(0, len(order), args.batch):
            b = order[s:s + args.batch]
            y = torch.tensor([TRUE if x["need"] else 1 - TRUE for x in b], device=d.device)
            with torch.autocast(d.device.type, dtype=torch.bfloat16, enabled=d.device.type == "cuda"):
                logits = forward(d, [{"state": x["state"], "questions": {"need": QUESTION}} for x in b])
            loss = torch.nn.functional.cross_entropy(logits / d.temperature, y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1; hist.append(loss.item())
            if step % 50 == 0 or step == steps:
                print(f"{ck.name}: epoch {ep + 1} step {step}/{steps} | loss {np.mean(hist[-50:]):.4f} | {(time.time() - t0) / 60:.1f} min")
    d.backbone.eval(); d.readout.eval(); d.backbone.requires_grad_(False); d.readout.requires_grad_(False)
    ck.mkdir(parents=True, exist_ok=True)
    torch.save({"lora": get_peft_model_state_dict(d.backbone), "readout": d.readout.state_dict()}, ck / "adapter.pt")
    return d


SC = {k: {"Laya": D.extra["P"]} for k, D in r.sets.items()}
for k, D in r.sets.items():
    f1 = r.rd / "scores" / f"{k}_pplxft_top{args.k}_val.npy"
    if f1.exists():
        SC[k]["pplx v1"] = np.load(f1)
for cust, ks in cfg["customers"].items():
    d = train_pplx(cust)
    for k in ks:
        D, f = r.sets[k], r.rd / "scores" / f"{k}_{TAG}_top{args.k}_val.npy"
        P = np.load(f) if f.exists() else np.full((len(D.Q), D.n), np.nan)
        todo = [(i, int(t)) for i in r.SPLIT[k]["val"] for t in np.argsort(-D.extra["hyb"][i], kind="stable")[:args.k] if np.isnan(P[i, t])]
        for s in range(0, len(todo), 16):
            chunk = todo[s:s + 16]
            with torch.inference_mode():
                res = d.decide([{"state": scorer.state(D, D.Q[i][1], t, cfg), "questions": {"need": QUESTION}} for i, t in chunk])
            for (i, t), x in zip(chunk, res):
                P[i, t] = float(x["answers"]["need"]["noul"])
            if s % 3200 == 0:
                np.save(f, P); print(f"{k}: scored {s + len(chunk)}/{len(todo)}")
        np.save(f, P)
        SC[k]["pplx v2"] = P
    del d; scorer._free()

for k, D in r.sets.items():                          # every scorer sees the same search top-k
    for name in list(SC[k]):
        P = np.full(D.extra["P"].shape, budget.FLOOR)
        for i in r.SPLIT[k]["val"]:
            top = np.argsort(-D.extra["hyb"][i], kind="stable")[:args.k]
            P[i, top] = SC[k][name][i, top]
        SC[k][name] = P

# ================================================================ lever 2: features
def col_names(card):
    return [c.strip() for c in card.split(":", 1)[1].split(",") if c.strip()] if ":" in card else []


if os.environ.get("NIFT_FAKE_EMBED") == "1":
    from fake_data import hash_embedder as EMB_FOR    # noqa: E402
    EMB = {c: EMB_FOR for c in cfg["customers"]}
else:
    from sentence_transformers import SentenceTransformer   # noqa: E402
    EMB = {}
    for c in cfg["customers"]:
        p = r.rd / "checkpoints" / f"bge_{c}"
        m = SentenceTransformer(str(p) if (p / "modules.json").exists() else cfg["search"]["model"], device=DEVICE)
        EMB[c] = (lambda mm: (lambda xs: mm.encode(list(xs), normalize_embeddings=True, batch_size=256, show_progress_bar=False)))(m)
SEM = {}
for cust, ks in cfg["customers"].items():
    for k in ks:
        D = r.sets[k]
        names = [c.split(":", 1)[0] for c in D.cards]
        cols = [col_names(c) for c in D.cards]
        flat = [x for cs in cols for x in cs] or ["_"]
        owner = np.array([t for t, cs in enumerate(cols) for _ in cs])
        val = r.SPLIT[k]["val"]
        QV = np.asarray(EMB[cust]([D.Q[i][1] for i in val])); NV = np.asarray(EMB[cust]([n.replace("_", " ") for n in names]))
        CV = np.asarray(EMB[cust]([c.replace("_", " ") for c in flat]))
        name_cos = QV @ NV.T
        col_cos = np.full((len(val), D.n), -1.0)
        if len(owner):
            S_ = QV @ CV.T
            for t in range(D.n):
                m = owner == t
                if m.any():
                    col_cos[:, t] = S_[:, m].max(1)
        SEM[k] = {i: (name_cos[j], col_cos[j]) for j, i in enumerate(val)}
        D.extra["_names"], D.extra["_cols"] = names, cols


def pre(w):
    return {x[:4] for x in w}


HIST = re.compile(r"(hist|_bkp|backup|_old|_tmp|stg|staging|archive|_log$|audit)", re.I)
CORE11 = ["log_rank_hyb", "dense", "bm25", "db_best_rank_hyb", "logit_p", "log_rank_cls", "disagree", "db_best_rank_cls",
          "join_best_logit", "join_in_top10", "join_bridge"]
EXTRA = ["gap_to_best", "z_in_question", "mass_share", "rank_vs_expected", "dense_gap", "same_db_as_top1", "db_share_top10",
         "hops_to_top1", "log_degree", "log_n_columns", "historyish_name", "q_words_in_name", "q_words_in_columns",
         "cos_q_name", "max_cos_q_column"]
BINARY = {"join_bridge", "same_db_as_top1", "historyish_name"}


def feats(D, i, P):
    n, dense, bm, hyb = D.n, D.extra["dense"][i], D.extra["bm25"][i], D.extra["hyb"][i]
    ho = np.argsort(-hyb, kind="stable"); rh = np.empty(n, int); rh[ho] = np.arange(n)
    p = np.clip(P[i].astype(float), 1e-6, 1 - 1e-6); lp = np.log(p / (1 - p))
    co = np.argsort(-p, kind="stable"); rp = np.empty(n, int); rp[co] = np.arange(n)
    dr, dc, dbof = np.zeros(n), np.zeros(n), np.zeros(n, int)
    for j, ix in enumerate(D.DBI.values()):
        dr[ix], dc[ix], dbof[ix] = rh[ix].min(), rp[ix].min(), j
    cand = np.array(sorted(set(ho[:TOP].tolist()) | set(co[:TOP].tolist())))
    nb = [D.adj.get(int(t), set()) for t in cand]
    t10, t5 = set(co[:10].tolist()), set(co[:5].tolist()) | set(ho[:5].tolist())
    scored = p > 1e-6 * 1.01
    ls = lp[scored] if scored.any() else lp
    top1 = int(co[0])
    hops = {top1: 0}; frontier = [top1]
    for h in (1, 2, 3):
        frontier = [u for v in frontier for u in D.adj.get(v, set()) if u not in hops]
        for u in frontier:
            hops[u] = h
    top10_db = collections.Counter(dbof[co[:10]])
    qw = pre(search.words(D.Q[i][1])) or {"_"}
    nc, cc = SEM[D.name][i]
    F = {
        "log_rank_hyb": np.log1p(rh[cand]), "dense": dense[cand], "bm25": bm[cand], "db_best_rank_hyb": np.log1p(dr[cand]),
        "logit_p": lp[cand], "log_rank_cls": np.log1p(rp[cand]), "disagree": np.abs(np.log1p(rp[cand]) - np.log1p(rh[cand])),
        "db_best_rank_cls": np.log1p(dc[cand]),
        "join_best_logit": np.array([max((lp[u] for u in s), default=-14.0) for s in nb]),
        "join_in_top10": np.array([sum(u in t10 for u in s) for s in nb], float),
        "join_bridge": np.array([float(sum(u in t5 for u in s) >= 2) for s in nb]),
        "gap_to_best": lp[cand] - lp.max(),
        "z_in_question": (lp[cand] - ls.mean()) / (ls.std() + 1e-6),
        "mass_share": p[cand] / p[scored].sum() if scored.any() else p[cand],
        "rank_vs_expected": np.log1p(rp[cand]) - np.log1p(p[scored].sum() if scored.any() else 1.0),
        "dense_gap": dense[cand] - dense.max(),
        "same_db_as_top1": (dbof[cand] == dbof[top1]).astype(float),
        "db_share_top10": np.array([top10_db[dbof[t]] / 10 for t in cand]),
        "hops_to_top1": np.array([hops.get(int(t), 4) for t in cand], float),
        "log_degree": np.log1p([len(s) for s in nb]),
        "log_n_columns": np.log1p([len(D.extra["_cols"][t]) for t in cand]),
        "historyish_name": np.array([float(bool(HIST.search(D.extra["_names"][t]))) for t in cand]),
        "q_words_in_name": np.array([len(qw & pre(search.words(D.extra["_names"][t]))) / len(qw) for t in cand]),
        "q_words_in_columns": np.array([len(qw & pre([w for c in D.extra["_cols"][t] for w in search.words(c)])) / len(qw) for t in cand]),
        "cos_q_name": nc[cand], "max_cos_q_column": cc[cand],
    }
    g = D.Q[i][2]
    return dict(cand=cand, F=np.nan_to_num(np.column_stack([F[c] for c in CORE11 + EXTRA])), ho=ho,
                y=np.array([t in g for t in cand], float), gold=set(g))


def model(kind, ncols, names):
    if kind == "LR splines":
        cont = [j for j, c in enumerate(names) if c not in BINARY]
        bins = [j for j, c in enumerate(names) if c in BINARY]
        ct = ColumnTransformer([("s", make_pipeline(StandardScaler(), SplineTransformer(n_knots=5, degree=3, extrapolation="constant")), cont),
                                ("b", "passthrough", bins)])
        return make_pipeline(ct, StandardScaler(), LogisticRegression(C=0.3, max_iter=5000))
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))


SETUPS = [("LR, core 11", CORE11, "LR"), ("LR, 26 features", CORE11 + EXTRA, "LR"), ("LR splines, 26 features", CORE11 + EXTRA, "LR splines")]
ALLN = CORE11 + EXTRA
rows = []
for scorer_name in [s for s in ("Laya", "pplx v1", "pplx v2") if all(s in SC[k] for k in SC)]:
    for cust, ks in cfg["customers"].items():
        qs = [(k, i) for k in ks for i in r.SPLIT[k]["val"]]
        E = {(k, i): feats(r.sets[k], i, SC[k][scorer_name]) for k, i in qs}
        perm = np.random.default_rng(1).permutation(len(qs)); half = [set(perm[0::2].tolist()), set(perm[1::2].tolist())]
        for label, names, kind in SETUPS:
            idx = [ALLN.index(c) for c in names]
            for a in (0, 1):
                tr = [E[q] for j, q in enumerate(qs) if j in half[a]]
                m = model(kind, len(idx), names).fit(np.vstack([e["F"][:, idx] for e in tr]), np.concatenate([e["y"] for e in tr]))
                for j, (k, i) in enumerate(qs):
                    if j in half[1 - a]:
                        e = E[(k, i)]
                        rk = np.empty(len(e["ho"]), int); rk[e["ho"]] = np.arange(len(e["ho"]))
                        fo = e["cand"][np.lexsort((rk[e["cand"]], -m.predict_proba(e["F"][:, idx])[:, 1]))].tolist()
                        o = fo + [int(t) for t in e["ho"] if int(t) not in set(fo)]
                        rows.append(dict(set=k, qi=i, group="|".join(map(str, sorted(e["gold"]))), method=f"{label} | {scorer_name}",
                                         **evaluate.metrics(o, e["gold"], cfg)))
        if scorer_name == "pplx v2":                  # which new features carry weight (fit on all validation)
            X = np.vstack([E[q]["F"] for q in qs]); y = np.concatenate([E[q]["y"] for q in qs])
            w = model("LR", len(ALLN), ALLN).fit(X, y)[-1].coef_[0]
            print(f"\nstandardized weights, LR 26 features | pplx v2 | {cust}: " +
                  ", ".join(f"{c} {v:+.2f}" for c, v in sorted(zip(ALLN, w), key=lambda x: -abs(x[1]))))
R = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print("\n" + "=" * 110 + "\nSCORER QUALITY on the same search top-k validation pairs\n" + "=" * 110)
q = []
for k, D in r.sets.items():
    for s in SC[k]:
        ps, ys, first = [], [], []
        for i in r.SPLIT[k]["val"]:
            top = np.argsort(-D.extra["hyb"][i], kind="stable")[:args.k]
            ps.append(SC[k][s][i, top]); ys.append(np.array([t in D.Q[i][2] for t in top], float))
            first.append(float(top[np.argmax(SC[k][s][i, top])] in D.Q[i][2]))
        p_, y_ = np.concatenate(ps), np.concatenate(ys)
        q.append(dict(set=k, scorer=s, auc=roc_auc_score(y_, p_), avg_precision=average_precision_score(y_, p_), top1_gold=100 * np.mean(first)))
print(pd.DataFrame(q).set_index(["set", "scorer"]).round(3).to_string())
print("\n" + "=" * 110 + f"\nFUSION, seed {args.seed} validation (2-fold), search top-{args.k}\n" + "=" * 110)
T = R.groupby(["method", "set"])[["c@8", "c@15", "c@25", "R@10"]].mean().mul(100).round(1).unstack("set")
T = T.assign(**{"mean c@8": T["c@8"].mean(axis=1).round(1), "mean c@15": T["c@15"].mean(axis=1).round(1)})
print(T.sort_values("mean c@15", ascending=False).to_string())
base = "LR, core 11 | pplx v1" if "LR, core 11 | pplx v1" in set(R.method) else "LR, core 11 | Laya"
print(f"\nchange vs '{base}', c@8 / c@15 [95% CI over gold-set groups]")
for m in sorted(set(R.method) - {base}):
    cells, d8, d15 = [], [], []
    for k in sorted(R.set.unique()):
        A_ = R[(R.set == k) & (R.method == m)].sort_values("qi"); B_ = R[(R.set == k) & (R.method == base)].sort_values("qi")
        a8 = evaluate.group_boot(A_["c@8"].values - B_["c@8"].values, A_.group.tolist(), 2000)
        a15 = evaluate.group_boot(A_["c@15"].values - B_["c@15"].values, A_.group.tolist(), 2000)
        d8.append(a8[0]); d15.append(a15[0])
        cells.append(f"{k} {a8[0]:+.1f} [{a8[1]:+.1f},{a8[2]:+.1f}] / {a15[0]:+.1f} [{a15[1]:+.1f},{a15[2]:+.1f}]")
    print(f"{m:40s} mean {np.mean(d8):+5.1f} / {np.mean(d15):+5.1f} | " + " | ".join(cells))
