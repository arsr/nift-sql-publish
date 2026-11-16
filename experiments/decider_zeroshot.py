"""Zero-shot decision models (no training on our data) as the table scorer, against fine-tuned Laya (421M).

    python experiments/decider_zeroshot.py --scorer pplx   --run arjun_v3j --seed 0 --config nift_ftj --data D --out O
    python experiments/decider_zeroshot.py --scorer darwin --run arjun_v3j --seed 0 --config nift_ftj --data D --out O

  pplx   = perplexity-ai/pplx-decider-flash-v1.2 (Qwen3.5-4B, about 9 GiB BF16; gated: HF token with access)
  darwin = FINAL-Bench/Darwin-27B-ZTC-v9 (26B, about 60 GB BF16: an 80 GB GPU)

Validation only (test is not used). Both scorers see exactly the same input (question + joins + table card, the same
'need' question) and score the same tables: the search top-k per validation question. Laya's scores come from the
run's cache; the other scorer's scores are cached in <run>/scores/<set>_<scorer>_top<k>_val.npy (resumable).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nift import budget, evaluate, scorer   # noqa: E402
from nift.config import REPO                # noqa: E402
from nift.pipeline import Run               # noqa: E402

MODELS = {"pplx": ("perplexity-ai/pplx-decider-flash-v1.2", "pplx-decider-flash 4B (zero-shot)", 14),
          "darwin": ("FINAL-Bench/Darwin-27B-ZTC-v9", "Darwin-27B (zero-shot)", 58)}
ap = argparse.ArgumentParser()
ap.add_argument("--scorer", choices=list(MODELS), required=True)
ap.add_argument("--run", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--config", default="nift_ftj")
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--k", type=int, default=100, help="tables scored per question: the search top k")
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--model", default=None, help="override the model repo or a local folder (tests)")
args = ap.parse_args()
repo, LABEL, NEED_GB = MODELS[args.scorer]
repo = args.model or repo

r = Run(args.seed, args.run, args.data, args.out, config_path=REPO / "configs" / f"{args.config}.yaml")
r.load(check_descriptions=False); r.split(); r.search(); r.build_rows(show=0)
r.train(); r.train_search(); r.score()
QUESTION = scorer.NEED_Q["need"]

# ---------------------------------------------------------------- load the zero-shot scorer
import torch                                         # noqa: E402
from huggingface_hub import snapshot_download        # noqa: E402
scorer._free()
free = torch.cuda.mem_get_info()[0] / 2**30 if torch.cuda.is_available() else 0
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'} | free {free:.0f} GB")
local = Path(repo).exists()
assert free >= NEED_GB or local, f"{LABEL} needs about {NEED_GB} GB of free GPU memory on this runtime"
path = Path(repo) if local else Path(snapshot_download(repo))

if args.scorer == "darwin":
    sys.path.insert(0, str(path))
    from autojev.model import DecisionModel, answer      # noqa: E402  (code shipped in the model repo)
    model = DecisionModel(checkpoint=path).eval()

    def p_yes(raw):
        """P(yes) from one Darwin prediction; the format is read off the first prediction and checked."""
        a = answer(QUESTION, raw)
        for obj in (a, raw):
            if isinstance(obj, dict):
                for key in ("probabilities", "probs", "scores", "distribution", "options"):
                    d = obj.get(key)
                    if isinstance(d, dict):
                        for yes in ("true", "True", True, "yes", "Yes"):
                            if yes in d:
                                return float(d[yes])
                for yes in ("true", "True", "yes", "Yes"):
                    if yes in obj and isinstance(obj[yes], (int, float)):
                        return float(obj[yes])
                if isinstance(obj.get("noul"), (int, float)):
                    return float(obj["noul"])
        v = np.asarray(raw.detach().float().cpu() if hasattr(raw, "detach") else raw, dtype=float).ravel()
        if v.size == 2:                                   # noul options are rendered [false, true]
            return float(v[1])
        raise ValueError("cannot read P(yes) from Darwin's output; the first prediction was printed above")

    def predict(states):
        with torch.inference_mode():
            return [p_yes(o) for o in model.predict([{"state": s, "question": QUESTION} for s in states])]
else:
    req = path / "requirements.txt"
    if req.exists() and not local:                     # the repo's pins, minus torch: keep the runtime's CUDA build
        import subprocess
        keep = [l for l in req.read_text().splitlines() if l.strip() and not l.strip().lower().startswith(("torch", "#"))]
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *keep], check=True)
    sys.path.insert(0, str(path / "source" / "src"))
    from pplx_decider import Decider                    # noqa: E402  (code shipped in the model repo)
    decider = Decider.from_pretrained(path, device="cuda" if torch.cuda.is_available() else "cpu")

    def predict(states):                                # one question per request, as the model card recommends
        with torch.inference_mode():
            res = decider.decide([{"state": s, "questions": {"need": QUESTION}} for s in states])
        return [float(x["answers"]["need"]["noul"]) for x in res]

first = predict([scorer.state(r.sets["beaver"], "Which buildings are on the main campus?", 0, r.cfg)])
print(f"{LABEL}: first P(need) = {first[0]:.4f}")
assert 0.0 <= first[0] <= 1.0

# ---------------------------------------------------------------- score the search top-k of every validation question
timing = {}
for k, D in r.sets.items():
    f = r.rd / "scores" / f"{k}_{args.scorer}_top{args.k}_val.npy"
    val = r.SPLIT[k]["val"]
    P = np.load(f) if f.exists() else np.full((len(D.Q), D.n), np.nan)
    todo = [(i, int(t)) for i in val for t in np.argsort(-D.extra["hyb"][i], kind="stable")[:args.k] if np.isnan(P[i, t])]
    todo.sort(key=lambda it: len(json.dumps(scorer.state(D, D.Q[it[0]][1], it[1], r.cfg))))
    t0, done = time.time(), 0
    for s in range(0, len(todo), args.batch):
        chunk = todo[s:s + args.batch]
        out = predict([scorer.state(D, D.Q[i][1], t, r.cfg) for i, t in chunk])
        for (i, t), o in zip(chunk, out):
            P[i, t] = o
        done += len(chunk)
        if s == 0 or done % (args.batch * 50) < args.batch or done == len(todo):
            el = time.time() - t0
            print(f"{k}: {done}/{len(todo)} pairs | {1000 * el / done:.0f} ms per table | ETA {el / done * (len(todo) - done) / 60:.0f} min")
            np.save(f, P)
    np.save(f, P)
    if todo:
        timing[k] = 1000 * (time.time() - t0) / len(todo)
    D.extra["P_other"] = P

# ---------------------------------------------------------------- compare on validation, both scorers on the same top-k tables
rows, full = [], {k: D.extra["P"] for k, D in r.sets.items()}
for scorer_name in ("Laya (fine-tuned, 421M)", LABEL):
    for k, D in r.sets.items():
        src = full[k] if scorer_name.startswith("Laya") else D.extra["P_other"]
        P = np.full_like(full[k], budget.FLOOR)
        for i in r.SPLIT[k]["val"]:
            top = np.argsort(-D.extra["hyb"][i], kind="stable")[:args.k]
            P[i, top] = src[i, top]
        D.extra["P"] = P
        for i in r.SPLIT[k]["val"]:                    # scorer alone: scored tables by score, then the rest by search
            top = set(np.argsort(-D.extra["hyb"][i], kind="stable")[:args.k].tolist())
            order = sorted(top, key=lambda t: -P[i, t]) + [int(t) for t in np.argsort(-D.extra["hyb"][i], kind="stable") if int(t) not in top]
            rows.append(dict(set=k, qi=i, group="|".join(map(str, sorted(D.Q[i][2]))), method=f"{scorer_name} alone",
                             **evaluate.metrics(order, set(D.Q[i][2]), r.cfg)))
    rows += [dict(x, method=f"NIFT-SQL fusion with {scorer_name}") for x in budget.two_fold(r, r.cfg, "val")]
for k, D in r.sets.items():
    D.extra["P"] = full[k]
R = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print("\n" + "=" * 110 + f"\nSEED {args.seed} VALIDATION, scorers on the search top {args.k} tables (fusion 2-fold inside validation)\n" + "=" * 110)
print(R.groupby(["method", "set"])[["c@8", "c@15", "c@25"]].mean().mul(100).round(1).unstack("set").to_string())
a, b = f"NIFT-SQL fusion with {LABEL}", "NIFT-SQL fusion with Laya (fine-tuned, 421M)"
cells = []
for k in r.sets:
    A_ = R[(R.set == k) & (R.method == a)].sort_values("qi"); B_ = R[(R.set == k) & (R.method == b)].sort_values("qi")
    m8, lo8, hi8 = evaluate.group_boot(A_["c@8"].values - B_["c@8"].values, A_.group.tolist(), 3000)
    m15, lo15, hi15 = evaluate.group_boot(A_["c@15"].values - B_["c@15"].values, A_.group.tolist(), 3000)
    cells.append(f"{k} c@8 {m8:+.1f} [{lo8:+.1f},{hi8:+.1f}]  c@15 {m15:+.1f} [{lo15:+.1f},{hi15:+.1f}]")
print(f"\n{LABEL} fusion minus Laya fusion:\n  " + "\n  ".join(cells))
if timing:
    print(f"\n{LABEL} ms per table: " + ", ".join(f"{k} {v:.0f}" for k, v in timing.items()) + "   (Laya, from this study: about 3-6 ms per table)")
