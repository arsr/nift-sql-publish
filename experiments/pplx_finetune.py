"""Fine-tune pplx-decider-flash-v1.2 (Qwen3.5-4B decision model) per customer on Laya's training rows, then compare.

    python experiments/pplx_finetune.py --run arjun_v3j --seed 0 --config nift_ftj --data D --out O

Curiosity experiment, validation only (test is not used). The model ships no training code, so this trains LoRA
adapters (rank 16, MLP and attention projections of the language model) plus its decision head, through the same
forward pass its own reference path uses (left-padded batch, noncausal full attention, readout at the marker).
Training data: exactly Laya's training rows for this run (same questions, same gold / hard / random negatives).
Settings are fixed in advance, not tuned: 2 epochs, lr 1e-4, batch 8, cosine schedule with 5% warmup.
Scored tables: the search top-k per validation question, as in decider_zeroshot.py, so all scorers are compared on
the same tables. Adapters are cached in <run>/checkpoints/pplx_<customer>/ and scores in
<run>/scores/<set>_pplxft_top<k>_val.npy (resumable).
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import importlib
import importlib.metadata as _md
import subprocess

try:                                                 # Colab ships torchao 0.10; new peft refuses any torchao < 0.16 when
    _old = tuple(int(x) for x in _md.version("torchao").split(".")[:2]) < (0, 16)   # it builds LoRA layers. Unused here.
except _md.PackageNotFoundError:
    _old = False
if _old:
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=True)
    importlib.invalidate_caches()
    print("removed the old torchao that peft refuses (not used by this experiment)")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nift import budget, evaluate, scorer   # noqa: E402
from nift.config import REPO                # noqa: E402
from nift.pipeline import Run               # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--config", default="nift_ftj")
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--k", type=int, default=100)
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--model", default="perplexity-ai/pplx-decider-flash-v1.2")
args = ap.parse_args()

r = Run(args.seed, args.run, args.data, args.out, config_path=REPO / "configs" / f"{args.config}.yaml")
r.load(check_descriptions=False); r.split(); r.search(); r.build_rows(show=0)
r.train(); r.train_search(); r.score()
QUESTION = scorer.NEED_Q["need"]

import torch                                         # noqa: E402
from huggingface_hub import snapshot_download        # noqa: E402
scorer._free()
path = Path(args.model) if Path(args.model).exists() else Path(snapshot_download(args.model))
req = path / "requirements.txt"
if req.exists() and not Path(args.model).exists():   # the repo's pins, minus torch: keep the runtime's CUDA build
    import subprocess
    keep = [l for l in req.read_text().splitlines() if l.strip() and not l.strip().lower().startswith(("torch", "#"))]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft>=0.13", *keep], check=True)
sys.path.insert(0, str(path / "source" / "src"))
from pplx_decider import Decider                     # noqa: E402
from pplx_decider.protocol import left_pad, options  # noqa: E402
from peft import LoraConfig, get_peft_model_state_dict, inject_adapter_in_model, set_peft_model_state_dict  # noqa: E402

names = [str(x).lower() for x in options(QUESTION)[0]]
TRUE = names.index("true")
print(f"noul options as the model sees them: {names} -> 'true' is index {TRUE}")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LORA = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                  target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")


def fresh():
    d = Decider.from_pretrained(path, device=DEVICE)
    inject_adapter_in_model(LORA, d.backbone)
    d.backbone.to(DEVICE)
    return d


def forward(d, requests):
    """The reference path's computation (model.py reference_logits), with gradients."""
    chunks = [d.encoder.plan(q, d.max_length)[0] for q in requests]
    inputs = {k: v.to(d.device) for k, v in left_pad(chunks, d.encoder.pad_id).items()}
    hidden = d.backbone(**inputs, use_cache=False).last_hidden_state
    width = inputs["input_ids"].shape[1]
    cols = torch.stack([c.markers[0] + (width - c.length) for c in chunks]).to(d.device)
    rows = torch.arange(len(chunks), device=d.device)
    return d.readout(hidden[rows, cols]).float()[:, :2]          # the noul's two options


def train(d, rows, ck: Path) -> float:
    if (ck / "adapter.pt").exists():
        sd = torch.load(ck / "adapter.pt", map_location="cpu")
        set_peft_model_state_dict(d.backbone, sd["lora"]); d.readout.load_state_dict(sd["readout"])
        print(f"{ck.name}: adapter exists, loaded")
        return 0.0
    for n, p in d.backbone.named_parameters():
        p.requires_grad_("lora_" in n)
    d.readout.requires_grad_(True)
    d.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    d.backbone.enable_input_require_grads()
    d.backbone.train(); d.readout.train()
    params = [p for p in d.backbone.parameters() if p.requires_grad] + list(d.readout.parameters())
    print(f"{ck.name}: training {sum(p.numel() for p in params) / 1e6:.1f}M parameters on {len(rows)} rows")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps = args.epochs * math.ceil(len(rows) / args.batch)
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    rng, t0, step = random.Random(args.seed), time.time(), 0
    for ep in range(args.epochs):
        order = rows[:]; rng.shuffle(order); run_loss = []
        for s in range(0, len(order), args.batch):
            b = order[s:s + args.batch]
            y = torch.tensor([TRUE if x["expected"]["need"] else 1 - TRUE for x in b], device=d.device)
            with torch.autocast(d.device.type, dtype=torch.bfloat16, enabled=d.device.type == "cuda"):
                logits = forward(d, [{"state": x["state"], "questions": {"need": QUESTION}} for x in b])
            loss = torch.nn.functional.cross_entropy(logits / d.temperature, y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            run_loss.append(loss.item())
            if step % 50 == 0 or step == steps:
                print(f"{ck.name}: epoch {ep + 1} step {step}/{steps} | loss {np.mean(run_loss[-50:]):.4f} | {(time.time() - t0) / 60:.1f} min")
    d.backbone.eval(); d.readout.eval()
    d.backbone.requires_grad_(False); d.readout.requires_grad_(False)
    ck.mkdir(parents=True, exist_ok=True)
    torch.save({"lora": get_peft_model_state_dict(d.backbone), "readout": d.readout.state_dict()}, ck / "adapter.pt")
    return time.time() - t0


timing = {}
for cust, ks in r.cfg["customers"].items():
    d = fresh()
    sec = train(d, r.rows[cust], r.rd / "checkpoints" / f"pplx_{cust}")
    if sec:
        timing[f"train {cust}"] = sec / 60
    for k in ks:
        D, f = r.sets[k], r.rd / "scores" / f"{k}_pplxft_top{args.k}_val.npy"
        P = np.load(f) if f.exists() else np.full((len(D.Q), D.n), np.nan)
        todo = [(i, int(t)) for i in r.SPLIT[k]["val"] for t in np.argsort(-D.extra["hyb"][i], kind="stable")[:args.k] if np.isnan(P[i, t])]
        t0 = time.time()
        for s in range(0, len(todo), 16):
            chunk = todo[s:s + 16]
            with torch.inference_mode():
                res = d.decide([{"state": scorer.state(D, D.Q[i][1], t, r.cfg), "questions": {"need": QUESTION}} for i, t in chunk])
            for (i, t), x in zip(chunk, res):
                P[i, t] = float(x["answers"]["need"]["noul"])
            if s % 1600 == 0:
                np.save(f, P)
        np.save(f, P)
        if todo:
            timing[f"score {k}"] = 1000 * (time.time() - t0) / len(todo)
            print(f"{k}: scored {len(todo)} pairs, {timing[f'score {k}']:.0f} ms per table")
        D.extra["P_pplxft"] = P
    del d; scorer._free()

# ---------------------------------------------------------------- compare on validation, same top-k tables for every scorer
scorers = {"Laya (fine-tuned, 421M)": None, "pplx 4B (fine-tuned, LoRA)": "P_pplxft"}
for k, D in r.sets.items():
    z = r.rd / "scores" / f"{k}_pplx_top{args.k}_val.npy"
    if z.exists():
        D.extra["P_pplx0"] = np.load(z)
if all("P_pplx0" in D.extra for D in r.sets.values()):
    scorers["pplx 4B (zero-shot)"] = "P_pplx0"
full, rows = {k: D.extra["P"] for k, D in r.sets.items()}, []
for label, key in scorers.items():
    for k, D in r.sets.items():
        src = full[k] if key is None else D.extra[key]
        P = np.full_like(full[k], budget.FLOOR)
        for i in r.SPLIT[k]["val"]:
            top = np.argsort(-D.extra["hyb"][i], kind="stable")[:args.k]
            P[i, top] = src[i, top]
            ts = set(top.tolist())
            order = sorted(ts, key=lambda t: -P[i, t]) + [int(t) for t in np.argsort(-D.extra["hyb"][i], kind="stable") if int(t) not in ts]
            rows.append(dict(set=k, qi=i, group="|".join(map(str, sorted(D.Q[i][2]))), method=f"{label} alone",
                             **evaluate.metrics(order, set(D.Q[i][2]), r.cfg)))
        D.extra["P"] = P
    rows += [dict(x, method=f"NIFT-SQL fusion with {label}") for x in budget.two_fold(r, r.cfg, "val")]
for k, D in r.sets.items():
    D.extra["P"] = full[k]
R = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print("\n" + "=" * 110 + f"\nSEED {args.seed} VALIDATION, scorers on the search top {args.k} tables (fusion 2-fold inside validation)\n" + "=" * 110)
print(R.groupby(["method", "set"])[["c@8", "c@15", "c@25"]].mean().mul(100).round(1).unstack("set").to_string())
for other in [s for s in scorers if s != "Laya (fine-tuned, 421M)"]:
    cells = []
    for k in r.sets:
        A_ = R[(R.set == k) & (R.method == f"NIFT-SQL fusion with {other}")].sort_values("qi")
        B_ = R[(R.set == k) & (R.method == "NIFT-SQL fusion with Laya (fine-tuned, 421M)")].sort_values("qi")
        m8, lo8, hi8 = evaluate.group_boot(A_["c@8"].values - B_["c@8"].values, A_.group.tolist(), 3000)
        m15, lo15, hi15 = evaluate.group_boot(A_["c@15"].values - B_["c@15"].values, A_.group.tolist(), 3000)
        cells.append(f"{k} c@8 {m8:+.1f} [{lo8:+.1f},{hi8:+.1f}]  c@15 {m15:+.1f} [{lo15:+.1f},{hi15:+.1f}]")
    print(f"\n{other} fusion minus Laya fusion:\n  " + "\n  ".join(cells))
print("\ntiming: " + ", ".join(f"{k} {v:.1f} {'min' if k.startswith('train') else 'ms/table'}" for k, v in timing.items()))
