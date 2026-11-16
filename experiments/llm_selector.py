"""LLM baseline: a large instruction-tuned LLM (default Gemma 4 31B, zero-shot) picks the needed tables from a shortlist.

    python experiments/llm_selector.py --seeds 1 --data D --out O [--model google/gemma-4-31B-it] [--shortlist search --k 50]

This is the usual LLM schema-linking setup: retrieve a shortlist, show the LLM every shortlisted table (names-only card +
joins, the same text the scorers read), ask for the tables needed to write the SQL, most necessary first.
Ranking for complete@K: the LLM's picks in its order, then every other table in the order of the ranking the shortlist
came from (search, or NIFT-SQL). TEST questions of the finished main runs (nift_base), compared on the same questions with NIFT-SQL (Laya) and,
when present, NIFT-SQL with the fine-tuned cross-encoder (nift_ce). 95% intervals clustered by gold-table set.
  --shortlist search   the hybrid search top k (an LLM-only pipeline, no customer training)
  --shortlist nift_ce  the top k of NIFT-SQL with the fine-tuned cross-encoder (an LLM picker on top of our retriever)
Answers are cached per (model, shortlist, k, question) in <out>/results/.cache/llm/, so the run resumes.
Gemma 4 31B needs about 62 GB in bf16; on a 40 GB GPU use --load 4bit (bitsandbytes).
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nift import evaluate, fusion, scorer      # noqa: E402
from nift.pipeline import Run                  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[1])
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="google/gemma-4-31B-it")
ap.add_argument("--load", choices=["auto", "bf16", "4bit"], default="auto")
ap.add_argument("--shortlist", choices=["search", "nift_ce"], default="search")
ap.add_argument("--k", type=int, default=50)
ap.add_argument("--max-new", type=int, default=160)
args = ap.parse_args()
TAG = args.model.rstrip("/").split("/")[-1]
LABEL = f"{TAG} picks from {args.shortlist} top {args.k}"

PROMPT = ("You are given a question about a company's database and a numbered list of candidate tables, each shown as "
          "'table name: column names', sometimes followed by its join keys.\n"
          "Select every table that is needed to write the SQL query that answers the question: tables that hold answer "
          "columns, filter columns or values, and tables needed to join such tables.\n"
          "Answer with a JSON list of the table numbers, most necessary first, for example [4, 17, 2]. Include every needed "
          "table and no others. Output only the JSON list.\n\nQuestion: {q}\n\nCandidate tables:\n{tables}")

# ---------------------------------------------------------------- model
if args.model == "fake":                                       # tests only: picks the first three shortlisted tables
    def ask(prompt):
        return "[1, 2, 3]"
else:
    import torch
    import transformers
    from transformers import AutoProcessor
    free = torch.cuda.mem_get_info()[0] / 2**30 if torch.cuda.is_available() else 0
    load = args.load if args.load != "auto" else ("bf16" if free > 70 else "4bit")
    kw = dict(device_map="auto")
    if load == "4bit":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                                       bnb_4bit_quant_type="nf4")
    else:
        kw["dtype"] = torch.bfloat16
    cls = next(getattr(transformers, c) for c in ("AutoModelForMultimodalLM", "AutoModelForImageTextToText", "AutoModelForCausalLM")
               if hasattr(transformers, c))
    print(f"loading {args.model} ({load}, {cls.__name__}) | GPU free {free:.0f} GB")
    proc = AutoProcessor.from_pretrained(args.model)
    model = cls.from_pretrained(args.model, **kw).eval()

    @torch.inference_mode()
    def ask(prompt):
        msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        try:
            inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                           return_tensors="pt", enable_thinking=False)
        except TypeError:
            inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        inp = inp.to(model.device)
        out = model.generate(**inp, max_new_tokens=args.max_new, do_sample=False)
        return proc.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)


def picks(text, k):
    """Table numbers from the answer, in order, valid and unique."""
    m = re.search(r"\[[^\]]*\]", text)
    nums = [int(x) for x in re.findall(r"\d+", m.group(0) if m else text)]
    seen, out = set(), []
    for n in nums:
        if 1 <= n <= k and n not in seen:
            seen.add(n); out.append(n)
    return out


def card_text(D, t, cfg):
    st = scorer.state(D, "", t, cfg)
    return st["table"] + (("  | joins: " + "; ".join(st["joins"])) if st["joins"] else "")


cache_dir = Path(args.out) / "results" / ".cache" / "llm"; cache_dir.mkdir(parents=True, exist_ok=True)
cache_f = cache_dir / f"{TAG}_{args.shortlist}_top{args.k}.json"
cache = json.loads(cache_f.read_text()) if cache_f.exists() else {}
rows, secs, n_bad = [], [], 0
for seed in args.seeds:
    r = Run(seed, "nift_base", args.data, args.out)
    main = pd.read_csv(r.rd / "results.csv")
    r.load(check_descriptions=False); r.split(); r.search()
    ce = None
    if args.shortlist == "nift_ce" or (Path(args.out) / "results" / f"nift_ce_seed{seed}" / "results.csv").exists():
        ce_dir = Path(args.out) / "results" / f"nift_ce_seed{seed}"
        if args.shortlist == "nift_ce":
            assert ce_dir.exists(), "run experiments/supervised_crossencoder.py first"
            for k, D in r.sets.items():
                D.extra["P"] = np.load(next(ce_dir.glob(f"scores/{k}_ce_*.npy")))
            r.fit_fusion()
        if (ce_dir / "results.csv").exists():
            ce = pd.read_csv(ce_dir / "results.csv")
    for cust, names in r.cfg["customers"].items():
        for k in names:
            D = r.sets[k]
            for i in r.SPLIT[k]["test"]:
                ho = np.argsort(-D.extra["hyb"][i], kind="stable")
                full = ho.tolist() if args.shortlist == "search" else fusion.rank(r.FUSION[cust], r.E[k][i])
                sl = [int(t) for t in full[:args.k]]
                key = f"{k}|{D.Q[i][0]}" + ("" if args.shortlist == "search" else f"|seed{seed}")   # NIFT shortlists differ by seed
                if key not in cache:
                    tables = "\n".join(f"[{j + 1}] {card_text(D, t, r.cfg)}" for j, t in enumerate(sl))
                    t0 = time.time(); ans = ask(PROMPT.format(q=D.Q[i][1], tables=tables)); secs.append(time.time() - t0)
                    cache[key] = {"answer": ans, "shortlist": [int(t) for t in sl]}
                    if len(cache) % 20 == 0:
                        cache_f.write_text(json.dumps(cache)); print(f"{len(cache)} questions answered | {np.mean(secs):.1f} s per question", flush=True)
                got = picks(cache[key]["answer"], len(sl))
                n_bad += not got
                chosen = [sl[j - 1] for j in got]
                order = chosen + [int(t) for t in full if int(t) not in set(chosen)]     # then the shortlist's own ranking
                g = set(D.Q[i][2])
                rows.append(dict(seed=seed, set=k, qi=i, group="|".join(map(str, sorted(g))), method=LABEL,
                                 n_picked=len(got), **evaluate.metrics(order, g, r.cfg)))
    cache_f.write_text(json.dumps(cache))
    keep = ["search alone", "NIFT-SQL (fusion)"]
    rows += main[main.method.isin(keep)].assign(seed=seed)[["seed", "set", "qi", "group", "method"] +
             [c for c in main.columns if c.startswith(("c@", "R@"))]].to_dict("records")
    if ce is not None:
        rows += ce.assign(seed=seed)[["seed", "set", "qi", "group", "method"] + [c for c in ce.columns if c.startswith(("c@", "R@"))]].to_dict("records")

A = pd.DataFrame(rows)
pd.set_option("display.width", 250)
lines = [f"LLM SELECTOR | {LABEL} | test, seeds {args.seeds} | answers without a usable list: {n_bad}"
         + (f" | {np.mean(secs):.1f} s per question (this session)" if secs else ""), ""]
lines.append(A.groupby(["set", "method"])[["c@5", "c@8", "c@15", "c@25", "R@10"]].mean().mul(100).round(1).to_string())
lines.append(f"\ntables picked per question (mean): " + ", ".join(f"{k} {v:.1f}" for k, v in A[A.method == LABEL].groupby("set").n_picked.mean().items()))
lines.append("\nNIFT-SQL minus the LLM selector, c@8 / c@15 / R@10 [95% CI, clustered by gold-table set]")
for other in [m for m in A.method.unique() if m.startswith("NIFT-SQL")]:
    for k in sorted(A.set.unique()):
        X = A[(A.set == k) & (A.method == other)].sort_values(["seed", "qi"]); Y = A[(A.set == k) & (A.method == LABEL)].sort_values(["seed", "qi"])
        assert (X.qi.values == Y.qi.values).all()
        cells = []
        for key in ("c@8", "c@15", "R@10"):
            m_, lo, hi = evaluate.group_boot(X[key].values - Y[key].values, X.group.tolist(), 3000)
            cells.append(f"{key} {m_:+5.1f} [{lo:+.1f}, {hi:+.1f}]")
        lines.append(f"  {other:45s} | {k:20s} " + "   ".join(cells))
txt = "\n".join(lines)
print("\n" + "=" * 110 + "\n" + txt)
out = Path(args.out) / "results" / f"LLM_SELECTOR_{TAG}_{args.shortlist}_top{args.k}_seeds{''.join(map(str, args.seeds))}.txt"
out.write_text(txt)
print(f"\nsaved: {out}")
