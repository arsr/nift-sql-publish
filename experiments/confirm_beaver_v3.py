"""Confirmation run: NIFT-SQL, frozen as in the paper, scored ONCE on data it has never seen. Read top to bottom.

    python experiments/confirm_beaver_v3.py --beaver /content/drive/MyDrive/beaver_v3 \
        --keys /content/drive/MyDrive/nift_v2/schema_keys.json --out /content/drive/MyDrive/nift_confirm [--seeds 1]

Data: the current public BEAVER release (dw, nova, neutron; 381 tables pooled, no database identifier; 7,857 synthetic
questions and 121 real questions from query logs). The paper's BEAVER is an older 209-question retrieval release, so no
question here was used to develop the method. Declared foreign keys come from the dumps (v2/schema_keys.py): the paper's
names-and-keys setting passes declared keys where they exist (nova and neutron declare some, dw none).

Pre-registered before any test scoring (written to PREREGISTRATION.txt in the run folder, refused once test is scored):
  method   NIFT-SQL exactly as in configs/ftsearch.yaml (configs/beaver_v3.yaml differs only in the data section):
           fine-tuned bge-reranker-large (4 epochs, lr 2e-5, batch 16, BCE, rows = gold + 4 hard + 1 random negatives from
           off-the-shelf hybrid search), hybrid search over BGE fine-tuned on the same rows (2 epochs, contrastive), and the
           8-feature logistic regression (C = 1) fitted on validation; candidates = scorer top 50 + search top 50.
  split    50 / 30 / 20 of the synthetic questions, grouped by gold-table set and question text (nift/split.py).
           The 121 real questions are test only.
  primary  H1  NIFT-SQL minus off-the-shelf hybrid search, complete@15, synthetic test: interval above zero.
           H2  NIFT-SQL minus the retriever fine-tuned on training + validation questions (same labels), complete@15,
               synthetic test: interval above zero.
           Both are also reported on the 121 real questions and at complete@8; 95% intervals clustered by gold-table set.
  scored   once per seed; a second test scoring is refused.

Steps:
  1  Convert the release into the paper's data format (corpus.jsonl / queries.jsonl / qrels), names and declared keys only.
  2  Load and split with the paper's code (nift.data, nift.split); off-the-shelf hybrid search (nift.search).
  3  Training rows (nift.scorer.training_rows) from the training questions; the matched retriever's rows from training +
     validation questions.
  4  Train the cross-encoder; score every table for validation, test and real questions.
  5  Fine-tune BGE on the training rows (NIFT-SQL's retriever) and on training + validation rows (matched retriever).
  6  Fit the fusion on validation, once with off-the-shelf search (NIFT-SQL-base) and once with fine-tuned search (NIFT-SQL).
  7  Score test and real once; report.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nift import data, evaluate, fusion, scorer, search    # noqa: E402
from nift.config import REPO                               # noqa: E402
from nift.pipeline import Run                              # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--beaver", required=True, help="folder with table_<db>.parquet and query_<split>.parquet")
ap.add_argument("--keys", default=None, help="schema_keys.json from v2/schema_keys.py (declared keys); omit for none")
ap.add_argument("--out", required=True)
ap.add_argument("--seeds", type=int, nargs="+", default=[1])
ap.add_argument("--run", default="confirm_v3")
ap.add_argument("--fake", action="store_true", help="tests only: hash embedder, tiny random cross-encoder, no downloads")
ap.add_argument("--overwrite", action="store_true", help="only if an earlier test scoring was broken")
args = ap.parse_args()
OUT = Path(args.out)
CFG_PATH = REPO / "configs" / "beaver_v3.yaml"
DBS = ("dw", "nova", "neutron")
CE_MODEL, EPOCHS, LR, BATCH, MAX_LEN = "BAAI/bge-reranker-large", 4, 2e-5, 16, 512     # as experiments/supervised_crossencoder.py


# ---------------------------------------------------------------- 1  the release in the paper's data format
def build_suite(src: Path, keys: Path | None, dst: Path) -> Path:
    """dst/beaver_v3 (synthetic questions) and dst/beaver_v3_real (the 121 real questions), same corpus."""
    if (dst / "beaver_v3_real" / "qrels" / "test.tsv").exists():
        return dst
    K = json.loads(keys.read_text()) if keys and keys.exists() else {}
    docs, orig = [], {}
    for db in DBS:
        T = pd.read_parquet(src / f"table_{db}.parquet")
        for _, r in T.iterrows():
            t = str(r["table_name"])
            cols = [str(c) for c in (json.loads(r["column_names"]) if isinstance(r["column_names"], str) else r["column_names"])]
            orig[(db, t.upper())] = (t, {c.upper(): c for c in cols})
            docs.append((db, t, cols))
    spaced = sum(any(ch.isspace() for ch in c) for _, _, cols in docs for c in cols)
    print(f"catalog: {len(docs)} tables | column names with whitespace (the parser would cut them): {spaced}")
    out = []
    for db, t, cols in docs:
        lines = [f'  "{c}" TEXT' for c in cols]
        for a, ref, b in K.get(db, {}).get(t.upper(), {}).get("fk", []):
            if (db, ref) in orig and len(a) == len(b):
                rt, rcols = orig[(db, ref)]
                for x, y in zip(a, b):
                    lines.append(f'  FOREIGN KEY ("{orig[(db, t.upper())][1].get(x, x)}") REFERENCES "{rt}" ("{rcols.get(y, y)}")')
        out.append({"_id": f"{db}__{t}", "text": f'CREATE TABLE "{t}" (\n' + ",\n".join(lines) + "\n)"})
    for sub, splits in (("beaver_v3", DBS), ("beaver_v3_real", ("dw_real",))):
        d = dst / sub; (d / "qrels").mkdir(parents=True, exist_ok=True)
        (d / "corpus.jsonl").write_text("".join(json.dumps(x) + "\n" for x in out))
        qs, qrels, dropped = [], ["query-id\tcorpus-id\tscore"], 0
        for sp in splits:
            Q = pd.read_parquet(src / f"query_{sp}.parquet")
            for _, r in Q.iterrows():
                db = str(r["db"]); tabs = json.loads(r["tables"]) if isinstance(r["tables"], str) else list(r["tables"])
                ids = [f"{db}__{orig[(db, x.upper())][0]}" for x in tabs if (db, x.upper()) in orig]
                if not tabs or len(ids) != len(tabs):
                    dropped += 1; continue
                qid = f"{sp}_{r['id']}"
                qs.append({"_id": qid, "text": str(r["question"])})
                qrels += [f"{qid}\t{c}\t1" for c in ids]
        (d / "queries.jsonl").write_text("".join(json.dumps(x) + "\n" for x in qs))
        (d / "qrels" / "test.tsv").write_text("\n".join(qrels) + "\n")
        print(f"{sub}: {len(out)} tables, {len(qs)} questions ({dropped} dropped: gold table not in the public schema)")
    return dst


SUITE = build_suite(Path(args.beaver), Path(args.keys) if args.keys else None, OUT / "beaver_v3_suite")

# ---------------------------------------------------------------- the cross-encoder (as experiments/supervised_crossencoder.py)
import torch                                                                   # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer    # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
AMP = dict(device_type="cuda", dtype=torch.bfloat16) if DEV == "cuda" else None


def doc(st):
    return st["table"] + (("\nJoins: " + "; ".join(st["joins"])) if st["joins"] else "")


def logits(model, tok, pairs):
    enc = tok([q for q, _ in pairs], [d for _, d in pairs], padding=True, truncation="longest_first",
              max_length=MAX_LEN, return_tensors="pt").to(DEV)
    if AMP:
        with torch.autocast(**AMP):
            return model(**enc).logits.float().view(-1)
    return model(**enc).logits.float().view(-1)


def fresh_ce():
    if not args.fake:
        return AutoModelForSequenceClassification.from_pretrained(CE_MODEL, num_labels=1).to(DEV), AutoTokenizer.from_pretrained(CE_MODEL)
    import string, tempfile                                                    # offline test double
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
    d = Path(tempfile.mkdtemp()); vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + list(string.ascii_lowercase + string.digits + ",:_.-;=")
    (d / "vocab.txt").write_text("\n".join(vocab))
    cfg = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64, num_labels=1)
    return BertForSequenceClassification(cfg).to(DEV), BertTokenizerFast(vocab_file=str(d / "vocab.txt"))


def train_ce(rows, ck: Path):
    if (ck / "config.json").exists():
        print(f"{ck.name}: trained cross-encoder exists, loading")
        return AutoModelForSequenceClassification.from_pretrained(str(ck)).to(DEV).eval(), AutoTokenizer.from_pretrained(str(ck))
    model, tok = fresh_ce()
    data_ = [(r["state"]["question"], doc(r["state"]), float(r["expected"]["need"])) for r in rows]
    steps = EPOCHS * math.ceil(len(data_) / BATCH); warm = max(1, int(0.1 * steps))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (steps - s) / max(1, steps - warm)))
    lossf, rng, t0 = torch.nn.BCEWithLogitsLoss(), random.Random(0), time.time()
    model.train()
    for ep in range(EPOCHS):
        rng.shuffle(data_); hist = []
        for s in range(0, len(data_), BATCH):
            b = data_[s:s + BATCH]
            loss = lossf(logits(model, tok, [(q, d) for q, d, _ in b]), torch.tensor([y for _, _, y in b], device=DEV))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); hist.append(loss.item())
        print(f"{ck.name}: epoch {ep + 1}/{EPOCHS} | {len(data_)} rows | loss {np.mean(hist):.4f} | {(time.time() - t0) / 60:.1f} min", flush=True)
    model.eval(); ck.mkdir(parents=True, exist_ok=True); model.save_pretrained(str(ck)); tok.save_pretrained(str(ck))
    return model, tok


@torch.inference_mode()
def score_ce(model, tok, D, qidx, cfg, f: Path):
    """P(needed) for every table of every question in qidx, cached."""
    if f.exists():
        return np.load(f)
    pairs = [(i, t) for i in qidx for t in range(D.n)]
    docs = [doc(scorer.state(D, "", t, cfg)) for t in range(D.n)]
    order = sorted(range(len(pairs)), key=lambda j: len(docs[pairs[j][1]]))
    out, t0 = np.empty(len(pairs)), time.time()
    for s in range(0, len(order), 256):
        js = order[s:s + 256]
        out[js] = torch.sigmoid(logits(model, tok, [(D.Q[pairs[j][0]][1], docs[pairs[j][1]]) for j in js])).cpu().numpy()
    P = np.full((len(D.Q), D.n), np.nan)
    P[[i for i, _ in pairs], [t for _, t in pairs]] = out
    print(f"{D.name}: {len(pairs)} pairs scored, {1000 * (time.time() - t0) / max(1, len(pairs)):.2f} ms per table", flush=True)
    np.save(f, P)
    return P


def embedder(model_or_none):
    if args.fake:
        sys.path.insert(0, str(REPO / "tests"))
        from fake_data import hash_embedder
        return hash_embedder
    return search.st_embedder(model_or_none)


def snapshot(D):
    return {k: D.extra[k].copy() for k in ("dense", "bm25", "hyb")}


def finetuned_search(rows, ck: Path, cfg, seed, sets):
    """Fine-tune BGE on rows (nift.search.finetune_bge) and return hybrid scores for each dataset; restores D.extra."""
    keep = {D.name: snapshot(D) for D in sets}
    model = None if args.fake else search.finetune_bge(rows, cfg, ck, seed, DEV)[0]
    emb, out = embedder(model), {}
    for D in sets:
        search.hybrid_scores(D, emb, cfg)
        out[D.name] = snapshot(D); D.extra.update(keep[D.name])
    del model; scorer._free()
    return out


# ---------------------------------------------------------------- 2 to 7, per seed
all_rows = []
for seed in args.seeds:
    r = Run(seed, args.run, str(SUITE), str(OUT), config_path=CFG_PATH)
    if (r.rd / "results.csv").exists() and not args.overwrite:
        raise FileExistsError(f"{r.rd}/results.csv exists: this seed's test part was already scored once.")
    pre = r.rd / "PREREGISTRATION.txt"
    pre.write_text(__doc__.split("Steps:")[0] + f"\ncommit {r.manifest['git_commit']} | config {r.cfg['_config_sha']} | seed {seed}\n")
    r.load(check_descriptions=False); r.split()
    cfg, D, S = r.cfg, r.sets["beaver_v3"], r.SPLIT["beaver_v3"]
    Dr = data.load_subset(str(SUITE), "beaver_v3_real", "beaver_v3_real"); Dr.adj = data.adjacency(Dr)
    assert Dr.cards == D.cards and Dr.ids == D.ids, "the real questions must use the same catalog"
    real = list(range(len(Dr.Q)))
    print(f"seed {seed}: train {len(S['train'])} | val {len(S['val'])} | test {len(S['test'])} | real {len(real)} questions | {D.n} tables")

    if args.fake:                                                          # 2  off-the-shelf hybrid search
        for X in (D, Dr):
            search.hybrid_scores(X, embedder(None), cfg)
    else:
        r.search()
        search.hybrid_scores(Dr, search.bge_embedder(cfg["search"]["model"], DEV), cfg); scorer._free()
    base = {X.name: snapshot(X) for X in (D, Dr)}

    r.build_rows(show=1)                                                   # 3  rows from off-the-shelf search, as in the paper
    rows_tr = r.rows["beaver_v3"]
    rows_tv = scorer.training_rows(r.sets, [("beaver_v3", i) for i in S["train"] + S["val"]], cfg, seed)

    model, tok = train_ce(rows_tr, r.rd / "checkpoints" / "ce_beaver_v3")  # 4  the scorer
    D.extra["P"] = score_ce(model, tok, D, S["val"] + S["test"], cfg, r.rd / "scores" / "beaver_v3_ce.npy")
    Dr.extra["P"] = score_ce(model, tok, Dr, real, cfg, r.rd / "scores" / "beaver_v3_real_ce.npy")
    del model; scorer._free()

    ft = finetuned_search(rows_tr, r.rd / "checkpoints" / "bge_train", cfg, seed, (D, Dr))       # 5  NIFT-SQL's retriever
    tv = finetuned_search(rows_tv, r.rd / "checkpoints" / "bge_trval", cfg, seed, (D, Dr))       #    matched retriever

    orders = {}                                                            # 6  fusion on validation, then 7  test + real
    for name, srch in (("NIFT-SQL-base (off-the-shelf search)", base), ("NIFT-SQL", ft)):
        for X in (D, Dr):
            X.extra.update(srch[X.name])
        m = fusion.fit([fusion.features(D, i, cfg) for i in S["val"]], cfg)
        print(f"{name}: fusion fitted on {len(S['val'])} validation questions | weights {fusion.weights(m, cfg['fusion']['features'])}")
        for X, idx in ((D, S["test"]), (Dr, real)):
            for i in idx:
                orders.setdefault((X.name, i), {})[name] = fusion.rank(m, fusion.features(X, i, cfg))
    for X, idx in ((D, S["test"]), (Dr, real)):
        for i in idx:
            o = orders[(X.name, i)]
            o["search (off-the-shelf BGE)"] = np.argsort(-base[X.name]["hyb"][i], kind="stable").tolist()
            o["fine-tuned retriever, training questions"] = np.argsort(-ft[X.name]["hyb"][i], kind="stable").tolist()
            o["fine-tuned retriever, training + validation (matched labels)"] = np.argsort(-tv[X.name]["hyb"][i], kind="stable").tolist()
            o["cross-encoder alone"] = np.argsort(-X.extra["P"][i], kind="stable").tolist()
            g = X.Q[i][2]; dbs = sorted({X.dbo[t] for t in g})
            for meth, ordr in o.items():
                all_rows.append(dict(seed=seed, part="test" if X is D else "real", qi=i, db="+".join(dbs), n_gold=len(g),
                                     group=f"{X.name}|" + "|".join(map(str, sorted(g))), method=meth, **evaluate.metrics(ordr, g, cfg)))
    pd.DataFrame([x for x in all_rows if x["seed"] == seed]).to_csv(r.rd / "results.csv", index=False)

# ---------------------------------------------------------------- report
R = pd.DataFrame(all_rows)
METHODS = ["search (off-the-shelf BGE)", "fine-tuned retriever, training questions",
           "fine-tuned retriever, training + validation (matched labels)", "cross-encoder alone",
           "NIFT-SQL-base (off-the-shelf search)", "NIFT-SQL"]
cols = [c for c in R.columns if c.startswith(("c@", "R@"))]
pd.set_option("display.width", 250)
L = [f"NIFT-SQL confirmation on the current BEAVER release | seeds {' '.join(map(str, args.seeds))} | commit {r.manifest['git_commit']} "
     f"| config {r.cfg['_config_sha']}", "Pre-registered: H1 NIFT-SQL > search, H2 NIFT-SQL > matched retriever, complete@15, synthetic test.", ""]
for part in ("test", "real"):
    X = R[R.part == part]
    L += [f"== {part}: {X.qi.nunique()} questions per seed, mean over seeds, %",
          X.groupby(["method", "seed"])[cols].mean().mul(100).groupby(level=0).mean().round(1).reindex(METHODS).to_string(), ""]
L += ["== test, complete@15 by database and by number of gold tables",
      R[R.part == "test"].groupby(["method", "db"])["c@15"].mean().mul(100).round(1).unstack().reindex(METHODS).to_string(),
      R[R.part == "test"].assign(size=lambda d: pd.cut(d.n_gold, [0, 2, 4, 100], labels=["1-2", "3-4", "5+"]))
      .groupby(["method", "size"], observed=True)["c@15"].mean().mul(100).round(1).unstack().reindex(METHODS).to_string(), "",
      "== NIFT-SQL minus each method, pooled over seeds, 95% interval clustered by gold-table set (* excludes zero)"]
for b in METHODS[:-1]:
    for part in ("test", "real"):
        A = R[(R.part == part) & (R.method == "NIFT-SQL")].sort_values(["seed", "qi"])
        B = R[(R.part == part) & (R.method == b)].sort_values(["seed", "qi"])
        cells = []
        for k in ("c@8", "c@15"):
            m_, lo, hi = evaluate.group_boot(A[k].values - B[k].values, A.group.tolist(), 3000)
            cells.append(f"{k} {m_:+5.1f} [{lo:+.1f}, {hi:+.1f}]{' *' if lo > 0 or hi < 0 else '  '}")
        tag = " (H1)" if b.startswith("search") and part == "test" else " (H2)" if "matched" in b and part == "test" else ""
        L.append(f"  minus {b:62s} | {part:4s} " + "   ".join(cells) + tag)
(OUT / "CONFIRM.txt").write_text("\n".join(L) + "\n")
print("\n".join(L))
