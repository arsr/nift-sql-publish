"""Write a tiny fake dataset in the Snowflake suite layout, plus a deterministic offline embedder."""
import hashlib
import json
import random
from pathlib import Path

import numpy as np

SUBSETS = {"beaver": 3, "livesqlbench_table": 2, "livesqlbench_large_table": 3}   # databases per subset
WORDS = "order customer invoice ledger vendor plant material stock price account region payment shipment employee asset".split()


def make(root: Path, seed: int = 0, n_tables: int = 12, n_q: int = 60) -> Path:
    rng = random.Random(seed)
    shared_q = []
    for sub, ndb in SUBSETS.items():
        d = Path(root) / sub
        (d / "qrels").mkdir(parents=True, exist_ok=True)
        corpus, ids = [], []
        for db in range(ndb):
            for t in range(n_tables):
                name, cols = f"T{db}{t:02d}{WORDS[t % len(WORDS)].upper()[:4]}", [f"{WORDS[(t + j) % len(WORDS)][:4].upper()}_{j}" for j in range(4)]
                ref = f"T{db}{(t + 1) % n_tables:02d}{WORDS[(t + 1) % n_tables % len(WORDS)].upper()[:4]}"
                text = (f'CREATE TABLE "{name}" (\n' + "".join(f'  "{c}" VARCHAR(20),\n' for c in cols)
                        + "  -- a comment line\n" + f'  FOREIGN KEY ({cols[0]}) REFERENCES "{ref}" ({cols[0]})\n);')
                ids.append(f"db{db}__{name}")
                corpus.append({"_id": ids[-1], "text": text})
        qs, qrels = [], []
        for q in range(n_q):
            if sub == "livesqlbench_large_table" and q < len(shared_q):
                text, gold = shared_q[q]                      # the large set repeats some questions of the small one
                gold = [g for g in gold if g in ids] or [ids[0]]
            else:
                db = rng.randrange(ndb)
                k = rng.choice([1, 2, 2, 3])
                start = rng.randrange(n_tables - 3)
                gold = [f"db{db}__T{db}{start + j:02d}{WORDS[(start + j) % len(WORDS)].upper()[:4]}" for j in range(k)]
                text = "show " + " and ".join(WORDS[(start + j) % len(WORDS)] for j in range(k)) + f" totals {q % 5}"
                if sub == "livesqlbench_table" and q < 20:
                    shared_q.append((text, gold))
            qs.append({"_id": f"{sub}_q{q}", "text": text})
            qrels += [(f"{sub}_q{q}", g) for g in gold]
        (d / "corpus.jsonl").write_text("\n".join(json.dumps(x) for x in corpus))
        (d / "queries.jsonl").write_text("\n".join(json.dumps(x) for x in qs))
        (d / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\n" + "".join(f"{a}\t{b}\t1\n" for a, b in qrels))
    return Path(root)


def hash_embedder(xs, dim: int = 64) -> np.ndarray:
    out = np.zeros((len(xs), dim))
    for r, s in enumerate(xs):
        for w in str(s).lower().replace("_", " ").replace(":", " ").replace(",", " ").split():
            out[r, int(hashlib.md5(w[:4].encode()).hexdigest(), 16) % dim] += 1
    return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9)
