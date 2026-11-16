"""NIFT-SQL enterprise: one experiment, top to bottom. Start reviewing here.

    python main.py --seed 1 --name arjun --data <DATA_ROOT> --out <OUT_ROOT> [--config default]

Every step below is one call. The comment above each says which file holds the code.
Settings come from configs/<config>.yaml; nothing is hard-coded here.
"""
import argparse

from nift.config import REPO
from nift.pipeline import Run

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True, help="Arjun: 1, 2 | nishant: 3, 4 | 0 = development only")
ap.add_argument("--name", required=True, help="run name, lowercase, e.g. arjun")
ap.add_argument("--data", required=True, help="folder with beaver/, livesqlbench_table/, livesqlbench_large_table/")
ap.add_argument("--out", required=True, help="results go to <out>/results/<name>_seed<seed>/")
ap.add_argument("--config", default="default", help="file in configs/ without .yaml; a changed config needs a new --name")
args = ap.parse_args()

# Setup: load configs/<config>.yaml, make the run folder, record commit + config hash + GPU   (nift/config.py)
run = Run(args.seed, args.name, args.data, args.out, config_path=REPO / "configs" / f"{args.config}.yaml")

# Step 1. Load BEAVER, LiveSQLBench, LiveSQLBench-Large as names-only cards + joins,
#         and report whether table documents contain any descriptions                       (nift/data.py)
run.load()

# Step 2. Split 50/30/20 train/val/test, grouped by gold-table set; shared questions on one side (nift/split.py)
run.split()

# Step 3. Hybrid search over every table: 0.6 BGE dense + 0.4 BM25                          (nift/search.py)
run.search()

# Step 4. Build Laya training rows from TRAIN questions and print what they look like        (nift/scorer.py)
run.build_rows(show=2)

# Step 5. Train one Laya model per customer from the base checkpoint (cached)                (nift/scorer.py)
run.train()

# Step 5b. Fine-tune the search model (BGE) per customer on the SAME training rows, then redo search.
#          Only when the config has search.finetune (configs/nift_ft.yaml); a no-op for default.yaml  (nift/search.py)
run.train_search()

# Step 6. Laya P(need) for every validation and test question x every table                 (nift/scorer.py)
run.score()

# Step 7. Fit the fusion (logistic regression, 8 core features) on VALIDATION, per customer  (nift/fusion.py)
run.fit_fusion()

# Step 8. Score TEST once: search alone, Laya alone, rank average (no learning), NIFT-SQL;
#         plus base-BGE search when the search model was fine-tuned. Group-bootstrap CIs      (nift/evaluate.py)
run.test()
