"""Supervised-retriever baseline: hybrid search with BGE fine-tuned per customer, on the same training rows the scorers use.

    python experiments/finetuned_search_baseline.py --data D --out O [--seeds 1 2 3 4] [--replicate R] [--with-val]

Config configs/ftsearch.yaml = configs/default.yaml + search.finetune (2 epochs, contrastive, as in Appendix A).
Same seeds and therefore the same splits as the main runs (checked against nift_base_seed<N>/split.json).

  --replicate R   retrain with another training seed (1000 * seed + R): data order and dropout change, the split does not.
                  R = 0 is the original run. Run name nift_ftsearch[_trval]_rep<R>.
  --with-val      matched label budget: BGE is trained on the training AND validation questions, i.e. every labeled question
                  NIFT-SQL sees (the scorer's training questions plus the fusion's validation questions). Run name
                  nift_ftsearch_trval[_rep<R>]. Test questions are untouched.

Writes results/<run>_seed<N>/scores/<set>_hyb_ft.npy (fine-tuned hybrid scores for every question and table);
experiments/additional_analyses.py turns them into rankings and paired comparisons. About 2 to 4 GPU-minutes per seed.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nift.config import REPO           # noqa: E402
from nift.pipeline import Run          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--main", default="nift_base")
ap.add_argument("--config", default="ftsearch")
ap.add_argument("--replicate", type=int, default=0)
ap.add_argument("--with-val", action="store_true")
args = ap.parse_args()
name = "nift_ftsearch" + ("_trval" if args.with_val else "") + (f"_rep{args.replicate}" if args.replicate else "")

for seed in args.seeds:
    r = Run(seed, name, args.data, args.out, config_path=REPO / "configs" / f"{args.config}.yaml")
    out = {k: r.rd / "scores" / f"{k}_hyb_ft.npy" for k in ("beaver", "livesqlbench", "livesqlbench_large")}
    if all(f.exists() for f in out.values()):
        print(f"{name} seed {seed}: fine-tuned search scores exist, skipping"); continue
    r.load(check_descriptions=False); r.split()
    main_split = Path(args.out) / "results" / f"{args.main}_seed{seed}" / "split.json"
    assert json.loads(main_split.read_text()) == json.loads((r.rd / "split.json").read_text()), "split differs from the main run"
    if args.with_val:                                   # every labeled question NIFT-SQL uses; test stays held out
        for k in r.SPLIT:
            test = set(r.SPLIT[k]["test"])
            r.SPLIT[k] = dict(r.SPLIT[k], train=sorted(set(r.SPLIT[k]["train"]) | set(r.SPLIT[k]["val"])))
            assert not test & set(r.SPLIT[k]["train"])
    r.search(); r.build_rows(show=0)
    r.train_search(train_seed=seed if not args.replicate else 1000 * seed + args.replicate)
    (r.rd / "scores").mkdir(parents=True, exist_ok=True)
    for k, f in out.items():
        np.save(f, r.sets[k].extra["hyb"])
    print(f"{name} seed {seed}: fine-tuned search scores written")
