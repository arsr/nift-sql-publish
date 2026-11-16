"""One experiment run, step by step. The notebook calls these steps in order; `python -m nift.pipeline` runs them all.

Protocol (agreed):
  per customer: Laya trained on TRAIN, fusion fit on VAL (Laya scores on VAL come from the TRAIN model),
  TEST scored once at the end. No transfer between customers. No LLM anywhere in the headline.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import data, evaluate, fusion, scorer, search, split
from .costs import CostLog

METHODS = ["search alone", "Laya alone", "rank average (no learning)", "NIFT-SQL (fusion)"]
BASE_SEARCH = "search alone, base BGE"            # reported only when the config fine-tunes the search model


class Run:
    def __init__(self, seed: int, run_name: str, data_root: str, out_root: str, device: str | None = None, config_path: str | None = None):
        self.cfg = C.load_config(config_path)
        self.seed, self.run_name = int(seed), run_name
        self.data_root, self.out_root = data_root, out_root
        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self.device = device
        self.rd = C.run_dir(out_root, run_name, self.seed)
        old = self.rd / "manifest.json"
        if old.exists() and json.loads(old.read_text()).get("config_sha") != self.cfg["_config_sha"]:
            raise RuntimeError(f"{self.rd} was made with a different config, and its cached models would be reused. "
                               "Use a new RUN_NAME for a changed config, e.g. 'arjun_ep6'.")
        self.costs = CostLog(self.rd / "costs.json", self.cfg["costs"]["gpu_price_per_hour"])
        C.seed_everything(self.seed)
        if (self.rd / "results.csv").exists() and old.exists():           # a finished run keeps the record of the code that scored it
            self.manifest = json.loads(old.read_text())
        else:
            self.manifest = C.write_manifest(self.rd, self.cfg, self.seed, run_name, {"device": device})
        print(f"run {run_name} seed {self.seed} | device {device} | commit {self.manifest['git_commit']} | "
              f"config {self.cfg['_config_sha']} | writing to {self.rd}")

    # ---------------------------------------------------------------- step 1
    def load(self, check_descriptions: bool = True) -> None:
        self.sets = data.load_all(self.data_root, self.cfg)
        data.summary(self.sets)
        import hashlib                                                     # cached models are only valid for the same cards
        sha = hashlib.sha256(json.dumps({k: [D.cards, sorted((t, j) for t, js in D.joins.items() for j in js)]
                                         for k, D in self.sets.items()}).encode()).hexdigest()[:12]
        f = self.rd / "cards_sha.txt"
        if f.exists() and f.read_text().strip() != sha:
            raise RuntimeError(f"{self.rd} was built from different table cards (data or parser changed). Use a new RUN_NAME.")
        f.write_text(sha)
        print(f"cards fingerprint: {sha}")
        if check_descriptions:
            rep = data.description_report(self.data_root, self.cfg)
            (self.rd / "description_report.json").write_text(json.dumps(rep, indent=1))

    # ---------------------------------------------------------------- step 2
    def split(self) -> None:
        self.SPLIT = split.make_split(self.sets, self.cfg, self.seed)
        self.SPLIT = split.subsample_train(self.sets, self.SPLIT, self.cfg, self.seed)   # no-op unless split.train_use < 1
        split.split_report(self.sets, self.SPLIT)
        (self.rd / "split.json").write_text(json.dumps(self.SPLIT))

    # ---------------------------------------------------------------- step 3
    def search(self, embed=None) -> None:
        cache_model = None
        if embed is None and os.environ.get("NIFT_FAKE_EMBED") == "1":      # tests only: offline embedder
            from fake_data import hash_embedder as embed
        elif embed is None:
            cache_model = self.cfg["search"]["model"]
            embed = search.bge_embedder(cache_model, self.device)
        cache_dir = Path(self.out_root) / "results" / ".cache" / "search_embeddings"
        with self.costs.stage("search (embed all cards + questions)"):
            for D in self.sets.values():
                cache_path = search.embedding_cache_path(cache_dir, cache_model, D) if cache_model else None
                search.hybrid_scores(D, embed, self.cfg, cache_path)

    # ---------------------------------------------------------------- step 4
    def build_rows(self, show: int = 2) -> None:
        self.rows = {}
        for cust, names in self.cfg["customers"].items():
            qs = [(k, i) for k in names for i in self.SPLIT[k]["train"]]
            self.rows[cust] = scorer.training_rows(self.sets, qs, self.cfg, self.seed)
            scorer.write_rows(self.rd / "rows" / f"{cust}.jsonl", self.rows[cust])
            print("\n" + "=" * 100 + f"\nTRAINING DATA, customer {cust} ({' + '.join(names)})\n" + "=" * 100)
            scorer.show_rows(self.rows[cust], show)

    # ---------------------------------------------------------------- step 5
    def train(self) -> None:
        self.CK = {}
        for cust in self.cfg["customers"]:
            ck, sec = scorer.train(self.rd / "rows" / f"{cust}.jsonl", self.rd / "checkpoints" / f"laya_{cust}", self.cfg, self.seed, self.device)
            self.CK[cust] = ck
            if sec:
                self.costs.add(f"train Laya, {cust}", sec)
            print(f"{cust}: {len(self.rows[cust]) if hasattr(self, 'rows') else '?'} rows -> {ck}")

    # ---------------------------------------------------------------- step 5b
    def train_search(self, train_seed: int | None = None) -> None:
        """Fine-tune the dense search model per customer on the same training rows (config search.finetune), then redo search.

        Off in configs without search.finetune. Laya's rows were built from the base search, so Laya is unaffected.
        The base search scores are kept as 'hyb_base' for the base-search baseline.
        """
        if not self.cfg["search"].get("finetune"):
            print("search fine-tuning is off in this config")
            return
        scorer._free()                                                     # release the base BGE before training
        for cust, names in self.cfg["customers"].items():
            model, sec = search.finetune_bge(self.rows[cust], self.cfg, self.rd / "checkpoints" / f"bge_{cust}",
                                             self.seed if train_seed is None else train_seed, self.device)
            if sec:
                self.costs.add(f"fine-tune BGE, {cust}", sec)
            embed = search.st_embedder(model)
            with self.costs.stage(f"search with fine-tuned BGE, {cust}"):
                for k in names:
                    D = self.sets[k]
                    D.extra["hyb_base"] = D.extra["hyb"]
                    search.hybrid_scores(D, embed, self.cfg)
            for k in names:
                D, val = self.sets[k], self.SPLIT[k]["val"]
                hit = lambda h: 100 * np.mean([D.Q[i][2] <= set(np.argsort(-h[i])[:50].tolist()) for i in val])  # noqa: E731
                print(f"{k}: validation questions with every gold table in search top 50: base {hit(D.extra['hyb_base']):.1f}% "
                      f"-> fine-tuned {hit(D.extra['hyb']):.1f}%")
            del model
            scorer._free()

    # ---------------------------------------------------------------- step 6
    def score(self) -> None:
        for cust, names in self.cfg["customers"].items():
            for k in names:
                D, pf = self.sets[k], self.rd / "scores" / f"{k}_laya.npy"
                qs = self.SPLIT[k]["val"] + self.SPLIT[k]["test"]
                if pf.exists():
                    D.extra["P"] = np.load(pf)
                    print(f"{k}: cached Laya scores")
                    continue
                P = np.full((len(D.Q), D.n), np.nan)
                with self.costs.stage(f"score Laya, {k} (val + test, all tables)", len(qs)):
                    P[qs] = scorer.score(self.CK[cust], D, qs, self.cfg, self.device)
                np.save(pf, P)
                D.extra["P"] = P
                G = np.zeros((len(qs), D.n), bool)
                for r, i in enumerate(qs):
                    G[r, list(D.Q[i][2])] = True
                print(f"{k}: mean P(need) gold {P[qs][G].mean():.3f} vs other {P[qs][~G].mean():.3f}")

    # ---------------------------------------------------------------- step 7
    def fit_fusion(self) -> None:
        self.FUSION, self.E = {}, {}
        for cust, names in self.cfg["customers"].items():
            for k in names:
                D = self.sets[k]
                self.E[k] = {i: fusion.features(D, i, self.cfg) for i in self.SPLIT[k]["val"] + self.SPLIT[k]["test"]}
                ceil = np.mean([self.E[k][i]["gold"] <= set(self.E[k][i]["cand"].tolist()) for i in self.SPLIT[k]["test"]])
                print(f"{k:20s} test questions with every gold table inside the candidates: {100 * ceil:.1f}%")
            val = [self.E[k][i] for k in names for i in self.SPLIT[k]["val"]]
            self.FUSION[cust] = fusion.fit(val, self.cfg)
            print(f"fusion {cust}: fit on {len(val)} validation questions | weights (standardized): {fusion.weights(self.FUSION[cust], self.cfg['fusion']['features'])}")
        (self.rd / "fusion_weights.json").write_text(json.dumps({c: fusion.weights(m, self.cfg["fusion"]["features"]) for c, m in self.FUSION.items()}, indent=1))

    # ---------------------------------------------------------------- step 8
    def test(self, overwrite: bool = False) -> pd.DataFrame:
        """Score the TEST part once. Writes results.csv and summary.txt. Refuses to overwrite a finished run."""
        if (self.rd / "results.csv").exists() and not overwrite:
            raise FileExistsError(f"{self.rd}/results.csv already exists: this seed's test part was already scored. "
                                  "Use a new RUN_NAME, or call test(overwrite=True) only if the earlier run was broken.")
        rows, ev = [], self.cfg["evaluate"]
        tuned = bool(self.cfg["search"].get("finetune"))
        methods = ([BASE_SEARCH] if tuned else []) + METHODS
        for cust, names in self.cfg["customers"].items():
            for k in names:
                for i in self.SPLIT[k]["test"]:
                    e = self.E[k][i]
                    orders = {"search alone": e["ho"].tolist(), "Laya alone": e["co"].tolist(),
                              "rank average (no learning)": fusion.rank_average(e), "NIFT-SQL (fusion)": fusion.rank(self.FUSION[cust], e)}
                    if tuned:
                        orders[BASE_SEARCH] = np.argsort(-self.sets[k].extra["hyb_base"][i], kind="stable").tolist()
                    for m, o in orders.items():
                        rows.append(dict(set=k, customer=cust, qi=i, qid=self.sets[k].Q[i][0], n_gold=len(e["gold"]),
                                         group="|".join(map(str, sorted(e["gold"]))), method=m, **evaluate.metrics(o, e["gold"], self.cfg)))
        R = pd.DataFrame(rows)
        R.to_csv(self.rd / "results.csv", index=False)
        cols = [f"c@{K}" for K in ev["K"]] + [f"R@{N}" for N in ev["N"]]
        lines = [f"run {self.run_name} seed {self.seed} | commit {self.manifest['git_commit']} | config {self.cfg['_config_sha']}"]
        pd.set_option("display.width", 250)
        for k in self.sets:
            X = R[R.set == k]
            lines += ["", "=" * 100, f"{k}: {X.qi.nunique()} test questions, {X.group.nunique()} gold-set groups", "=" * 100,
                      evaluate.table(X, methods, cols).to_string()]
            for b in methods[:-1]:
                cells = [f"{key} {m:+5.1f} [{lo:+.1f}, {hi:+.1f}]" for key in ev["headline"]
                         for m, lo, hi in [evaluate.compare(X, "NIFT-SQL (fusion)", b, key, ev["bootstrap_reps"])]]
                lines.append(f"  NIFT-SQL (fusion) vs {b:26s} " + "   ".join(cells) + "   (95% CI over gold-set groups)")
        R["size"] = pd.cut(R.n_gold, [0, 1, 2, 4, 100], labels=["1", "2", "3-4", "5+"])
        lines += ["", "complete@15 by number of gold tables",
                  R.groupby(["set", "method", "size"], observed=True)["c@15"].mean().mul(100).round(1).unstack().to_string()]
        txt = "\n".join(lines)
        (self.rd / "summary.txt").write_text(txt)
        print(txt)
        print()
        self.costs.show()
        return R


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run one NIFT-SQL enterprise experiment (one person, one seed).")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--name", required=True, help="your run name, e.g. arjun")
    ap.add_argument("--data", required=True, help="folder with beaver/, livesqlbench_table/, livesqlbench_large_table/")
    ap.add_argument("--out", required=True, help="output root; results go to <out>/results/<name>_seed<seed>/")
    a = ap.parse_args(argv)
    r = Run(a.seed, a.name, a.data, a.out)
    r.load(); r.split(); r.search(); r.build_rows(); r.train(); r.train_search(); r.score(); r.fit_fusion(); r.test()


if __name__ == "__main__":
    main()
