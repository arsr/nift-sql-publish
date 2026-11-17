# Experiments

Scripts that read finished runs (made by `main.py`) and produce the paper's baselines, analyses and appendix studies.
`D` = the data folder, `O` = the output root, as for `main.py`. `reproduce_paper.py` at the repository root runs the
scripts marked **paper** in order.

| Script | Paper | What it does | GPU | Output in this repo |
|---|---|---|---|---|
| `paper_tables.py` | **paper**, Tables 1 to 6, A1, F1 | Every paper table in one file, from finished runs and caches. Also data and protocol checks (T0) and the uncached search cost (T6b). | optional (only T6b) | `results/paper/PAPER_TABLES.txt` |
| `reranker_baseline.py` | **paper**, Tables 2, 3, 5 | Hybrid search top 100 reordered by zero-shot Qwen3-Reranker-8B, on the main runs' test questions. | yes, about 20 GB | in `PAPER_TABLES.txt` |
| `shortlist_coverage.py` | **paper**, Table 4 | How often a shortlist (search, Laya, NIFT-SQL candidates) holds every gold table. Same numbers as T5 in `paper_tables.py`. | no | in `PAPER_TABLES.txt` |
| `supervised_crossencoder.py` | revision, headline scorer | bge-reranker-large fine-tuned on exactly Laya's training rows, scoring every table, alone and inside the same fusion; test, seeds 1 to 4, compared with `nift_base`. Variants: `--zero-shot` (run `nift_ce0`), `--no-joins` (`nift_ce_nojoin`), `--config train10/20/30` (`nift_ce_train10/20/30`). | yes, about 16 GB | `SUPERVISED_CROSSENCODER_<model>.txt` in the output folder |
| `llm_selector.py` | revision, LLM baseline | Gemma 4 31B (zero-shot) picks the needed tables from the search top 50 (`--shortlist search`) or from the top 50 of NIFT-SQL with the fine-tuned cross-encoder (`--shortlist nift_ce`); test, compared on the same questions with `nift_base` and `nift_ce`. Answers cached, resumable. | yes, about 62 GB bf16, or `--load 4bit` on 40 GB | `LLM_SELECTOR_<model>_<shortlist>_top<k>_seeds<N>.txt` in the output folder |
| `ce_analysis.py` | revision, Tables 3 to 6 with the cross-encoder | CPU only, from cached scores: reranker-row intervals, pooled tests, cross-encoder + rank average, zero-shot vs fine-tuned cross-encoder (same model), no-joins ablation, shortlist coverage, breakdown by number of tables, tables that share no word with the question, training-size curve, unique labelled-question counts, GPU, database-given evaluation and routing hit rate, fusion without the two database features, cross-encoder input length vs 512 tokens. Pooled intervals cluster gold-table sets linked across the two LiveSQLBench sets by question text. Checks every rebuilt ranking against the run's own results.csv. | no | `CE_ANALYSIS.txt` in the output folder |
| `additional_analyses.py` | Sections 6.5 to 7 and Appendix G | CPU only, from cached scores: training-free database grouping for every baseline, database-given intervals, the reranker over the whole known database, fusion fitted on 10/20/40 validation questions, cross-encoder vs Laya at each training size, retraining replicates and the deterministic-mode check, inputs cut by the card and join caps. | no | `ADDITIONAL_ANALYSES.txt` |
| `fusion_extensions.py` | Section 6.8 | Two extensions fixed before test scoring: (A) NIFT-SQL with the fine-tuned retriever in place of base search, (B) NIFT-SQL plus two Qwen3-Reranker-8B features (high-accuracy mode), (C) both; every cross-encoder training run paired with fine-tuned-search run R, two-level intervals. Reranker scores computed fresh for exactly each question's candidates (the earlier caches' coverage depended on the gold database). Validation check, then test once. | yes, about 20 GB | `FUSION_EXTENSIONS.txt`, `fusion_ext/` |
| `final_breakdowns.py` | Tables 7 and 8, note to Table G5 | From the caches of `fusion_extensions.py` only: rebuilds its rankings, checks every test row against `fusion_ext/test_rows.csv` (stops on any difference), then (1) checks the reranker routed by its own ranking against the plain reranker question by question, (2) complete@15 by number of gold tables and (3) gold tables sharing no word with the question, both against the validation-chosen reranker, the train + validation retriever, NIFT-SQL-base, NIFT-SQL and the high-accuracy mode. | no | `FINAL_BREAKDOWNS.txt` |
| `run_additional_analyses.py` | runs all of the above in order, resumable | input hashes, deterministic BEAVER runs (`supervised_crossencoder.py --deterministic --score-only`), two retrained replicates (`--replicate R`), the reranker over the known database (`reranker_baseline.py --known-db`), the fine-tuned-BGE baseline (`finetuned_search_baseline.py`), then `additional_analyses.py`. About 2 GPU hours on an A100 from scratch. | yes | logs in `results/logs/additional_*` |
| `join_check.py` | Appendix A | Seed-0 validation decision for `configs/nift_ftj.yaml` against `nift_ft`. | no | `results/exploratory/A_join_completion_seed0_validation.txt` |
| `laya_budget.py` | Appendix B | Laya scoring only the search top K tables, simulated exactly from cached scores. Validation. | no | `results/exploratory/B_laya_budget_validation.txt` |
| `decider_zeroshot.py` | Appendix C | A zero-shot decision model as the scorer (`--scorer pplx`; the `darwin` option exists but was never run). Validation. | yes | `results/exploratory/C_pplx_zero_shot_validation.txt` |
| `pplx_finetune.py` | Appendix C | LoRA fine-tune of pplx-decider-flash on Laya's training rows, then the same comparison. Validation. | yes, about 24 GB | `results/exploratory/C_pplx_finetuned_validation.txt` |
| `fuser_compare.py` | Appendix D | Scorer quality (AUC, average precision, calibration) and fusers: logistic regression, MLP, gradient-boosted trees, LambdaRank. Validation. | no | `results/exploratory/D_fusers_validation.txt` |
| `enterprise_v2.py` | Appendix D | 15 extra fusion features, spline fusion, and pplx v2 (negatives from the fine-tuned search). Validation. | yes | `results/exploratory/D_features_and_pplx_v2_validation.txt` |

Appendix B to D studies are exploratory: seed-0 validation only (B also seeds 1 and 2), run on the
`nift_ftj` variant (runs named `arjun_v3j`), with settings fixed before each run. They are not test results.

## Commands as run

```
python experiments/reranker_baseline.py --run nift_base --seeds 1 2 3 4 --model Qwen/Qwen3-Reranker-8B --data D --out O
python experiments/shortlist_coverage.py --run nift_base --seeds 1 2 3 4 --data D --out O
python experiments/paper_tables.py --data D --out O                       # appendix runs named arjun_v3j by default
python experiments/join_check.py --cand arjun_v3j --base arjun_v2 --data D --out O
python experiments/laya_budget.py --runs arjun_v3j:0 arjun_v3j:1 arjun_v3j:2 --config nift_ftj --data D --out O
python experiments/decider_zeroshot.py --scorer pplx --run arjun_v3j --seed 0 --config nift_ftj --data D --out O
python experiments/pplx_finetune.py --run arjun_v3j --seed 0 --config nift_ftj --data D --out O
python experiments/fuser_compare.py --run arjun_v3j --seed 0 --config nift_ftj --data D --out O
python experiments/enterprise_v2.py --run arjun_v3j --seed 0 --config nift_ftj --data D --out O
python experiments/fusion_extensions.py --data D --out O                  # Section 6.8, after run_additional_analyses.py
python experiments/final_breakdowns.py --data D --out O                   # Tables 7, 8 and the routing check; CPU, after fusion_extensions.py
```

The pplx scripts need a Hugging Face token with access to `perplexity-ai/pplx-decider-flash-v1.2` in `HF_TOKEN`.
They run third-party code shipped in that model repository.
