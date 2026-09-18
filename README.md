# Replication package: From Prompting Recipes to Harnesses

Replication package for the article *"From Prompting Recipes to Harnesses: What
Still Matters in LLM Code Generation, and Where It Breaks"* (Avalos et al., submitted to IEEE
Access; revision of manuscript Access-2026-35607).

The study compares twelve prompting conditions (direct prompting, three
paraphrases of it, persona, few-shot, chain-of-thought, Plan-and-Solve,
test-case-guided prompting, a contract-restatement control, a stacked recipe,
and one round of execution-feedback repair) on ten current LLMs, using
HumanEval and a difficulty-stratified 180-problem LiveCodeBench sample graded
with the official LiveCodeBench evaluator. It also quantifies how much each
evaluation-harness decision (input/output contract, extraction rule, hidden
tests, timeouts, output caps, wording) moves the measured pass rate.

## Reproduce the numbers, tables, and figures (no API keys)

The statistics computed from the stored verdicts take minutes and need no
network access:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python make_paper_analyses_v3.py      # four core conditions, 10 models (Table: per-model)
python make_paper_analyses_v4.py      # recipe study, noise floor, equivalence tests (Table: recipes)
python harness_sensitivity_v3.py --public   # public-vs-hidden inflation from stored verdicts
python harness_sensitivity_humaneval.py     # extraction rule, seeds, wording on HumanEval
python humaneval_tests.py             # HumanEval range and McNemar/Holm contrasts quoted in the text
python analyze_repair_rounds.py       # feedback-loop curve (public vs hidden pass by round)
python audit_tcgp_scenarios_v3.py     # correctness of generated test scenarios (optional)
python make_harness_budget.py         # harness budget table/figure inputs -> results/harness_budget.json
python make_paper_figures_v3.py --outdir figures_paper
python build_manifest_v3.py           # per-file manifest: parameters, caps, dates, hashes
```

Three rows of the harness budget re-execute generated programs and take
longer (an hour or more on a laptop; they need the LiveCodeBench sample,
cached on first run, and the extraction-rule and flakiness rows need the
programs to be graded again):

```bash
python harness_sensitivity_v3.py --extract   # first-block vs last-block extraction rule
python harness_sensitivity_v3.py --flaky     # verdict stability under repeated isolated grading
git clone https://github.com/LiveCodeBench/LiveCodeBench && (cd LiveCodeBench && git checkout 28fef95)
python characterize_crosscheck_v3.py LiveCodeBench   # official entry point vs our verdicts, checkpointed per file
```

`make_harness_budget.py` reads whichever of these outputs exist and reports
which rows it could fill. Two numbers in the paper are not produced by these
scripts: the leaderboard pass rates quoted for validation (read from the public
LiveCodeBench leaderboard on the dates in the manifest) and the token counts,
which are the providers' usage fields stored in each record.

## Contents

| Path | What it is |
|---|---|
| `lcb_eval/` | The official LiveCodeBench evaluator (`testing_util.py`, commit 28fef95, MIT), vendored unchanged, plus a thin isolated-subprocess wrapper, the official few-shot examples, and the official extraction rule |
| `run_livecodebench_v3.py` | Corrected LiveCodeBench harness: official prompt format block, all public+private tests, per-provider output caps, per-problem resume, full raw and token logging; all twelve conditions |
| `run_repair_v3.py`, `run_repair_rounds_v3.py`, `run_sampling_v3.py` | One-round execution-feedback repair; the multi-round loop (public-test feedback, hidden-test judgment); k-sample selection (prepared, not run) |
| `run_livecodebench_v2.py`, `run_livecodebench.py`, `run_tcgp_vs_cot.py`, `prompts/` | The original (stdin-serialization) harness, kept for the artifact analysis and as import dependencies |
| `run_humaneval_v2.py` | HumanEval harness (unchanged; HumanEval grades by direct function call) |
| `validate_lcb_v3.py`, `crosscheck_lcb_v3.py`, `characterize_crosscheck_v3.py` | Harness validation: known solutions, official entry-point cross-check on all generations, serial re-grading of disagreements |
| `regrade_timeouts_v3.py`, `purge_v3_records.py`, `rescore_v2_official.py`, `backfill_full_code_v3.py` | Isolated timeout re-grade; API-error purge/resume; re-scoring of the original outputs with the official evaluator; one-off backfill of full extracted programs and hashes into records written before the runners stored them |
| `make_paper_analyses_v3.py`, `make_paper_analyses_v4.py`, `harness_sensitivity_v3.py`, `harness_sensitivity_humaneval.py`, `humaneval_tests.py`, `analyze_repair_rounds.py`, `audit_tcgp_scenarios_v3.py`, `make_harness_budget.py`, `make_paper_figures_v3.py`, `build_manifest_v3.py` | The statistics, tables, and figures in the paper, and the run manifest |
| `results/livecodebench_v3_strat/` | 20,160 raw records: 10 models x 4 core conditions + 9 models x 8 further conditions x 180 problems, with raw responses, the full extracted program and its SHA-256, per-step token usage, verdicts, public-only verdicts; plus 9 `repairR_*` files with the multi-round loop (round 0 and up to three repair rounds per problem) |
| `results/livecodebench_v3_cap4k/` | Small-output-cap runs for the truncation row of the harness budget |
| `results/livecodebench_v2_strat/`, `results/livecodebench_v2_strat_rescored/` | The original submission's outputs and their re-grading with the official evaluator (the artifact analysis) |
| `results/humaneval_v2/` | HumanEval: 10 models x 3 conditions x 3 seeds, plus three paraphrases of the direct prompt on 9 models (seed 42), with `manifest.json` (deployments, endpoints, parameters, dates) |
| `results/paper_analyses_v3.json`, `paper_analyses_v4.json`, `harness_*_v3.json`, `harness_budget.json`, `humaneval_tests.json`, `repair_rounds.json`, `scenario_audit_v3.json`, `manifest_v3.json` | Analysis outputs and the manifest |
| `data/humaneval/humaneval.jsonl` | HumanEval problems; the LiveCodeBench sample is cached on first run from `bzantium/livecodebench` |
| `archive/incorrect-harness-v1_*.zip`, `ABLATION_DESIGNS_V1.md` | Provenance of an even earlier harness (body extraction), not used in the paper |
| `EXPERIMENT_DESIGN_V2.md`, `RESULTS_V2_SUMMARY.md` | Design notes and the (superseded) first-submission numbers, kept for the record |

## Re-run the experiments (API keys and budget)

```bash
cp .env.example .env    # add your keys
python run_livecodebench_v3.py --models gpt-4o:azure --conditions direct cot tcgp restate --tag strat
python run_livecodebench_v3.py --models gpt-4o:azure --conditions para1 para2 para3 persona fewshot stacked plansolve --tag strat
python run_repair_v3.py --models gpt-4o:azure --tag strat
```

Runs resume per problem. Provider-side model updates mean exact pass rates
are not guaranteed to reproduce over time; deployments and query dates are in
the manifest. Version 1 of this package (the original submission) is archived
at https://doi.org/10.5281/zenodo.21421552; the version matching the revised
article is v2.3, archived at https://doi.org/10.5281/zenodo.22837940.

## License

MIT (see `LICENSE`).
