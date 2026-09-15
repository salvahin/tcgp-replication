# Replication package: How Much of a Prompting Result Is the Harness?

Replication package for the article *"How Much of a Prompting Result Is the
Harness? A Quantified Budget, a Validated Protocol, and a Reporting Card for
Execution-Based Code-Generation Evaluation"* (Avalos et al., submitted to IEEE
Access; revision of manuscript Access-2026-35607).

The study compares twelve prompting conditions (direct prompting, three
paraphrases of it, persona, few-shot, chain-of-thought, Plan-and-Solve,
test-case-guided prompting, a contract-restatement control, a stacked recipe,
and one round of execution-feedback repair) on ten current LLMs, using
HumanEval and a difficulty-stratified 180-problem LiveCodeBench sample graded
with the official LiveCodeBench evaluator. It also quantifies how much each
evaluation-harness decision (input/output contract, extraction rule, hidden
tests, timeouts, output caps, wording) moves the measured pass rate.

## Reproduce every number, table, and figure (no API keys, minutes)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python make_paper_analyses_v3.py      # four core conditions, 10 models
python make_paper_analyses_v4.py      # recipe study, noise floor, equivalence tests
python harness_sensitivity_v3.py --public   # public-vs-hidden tests (add --extract / --flaky to re-grade)
python harness_sensitivity_humaneval.py     # extraction rule, seeds, wording on HumanEval
python audit_tcgp_scenarios_v3.py     # correctness of generated test scenarios
python make_paper_figures_v3.py --outdir figures_paper
```

## Contents

| Path | What it is |
|---|---|
| `lcb_eval/` | The official LiveCodeBench evaluator (`testing_util.py`, commit 28fef95, MIT), vendored unchanged, plus a thin isolated-subprocess wrapper, the official few-shot examples, and the official extraction rule |
| `run_livecodebench_v3.py` | Corrected LiveCodeBench harness: official prompt format block, all public+private tests, per-provider output caps, per-problem resume, full raw and token logging; all twelve conditions |
| `run_repair_v3.py`, `run_sampling_v3.py` | One-round execution-feedback repair; k-sample selection (prepared, not run) |
| `run_livecodebench_v2.py`, `run_livecodebench.py`, `run_tcgp_vs_cot.py`, `prompts/` | The original (stdin-serialization) harness, kept for the artifact analysis and as import dependencies |
| `run_humaneval_v2.py` | HumanEval harness (unchanged; HumanEval grades by direct function call) |
| `validate_lcb_v3.py`, `crosscheck_lcb_v3.py`, `characterize_crosscheck_v3.py` | Harness validation: known solutions, official entry-point cross-check on all generations, serial re-grading of disagreements |
| `regrade_timeouts_v3.py`, `purge_v3_records.py`, `rescore_v2_official.py` | Isolated timeout re-grade; API-error purge/resume; re-scoring of the original outputs with the official evaluator |
| `make_paper_analyses_v3.py`, `make_paper_analyses_v4.py`, `harness_sensitivity_v3.py`, `audit_tcgp_scenarios_v3.py`, `make_paper_figures_v3.py` | Every statistic, table, and figure in the paper |
| `results/livecodebench_v3_strat/` | 20,160 raw records: 10 models x 4 core conditions + 9 models x 8 further conditions x 180 problems, with raw responses, extracted code, per-step token usage, verdicts, public-only verdicts |
| `results/livecodebench_v3_cap4k/` | Small-output-cap runs for the truncation row of the harness budget |
| `results/livecodebench_v2_strat/`, `results/livecodebench_v2_strat_rescored/` | The original submission's outputs and their re-grading with the official evaluator (the artifact analysis) |
| `results/humaneval_v2/` | HumanEval: 10 models x 3 conditions x 3 seeds, plus three paraphrases of the direct prompt on 9 models (seed 42), with `manifest.json` (deployments, endpoints, parameters, dates) |
| `results/paper_analyses_v3.json`, `paper_analyses_v4.json`, `harness_sensitivity_v3.json`, `scenario_audit_v3.json` | Analysis outputs |
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
article is v2.0, archived at https://doi.org/10.5281/zenodo.22714011.

## License

MIT (see `LICENSE`).
