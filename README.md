# Replication package: Structure Over Reasoning — Test-Case-Guided Prompting for Hard LLM Code Generation

This repository contains the full replication package for the paper
*"Structure Over Reasoning: Test-Case-Guided Prompting for Hard LLM Code
Generation"* (Avalos et al.).

The study compares three prompting strategies — direct prompting,
chain-of-thought (CoT), and test-case-guided prompting (TCGP) — on ten
current LLMs, using HumanEval (easy, saturated) and a difficulty-stratified
180-problem LiveCodeBench sample (hard).

## Contents

| Path | What it is |
|---|---|
| `run_humaneval_v2.py` | HumanEval harness (complete-function protocol, per-problem resume, full raw + token logging) |
| `run_livecodebench_v2.py` | LiveCodeBench harness (same protocol, difficulty-stratified sampling) |
| `run_tcgp_vs_cot.py`, `run_livecodebench.py`, `prompts/` | Legacy modules the V2 harnesses import (test executors); kept for that reason |
| `make_paper_figures.py` | Regenerates every figure in the paper from the logged results |
| `make_paper_analyses.py` | Regenerates every statistical analysis (McNemar + Holm, partial correctness, reasoning tokens, sample robustness, seed variance) |
| `results/humaneval_v2/` | Raw per-problem records: 10 models x 3 conditions x 3 seeds x 164 problems, with full model responses and per-step token usage, plus `manifest.json` (deployments, endpoints, parameters, query dates) |
| `results/livecodebench_v2/` | Raw records for the 50-problem pilot sample (seed 42) |
| `results/livecodebench_v2_strat/` | Raw records for the stratified 180-problem sample (60 easy / 60 medium / 60 hard) |
| `results/paper_analyses.json` | Output of `make_paper_analyses.py` |
| `data/humaneval/humaneval.jsonl` | HumanEval problems (input to the harness) |
| `archive/incorrect-harness-v1_*.zip` | Results from a retired earlier harness, kept for provenance only (see `ABLATION_DESIGNS_V1.md`) |
| `EXPERIMENT_DESIGN_V2.md` | The experiment design document |
| `RESULTS_V2_SUMMARY.md` | Headline numbers used in the manuscript |
| `ABLATION_DESIGNS_V1.md` | Ablation designs from an earlier iteration, candidates for future re-runs |

## Reproduce the paper's figures and analyses (no API keys needed, minutes)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python make_paper_analyses.py     # statistics -> results/paper_analyses.json + console report
python make_paper_figures.py      # figures    -> figures_paper/*.pdf
```

Every number and figure in the paper derives from the raw records in
`results/` through these two scripts.

## Re-run the experiments (API keys + budget, hours)

```bash
cp .env.example .env   # add your keys; see the provider notes inside

# HumanEval, one model, one seed
python run_humaneval_v2.py --models gpt-4o:azure --conditions direct cot tcgp --seed 42

# LiveCodeBench, stratified sample
python run_livecodebench_v2.py --models gpt-4o:azure --conditions direct cot tcgp \
    --seed 42 --per-difficulty 60 --tag strat
```

Both harnesses resume per problem: re-running the same command skips
completed records. Model deployments and providers used in the paper are
listed in `results/humaneval_v2/manifest.json`. Note that provider-side
model updates and deployment retirements mean exact pass rates are not
guaranteed to reproduce over time.

## License

MIT (see `LICENSE`).
