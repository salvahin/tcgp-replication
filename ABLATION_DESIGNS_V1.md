# Ablation designs salvaged from the V1 supplementary (designs only)

The V1 supplementary described four ablations (E1-E4). The V1 manuscript and
supplementary were retired when the body-extraction harness was found to corrupt
results, and these ablations were run on that flawed harness.

**Do not cite the V1 numbers.** Several of the recorded V1 failure modes are the
extraction artifact itself (IndentationError from re-indentation, violations of
the "function body only" contract that V2 no longer uses). The *designs*, however,
remain good candidates to re-run on the V2 complete-function harness if a future
revision needs them.

## E1. Plan-and-Solve CoT variant
Replace free-form CoT step 1 with Plan-and-Solve prompting (Wang et al., ACL 2023):
first devise a plan, then carry it out. Tests whether a more structured reasoning
format closes the gap to TCGP.

## E2a. Extended Direct (token-budget control)
Direct prompting plus brief "think before coding" instructions and a larger output
budget. Tests the counterfactual that TCGP's gain is just extra tokens rather than
its structure. On V2, compare measured token usage, not caps.

## E2b. Minimal TCGP (scenario-count ablation)
TCGP with only 2 scenarios (one normal, one edge) instead of 3-4. Tests whether the
benefit needs comprehensive scenario coverage or just any concrete example. A V2
version could sweep 1/2/4/6 scenarios.

## E3. Hybrid TCGP+CoT (composition)
Single step-1 prompt asking for both test scenarios and step-by-step reasoning,
then code conditioned on both. Tests whether the two kinds of intermediate
structure compose. V1 observed severe degradation, but the recorded failure modes
overlap with the extraction artifact, so the conclusion needs re-testing on V2.

## E4. Self-consistency CoT (sampling variant)
Sample N=5 CoT completions at temperature 0.7; report Pass@1 and Pass@5
(Chen et al. unbiased estimator). Tests whether sampling diversity rescues CoT.
V1 saw degradation, again confounded by the V1 body-only output contract.

## Where the V1 material lives
- Full V1 supplementary text: git history of the paper repo
  (salvahin/BDD-LLM-Comia, file supplementary.tex, removed after the V2 rewrite).
- V1 raw results: archive/incorrect-harness-v1_2026-07-02.zip in this repo.
- V1 ablation runner: run_humaneval_ablation.py (this repo).
