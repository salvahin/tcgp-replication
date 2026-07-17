# Experiment Design V2 — Corrected Harness, Reasoning-Contrast Study

**Date:** 2026-07-02 · **Trigger:** Peer-review W1 (harness artifact confirmed; see
`phaseA_harness_artifact_findings_2026-07-02.md`) · **Goal:** rebuild the empirical core on a
validated harness, with raw outputs stored for full reproducibility, framed around the
reasoning-vs-non-reasoning contrast (answers reviewer W5).

---

## 1. Research questions (reframed)

- **RQ1.** Under a corrected, format-robust harness, does prompted Chain-of-Thought help,
  hurt, or not affect code generation relative to Direct prompting?
- **RQ2.** Does test-case-guided prompting (TCGP) provide more reliable gains than free-form
  CoT — and at what token/cost overhead?
- **RQ3 (new, W5).** Does the answer differ between *reasoning-trained* models and their
  *non-reasoning* counterparts? Centerpiece: the matched pair
  `grok-4-20-reasoning` ↔ `grok-4-20-non-reasoning` (same base model, reasoning toggled),
  plus the GPT-5.x reasoning line vs non-reasoning anchors (gpt-4o, gpt-4.1).

## 2. Models (10) and routing

| Model | Provider path | Endpoint | Class |
|---|---|---|---|
| gpt-4o | `azure` (classic) | models4325218527.openai.azure.com | non-reasoning anchor |
| gpt-4.1 | `azure` (classic) | same | non-reasoning anchor |
| gemini-2.5-flash | `gemini` | Google AI | thinking (hybrid) |
| grok-4-20-reasoning | `foundry` | models4325218527.services.ai.azure.com/openai/v1 | reasoning (matched pair) |
| grok-4-20-non-reasoning | `foundry` | same | non-reasoning (matched pair) |
| gpt-5.4 | `foundry` | same | reasoning |
| gpt-5.4-mini-2 | `foundry` | same | reasoning (mid) |
| gpt-5.4-nano | `foundry` | same | reasoning (small) |
| gpt-5.5 | `foundry` | same | reasoning (frontier) |
| gpt-5.3-codex | `foundry` | same | code-specialized |

Auth: single `AZURE_OPENAI_KEY` works for both Azure endpoints (verified live 2026-07-02);
`GEMINI_API_KEY` for Gemini. Anthropic excluded (subscription does not allow deployment).

## 3. Conditions (same three, corrected implementation)

- **Direct:** one step. Model asked for the **complete function** in a fenced block.
- **CoT:** step 1 free-form reasoning; step 2 complete function conditioned on the reasoning.
- **TCGP:** step 1 generate 3–4 concrete input/output scenarios; step 2 complete function
  that must satisfy them.

### Harness corrections (vs V1, which produced the artifact)
1. **Complete-function generation** — no body-only requests, no heuristic re-indentation.
   Fenced code is extracted verbatim; execution = prompt preamble (imports/helpers) + model
   function + official tests. Validated: Gemini-2.5-Flash 16.5% (V1) → 15/15 (V2 prototype).
2. **Generous completion budgets** so hidden thinking cannot truncate code:
   step 1 cap 4,000; step 2 / Direct cap 8,000 completion tokens. Caps are upper bounds;
   actual usage is recorded per call. The V1 equal-budget-cap design is *replaced* by
   equal-generous-caps + measured-usage accounting (reasoning models make fixed small caps
   incomparable). Token/cost comparisons use measured usage.
3. **Per-model parameter adaptation, recorded in a manifest**: `max_completion_tokens` vs
   `max_tokens`, temperature 0.2 where accepted (recorded fallback to provider default where
   rejected), seed forwarded where supported. Manifest stores exact deployment name, endpoint
   host, params, and query dates (answers W7's model-provenance request).

## 4. Benchmarks and scale

- **HumanEval (164, primary):** 10 models × 3 conditions × 3 seeds (42, 123, 2024).
  Run order: seed 42 across all models first (full single-seed picture), then 123, 2024.
- **LiveCodeBench (n=50 subset, secondary, hard tasks):** foundry models × 3 conditions
  × 1 seed, using the existing (already full-program) LCB harness with the foundry provider
  added. Existing gpt-4o/gemini LCB data retained for comparison; NO_OUTPUT decomposition
  re-checked under generous budgets.

## 5. Storage & reproducibility (raw answers)

Every generation stores the **full, untruncated raw model response** for both steps:

```
results/humaneval_v2/{condition}_{model}_s{seed}.jsonl   # per-problem records
results/humaneval_v2/manifest.json                        # model/endpoint/params/dates
```

Record schema: `task_id, model, condition, seed, raw_step1, raw_step2, extracted_code,
passed, error, tokens_step1, tokens_step2, params_used, duration_s, timestamp`.

Stable filenames + per-problem resume (a problem is done only when its record is written);
runs are idempotent and interruption-safe. Nothing from V1 is overwritten — V1 results stay
under `results/bdd_vs_cot/` for provenance and for documenting the artifact.

## 6. Statistics (answers W2)

- Per-condition Pass@1 with Wilson 95% CIs; per-model TCGP−CoT and CoT−Direct margins with
  seed-level mean ± CI; a winner is claimed **only when the margin CI excludes zero**.
- Effect sizes as **Cohen's h** (proportions), replacing d.
- Multiplicative framings ("N×") replaced by absolute pp differences with CIs.

## 7. What this supports in the rewritten paper

- Corrected multi-model comparison (Table 2 replacement) with valid measurements.
- Reasoning-contrast finding (matched grok pair + GPT-5.x line) — new centerpiece, W5.
- Token/cost accounting from measured usage (kept from V1 in spirit, now valid).
- LCB difficulty dimension + budget-sweep control (already run for gpt-4o).
- V1 artifact documented as a Threats/lessons item, with the Phase-A findings report.
