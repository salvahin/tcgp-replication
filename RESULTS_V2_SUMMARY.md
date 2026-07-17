# V2 Corrected Results (manuscript numbers)

Harness: complete-function generation + verbatim extraction + generous budgets + full raw logging.
HumanEval: 10 models x 3 conditions x 3 seeds (n=492/cond).
LiveCodeBench: difficulty-stratified n=180 (60 easy/60 medium/60 hard), seed 42. Clean: 0 API errors, 2 no-output / 5400 records.

## HumanEval (saturated) — all models 90-100%; effects negligible (|Cohen h|<0.1 typical).

## LiveCodeBench (hard) — per-difficulty, pooled across 10 models (n=600/cell), Pass@1 [95% CI]
diff     Direct        CoT           TCGP
easy     40 [36,44]    42 [38,46]    60 [56,64]
medium   30 [26,34]    33 [30,37]    45 [41,49]
hard     23 [20,27]    27 [24,31]    35 [31,39]
ALL      31 [29,33]    34 [32,37]    46 [44,49]   <- TCGP vs Direct: non-overlapping CIs, +15pp

## LiveCodeBench per-model (n=180) Direct / CoT / TCGP  (TCGP-Direct)
GPT-5.5           85 / 88 / 87  (+2, near ceiling)
GPT-5.4           68 / 66 / 73  (+4; CoT -2)
GPT-5.3-codex     54 / 61 / 78  (+23)
GPT-5.4-mini      39 / 28 / 67  (+28; CoT HURTS -11)
GPT-5.4-nano      24 / 23 / 20  (-4; TCGP slightly worse)
GPT-4.1           13 / 27 / 42  (+29)
Grok-nonreasoning  9 / 15 / 34  (+25)
Grok-reasoning     8 / 21 / 32  (+24)
Gemini-2.5-Flash   4 / 12 / 29  (+24)
GPT-4o             4 /  2 /  4  (flat, weakest on LCB)

Summary: TCGP best-or-tied 8/10; TCGP>=CoT 8/10; CoT<Direct on 4/10 (inconsistent).
Tokens (HumanEval avg/problem): direct 668, cot 2643, tcgp 1511 -> TCGP ~43% cheaper than CoT.

## Narrative
- TCGP reliably + substantially helps on hard code tasks (~1.5x direct at every difficulty; +29pp max).
- CoT is inconsistent (helps 6, hurts 4; -11pp on GPT-5.4-mini). => prefer TCGP for reliability.
- Matched Grok pair: both variants gain ~equally from TCGP (+24/+25). Reasoning contrast is WEAK
  (CoT helps the reasoning variant slightly MORE here) -- do not over-claim a "reasoning gap".
