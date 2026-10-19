#!/usr/bin/env python3
"""contract_mechanism_v2.py -- Mechanism check for the standard-input contract
artifact (Section V-B). Under the stdin harness (results/livecodebench_v2_strat),
for every (model, problem) where TCGP passed and CoT failed, count whether the
TCGP program contains MORE input-parsing calls than the CoT program. Rule: the
number of occurrences of the tokens below in the extracted program.
Nine-model cohort (Gemini excluded). Writes results/contract_mechanism_v2.json."""
import json, glob, os, re
from pathlib import Path
HERE = Path(__file__).resolve().parent; D = HERE / "results" / "livecodebench_v2_strat"
TOKENS = ["input(", "sys.stdin", "readline", ".strip(", ".split(", "json.loads", "literal_eval", "eval(", "int(", "map("]
def count(code): return sum(len(re.findall(re.escape(t), code or "")) for t in TOKENS)
def load(c, m): return {json.loads(l)["question_id"]: json.loads(l) for l in open(D / f"{c}_{m}_s42.jsonl") if l.strip()}
models = sorted({os.path.basename(f)[len("tcgp_"):-len("_s42.jsonl")] for f in glob.glob(str(D / "tcgp_*_s42.jsonl"))} - {"gemini-2.5-flash"})
pairs = more = equal = 0; per = {}
for m in models:
    T, C = load("tcgp", m), load("cot", m); a = b = e = 0
    for q in T:
        if T[q].get("passed") and q in C and not C[q].get("passed"):
            a += 1; ct, cc = count(T[q].get("extracted_code") or T[q].get("generated_code")), count(C[q].get("extracted_code") or C[q].get("generated_code"))
            b += ct > cc; e += ct == cc
    per[m] = {"tcgp_pass_cot_fail": a, "tcgp_more_parsing": b, "equal": e}; pairs += a; more += b; equal += e
out = {"rule": f"count of {TOKENS} in the extracted program; 'more' means strictly greater in the TCGP program", "models": models,
       "pairs": pairs, "tcgp_more_parsing": more, "equal": equal, "share_more": more / pairs if pairs else None, "per_model": per}
json.dump(out, open(HERE / "results" / "contract_mechanism_v2.json", "w"), indent=1)
print(f"pairs={pairs} tcgp has more parsing code in {more} ({100*more/pairs:.0f}%), equal in {equal}")
