#!/usr/bin/env python3
"""humaneval_tests.py -- Paired tests on HumanEval (10 models x {direct, cot,
tcgp} x 3 seeds). Two families, both reported:
  (A) per model, seeds pooled: McNemar on discordant (problem, seed) pairs for
      cot-vs-direct, tcgp-vs-direct, tcgp-vs-cot; Holm over the 30 tests;
  (B) per model and seed: the same three contrasts; Holm over the 90 tests.
Also the range of Pass@1 over every (model, condition, seed) cell.
Writes results/humaneval_tests.json."""
import json, glob, os, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
def mcn(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c); return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
def holm(ps):
    idx = sorted(range(len(ps)), key=lambda i: ps[i]); out = [0] * len(ps); run = 0; m = len(ps)
    for rk, i in enumerate(idx): run = max(run, (m - rk) * ps[i]); out[i] = min(1.0, run)
    return out
R = {}
for f in glob.glob(str(HERE / "results/humaneval_v2/*_s*.jsonl")):
    b = os.path.basename(f)[:-6]; c, rest = b.split("_", 1); m, s = rest.rsplit("_s", 1)
    if c in ("direct", "cot", "tcgp"): R[(m, c, s)] = {json.loads(l)["task_id"]: bool(json.loads(l).get("passed")) for l in open(f) if l.strip()}
models = sorted({k[0] for k in R}); seeds = ("42", "123", "2024"); contrasts = (("cot", "direct"), ("tcgp", "direct"), ("tcgp", "cot"))
cells = {f"{m}|{c}|{s}": 100 * sum(v.values()) / len(v) for (m, c, s), v in R.items()}
A = []; B = []
for m in models:
    for a, b in contrasts:
        n01 = n10 = 0
        for s in seeds:
            X, Y = R[(m, a, s)], R[(m, b, s)]; x01 = sum(1 for t in X if X[t] and not Y[t]); x10 = sum(1 for t in X if Y[t] and not X[t])
            B.append({"model": m, "seed": s, "contrast": f"{a}_vs_{b}", "n01": x01, "n10": x10, "p": mcn(x01, x10)}); n01 += x01; n10 += x10
        A.append({"model": m, "contrast": f"{a}_vs_{b}", "n01": n01, "n10": n10, "p": mcn(n01, n10)})
for fam in (A, B):
    for t, h in zip(fam, holm([t["p"] for t in fam])): t["holm_p"] = h
out = {"cell_min": min(cells.values()), "cell_max": max(cells.values()), "cells": cells, "family_A_seeds_pooled": A, "family_B_per_seed": B,
       "significant_A": [t for t in A if t["holm_p"] < 0.05], "significant_B": [t for t in B if t["holm_p"] < 0.05]}
json.dump(out, open(HERE / "results/humaneval_tests.json", "w"), indent=1)
print(f"cells: min {out['cell_min']:.1f}%  max {out['cell_max']:.1f}%")
print("family A (30 tests, seeds pooled) significant:", [(t["model"], t["contrast"], t["n01"], t["n10"], round(t["holm_p"], 4)) for t in out["significant_A"]])
print("family B (90 per-seed tests) significant:", len(out["significant_B"]))
