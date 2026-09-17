#!/usr/bin/env python3
"""make_harness_budget.py -- Every number of the harness-budget table and figure,
derived from retained artifacts into results/harness_budget.json. Inputs: raw
records (v2 standard-input runs, their official re-grading, v3 runs, cap runs),
the per-analysis JSONs (extraction rule, public-only, flakiness, cross-check,
HumanEval rows, recipe study) and the archived body-extraction runs."""
import json, glob, os, random, zipfile, io
from pathlib import Path
HERE = Path(__file__).resolve().parent; R = HERE / "results"
def J(name):
    p = R / name; return json.load(open(p)) if p.exists() else None
def rate_file(f, key="passed"):
    rows = [json.loads(l) for l in open(f) if l.strip()]; return 100 * sum(bool(r.get(key)) for r in rows) / len(rows), len(rows)
def load(d):
    out = {}
    for f in glob.glob(str(R / d / "*_s42.jsonl")):
        b = os.path.basename(f)[:-len("_s42.jsonl")]; c, m = b.split("_", 1)
        if c in ("direct", "cot", "tcgp"): out[(m, c)] = {json.loads(l)["question_id"]: bool(json.loads(l).get("passed")) for l in open(f) if l.strip()}
    return out
def pooled(Rr, c):
    ms = sorted({m for m, _ in Rr}); n = sum(len(Rr[(m, c)]) for m in ms); return 100 * sum(sum(Rr[(m, c)].values()) for m in ms) / n
def paired(Rr, a="tcgp", b="direct"):
    ms = sorted({m for m, _ in Rr}); q = sorted(set.intersection(*[set(Rr[(m, c)]) for m in ms for c in (a, b)]))
    d = [sum(Rr[(m, a)][x] - Rr[(m, b)][x] for m in ms) for x in q]; obs = sum(d); n = len(ms) * len(q)
    rng = random.Random(0); ge = sum(1 for _ in range(20000) if abs(sum(v if rng.random() < .5 else -v for v in d)) >= abs(obs))
    rb = random.Random(1); bs = sorted(100 * sum(d[rb.randrange(len(d))] for _ in d) / n for _ in range(10000))
    return {"diff_pp": 100 * obs / n, "perm_p": (ge + 1) / 20001, "boot90": [bs[500], bs[9499]], "models": len(ms), "problems": len(q)}
out = {}
# 1. input/output contract
v2, resc, v3 = load("livecodebench_v2_strat"), load("livecodebench_v2_strat_rescored"), load("livecodebench_v3_strat")
out["contract"] = {k: {c: pooled(Rr, c) for c in ("direct", "cot", "tcgp")} | {"tcgp_vs_direct": paired(Rr)} for k, Rr in (("stdin_contract", v2), ("same_outputs_official", resc), ("official_fresh", v3))}
out["contract"]["deflation_direct_pp"] = out["contract"]["stdin_contract"]["direct"] - out["contract"]["official_fresh"]["direct"]
# 2. body extraction (archived runs vs complete-function runs, HumanEval, seed 42)
be = {}
z = HERE / "archive" / "incorrect-harness-v1_2026-07-02.zip"
if z.exists():
    with zipfile.ZipFile(z) as Z:
        for m in ("gemini-2.5-flash", "gpt-4.1", "gpt-4o"):
            name = f"results/bdd_vs_cot/direct_{m}_s42.jsonl"
            if name in Z.namelist():
                rows = [json.loads(l) for l in io.TextIOWrapper(Z.open(name)) if l.strip()]
                ok = [r for r in rows if "API" not in str(r.get("error") or "") and "Error code" not in str(r.get("error") or "")]
                if len(ok) == 164:
                    old = 100 * sum(bool(r.get("passed") or r.get("success")) for r in rows) / 164; new, _ = rate_file(R / "humaneval_v2" / f"direct_{m}_s42.jsonl")
                    be[m] = {"body_extraction": old, "complete_function": new, "diff_pp": old - new, "indentation_errors": sum(1 for r in rows if "IndentationError" in str(r.get("error") or ""))}
out["body_extraction_humaneval"] = be
# 3-5. extraction rule, public-only, flakiness, cross-check
ex = J("harness_extraction_rule_v3.json"); tot = {}
for f in glob.glob(str(R / "livecodebench_v3_strat" / "*_s42.jsonl")):
    c = os.path.basename(f).split("_")[0]; tot[c] = tot.get(c, 0) + sum(1 for l in open(f) if l.strip())
if ex: out["extraction_rule"] = {c: v | {"n_condition": tot.get(c), "delta_first_block_pp": 100 * (v["pass_first_block"] - v["pass_last_block"]) / tot[c]} for c, v in ex.items()}
po = J("harness_public_only_v3.json")
if po: out["public_only_inflation_pp"] = {c: v["public_only_pass1"] - v["hidden_pass1"] for c, v in po.items()}
out["flakiness"] = J("harness_flakiness_v3.json"); out["crosscheck"] = J("harness_crosscheck_v3.json")
# 6. output caps (retained runs only)
caps = {}
for f in glob.glob(str(R / "livecodebench_v3_cap4k" / "direct_*_s42.jsonl")):
    m = os.path.basename(f)[len("direct_"):-len("_s42.jsonl")]; r4, n = rate_file(f); r12, _ = rate_file(R / "livecodebench_v3_strat" / f"direct_{m}_s42.jsonl")
    rows = [json.loads(l) for l in open(f) if l.strip()]
    caps[m] = {"cap_4k": r4, "cap_12k": r12, "diff_pp": r4 - r12, "flagged_truncated": sum(1 for r in rows if (r.get("usage_step2") or {}).get("truncated")), "no_output": sum(1 for r in rows if r.get("no_output"))}
out["output_cap"] = caps
# 7. wording, recipes, HumanEval rows
v4 = J("paper_analyses_v4.json")
if v4:
    out["wording_lcb"] = {"pooled_range_pp": v4["noise_floor"]["pooled_range"], "per_model_range_pp": v4["noise_floor"]["per_model_range"]}
    rec = {c: v["diff_pp"] for c, v in v4["vs_direct"].items() if c in ("persona", "fewshot", "plansolve", "tcgp", "stacked", "cot")}
    out["largest_recipe_effect"] = {"condition": max(rec, key=lambda c: abs(rec[c])), "diff_pp": max(rec.values(), key=abs)}
out["humaneval_rows"] = J("harness_sensitivity_humaneval.json")
json.dump(out, open(R / "harness_budget.json", "w"), indent=1)
c = out["contract"]; print(f"contract: stdin {c['stdin_contract']['direct']:.1f}/{c['stdin_contract']['tcgp']:.1f} -> official fresh {c['official_fresh']['direct']:.1f}/{c['official_fresh']['tcgp']:.1f}; deflation {c['deflation_direct_pp']:.1f}; strategy effect {c['stdin_contract']['tcgp_vs_direct']['diff_pp']:+.1f} -> {c['same_outputs_official']['tcgp_vs_direct']['diff_pp']:+.1f}")
print("body extraction:", {m: round(v["diff_pp"], 1) for m, v in be.items()})
print("caps:", {m: round(v["diff_pp"], 2) for m, v in caps.items()}); print("extraction rule present:", bool(ex), "| crosscheck present:", bool(out["crosscheck"]))
