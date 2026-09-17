#!/usr/bin/env python3
"""
harness_sensitivity_v3.py -- Two harness decisions quantified on the logged outputs.
  --flaky   : records whose verdict differs between isolated runs are re-graded
              3x; reports the share of unstable verdicts and their error codes.
  --extract : for every record where the FIRST fenced block differs from the
              LAST (the official rule), grade the first-block code too and
              report the pass-rate change per condition under the alternative
              extraction rule.
  --public  : pooled Pass@1 per condition if only PUBLIC tests were used.
Writes results/harness_sensitivity_v3.json (merged).
"""
import sys, json, glob, os, importlib.util
from collections import defaultdict
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from lcb_eval import grade
OUT = HERE / "results" / "harness_sensitivity_v3.json"
def out_for(key): return HERE / "results" / f"harness_{key}_v3.json"

def blocks(t):
    lines = (t or "").split("\n"); idx = [i for i, l in enumerate(lines) if "```" in l]
    if len(idx) < 2: return []
    return ["\n".join(lines[idx[k]+1:idx[k+1]]) for k in range(0, len(idx)-1, 2)]

def load_all():
    rows = []
    for f in glob.glob(str(HERE / "results/livecodebench_v3_strat/*_s42.jsonl")):
        cond = os.path.basename(f).split("_")[0]
        for l in open(f):
            if l.strip(): r = json.loads(l); r["_cond"] = cond; rows.append(r)
    return rows

def main():
    spec = importlib.util.spec_from_file_location("v3", HERE / "run_livecodebench_v3.py"); v3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v3)
    v3._init_heavy(); probs = {str(p["question_id"]): p for p in v3.load_problems()}
    res = json.load(open(OUT)) if OUT.exists() else {}
    rows = load_all()
    if "--public" in sys.argv:
        agg = defaultdict(lambda: [0, 0, 0])
        for r in rows: a = agg[r["_cond"]]; a[0] += 1; a[1] += bool(r.get("public_pass")); a[2] += bool(r.get("passed"))
        res["public_only"] = {c: {"n": v[0], "public_only_pass1": 100*v[1]/v[0], "hidden_pass1": 100*v[2]/v[0]} for c, v in agg.items()}
        print("public-only vs hidden Pass@1:", {c: f"{v['public_only_pass1']:.1f} vs {v['hidden_pass1']:.1f}" for c, v in res["public_only"].items()})
    if "--extract" in sys.argv:
        agg = defaultdict(lambda: [0, 0, 0]); n = 0
        for r in rows:
            b = blocks(r.get("raw_step2") or r.get("raw_repair") or "")
            if len(b) > 1 and b[0].strip() != b[-1].strip():
                g = grade(probs[r["question_id"]], b[0]); n += 1
                a = agg[r["_cond"]]; a[0] += 1; a[1] += bool(r.get("passed")); a[2] += bool(g["passed"])
                if n % 100 == 0: print(f"  extract: {n}", flush=True)
        res["extraction_rule"] = {c: {"n_affected": v[0], "pass_last_block": v[1], "pass_first_block": v[2]} for c, v in agg.items()}
        print("extraction rule (affected records: pass under last-block vs first-block):", {c: f"{v['pass_last_block']}/{v['n_affected']} vs {v['pass_first_block']}/{v['n_affected']}" for c, v in res["extraction_rule"].items()})
    if "--flaky" in sys.argv:
        # candidates: any record with a timeout verdict or a regrade flag -> re-grade 3x
        cands = [r for r in rows if r.get("regrade_timeout_isolated") or "time" in str(r.get("error_message", "")).lower()]
        unstable = 0; codes = defaultdict(int); checked = 0
        for r in cands:
            vs = [grade(probs[r["question_id"]], r.get("extracted_code") or "") for _ in range(3)]
            checked += 1
            if len({v["passed"] for v in vs} | {bool(r.get("passed"))}) > 1:
                unstable += 1
                for v in vs: codes[str(v.get("error_code"))] += 1
            if checked % 100 == 0: print(f"  flaky: {checked}/{len(cands)}", flush=True)
        res["flakiness"] = {"candidates": len(cands), "unstable_records": unstable, "share_of_all_records": unstable/len(rows), "error_codes": dict(codes)}
        print(f"flakiness: {unstable}/{len(cands)} timeout-adjacent records give different verdicts across 4 isolated runs ({100*unstable/len(rows):.2f}% of all records); codes {dict(codes)}")
    for k, v in res.items():
        json.dump(v, open(out_for(k), "w"), indent=1)   # one file per analysis: concurrent runs cannot overwrite each other
    json.dump(res, open(OUT, "w"), indent=1)

if __name__ == "__main__":
    main()
