#!/usr/bin/env python3
"""
harness_sensitivity_humaneval.py -- Harness-decision rows measured on HumanEval,
from the logged v2 records (10 models x {direct, cot, tcgp} x 3 seeds, plus
paraphrases when present):
  * extraction rule: the harness takes the LONGEST fenced block; for records
    with more than one block we also grade the FIRST and the LAST block;
  * cross-seed variation: range of Pass@1 across the three seeds per model and
    condition (a run-to-run noise row);
  * wording: spread across direct and its three paraphrases (seed 42), if run.
Writes results/harness_sensitivity_humaneval.json.
"""
import sys, json, glob, os, re, importlib.util
from collections import defaultdict
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("hv2", HERE / "run_humaneval_v2.py"); hv2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(hv2)

def blocks(raw):
    return [b.strip("\n") for b in re.findall(r"```(?:python)?[ \t]*\n(.*?)```", raw or "", re.DOTALL)]

def main():
    probs = {p["task_id"]: p for p in (json.loads(l) for l in open(HERE / "data/humaneval/humaneval.jsonl") if l.strip())}
    out = {}
    # ---- extraction rule ----
    agg = defaultdict(lambda: [0, 0, 0, 0]); n = 0
    for f in glob.glob(str(HERE / "results/humaneval_v2/*_s42.jsonl")):
        cond = os.path.basename(f).split("_")[0]
        for l in open(f):
            if not l.strip(): continue
            r = json.loads(l); b = blocks(r.get("raw_step2"))
            if len(b) < 2 or (b[0] == b[-1]): continue
            p = probs[r["task_id"]]
            first_ok, _ = hv2.execute_test(hv2.assemble(p, b[0]), p["test"], p["entry_point"])
            last_ok, _ = hv2.execute_test(hv2.assemble(p, b[-1]), p["test"], p["entry_point"])
            a = agg[cond]; a[0] += 1; a[1] += bool(r.get("passed")); a[2] += bool(first_ok); a[3] += bool(last_ok); n += 1
    tot_by_cond = defaultdict(int)
    for f in glob.glob(str(HERE / "results/humaneval_v2/*_s42.jsonl")):
        tot_by_cond[os.path.basename(f).split("_")[0]] += sum(1 for l in open(f) if l.strip())
    out["extraction_rule"] = {c: {"n_affected": v[0], "n_total": tot_by_cond[c], "pass_longest": v[1], "pass_first": v[2], "pass_last": v[3],
                                  "delta_first_pp": 100*(v[2]-v[1])/tot_by_cond[c], "delta_last_pp": 100*(v[3]-v[1])/tot_by_cond[c]} for c, v in agg.items()}
    # ---- cross-seed variation ----
    rates = defaultdict(dict)
    for f in glob.glob(str(HERE / "results/humaneval_v2/*_s*.jsonl")):
        b = os.path.basename(f)[:-6]; cond, rest = b.split("_", 1); model, seed = rest.rsplit("_s", 1)
        rows = [json.loads(l) for l in open(f) if l.strip()]
        if rows: rates[(model, cond)][seed] = 100*sum(bool(r.get("passed")) for r in rows)/len(rows)
    seedrange = {f"{m}|{c}": max(v.values())-min(v.values()) for (m, c), v in rates.items() if len(v) >= 3}
    vals = sorted(seedrange.values())
    out["seed_range"] = {"per_model_condition": seedrange, "median_pp": vals[len(vals)//2] if vals else None, "max_pp": max(vals) if vals else None}
    # ---- wording (direct + paraphrases, seed 42) ----
    W = ["direct", "para1", "para2", "para3"]; pooled = {}; per = defaultdict(dict)
    for c in W:
        n = p = 0
        for f in glob.glob(str(HERE / f"results/humaneval_v2/{c}_*_s42.jsonl")):
            m = os.path.basename(f)[len(c)+1:-len("_s42.jsonl")]
            rows = [json.loads(l) for l in open(f) if l.strip()]
            if len(rows) < 164 or m == "gemini-2.5-flash": continue
            n += len(rows); p += sum(bool(r.get("passed")) for r in rows); per[m][c] = 100*sum(bool(r.get("passed")) for r in rows)/len(rows)
        if n: pooled[c] = 100*p/n
    complete = {m: v for m, v in per.items() if len(v) == 4}
    out["wording"] = {"pooled": pooled, "pooled_range": (max(pooled.values())-min(pooled.values())) if len(pooled) == 4 else None,
                      "per_model_range": {m: max(v.values())-min(v.values()) for m, v in complete.items()}, "n_models_complete": len(complete)}
    json.dump(out, open(HERE / "results/harness_sensitivity_humaneval.json", "w"), indent=1)
    print("extraction rule (HumanEval):", {c: f"affected {v['n_affected']}/{v['n_total']}, first {v['delta_first_pp']:+.1f} pp, last {v['delta_last_pp']:+.1f} pp" for c, v in out["extraction_rule"].items()})
    print(f"cross-seed range: median {out['seed_range']['median_pp']:.1f} pp, max {out['seed_range']['max_pp']:.1f} pp")
    print("wording:", out["wording"])

if __name__ == "__main__":
    main()
