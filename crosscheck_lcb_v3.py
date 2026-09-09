#!/usr/bin/env python3
"""
crosscheck_lcb_v3.py -- Independent verification of v3 verdicts.
Re-grades the saved generations through the official LiveCodeBench entry point
(lcb_runner.evaluation.compute_code_generation_metrics.codegen_metrics, from a
fresh clone at commit 28fef95) and compares verdict-for-verdict with what the
harness recorded. Also prints per-difficulty Pass@1 for comparison with the
public leaderboard. Usage: crosscheck_lcb_v3.py <path-to-LiveCodeBench-clone> [file glob]
"""
import sys, json, glob, importlib.util
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from lcb_eval import make_sample

def main():
    lcb = sys.argv[1]; sys.path.insert(0, lcb)
    from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics
    from datasets import load_dataset
    spec = importlib.util.spec_from_file_location("lcbv2", HERE / "run_livecodebench_v2.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    ds = load_dataset("bzantium/livecodebench", split="test")
    probs = {str(p["question_id"]): p for p in m.select_problems(ds, 0, 60)}
    pat = sys.argv[2] if len(sys.argv) > 2 else "results/livecodebench_v3_strat/*_s42.jsonl"
    for f in sorted(glob.glob(str(HERE / pat))):
        rows = [json.loads(l) for l in open(f) if l.strip()]
        if len(rows) < 180: print(f"{Path(f).name}: only {len(rows)} rows, skipping"); continue
        samples = [make_sample(probs[r["question_id"]])[0] for r in rows]
        gens = [[r.get("extracted_code") or ""] for r in rows]
        metrics, results, _ = codegen_metrics(samples, gens, k_list=[1], num_process_evaluate=8, timeout=6)
        official = [all(x == True for x in results[i][0]) for i in range(len(rows))]
        agree = sum(o == bool(r["passed"]) for o, r in zip(official, rows))
        by = {}
        for r in rows: by.setdefault(r["difficulty"], [0, 0]); by[r["difficulty"]][0] += 1; by[r["difficulty"]][1] += r["passed"]
        print(f"{Path(f).name}: harness pass@1={100*sum(r['passed'] for r in rows)/len(rows):.1f}%  "
              f"official codegen_metrics pass@1={metrics['pass@1']:.1f}%  verdict agreement={agree}/{len(rows)} | "
              + " ".join(f"{d}={100*v[1]/v[0]:.0f}%" for d, v in by.items()))

if __name__ == "__main__":
    main()
