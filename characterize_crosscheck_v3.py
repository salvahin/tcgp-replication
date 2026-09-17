#!/usr/bin/env python3
"""For every v3 file: official codegen_metrics (parallel) vs harness verdicts;
each disagreement is re-graded serially through the official run_test (no
load). Reports how many disagreements remain after serial re-grading and the
official error codes of the parallel-path failures (-3 = time limit)."""
import sys, json, glob, importlib.util
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from lcb_eval import make_sample, check_correctness
def main():
    sys.path.insert(0, sys.argv[1])
    from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics
    from datasets import load_dataset
    spec = importlib.util.spec_from_file_location("lcbv2", HERE / "run_livecodebench_v2.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    probs = {str(p["question_id"]): p for p in m.select_problems(load_dataset("bzantium/livecodebench", split="test"), 0, 60)}
    tot = dis = resolved = 0; codes = {}
    for f in sorted(glob.glob(str(HERE / "results/livecodebench_v3_strat/*_s42.jsonl"))):
        rows = [json.loads(l) for l in open(f) if l.strip()]
        samples = [make_sample(probs[r["question_id"]])[0] for r in rows]
        _, results, metas = codegen_metrics(samples, [[r.get("extracted_code") or ""] for r in rows], k_list=[1], num_process_evaluate=8, timeout=6)
        for i, r in enumerate(rows):
            tot += 1
            off = all(x == True for x in results[i][0])
            if off == bool(r["passed"]): continue
            dis += 1
            mi = metas[i][0] if isinstance(metas[i], list) else metas[i]
            if isinstance(mi, str):
                try: mi = json.loads(mi)
                except Exception: mi = {}
            code = (mi or {}).get("error_code") if isinstance(mi, dict) else None
            codes[str(code)] = codes.get(str(code), 0) + 1
            res, _ = check_correctness(samples[i], r.get("extracted_code") or "", 6)   # serial, official run_test
            serial = bool(len(res) == len(json.loads(samples[i]["input_output"])["inputs"]) and all(x == True for x in res))
            if serial == bool(r["passed"]): resolved += 1
        print(f"{Path(f).name}: cumulative disagreements={dis} resolved-by-serial={resolved}", flush=True)
    print(f"TOTAL records={tot} parallel-path disagreements={dis} ({100*dis/tot:.2f}%) resolved by serial official re-grade={resolved} unresolved={dis-resolved}")
    print("official error codes among parallel-path disagreements:", codes)
    json.dump({"records": tot, "parallel_path_disagreements": dis, "resolved_by_serial_regrade": resolved, "unresolved": dis - resolved,
               "agreement_first_pass": 1 - dis / tot, "agreement_after_serial": 1 - (dis - resolved) / tot, "official_error_codes": codes},
              open(HERE / "results" / "harness_crosscheck_v3.json", "w"), indent=1)
if __name__ == "__main__": main()
