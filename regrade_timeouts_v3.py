#!/usr/bin/env python3
"""Re-grade, in isolation and sequentially, every v3 record whose verdict was a
timeout (per-test alarm or global). Under parallel load correct-but-slow
solutions can time out; the leaderboard grades under controlled load. Rewrites
the record in place with fields regrade_* recording the change."""
import sys, json, glob, importlib.util
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from lcb_eval import grade
def main():
    from datasets import load_dataset
    spec = importlib.util.spec_from_file_location("lcbv2", HERE / "run_livecodebench_v2.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    probs = {str(p["question_id"]): p for p in m.select_problems(load_dataset("bzantium/livecodebench", split="test"), 0, 60)}
    changed = checked = 0
    for f in sorted(glob.glob(str(HERE / "results/livecodebench_v3_strat/*_s42.jsonl"))):
        rows = [json.loads(l) for l in open(f) if l.strip()]; dirty = False
        for r in rows:
            em = str(r.get("error_message", "")).lower()
            if r.get("passed") or r.get("api_error") or not ("time" in em or "timeout" in em): continue
            checked += 1
            g = grade(probs[r["question_id"]], r.get("extracted_code") or "", timeout=6)
            r["regrade_timeout_isolated"] = True; r["regrade_prev_passed"] = r["passed"]
            if g["passed"] != r["passed"] or g["n_passed"] != r.get("n_passed"):
                r.update(g); dirty = True; changed += 1
        if dirty:
            with open(f, "w") as w:
                for r in rows: w.write(json.dumps(r) + "\n")
    print(f"timeout verdicts re-checked in isolation: {checked}; changed: {changed}")
if __name__ == "__main__": main()
