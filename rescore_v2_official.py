#!/usr/bin/env python3
"""Re-grade the old v2 LiveCodeBench outputs with the official evaluator (all
public+private tests). Diagnostic only: those programs were prompted to use
stdin, so this is a lower bound on what a correct prompt would achieve."""
import sys, json, glob, importlib.util
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from lcb_eval import grade

def main():
    from datasets import load_dataset
    spec = importlib.util.spec_from_file_location("lcbv2", HERE / "run_livecodebench_v2.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    ds = load_dataset("bzantium/livecodebench", split="test")
    probs = {str(p["question_id"]): p for p in m.select_problems(ds, 0, 60)}
    out = HERE / "results" / "livecodebench_v2_strat_rescored"; out.mkdir(exist_ok=True)
    files = sorted(glob.glob(str(HERE / "results/livecodebench_v2_strat/*_s42.jsonl")))
    if len(sys.argv) > 1: files = [f for f in files if any(k in f for k in sys.argv[1:])]
    for f in files:
        dst = out / Path(f).name
        done = set()
        if dst.exists():
            for l in open(dst):
                if l.strip(): done.add(json.loads(l)["question_id"])
        with open(dst, "a") as w:
            for l in open(f):
                if not l.strip(): continue
                r = json.loads(l); q = str(r["question_id"])
                if q in done: continue
                g = grade(probs[q], r.get("extracted_code") or "")
                w.write(json.dumps({"question_id": q, "condition": r["condition"], "model": r["model"],
                                    "difficulty": r.get("difficulty"), "old_passed": bool(r.get("passed")), **g}) + "\n"); w.flush()
        print("done", dst.name, flush=True)

if __name__ == "__main__":
    main()
