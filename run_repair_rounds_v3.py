#!/usr/bin/env python3
"""
run_repair_rounds_v3.py -- A minimal agentic loop: up to R rounds of
execution feedback on the PUBLIC tests only (the signal an agent harness has),
judged on ALL tests (the signal it is evaluated by). Starts from the direct-
prompting solution; each round shows the model its current code and the first
failing public test and asks for a fixed complete solution; the loop stops when
the public tests pass. Records, per problem, the hidden-test verdict after each
round, so that the overfitting-to-feedback curve can be plotted.
Output: repairR_{model}_s{seed}.jsonl (one record per problem, with a `rounds`
list). Resumes per problem.
"""
import argparse, json, sys, importlib.util
from datetime import datetime, timezone
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from lcb_eval import grade, extract_code, check_correctness, make_sample

def public_failure(problem, code):
    sample, n_pub, _ = make_sample(problem); io = json.loads(sample["input_output"])
    pub = {"input_output": json.dumps({"inputs": io["inputs"][:n_pub], "outputs": io["outputs"][:n_pub], "fn_name": io["fn_name"]})}
    if not code.strip(): return ("", "", "no code was produced")
    res, meta = check_correctness(pub, code, 6)
    if res and len(res) == n_pub and all(r is True or r == True for r in res): return None
    m = meta if isinstance(meta, dict) else {}
    return (str(m.get("inputs", ""))[:600], str(m.get("expected", ""))[:300], str(m.get("output", m.get("error", m.get("error_message", ""))))[:300])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--models", nargs="+", required=True); ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--tag", default="strat"); a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("v3", HERE / "run_livecodebench_v3.py"); v3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v3); v3._init_heavy()
    probs = {str(p["question_id"]): p for p in v3.load_problems()}
    out = HERE / "results" / f"livecodebench_v3_{a.tag}"
    for spec_ in a.models:
        model, provider = spec_.split(":"); memo = {}
        src = out / f"direct_{model}_s{a.seed}.jsonl"; dst = out / f"repairR_{model}_s{a.seed}.jsonl"; done = v3.load_done(dst)
        rows = [json.loads(l) for l in open(src) if l.strip()]
        cap1, cap2 = v3.PROVIDER_CAPS.get(provider, (v3.STEP1_CAP, v3.STEP2_CAP))
        with open(dst, "a") as w:
            for i, r in enumerate(rows, 1):
                q = r["question_id"]
                if q in done: continue
                p = probs[q]; starter = p.get("starter_code") or ""; code = r.get("extracted_code") or ""
                rec = {"question_id": q, "difficulty": p.get("difficulty"), "testtype": r.get("testtype"), "model": model, "provider": provider,
                       "seed": a.seed, "condition": "repairR", "timestamp": datetime.now(timezone.utc).isoformat(),
                       "round0": {"passed": bool(r.get("passed")), "public_pass": bool(r.get("public_pass"))}, "rounds": [], "api_error": None}
                try:
                    for rnd in range(1, a.rounds + 1):
                        fb = public_failure(p, code)
                        if fb is None: break
                        inp, exp, got = fb
                        prompt = (v3.p_direct(p["question_content"], starter) + f"### Your previous solution:\n```python\n{code}\n```\n\n"
                                  + f"### It failed a public test.\nInput:\n{inp}\nExpected output:\n{exp}\nYour output / error:\n{got}\n\n"
                                  + "### Fix the solution and return the complete corrected solution in the provided format, with backticks.\n\n")
                        raw, use, params = v3.call_llm(prompt, model, provider, cap2, a.seed, memo)
                        code = extract_code(raw or ""); g = grade(p, code)
                        rec["rounds"].append({"round": rnd, "feedback": {"input": inp, "expected": exp, "got": got}, "raw": raw, "code": code, "code_sha256": __import__("hashlib").sha256(code.encode()).hexdigest(),
                                              "usage": use, "passed": g["passed"], "public_pass": g["public_pass"], "n_passed": g["n_passed"], "n_tests": g["n_tests"]})
                except Exception as e:
                    rec["api_error"] = f"{type(e).__name__}: {str(e)[:200]}"
                w.write(json.dumps(rec) + "\n"); w.flush()
                if i % 20 == 0: print(f"  {model}/repairR: {i}/{len(rows)}", flush=True)
        print(f"[{model}/repairR] done", flush=True)

if __name__ == "__main__":
    main()
