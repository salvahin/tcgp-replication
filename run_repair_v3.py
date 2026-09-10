#!/usr/bin/env python3
"""
run_repair_v3.py -- One round of execution feedback ("self-repair") on top of the
direct condition. For each model, take the direct-condition solution; if it
fails a PUBLIC test (the only tests a user would have), show the model its code
and the first failing public test (input, expected, actual/error) and ask for a
fixed complete solution, once. Solutions that pass the public tests are kept
unchanged (no feedback available), so this is the realistic protocol. Grades
with all tests. Output: repair_{model}_s{seed}.jsonl (all 180 problems).
"""
import argparse, json, sys, importlib.util
from datetime import datetime, timezone
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from lcb_eval import grade, extract_code, check_correctness, make_sample

def public_failure(problem, code):
    """Return (inputs, expected, got) for the first failing PUBLIC test, or None."""
    sample, n_pub, _ = make_sample(problem)
    io = json.loads(sample["input_output"])
    pub = {"input_output": json.dumps({"inputs": io["inputs"][:n_pub], "outputs": io["outputs"][:n_pub], "fn_name": io["fn_name"]})}
    res, meta = check_correctness(pub, code, 6)
    if res and all(r is True or r == True for r in res) and len(res) == n_pub: return None
    m = meta if isinstance(meta, dict) else {}
    return (str(m.get("inputs", ""))[:600], str(m.get("expected", ""))[:300],
            str(m.get("output", m.get("error", m.get("error_message", ""))))[:300])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--models", nargs="+", required=True); ap.add_argument("--seed", type=int, default=42); ap.add_argument("--tag", default="strat")
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("v3", HERE / "run_livecodebench_v3.py"); v3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v3)
    v3._init_heavy()
    probs = {str(p["question_id"]): p for p in v3.load_problems()}
    out = HERE / "results" / f"livecodebench_v3_{a.tag}"
    for spec_ in a.models:
        model, provider = spec_.split(":"); memo = {}
        src = out / f"direct_{model}_s{a.seed}.jsonl"; dst = out / f"repair_{model}_s{a.seed}.jsonl"
        done = v3.load_done(dst)
        rows = [json.loads(l) for l in open(src) if l.strip()]
        cap1, cap2 = v3.PROVIDER_CAPS.get(provider, (v3.STEP1_CAP, v3.STEP2_CAP))
        with open(dst, "a") as w:
            for i, r in enumerate(rows, 1):
                q = r["question_id"]
                if q in done: continue
                p = probs[q]; starter = p.get("starter_code") or ""
                rec = dict(r); rec.update(condition="repair", timestamp=datetime.now(timezone.utc).isoformat(), repaired=False, feedback=None)
                code0 = r.get("extracted_code") or ""
                fb = public_failure(p, code0) if code0.strip() else ("", "", "no code was produced")
                if fb is not None:
                    inp, exp, got = fb
                    prompt = (v3.p_direct(p["question_content"], starter)
                              + f"### Your previous solution:\n```python\n{code0}\n```\n\n"
                              + f"### It failed a public test.\nInput:\n{inp}\nExpected output:\n{exp}\nYour output / error:\n{got}\n\n"
                              + "### Fix the solution and return the complete corrected solution in the provided format, with backticks.\n\n")
                    try:
                        raw, use, params = v3.call_llm(prompt, model, provider, cap2, a.seed, memo)
                        code = extract_code(raw or "")
                        g = grade(p, code)
                        rec.update(raw_repair=raw, extracted_code=code[:6000], usage_repair=use, params_used=params, repaired=True,
                                   feedback={"input": inp, "expected": exp, "got": got}, no_output=not code.strip(), api_error=None, **g)
                    except Exception as e:
                        rec.update(api_error=f"{type(e).__name__}: {str(e)[:200]}")
                w.write(json.dumps(rec) + "\n"); w.flush()
                if i % 20 == 0: print(f"  {model}/repair: {i}/{len(rows)}", flush=True)
        print(f"[{model}/repair] done", flush=True)

if __name__ == "__main__":
    main()
