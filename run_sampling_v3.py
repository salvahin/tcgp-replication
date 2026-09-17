#!/usr/bin/env python3
"""
run_sampling_v3.py -- The compute counterpoint: k=5 samples of the direct
prompt at temperature 0.7, then execution-based selection using the PUBLIC
tests only (the tests a user has). Reports, per problem: pass of the selected
sample (all tests), pass of the first sample, and any-of-5 (oracle upper bound).
Output: sample5_{model}_s{seed}.jsonl. Not run until approved (cost ~ 5x direct).
"""
import argparse, json, sys, importlib.util
from datetime import datetime, timezone
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from lcb_eval import grade, extract_code, check_correctness, make_sample

def public_score(problem, code):
    sample, n_pub, _ = make_sample(problem); io = json.loads(sample["input_output"])
    pub = {"input_output": json.dumps({"inputs": io["inputs"][:n_pub], "outputs": io["outputs"][:n_pub], "fn_name": io["fn_name"]})}
    res, _ = check_correctness(pub, code, 6) if code.strip() else ([], {})
    return sum(1 for r in res if r is True or r == True)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--models", nargs="+", required=True); ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7); ap.add_argument("--seed", type=int, default=42); ap.add_argument("--tag", default="strat")
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("v3", HERE / "run_livecodebench_v3.py"); v3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v3)
    v3._init_heavy()
    problems = v3.load_problems()
    out = HERE / "results" / f"livecodebench_v3_{a.tag}"
    for spec_ in a.models:
        model, provider = spec_.split(":")
        memo = {"strategies": [{"tok": "max_completion_tokens", "temp": a.temperature, "seed": False},
                               {"tok": "max_tokens", "temp": a.temperature, "seed": False},
                               {"tok": "max_completion_tokens", "temp": None, "seed": False}]}
        dst = out / f"sample{a.k}_{model}_s{a.seed}.jsonl"; done = v3.load_done(dst)
        cap1, cap2 = v3.PROVIDER_CAPS.get(provider, (v3.STEP1_CAP, v3.STEP2_CAP))
        with open(dst, "a") as w:
            for i, p in enumerate(problems, 1):
                q = str(p["question_id"])
                if q in done: continue
                starter = p.get("starter_code") or ""
                rec = {"question_id": q, "difficulty": p.get("difficulty"), "platform": p.get("platform"),
                       "testtype": "functional" if starter.strip() else "stdin", "condition": f"sample{a.k}", "model": model,
                       "provider": provider, "seed": a.seed, "temperature": a.temperature, "timestamp": datetime.now(timezone.utc).isoformat()}
                samples = []
                try:
                    for _ in range(a.k):
                        raw, use, params = v3.call_llm(v3.p_direct(p["question_content"], starter), model, provider, cap2, None, memo)
                        code = extract_code(raw or ""); g = grade(p, code)
                        samples.append({"raw": raw, "code": code, "code_sha256": __import__("hashlib").sha256(code.encode()).hexdigest(), "usage": use, "public_score": public_score(p, code), **g})
                    best = max(range(len(samples)), key=lambda j: (samples[j]["public_score"], -j))
                    rec.update(samples=samples, selected=best, passed=samples[best]["passed"], passed_first=samples[0]["passed"],
                               passed_any=any(s_["passed"] for s_ in samples), params_used=params, api_error=None)
                except Exception as e:
                    rec.update(samples=samples, passed=False, api_error=f"{type(e).__name__}: {str(e)[:200]}")
                w.write(json.dumps(rec) + "\n"); w.flush()
                if i % 20 == 0: print(f"  {model}/sample{a.k}: {i}/{len(problems)}", flush=True)
        print(f"[{model}/sample{a.k}] done", flush=True)

if __name__ == "__main__":
    main()
