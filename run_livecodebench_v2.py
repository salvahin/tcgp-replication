#!/usr/bin/env python3
"""
run_livecodebench_v2.py — Corrected LiveCodeBench harness (competitive programming, hard).

Same corrections as run_humaneval_v2.py, adapted to LCB:
  - Two-step CoT/TCGP (reason/scenarios first, then a COMPLETE stdin/stdout program),
    matching the paper's described methodology and preventing reasoning-model truncation
    (which produced the 96%-artifact "gpt-5.3-codex 13.5x" case in V1).
  - Generous completion budgets; full raw responses + token telemetry stored.
  - Verbatim fenced-code extraction; official public-test execution reused from V1.
  - Per-problem resume, stable per-(condition, model, seed) files.

USAGE
  python run_livecodebench_v2.py --models gpt-4o:azure grok-4-20-reasoning:foundry \
      --conditions direct cot tcgp --seed 42 --limit 50
"""

import argparse
import json
import os
import re
import time
import importlib.util
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset

HERE = Path(__file__).parent
OUT_DIR = HERE / "results" / "livecodebench_v2"

# Reuse the proven multi-provider caller (telemetry, foundry/reasoning params) and
# the proven stdin/stdout test executor + code extractor.
_v2 = importlib.util.module_from_spec(importlib.util.spec_from_file_location("hv2", HERE / "run_humaneval_v2.py"))
importlib.util.spec_from_file_location("hv2", HERE / "run_humaneval_v2.py").loader.exec_module(_v2)
call_llm = _v2.call_llm
extract_code = _v2.extract_code

_lcb = importlib.util.module_from_spec(importlib.util.spec_from_file_location("lcb1", HERE / "run_livecodebench.py"))
importlib.util.spec_from_file_location("lcb1", HERE / "run_livecodebench.py").loader.exec_module(_lcb)
run_test = _lcb.run_test

STEP1_CAP = 6000    # reasoning / scenarios (hard problems -> more room)
STEP2_CAP = 12000   # complete program


def parse_tests(problem):
    pts = problem.get("public_test_cases", [])
    if isinstance(pts, str):
        try:
            pts = json.loads(pts)
        except Exception:
            pts = []
    return pts


def p_direct(q, starter):
    s = f"\n## Starter code:\n```python\n{starter}\n```\n" if starter else ""
    return (f"Solve this competitive programming problem. Your solution must read from stdin "
            f"and print to stdout.{s}\n## Problem:\n{q}\n\nReturn ONLY the complete Python "
            f"solution inside one ```python code block.")


def p_reason(q, starter):
    return (f"Analyze this competitive programming problem: input/output format, algorithm, "
            f"edge cases, complexity. Do NOT write the final code yet.\n\n## Problem:\n{q}")


def p_scenarios(q, starter):
    return (f"For this competitive programming problem, write 3-4 concrete input -> expected "
            f"output test scenarios (exact values). Do NOT write the solution yet.\n\n## Problem:\n{q}")


def p_code_from(q, starter, prior, kind):
    s = f"\n## Starter code:\n```python\n{starter}\n```\n" if starter else ""
    return (f"Using your {kind} below, write the COMPLETE Python solution that reads from stdin "
            f"and prints to stdout.{s}\n## Your {kind}:\n{prior}\n\n## Problem:\n{q}\n\n"
            f"Return ONLY the complete Python solution inside one ```python code block.")


def evaluate(problem, condition, model, provider, seed, memo):
    q = problem.get("question_content", problem.get("question", ""))
    starter = problem.get("starter_code", "")
    rec = {"question_id": problem.get("question_id", problem.get("id", "unknown")),
           "difficulty": problem.get("difficulty", "unknown"),
           "model": model, "provider": provider, "condition": condition, "seed": seed,
           "timestamp": datetime.now(timezone.utc).isoformat()}
    t0 = time.time()
    try:
        if condition == "direct":
            raw1, use1 = "", None
            raw2, use2, params = call_llm(p_direct(q, starter), model, provider, STEP2_CAP, seed, memo)
        elif condition == "cot":
            raw1, use1, _ = call_llm(p_reason(q, starter), model, provider, STEP1_CAP, seed, memo)
            raw2, use2, params = call_llm(p_code_from(q, starter, raw1, "reasoning"), model, provider, STEP2_CAP, seed, memo)
        elif condition == "tcgp":
            raw1, use1, _ = call_llm(p_scenarios(q, starter), model, provider, STEP1_CAP, seed, memo)
            raw2, use2, params = call_llm(p_code_from(q, starter, raw1, "test scenarios"), model, provider, STEP2_CAP, seed, memo)
        else:
            raise ValueError(condition)
        code = extract_code(raw2)
        tests = parse_tests(problem)
        passed = False
        results = []
        if code.strip() and tests:
            all_ok = True
            for t in tests[:3]:
                ti = t.get("input", "")
                exp = t.get("output", t.get("expected_output", ""))
                if ti and exp:
                    ok, _actual, _err = run_test(code, ti, exp)
                    results.append(bool(ok))
                    if not ok:
                        all_ok = False
            passed = all_ok and len(results) > 0
        rec.update(raw_step1=raw1, raw_step2=raw2, extracted_code=code[:4000],
                   passed=passed, no_output=(code.strip() == ""),
                   n_tests=len(results), tests_passed=sum(results),
                   usage_step1=use1, usage_step2=use2,
                   truncated=bool((use1 or {}).get("truncated") or (use2 or {}).get("truncated")),
                   params_used=params, error=None)
    except Exception as e:
        rec.update(raw_step1=rec.get("raw_step1", ""), raw_step2="", extracted_code="",
                   passed=False, no_output=True, n_tests=0, tests_passed=0,
                   usage_step1=None, usage_step2=None, truncated=None, params_used=None,
                   error=f"API/HARNESS: {type(e).__name__}: {str(e)[:300]}")
    rec["duration_s"] = round(time.time() - t0, 2)
    return rec


def load_done(fp):
    done = {}
    if fp.exists():
        for line in fp.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                break
            if r.get("question_id") is not None:
                done[str(r["question_id"])] = r
        fp.write_text("".join(json.dumps(r) + "\n" for r in done.values()))
    return done


def select_problems(ds, limit, per_difficulty):
    """Deterministic problem selection.
    If per_difficulty>0, take that many of each difficulty (easy/medium/hard),
    ordered by question_id for reproducibility (a superset of the original
    first-50 is NOT guaranteed, so this is a distinct, difficulty-balanced set).
    Otherwise take the first `limit` in dataset order (back-compat)."""
    rows = list(ds)
    if per_difficulty and per_difficulty > 0:
        from collections import defaultdict
        buckets = defaultdict(list)
        for r in rows:
            buckets[str(r.get("difficulty", "unknown")).lower()].append(r)
        chosen = []
        for diff in ("easy", "medium", "hard"):
            b = sorted(buckets.get(diff, []), key=lambda r: str(r.get("question_id", "")))
            chosen.extend(b[:per_difficulty])
        return chosen
    return rows[:limit] if limit else rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--conditions", nargs="+", default=["direct", "cot", "tcgp"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--per-difficulty", type=int, default=0,
                    help="If >0, take this many easy+medium+hard each (difficulty-stratified).")
    ap.add_argument("--tag", default="", help="Suffix for the output dir, e.g. 'strat' -> results/livecodebench_v2_strat")
    args = ap.parse_args()

    global OUT_DIR
    if args.tag:
        OUT_DIR = OUT_DIR.parent / f"livecodebench_v2_{args.tag}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("bzantium/livecodebench", split="test")
    problems = select_problems(ds, args.limit, args.per_difficulty)
    n = len(problems)

    for spec in args.models:
        model, provider = spec.split(":", 1)
        memo = {}
        for condition in args.conditions:
            fp = OUT_DIR / f"{condition}_{model}_s{args.seed}.jsonl"
            done = load_done(fp)
            todo = [p for i, p in enumerate(problems)
                    if str(p.get("question_id", p.get("id", f"p{i}"))) not in done]
            passed = sum(1 for r in done.values() if r.get("passed"))
            print(f"\n== {model} [{condition}] s{args.seed}: {len(done)}/{n} done ==", flush=True)
            with open(fp, "a") as fh:
                for i, p in enumerate(todo):
                    r = evaluate(p, condition, model, provider, args.seed, memo)
                    fh.write(json.dumps(r) + "\n")
                    fh.flush()
                    passed += int(r["passed"])
                    mark = "PASS" if r["passed"] else ("api-ERR" if (r.get("error") or "").startswith("API") else ("no-out" if r.get("no_output") else "fail"))
                    print(f"  [{len(done)+i+1}/{n}] {r['question_id']}: {mark}", flush=True)
            print(f"  -> {model} [{condition}] s{args.seed}: {passed}/{n} = {100*passed/n:.1f}%", flush=True)


if __name__ == "__main__":
    main()
