#!/usr/bin/env python3
"""
run_livecodebench_v3.py -- LiveCodeBench harness, revision 1 (corrected).

Why v3: the previous harness instructed every model to read stdin and print
stdout and piped a home-made serialisation of the test inputs, although 171 of
the 180 sampled problems are LeetCode problems that the benchmark evaluates by
calling a method on `class Solution`. v3 evaluates exactly as the official
LiveCodeBench leaderboard does:

  * grading uses the official evaluator vendored verbatim in lcb_eval/
    (functional problems: instantiate Solution, call the method named in the
    problem metadata with JSON-decoded arguments; stdin problems: stdin/stdout),
  * ALL tests (public + private) must pass, as on the leaderboard,
  * the direct prompt is the official generic LiveCodeBench prompt, and every
    condition's final code step uses the official Format block, so the output
    contract is identical across conditions,
  * extraction is the official rule (last fenced block).

Conditions: direct, cot, tcgp, and `restate` (control: step 1 restates the
input/output contract without examples; isolates the extra call from the
content of test scenarios). Records keep raw responses, token usage, the
public-only verdict (for public/private disagreement analysis), and resume per
problem. Same deterministic 180-problem stratified sample as v2 (seed 42).
"""
import argparse, importlib.util, json, os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lcb_eval import grade, extract_code

# Heavy imports (API clients, datasets) are loaded lazily in main(): grading
# runs in spawned child processes that re-import this module, so its top
# level must stay light.
call_llm = None
select_problems = None

def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, HERE / fn)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def _init_heavy():
    global call_llm, select_problems
    from dotenv import load_dotenv
    load_dotenv(HERE / ".env")
    call_llm = _load("hv2", "run_humaneval_v2.py").call_llm
    select_problems = _load("lcbv2", "run_livecodebench_v2.py").select_problems

STEP1_CAP = 6000
STEP2_CAP = 12000
# Gemini 2.5 counts hidden thinking tokens against max_output_tokens (up to
# ~10k on hard problems), so it needs a much larger cap to avoid truncating
# the code; measured usage is what the paper reports, not the cap.
PROVIDER_CAPS = {"gemini": (24000, 48000)}
CONDITIONS = ("direct", "cot", "tcgp", "restate")

# ---- prompts: official LiveCodeBench wording for the code step ----
FMT_STARTER = ("You will use the following starter code to write the solution to the "
               "problem and enclose your code within delimiters.")
FMT_STDIN = ("Read the inputs from stdin solve the problem and write the answer to stdout "
             "(do not directly test on the sample inputs). Enclose your code within delimiters "
             "as follows. Ensure that when the python program runs, it reads the inputs, runs "
             "the algorithm and writes output to STDOUT.")

def fmt_block(starter):
    if starter:
        return f"### Format: {FMT_STARTER}\n```python\n{starter}\n```\n\n"
    return f"### Format: {FMT_STDIN}\n```python\n# YOUR CODE HERE\n```\n\n"

def p_direct(q, starter):
    # verbatim official generic template (lcb_runner/prompts/code_generation.py)
    return (f"### Question:\n{q}\n\n" + fmt_block(starter)
            + "### Answer: (use the provided format with backticks)\n\n")

def _starter_ref(starter):
    return (f"### Starter code (for reference; do not write the solution yet):\n"
            f"```python\n{starter}\n```\n\n") if starter else ""

def p_cot1(q, starter):
    return (f"### Question:\n{q}\n\n" + _starter_ref(starter)
            + "### Task: Think step by step about this problem: analyze the inputs and outputs, "
              "identify the algorithm, and consider edge cases. Do NOT write the solution code yet.\n\n")

def p_tcgp1(q, starter):
    how = ("using the exact argument values the starter code's method would receive and the exact "
           "value it must return" if starter else "using the exact stdin text and the exact stdout text")
    return (f"### Question:\n{q}\n\n" + _starter_ref(starter)
            + f"### Task: Before implementing, write 3-4 concrete test scenarios for this problem as "
              f"exact input -> expected output pairs, {how}. Do NOT write the implementation yet.\n\n")

def p_restate1(q, starter):
    what = ("the exact type and structure of each argument the starter code's method receives and "
            "the exact type and structure of the value it must return" if starter
            else "the exact stdin format and the exact stdout format")
    return (f"### Question:\n{q}\n\n" + _starter_ref(starter)
            + f"### Task: Restate precisely the input and output contract of this problem: {what}. "
              f"Do NOT give any examples and do NOT solve the problem.\n\n")

def p_step2(q, starter, prior, kind):
    return (f"### Question:\n{q}\n\n### Your {kind}:\n{prior}\n\n" + fmt_block(starter)
            + f"### Answer: Using your {kind} above, write the complete solution. "
              f"(use the provided format with backticks)\n\n")

STEP1 = {"cot": (p_cot1, "reasoning"), "tcgp": (p_tcgp1, "test scenarios"),
         "restate": (p_restate1, "input/output contract")}

def run_one(problem, condition, model, provider, seed, memo):
    q = problem["question_content"]; starter = problem.get("starter_code") or ""
    rec = {"question_id": str(problem["question_id"]), "difficulty": problem.get("difficulty"),
           "platform": problem.get("platform"), "testtype": "functional" if starter.strip() else "stdin",
           "condition": condition, "model": model, "provider": provider, "seed": seed,
           "timestamp": datetime.now(timezone.utc).isoformat()}
    cap1, cap2 = PROVIDER_CAPS.get(provider, (STEP1_CAP, STEP2_CAP))
    try:
        if condition == "direct":
            raw1, use1 = "", None
            raw2, use2, params = call_llm(p_direct(q, starter), model, provider, cap2, seed, memo)
        else:
            f1, kind = STEP1[condition]
            raw1, use1, _ = call_llm(f1(q, starter), model, provider, cap1, seed, memo)
            raw2, use2, params = call_llm(p_step2(q, starter, raw1, kind), model, provider, cap2, seed, memo)
        code = extract_code(raw2 or "")
        g = grade(problem, code)
        rec.update(raw_step1=raw1, raw_step2=raw2, extracted_code=code[:6000], no_output=not code.strip(),
                   usage_step1=use1, usage_step2=use2, params_used=params,
                   truncated=bool((use2 or {}).get("truncated")), **g, api_error=None)
    except Exception as e:
        rec.update(raw_step1="", raw_step2="", extracted_code="", no_output=True, passed=False,
                   public_pass=False, api_error=f"{type(e).__name__}: {str(e)[:200]}")
    return rec

def load_done(path):
    done = set()
    if path.exists():
        for l in open(path):
            if l.strip():
                try: done.add(json.loads(l)["question_id"])
                except Exception: pass
    return done

def git_commit():
    try: return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, text=True).strip()
    except Exception: return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="model:provider ...")
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-difficulty", type=int, default=60)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="strat")
    a = ap.parse_args()
    _init_heavy()
    from datasets import load_dataset
    out = HERE / "results" / f"livecodebench_v3_{a.tag}"; out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("bzantium/livecodebench", split="test")
    problems = select_problems(ds, a.limit, a.per_difficulty)
    dates = sorted(str(p.get("contest_date"))[:10] for p in problems)
    manifest_p = out / "manifest.json"
    manifest = json.load(open(manifest_p)) if manifest_p.exists() else {"runs": []}
    for spec in a.models:
        model, provider = spec.split(":")
        memo = {}
        for cond in a.conditions:
            path = out / f"{cond}_{model}_s{a.seed}.jsonl"
            done = load_done(path); todo = [p for p in problems if str(p["question_id"]) not in done]
            print(f"[{model}/{cond}] {len(done)} done, {len(todo)} to go", flush=True)
            with open(path, "a") as f:
                for i, p in enumerate(todo, 1):
                    rec = run_one(p, cond, model, provider, a.seed, memo)
                    f.write(json.dumps(rec) + "\n"); f.flush()
                    if i % 10 == 0:
                        print(f"  {model}/{cond}: {len(done)+i}/{len(problems)}", flush=True)
            manifest["runs"].append({"model": model, "provider": provider, "condition": cond, "seed": a.seed,
                "params_used": memo.get("params"), "step1_cap": PROVIDER_CAPS.get(provider, (STEP1_CAP, STEP2_CAP))[0], "step2_cap": PROVIDER_CAPS.get(provider, (STEP1_CAP, STEP2_CAP))[1],
                "evaluator": "official LiveCodeBench testing_util @ 28fef95 (vendored in lcb_eval/)",
                "dataset": "bzantium/livecodebench (test split, 1055 problems)",
                "n_problems": len(problems), "contest_date_window": [dates[0], dates[-1]],
                "harness_commit": git_commit(), "finished": datetime.now(timezone.utc).isoformat()})
            json.dump(manifest, open(manifest_p, "w"), indent=1)

if __name__ == "__main__":
    main()
