#!/usr/bin/env python3
"""
run_humaneval_v2.py — Corrected HumanEval harness (complete-function generation).

WHY V2 EXISTS
-------------
The V1 harness (run_tcgp_vs_cot.py) asked for a function *body* and re-indented it
heuristically, which mangled valid model output into non-running code (artifact rates
8-96% per model; see phaseA_harness_artifact_findings_2026-07-02.md). V2:

  1. Requests a COMPLETE function in a fenced block; extracts fenced code verbatim.
  2. Executes <prompt preamble (imports/helpers)> + <model function> + official tests.
  3. Uses generous completion caps (thinking models cannot truncate into artifacts);
     actual token usage is recorded per call.
  4. Stores the FULL RAW model response for every step (reproducibility requirement).
  5. Stable per-(condition, model, seed) files with per-problem resume.
  6. Per-model API params adapted with fallbacks and recorded in a manifest.

PROVIDERS
---------
  azure    - classic Azure OpenAI endpoint (AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_KEY)
  foundry  - Azure AI Foundry OpenAI-v1 endpoint (AZURE_FOUNDRY_ENDPOINT + AZURE_OPENAI_KEY)
  gemini   - Google AI (GEMINI_API_KEY)

USAGE
-----
  python run_humaneval_v2.py --models gpt-4o:azure grok-4-20-reasoning:foundry \
      --conditions direct cot tcgp --seed 42 [--samples 10]

OUTPUT
------
  results/humaneval_v2/{condition}_{model}_s{seed}.jsonl
  results/humaneval_v2/manifest.json
"""

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).parent
load_dotenv(HERE / ".env")

OUT_DIR = HERE / "results" / "humaneval_v2"
DATA_PATH = HERE / "data" / "humaneval" / "humaneval.jsonl"
MANIFEST_PATH = OUT_DIR / "manifest.json"

STEP1_CAP = 4000   # reasoning / scenario step
STEP2_CAP = 8000   # code step and Direct

# ---------------------------------------------------------------------------
# Execution of candidate solutions (reuse the proven executor from V1)
# ---------------------------------------------------------------------------
import importlib.util
_spec = importlib.util.spec_from_file_location("rtc_v1", HERE / "run_tcgp_vs_cot.py")
_v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v1)
execute_test = _v1.execute_test  # (full_code, test_code, entry_point) -> (passed, error)


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------
_clients = {}


def _client(provider):
    if provider in _clients:
        return _clients[provider]
    if provider == "azure":
        from openai import OpenAI
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/") + "/"
        c = OpenAI(base_url=endpoint, api_key=os.getenv("AZURE_OPENAI_KEY"),
                   default_query={"api-version": "preview"}, timeout=300.0, max_retries=2)
    elif provider == "foundry":
        from openai import OpenAI
        endpoint = os.getenv("AZURE_FOUNDRY_ENDPOINT",
                             "https://models4325218527.services.ai.azure.com/openai/v1")
        c = OpenAI(base_url=endpoint, api_key=os.getenv("AZURE_OPENAI_KEY"),
                   timeout=600.0, max_retries=2)
    elif provider == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        c = genai
    else:
        raise ValueError(f"Unknown provider: {provider}")
    _clients[provider] = c
    return c


def call_llm(prompt, model, provider, max_completion, seed=None, params_memo=None):
    """Call model; adapt params with fallbacks and simple 429 backoff.

    Returns (raw_text, usage_dict, params_used). usage_dict records every token
    metric the API exposes (prompt/completion/reasoning/thinking breakdown plus
    finish_reason and a truncated flag) so costs can be computed post-hoc from
    a price table without re-running anything.
    """
    memo = params_memo if params_memo is not None else {}
    if provider == "foundry_responses":
        # Azure Foundry root Responses route (verified for gpt-5.3-codex).
        import requests as _rq
        root = os.getenv("AZURE_FOUNDRY_ENDPOINT",
                         "https://models4325218527.services.ai.azure.com/openai/v1")
        root = re.sub(r"/openai/v1/?$", "", root)
        url = f"{root}/openai/responses?api-version=2025-04-01-preview"
        headers = {"Authorization": f"Bearer {os.getenv('AZURE_OPENAI_KEY')}",
                   "Content-Type": "application/json"}
        last = None
        for attempt in range(5):
            r = _rq.post(url, headers=headers, timeout=600,
                         json={"model": model, "input": prompt,
                               "max_output_tokens": max_completion})
            if r.status_code == 429 or r.status_code >= 500:
                last = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                time.sleep(15 * (attempt + 1))
                continue
            r.raise_for_status()
            d = r.json()
            text = "".join(c.get("text", "") for o in d.get("output", [])
                           for c in o.get("content", []) if isinstance(c, dict))
            u = d.get("usage", {})
            det = u.get("output_tokens_details", {}) or {}
            status = d.get("status")
            usage = {"prompt_tokens": u.get("input_tokens", 0),
                     "completion_tokens": u.get("output_tokens", 0),
                     "reasoning_tokens": det.get("reasoning_tokens", 0),
                     "total_tokens": u.get("total_tokens", 0),
                     "finish_reason": status,
                     "truncated": status == "incomplete"}
            return text, usage, {"api": "foundry/responses", "token_param": "max_output_tokens",
                                 "temperature": "provider-default", "seed_forwarded": False,
                                 "cap": max_completion}
        raise last
    if provider == "gemini":
        genai = _client(provider)
        gm = genai.GenerativeModel(model)
        resp = None
        for attempt in range(6):  # backoff on ResourceExhausted / 429 quota limits
            try:
                resp = gm.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.2, max_output_tokens=max_completion))
                break
            except Exception as e:
                msg = str(e)
                if "ResourceExhausted" in type(e).__name__ or "429" in msg or "quota" in msg.lower() or "exhausted" in msg.lower():
                    time.sleep(20 * (attempt + 1))
                    continue
                raise
        if resp is None:
            raise RuntimeError("gemini: exhausted retries on rate limit")
        try:
            text = resp.text
        except Exception:
            text = ""
            if resp.candidates and resp.candidates[0].content.parts:
                text = "".join(p.text for p in resp.candidates[0].content.parts if hasattr(p, "text"))
        um = getattr(resp, "usage_metadata", None)
        fin = None
        try:
            fin = str(resp.candidates[0].finish_reason) if resp.candidates else None
        except Exception:
            pass
        usage = {
            "prompt_tokens": getattr(um, "prompt_token_count", 0) if um else 0,
            "completion_tokens": getattr(um, "candidates_token_count", 0) if um else 0,
            "reasoning_tokens": getattr(um, "thoughts_token_count", 0) if um else 0,
            "total_tokens": getattr(um, "total_token_count", 0) if um else 0,
            "finish_reason": fin,
            "truncated": bool(fin and "MAX_TOKENS" in str(fin).upper()),
        }
        return text, usage, {"api": "gemini", "temperature": 0.2, "max_output_tokens": max_completion}

    client = _client(provider)
    # Param strategy order; first success is memoized per model.
    strategies = memo.get("strategies") or [
        {"tok": "max_completion_tokens", "temp": 0.2, "seed": True},
        {"tok": "max_completion_tokens", "temp": None, "seed": True},
        {"tok": "max_completion_tokens", "temp": None, "seed": False},
        {"tok": "max_tokens", "temp": 0.2, "seed": True},
        {"tok": "max_tokens", "temp": None, "seed": False},
    ]
    last_err = None
    for st in strategies:
        kwargs = {"model": model,
                  "messages": [{"role": "user", "content": prompt}],
                  st["tok"]: max_completion}
        if st["temp"] is not None:
            kwargs["temperature"] = st["temp"]
        if st["seed"] and seed is not None:
            kwargs["seed"] = seed
        for attempt in range(5):  # 429/503 backoff loop
            try:
                r = client.chat.completions.create(**kwargs)
                memo["strategies"] = [st]  # lock in the working strategy
                text = r.choices[0].message.content or ""
                u = r.usage
                det = getattr(u, "completion_tokens_details", None) if u else None
                fin = r.choices[0].finish_reason
                usage = {
                    "prompt_tokens": getattr(u, "prompt_tokens", 0) if u else 0,
                    "completion_tokens": getattr(u, "completion_tokens", 0) if u else 0,
                    "reasoning_tokens": getattr(det, "reasoning_tokens", 0) if det else 0,
                    "total_tokens": getattr(u, "total_tokens", 0) if u else 0,
                    "finish_reason": fin,
                    "truncated": fin == "length",
                }
                used = {"api": f"{provider}/chat.completions", "token_param": st["tok"],
                        "temperature": st["temp"] if st["temp"] is not None else "provider-default",
                        "seed_forwarded": bool(st["seed"] and seed is not None),
                        "cap": max_completion}
                return text, usage, used
            except Exception as e:
                msg = str(e)
                last_err = e
                if "429" in msg or "rate" in msg.lower() or "503" in msg:
                    time.sleep(15 * (attempt + 1))
                    continue
                break  # non-rate-limit error: leave backoff loop, try next strategy
        msg = str(last_err)
        # only fall through to next strategy on parameter-compat errors
        if not any(k in msg for k in ("Unsupported parameter", "unsupported_parameter",
                                      "does not support", "temperature", "seed",
                                      "max_tokens", "max_completion_tokens",
                                      "unknown_parameter", "Extra inputs")):
            raise last_err
    raise last_err


# ---------------------------------------------------------------------------
# Prompts (complete-function; V2)
# ---------------------------------------------------------------------------
def p_direct(problem):
    return (f"Write the complete Python function described below, including the exact "
            f"signature shown. Return ONLY the code inside one ```python code block, "
            f"with no explanation before or after.\n\n{problem['prompt']}")


def p_cot_step1(problem):
    return (f"Think step by step about how to solve this programming problem: analyze the "
            f"inputs and outputs, identify the algorithm, and consider edge cases. Do NOT "
            f"write the final code yet.\n\n{problem['prompt']}")


def p_cot_step2(problem, reasoning):
    return (f"Using your reasoning below, write the complete Python function, including the "
            f"exact signature shown. Return ONLY the code inside one ```python code block, "
            f"with no explanation.\n\nYOUR REASONING:\n{reasoning}\n\nPROBLEM:\n{problem['prompt']}")


def p_tcgp_step1(problem):
    return (f"Before implementing, write 3-4 concrete test scenarios for this function as "
            f"exact input -> expected output pairs (use real values). Do NOT write the "
            f"implementation yet.\n\n{problem['prompt']}")


def p_tcgp_step2(problem, scenarios):
    return (f"Write the complete Python function, including the exact signature shown. Your "
            f"implementation must pass the test scenarios below. Return ONLY the code inside "
            f"one ```python code block, with no explanation.\n\nTEST SCENARIOS:\n{scenarios}\n\n"
            f"PROBLEM:\n{problem['prompt']}")


# ---------------------------------------------------------------------------
# Extraction & assembly (verbatim; no re-indentation)
# ---------------------------------------------------------------------------
def extract_code(raw):
    blocks = re.findall(r"```(?:python)?[ \t]*\n(.*?)```", raw, re.DOTALL)
    if blocks:
        return max(blocks, key=len).strip("\n")
    return raw.strip("\n")


def assemble(problem, code):
    """Return runnable source: preamble imports/helpers + model function (verbatim)."""
    ep = problem["entry_point"]
    if re.search(rf"^\s*def\s+{re.escape(ep)}\s*\(", code, re.M):
        preamble = problem["prompt"].split("def " + ep)[0]
        return preamble + "\n" + code
    # model returned a body (rare in V2) -> attach under original signature
    return problem["prompt"] + code


# ---------------------------------------------------------------------------
# Main loop with per-problem resume
# ---------------------------------------------------------------------------
def run_one(problem, condition, model, provider, seed, memo):
    rec = {"task_id": problem["task_id"], "model": model, "provider": provider,
           "condition": condition, "seed": seed,
           "timestamp": datetime.now(timezone.utc).isoformat()}
    t0 = time.time()
    try:
        if condition == "direct":
            raw1, use1 = "", None
            raw2, use2, params = call_llm(p_direct(problem), model, provider, STEP2_CAP, seed, memo)
        elif condition == "cot":
            raw1, use1, _ = call_llm(p_cot_step1(problem), model, provider, STEP1_CAP, seed, memo)
            raw2, use2, params = call_llm(p_cot_step2(problem, raw1), model, provider, STEP2_CAP, seed, memo)
        elif condition == "tcgp":
            raw1, use1, _ = call_llm(p_tcgp_step1(problem), model, provider, STEP1_CAP, seed, memo)
            raw2, use2, params = call_llm(p_tcgp_step2(problem, raw1), model, provider, STEP2_CAP, seed, memo)
        else:
            raise ValueError(condition)
        code = extract_code(raw2)
        full = assemble(problem, code)
        passed, error = execute_test(full, problem["test"], problem["entry_point"])
        rec.update(raw_step1=raw1, raw_step2=raw2, extracted_code=code,
                   passed=bool(passed), error=None if passed else (error or "")[:500],
                   usage_step1=use1, usage_step2=use2,
                   truncated=bool((use1 or {}).get("truncated") or (use2 or {}).get("truncated")),
                   params_used=params)
    except Exception as e:
        rec.update(raw_step1=rec.get("raw_step1", ""), raw_step2="", extracted_code="",
                   passed=False, error=f"API/HARNESS: {type(e).__name__}: {str(e)[:300]}",
                   usage_step1=None, usage_step2=None, truncated=None, params_used=None)
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
            if r.get("task_id"):
                done[r["task_id"]] = r
        fp.write_text("".join(json.dumps(r) + "\n" for r in done.values()))
    return done


def update_manifest(model, provider, memo):
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    m = {}
    if MANIFEST_PATH.exists():
        try:
            m = json.loads(MANIFEST_PATH.read_text())
        except Exception:
            m = {}
    entry = m.get(model, {})
    try:
        import subprocess
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
                                capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        commit = "unknown"
    entry.update({"provider": provider,
                  "endpoint_host": {"azure": "models4325218527.openai.azure.com",
                                    "foundry": "models4325218527.services.ai.azure.com",
                                    "gemini": "generativelanguage.googleapis.com"}.get(provider, provider),
                  "params_strategy": memo.get("strategies"),
                  "caps": {"step1": STEP1_CAP, "step2": STEP2_CAP},
                  "harness_commit": commit,
                  "last_run": datetime.now(timezone.utc).isoformat()})
    m[model] = entry
    MANIFEST_PATH.write_text(json.dumps(m, indent=2))


def main():
    ap = argparse.ArgumentParser(description="HumanEval V2 (complete-function harness)")
    ap.add_argument("--models", nargs="+", required=True, help="model:provider pairs")
    ap.add_argument("--conditions", nargs="+", default=["direct", "cot", "tcgp"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--samples", type=int, default=None)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    problems = [json.loads(l) for l in open(DATA_PATH)]
    if args.samples:
        problems = problems[:args.samples]
    n = len(problems)

    for spec in args.models:
        model, provider = spec.split(":", 1)
        memo = {}
        for condition in args.conditions:
            fp = OUT_DIR / f"{condition}_{model}_s{args.seed}.jsonl"
            done = load_done(fp)
            todo = [p for p in problems if p["task_id"] not in done]
            passed = sum(1 for r in done.values() if r.get("passed"))
            print(f"\n== {model} [{condition}] seed={args.seed}: {len(done)}/{n} done"
                  f"{', resuming' if done and todo else ''} ==", flush=True)
            with open(fp, "a") as fh:
                for i, p in enumerate(todo):
                    r = run_one(p, condition, model, provider, args.seed, memo)
                    fh.write(json.dumps(r) + "\n")
                    fh.flush()
                    passed += int(r["passed"])
                    mark = "PASS" if r["passed"] else ("api-ERR" if (r.get("error") or "").startswith("API/HARNESS") else "fail")
                    print(f"  [{len(done)+i+1}/{n}] {p['task_id']}: {mark}", flush=True)
            print(f"  -> {model} [{condition}] s{args.seed}: {passed}/{n} = {100*passed/n:.1f}%", flush=True)
        update_manifest(model, provider, memo)


if __name__ == "__main__":
    main()
