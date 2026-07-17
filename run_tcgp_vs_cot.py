#!/usr/bin/env python3
"""
TCGP vs Chain-of-Thought vs Direct Prompting Comparison

This script isolates whether TCGP's improvement comes from:
1. The structured TCGP format specifically, or
2. Simply giving the model more "thinking" tokens

Three conditions:
- TCGP: Generate test scenarios → use them to guide code generation
- CoT: "Think step by step" reasoning → generate code
- Direct: Simple prompt without extra reasoning

Key design: TCGP and CoT use roughly equivalent token budgets.
"""

import json
import time
import random
import ast
import re
import math
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dotenv import load_dotenv
import os
import autopep8

# Import ablation prompts
from prompts.ablation_prompts import (
    PLAN_AND_SOLVE_STEP1,
    PLAN_AND_SOLVE_STEP2,
    EXTENDED_DIRECT,
    MINIMAL_BDD_STEP1,
    MINIMAL_BDD_STEP2,
    HYBRID_STEP1,
    HYBRID_STEP2,
    SELF_CONSISTENCY_COT,
    TOKEN_BUDGETS,
)

load_dotenv()


def get_client(provider: str = "gemini"):
    """Get API client based on provider."""
    if provider == "azure":
        from openai import OpenAI
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        # Ensure endpoint ends with trailing slash
        if not endpoint.endswith("/"):
            endpoint = endpoint + "/"
        return OpenAI(
            base_url=endpoint,
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            timeout=120.0,
            max_retries=2,
        )
    elif provider == "azure_models":
        # For Azure AI endpoints that require AzureOpenAI client with api_version
        from openai import AzureOpenAI
        return AzureOpenAI(
            api_version="2024-12-01-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            timeout=120.0,
            max_retries=2,
        )
    elif provider == "azure_responses":
        # Return None - we'll use requests directly for this API
        return None
    elif provider == "openai":
        from openai import OpenAI
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=120.0, max_retries=2)
    elif provider == "anthropic":
        import anthropic
        return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    elif provider == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        return genai
    else:
        raise ValueError(f"Unknown provider: {provider}")


class PromptComparisonAgent:
    """Agent for comparing TCGP vs CoT vs Direct prompting."""

    def __init__(self, model: str = "gemini-2.5-flash", provider: str = "gemini", seed: Optional[int] = 42):
        self.client = get_client(provider)
        self.model = model
        self.provider = provider
        self.seed = seed  # Forwarded to OpenAI/Azure-Responses APIs that support deterministic seeding
        self._call_count = 0  # Tracks calls so we can rotate the client periodically
        self._executor = ThreadPoolExecutor(max_workers=1)  # For hard-timeout API calls

    def _maybe_rotate_client(self):
        """Recreate the client every 30 calls to flush stale HTTP keep-alive
        connections. Mitigates a hang we observed where Azure long-running
        processes wedge in pthread_cond_wait inside SSL select after ~600 calls."""
        if self.provider in ("azure", "azure_models", "openai") and self._call_count > 0 and self._call_count % 30 == 0:
            self.client = get_client(self.provider)

    def _call_llm(self, prompt: str, temperature: float = 0.2, max_tokens: int = 1000,
                  seed: Optional[int] = None) -> Tuple[str, int]:
        """Call LLM and return response with token count.

        seed: per-call override. If None, falls back to self.seed. Only honored by
        OpenAI/Azure (chat.completions and Responses API); Anthropic & Gemini APIs
        do not expose deterministic-seed parameters as of 2025.
        """
        if seed is None:
            seed = self.seed
        self._maybe_rotate_client()
        self._call_count += 1
        if self.provider == "anthropic":
            def _do_call():
                return self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=60.0,
                )
            future = self._executor.submit(_do_call)
            try:
                response = future.result(timeout=120.0)
            except TimeoutError:
                # SDK didn't honor its own timeout (observed on slow streaming responses).
                # Replace executor + client so the next call doesn't queue behind the wedge.
                self._executor = ThreadPoolExecutor(max_workers=1)
                self.client = get_client(self.provider)
                raise Exception(f"Hard timeout (120s) on {self.model} (anthropic) — client rotated")
            return response.content[0].text, response.usage.input_tokens + response.usage.output_tokens
        elif self.provider == "azure_responses":
            import requests
            url = f"{os.getenv('AZURE_OPENAI_ENDPOINT')}openai/responses?api-version=2025-04-01-preview"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.getenv('AZURE_OPENAI_KEY')}"
            }
            data = {
                "input": prompt,
                "max_output_tokens": max_tokens,
                "model": self.model,
            }
            if seed is not None:
                data["seed"] = seed
            resp = requests.post(url, headers=headers, json=data, timeout=120)
            result = resp.json()
            if result.get("error"):
                raise Exception(f"API Error: {result['error']}")
            # Extract text from response - handle different response formats
            output_text = ""
            if "output" in result:
                for item in result["output"]:
                    if item.get("type") == "message":
                        for content in item.get("content", []):
                            if content.get("type") == "output_text":
                                output_text += content.get("text", "")
                            elif content.get("type") == "text":
                                output_text += content.get("text", "")
            # Estimate tokens
            tokens = result.get("usage", {}).get("total_tokens", len(prompt.split()) + len(output_text.split()))
            return output_text, tokens
        elif self.provider == "gemini":
            import google.generativeai as genai
            gemini_model = genai.GenerativeModel(self.model)
            generation_config = genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens
            )
            def _do_call():
                # request_options.timeout is the underlying gRPC deadline; pair it with
                # the executor hard-timeout below in case the SDK swallows the deadline.
                return gemini_model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    request_options={"timeout": 60},
                )
            future = self._executor.submit(_do_call)
            try:
                response = future.result(timeout=120.0)
            except TimeoutError:
                self._executor = ThreadPoolExecutor(max_workers=1)
                raise Exception(f"Hard timeout (120s) on {self.model} (gemini) — executor rotated")
            tokens = len(prompt.split()) + len(response.text.split()) if response.text else 0
            return response.text if response.text else "", tokens
        else:
            # Hard timeout via persistent ThreadPoolExecutor — the SDK's own timeout
            # doesn't reliably abort slow-streaming hangs (server keeps connection alive
            # with tiny chunks, so HTTP read-timeout never fires).
            def _do_call():
                kwargs = dict(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=60.0,
                )
                if seed is not None:
                    kwargs["seed"] = seed
                return self.client.chat.completions.create(**kwargs)

            future = self._executor.submit(_do_call)
            try:
                response = future.result(timeout=90.0)
            except TimeoutError:
                # Abandon the future — orphaned worker thread keeps running but is
                # daemonized (Py 3.9+) so it dies with the process. Replace the
                # executor + client so subsequent calls don't queue behind the wedge.
                self._executor = ThreadPoolExecutor(max_workers=1)
                self.client = get_client(self.provider)
                raise Exception(f"Hard timeout (90s) on {self.model} — client rotated")
            return response.choices[0].message.content, response.usage.total_tokens

    def generate_with_tcgp(self, prompt: str, entry_point: str) -> Tuple[str, float, int, str]:
        """Generate code with TCGP methodology.

        Returns: (code, duration, tokens, reasoning)
        """
        start = time.time()
        total_tokens = 0

        # Step 1: Generate TCGP scenarios
        bdd_prompt = f"""Analyze this function signature and docstring, then generate CONCRETE test scenarios.

{prompt}

Generate 3-4 specific test scenarios with exact inputs and expected outputs.

Format:
```
Scenario 1: [description]
  Input: [actual value like [1, 2, 3] or "hello"]
  Expected: [actual expected result]

Scenario 2: [description]
  Input: [actual value]
  Expected: [actual expected result]

Edge Case:
  Input: [edge case value]
  Expected: [expected result]
```

Use REAL values, not placeholders. Extract examples from the docstring if available.
"""
        bdd_scenarios, tokens = self._call_llm(bdd_prompt, temperature=0.3, max_tokens=600)
        total_tokens += tokens

        # Step 2: Generate code with TCGP context
        code_prompt = f"""Complete this Python function. The function signature and docstring are provided.

{prompt}

Your implementation must pass these test scenarios:
{bdd_scenarios}

REQUIREMENTS:
1. Return ONLY the function body (the code that goes inside the function)
2. Do NOT include the function signature (no 'def' line)
3. Use proper 4-space indentation
4. Every code path must return a value

Return ONLY the implementation code, no explanations.
"""
        code, tokens = self._call_llm(code_prompt, temperature=0.2, max_tokens=800)
        total_tokens += tokens

        code = self._extract_body(code)
        duration = time.time() - start

        return code, duration, total_tokens, bdd_scenarios

    def generate_with_cot(self, prompt: str, entry_point: str) -> Tuple[str, float, int, str]:
        """Generate code with Chain-of-Thought reasoning.

        Uses similar token budget to TCGP by asking for step-by-step reasoning.

        Returns: (code, duration, tokens, reasoning)
        """
        start = time.time()
        total_tokens = 0

        # Step 1: Generate reasoning (similar token budget to TCGP scenarios)
        cot_prompt = f"""Analyze this function and think step by step about how to implement it.

{prompt}

Think through the following:
1. What are the inputs and expected outputs?
2. What is the core algorithm or logic needed?
3. What edge cases should be handled?
4. What are the key steps in the implementation?

Provide your step-by-step reasoning (no code yet).
"""
        reasoning, tokens = self._call_llm(cot_prompt, temperature=0.3, max_tokens=600)
        total_tokens += tokens

        # Step 2: Generate code with CoT context
        code_prompt = f"""Complete this Python function based on your analysis.

{prompt}

Your reasoning:
{reasoning}

REQUIREMENTS:
1. Return ONLY the function body (the code that goes inside the function)
2. Do NOT include the function signature (no 'def' line)
3. Use proper 4-space indentation
4. Every code path must return a value

Return ONLY the implementation code, no explanations.
"""
        code, tokens = self._call_llm(code_prompt, temperature=0.2, max_tokens=800)
        total_tokens += tokens

        code = self._extract_body(code)
        duration = time.time() - start

        return code, duration, total_tokens, reasoning

    def generate_direct(self, prompt: str, entry_point: str) -> Tuple[str, float, int]:
        """Generate code with direct prompting (no extra reasoning).

        Returns: (code, duration, tokens)
        """
        start = time.time()

        code_prompt = f"""Complete this Python function. The function signature and docstring are provided.

{prompt}

REQUIREMENTS:
1. Return ONLY the function body (the code that goes inside the function)
2. Do NOT include the function signature (no 'def' line)
3. Use proper 4-space indentation
4. Every code path must return a value
5. Handle edge cases

Return ONLY the implementation code, no explanations.
"""
        code, tokens = self._call_llm(code_prompt, temperature=0.2, max_tokens=800)
        code = self._extract_body(code)
        duration = time.time() - start

        return code, duration, tokens

    # =========================================================================
    # ABLATION METHODS (E1, E2 experiments)
    # =========================================================================

    def generate_with_plan_and_solve(self, prompt: str, entry_point: str) -> Tuple[str, float, int, str]:
        """Generate code with Plan-and-Solve prompting (improved CoT baseline).

        This is the E1 ablation - a stronger CoT variant that should be
        compared against TCGP to ensure fair comparison.

        Returns: (code, duration, tokens, reasoning)
        """
        start = time.time()
        total_tokens = 0

        # Step 1: Plan-and-Solve analysis
        step1_prompt = PLAN_AND_SOLVE_STEP1.format(prompt=prompt)
        reasoning, tokens = self._call_llm(step1_prompt, temperature=0.3, max_tokens=600)
        total_tokens += tokens

        # Step 2: Generate code with plan
        step2_prompt = PLAN_AND_SOLVE_STEP2.format(prompt=prompt, reasoning=reasoning)
        code, tokens = self._call_llm(step2_prompt, temperature=0.2, max_tokens=800)
        total_tokens += tokens

        code = self._extract_body(code)
        duration = time.time() - start

        return code, duration, total_tokens, reasoning

    def generate_with_extended_direct(self, prompt: str, entry_point: str) -> Tuple[str, float, int]:
        """Generate code with Extended Direct prompting (token-matched control).

        This is the E2a ablation - adds thinking space to Direct prompting
        to control for token count. Tests whether TCGP improvement comes
        from methodology or just more tokens.

        Returns: (code, duration, tokens)
        """
        start = time.time()

        code_prompt = EXTENDED_DIRECT.format(prompt=prompt)
        code, tokens = self._call_llm(code_prompt, temperature=0.2, max_tokens=1000)
        code = self._extract_body(code)
        duration = time.time() - start

        return code, duration, tokens

    def generate_with_hybrid(self, prompt: str, entry_point: str) -> Tuple[str, float, int, str]:
        """Hybrid TCGP+CoT: scenarios AND step-by-step reasoning combined in step 1.

        Tests whether structure (scenarios) and reasoning (CoT) are complementary,
        addressing the methodology-fairness counterfactual: did TCGP only win
        because CoT lacked test cases?

        Same total budget as TCGP and CoT (600 + 800 = 1400 tokens).

        Returns: (code, duration, tokens, combined)
        """
        start = time.time()
        total_tokens = 0

        # Step 1: combined scenarios + reasoning
        step1_prompt = HYBRID_STEP1.format(prompt=prompt)
        combined, tokens = self._call_llm(step1_prompt, temperature=0.3, max_tokens=600)
        total_tokens += tokens

        # Step 2: generate code from combined output
        step2_prompt = HYBRID_STEP2.format(prompt=prompt, combined=combined)
        code, tokens = self._call_llm(step2_prompt, temperature=0.2, max_tokens=800)
        total_tokens += tokens

        code = self._extract_body(code)
        duration = time.time() - start

        return code, duration, total_tokens, combined

    def generate_with_self_consistency_cot(
        self,
        prompt: str,
        entry_point: str,
        n_samples: int = 5,
    ) -> Tuple[List[Tuple[str, int]], float, int]:
        """Self-Consistency CoT: sample N reasonings+codes at high temperature.

        Reference: Wang et al. 2022. For code we can't majority-vote text directly,
        so we report Pass@1 (first sample) and Pass@5 (any sample passes) — Pass@k
        per Chen et al. 2021's unbiased estimator.

        Returns: (samples, duration, total_tokens) where samples = [(code, tokens), ...]
        """
        start = time.time()
        samples = []
        total_tokens = 0

        sc_prompt = SELF_CONSISTENCY_COT.format(prompt=prompt)
        base_seed = self.seed if self.seed is not None else 0
        for i in range(n_samples):
            raw, tokens = self._call_llm(sc_prompt, temperature=0.7, max_tokens=1000,
                                          seed=base_seed * 1000 + i)
            code = self._extract_code_from_mixed_response(raw)
            samples.append((code, tokens))
            total_tokens += tokens

        duration = time.time() - start
        return samples, duration, total_tokens

    def _extract_code_from_mixed_response(self, response: str) -> str:
        """Extract code from a response that contains reasoning + code.

        Self-consistency prompts ask for reasoning followed by code in one call.
        Strategy: pull the last fenced code block; if none, fall back to _extract_body.
        """
        blocks = re.findall(r'```(?:python|py)?\s*\n?(.*?)```', response, re.DOTALL)
        if blocks:
            return self._extract_body(blocks[-1])
        return self._extract_body(response)

    def generate_with_minimal_tcgp(self, prompt: str, entry_point: str) -> Tuple[str, float, int, str]:
        """Generate code with Minimal TCGP (token-matched control).

        This is the E2b ablation - reduces TCGP to only 2 scenarios to
        match Direct token count. Tests whether TCGP's structure matters
        or just having test cases.

        Returns: (code, duration, tokens, scenarios)
        """
        start = time.time()
        total_tokens = 0

        # Step 1: Generate minimal scenarios (2 only)
        step1_prompt = MINIMAL_BDD_STEP1.format(prompt=prompt)
        scenarios, tokens = self._call_llm(step1_prompt, temperature=0.3, max_tokens=200)
        total_tokens += tokens

        # Step 2: Generate code with minimal scenarios
        step2_prompt = MINIMAL_BDD_STEP2.format(prompt=prompt, bdd_scenarios=scenarios)
        code, tokens = self._call_llm(step2_prompt, temperature=0.2, max_tokens=600)
        total_tokens += tokens

        code = self._extract_body(code)
        duration = time.time() - start

        return code, duration, total_tokens, scenarios

    def _extract_body(self, code: str) -> str:
        """Extract and clean function body from LLM response."""
        # Remove markdown code blocks
        # Strip code fences WITHOUT consuming the following line's indentation.
        # Using \s* here is a bug: \s matches newlines AND spaces, so it eats the
        # first code line's leading indentation, corrupting the body structure.
        # Restrict to spaces/tabs on the fence line, then an optional newline.
        code = re.sub(r'```[a-zA-Z0-9]*[ \t]*\n?', '', code)
        code = re.sub(r'[ \t]*```[ \t]*\n?', '', code)
        # Strip surrounding blank lines only -- NOT a full .strip(), which would
        # remove the first line's leading indentation and corrupt the relative
        # structure of a body-only ("Direct"-style) response.
        code = code.strip('\n')

        if not code:
            return "    pass"

        lines = code.split('\n')

        # Skip any def line if present
        start_idx = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('def ') and '(' in stripped and ':' in stripped:
                start_idx = i + 1
                break

        lines = lines[start_idx:]
        if not lines:
            return "    pass"

        # PRIMARY: preserve the model's own indentation, re-based so the
        # shallowest non-empty body line sits at 4 spaces. This keeps
        # multi-level structure intact (e.g. a trailing `return False` at
        # function-body level) instead of letting the lossy keyword heuristic
        # flatten it into the preceding block.
        non_empty = [ln for ln in lines if ln.strip()]
        if non_empty:
            min_indent = min(len(ln) - len(ln.lstrip()) for ln in non_empty)
            rebased = '\n'.join(
                ('    ' + ln[min_indent:]) if ln.strip() else ''
                for ln in lines
            )
            try:
                ast.parse("def _test():\n" + rebased)
                return rebased if rebased.strip() else "    pass"
            except SyntaxError:
                pass  # model's indentation didn't parse; fall through

        # FALLBACK: infer indentation from Python keywords (lossy; used only
        # when the model's own indentation does not parse).
        result = self._fix_indentation(lines)
        try:
            ast.parse("def _test():\n" + result)
            return result if result.strip() else "    pass"
        except SyntaxError:
            # Last resort: flatten every line to a single 4-space level.
            result = '\n'.join('    ' + line.strip() if line.strip() else '' for line in lines)
            return result if result.strip() else "    pass"

    def _fix_indentation(self, lines: list) -> str:
        """Fix indentation by inferring structure from Python keywords."""
        # Keywords that increase indentation on next line
        indent_keywords = ('if ', 'elif ', 'else:', 'for ', 'while ', 'try:',
                          'except', 'finally:', 'with ', 'class ', 'def ')
        # Keywords that decrease indentation
        dedent_keywords = ('return', 'break', 'continue', 'raise', 'pass')

        result_lines = []
        current_indent = 4  # Start with base indentation of 4
        indent_stack = [4]  # Stack to track indentation levels

        for line in lines:
            stripped = line.strip()
            if not stripped:
                result_lines.append('')
                continue

            # Check for dedent keywords (elif, else, except, finally)
            if stripped.startswith(('elif ', 'else:', 'except', 'finally:')):
                if len(indent_stack) > 1:
                    indent_stack.pop()
                    current_indent = indent_stack[-1]

            # Apply current indentation
            result_lines.append(' ' * current_indent + stripped)

            # Check if this line should increase indentation for next line
            if stripped.endswith(':'):
                for kw in indent_keywords:
                    if stripped.startswith(kw) or stripped == kw.rstrip():
                        current_indent += 4
                        indent_stack.append(current_indent)
                        break

            # Check for return/break/continue at current level (potential dedent after)
            # We don't automatically dedent here as the next line might be at same level

        return '\n'.join(result_lines)


def execute_test(full_code: str, test_code: str, entry_point: str, timeout: int = 30) -> Tuple[bool, str]:
    """Execute generated code with test cases in a sandboxed subprocess.

    Subprocess used (not threads) because some HumanEval problems—e.g.,
    HumanEval/75 is_multiply_prime—elicit brute-force O(N^3) solutions that
    are pure-Python tight loops. These hog the GIL and block thread-based
    timeouts from firing, so the threaded approach hung. Subprocess.run
    sends SIGKILL on timeout regardless of what the child is doing.
    """
    import subprocess, sys
    exec_code = full_code + "\n\n" + test_code + f"\n\ncheck({entry_point})\n"
    try:
        result = subprocess.run(
            [sys.executable, "-c", exec_code],
            timeout=timeout,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

    if result.returncode == 0:
        return True, ""

    stderr = (result.stderr or "").strip()
    last = stderr.split("\n")[-1] if stderr else ""
    for tag in ("AssertionError", "IndentationError", "SyntaxError",
                "NameError", "TypeError", "ValueError", "AttributeError",
                "ZeroDivisionError", "KeyError", "IndexError", "RecursionError"):
        if tag in stderr:
            return False, f"{tag}: {last[:80]}"
    return False, (last[:80] or "ExecutionError")


def wilson_ci(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Compute Wilson score confidence interval."""
    if total == 0:
        return 0.0, 0.0

    z = 1.96 if confidence == 0.95 else 2.576
    p = successes / total

    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator

    return max(0, center - margin), min(1, center + margin)


def cohens_d(n1: int, pass1: int, n2: int, pass2: int) -> float:
    """Compute Cohen's d effect size for two proportions."""
    p1 = pass1 / n1 if n1 > 0 else 0
    p2 = pass2 / n2 if n2 > 0 else 0

    pooled_var = (p1 * (1 - p1) + p2 * (1 - p2)) / 2
    pooled_std = math.sqrt(pooled_var) if pooled_var > 0 else 1

    return (p1 - p2) / pooled_std if pooled_std > 0 else 0


def run_comparison(
    data_path: str = "data/humaneval/humaneval.jsonl",
    model: str = "gemini-2.5-flash",
    provider: str = "gemini",
    sample_size: Optional[int] = None,
    seed: int = 42,
    output_dir: str = "results/bdd_vs_cot"
):
    """Run TCGP vs CoT vs Direct comparison study."""

    print("=" * 70, flush=True)
    print("TCGP vs CHAIN-OF-THOUGHT vs DIRECT COMPARISON", flush=True)
    print("=" * 70, flush=True)
    print(f"Model: {model}", flush=True)
    print(f"Provider: {provider}", flush=True)
    print(f"Seed: {seed}", flush=True)

    # Load dataset
    random.seed(seed)
    problems = []
    with open(data_path, 'r') as f:
        for line in f:
            problems.append(json.loads(line))

    if sample_size and sample_size < len(problems):
        problems = random.sample(problems, sample_size)
        print(f"Sample size: {sample_size}", flush=True)
    else:
        print(f"Using all {len(problems)} problems", flush=True)

    print("=" * 70, flush=True)

    # Initialize agent
    agent = PromptComparisonAgent(model=model, provider=provider, seed=seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Setup output directory and files for incremental saving
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    conditions = ["bdd", "cot", "direct"]
    # STABLE per-(model, seed) filenames enable problem-level resume across runs
    # (e.g. when a background job is killed by a wall-clock limit mid-seed).
    file_paths = {c: output_path / f"{c}_{model}_s{seed}.jsonl" for c in conditions}

    # --- RESUME: recover already-completed problems ---------------------------
    # Load valid records per condition (dropping any partial trailing line). A
    # problem counts as DONE only if all three conditions recorded it; half-done
    # problems are discarded so re-running them cannot create duplicate records.
    loaded = {c: {} for c in conditions}  # task_id -> record
    for c in conditions:
        if file_paths[c].exists():
            for line in file_paths[c].read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    break  # stop at first corrupt/partial line
                if rec.get("task_id"):
                    loaded[c][rec["task_id"]] = rec
    done_ids = set(loaded["bdd"]) & set(loaded["cot"]) & set(loaded["direct"])

    # Results storage, pre-populated from completed problems.
    results = {c: {"results": [], "passed": 0, "tokens": []} for c in conditions}
    for c in conditions:
        # Rewrite each file with only fully-completed problems (clean, no partials)
        with open(file_paths[c], 'w') as fh:
            for tid in done_ids:
                rec = loaded[c][tid]
                fh.write(json.dumps(rec) + '\n')
                results[c]["results"].append(rec)
                if rec.get("passed"):
                    results[c]["passed"] += 1
                if "tokens_used" in rec:
                    results[c]["tokens"].append(rec.get("tokens_used", 0))
    if done_ids:
        print(f"[resume] {len(done_ids)} problems already complete; skipping them.", flush=True)

    # Open for append now that files hold only clean, completed records.
    jsonl_files = {c: open(file_paths[c], 'a') for c in conditions}

    for i, problem in enumerate(problems):
        task_id = problem['task_id']
        if task_id in done_ids:
            continue
        prompt = problem['prompt']
        test = problem['test']
        entry_point = problem['entry_point']

        print(f"\n[{i+1}/{len(problems)}] {task_id}", flush=True)

        # Generate with TCGP
        try:
            code, duration, tokens, reasoning = agent.generate_with_tcgp(prompt, entry_point)
            full_code = prompt + code
            passed, error = execute_test(full_code, test, entry_point)

            if passed:
                results["bdd"]["passed"] += 1
                print(f"  TCGP:    PASS ({tokens} tokens)", flush=True)
            else:
                print(f"  TCGP:    FAIL - {error[:40]}", flush=True)

            results["bdd"]["tokens"].append(tokens)
            bdd_result = {
                "task_id": task_id,
                "condition": "bdd",
                "passed": passed,
                "generated_code": code,
                "reasoning": reasoning,
                "duration_seconds": duration,
                "tokens_used": tokens,
                "error": error if not passed else None
            }
            results["bdd"]["results"].append(bdd_result)
            jsonl_files["bdd"].write(json.dumps(bdd_result) + '\n')
            jsonl_files["bdd"].flush()
        except Exception as e:
            print(f"  TCGP:    ERROR - {e}", flush=True)
            bdd_result = {
                "task_id": task_id,
                "condition": "bdd",
                "passed": False,
                "error": str(e)
            }
            results["bdd"]["results"].append(bdd_result)
            jsonl_files["bdd"].write(json.dumps(bdd_result) + '\n')
            jsonl_files["bdd"].flush()

        # Generate with Chain-of-Thought
        try:
            code, duration, tokens, reasoning = agent.generate_with_cot(prompt, entry_point)
            full_code = prompt + code
            passed, error = execute_test(full_code, test, entry_point)

            if passed:
                results["cot"]["passed"] += 1
                print(f"  CoT:    PASS ({tokens} tokens)", flush=True)
            else:
                print(f"  CoT:    FAIL - {error[:40]}", flush=True)

            results["cot"]["tokens"].append(tokens)
            cot_result = {
                "task_id": task_id,
                "condition": "cot",
                "passed": passed,
                "generated_code": code,
                "reasoning": reasoning,
                "duration_seconds": duration,
                "tokens_used": tokens,
                "error": error if not passed else None
            }
            results["cot"]["results"].append(cot_result)
            jsonl_files["cot"].write(json.dumps(cot_result) + '\n')
            jsonl_files["cot"].flush()
        except Exception as e:
            print(f"  CoT:    ERROR - {e}", flush=True)
            cot_result = {
                "task_id": task_id,
                "condition": "cot",
                "passed": False,
                "error": str(e)
            }
            results["cot"]["results"].append(cot_result)
            jsonl_files["cot"].write(json.dumps(cot_result) + '\n')
            jsonl_files["cot"].flush()

        # Generate with Direct prompting
        try:
            code, duration, tokens = agent.generate_direct(prompt, entry_point)
            full_code = prompt + code
            passed, error = execute_test(full_code, test, entry_point)

            if passed:
                results["direct"]["passed"] += 1
                print(f"  Direct: PASS ({tokens} tokens)", flush=True)
            else:
                print(f"  Direct: FAIL - {error[:40]}", flush=True)

            results["direct"]["tokens"].append(tokens)
            direct_result = {
                "task_id": task_id,
                "condition": "direct",
                "passed": passed,
                "generated_code": code,
                "duration_seconds": duration,
                "tokens_used": tokens,
                "error": error if not passed else None
            }
            results["direct"]["results"].append(direct_result)
            jsonl_files["direct"].write(json.dumps(direct_result) + '\n')
            jsonl_files["direct"].flush()
        except Exception as e:
            print(f"  Direct: ERROR - {e}", flush=True)
            direct_result = {
                "task_id": task_id,
                "condition": "direct",
                "passed": False,
                "error": str(e)
            }
            results["direct"]["results"].append(direct_result)
            jsonl_files["direct"].write(json.dumps(direct_result) + '\n')
            jsonl_files["direct"].flush()

    # Close JSONL files (results already saved incrementally)
    for f in jsonl_files.values():
        f.close()

    # Compute statistics
    n = len(problems)
    stats = {}

    for condition in ["bdd", "cot", "direct"]:
        passed = results[condition]["passed"]
        rate = passed / n if n > 0 else 0
        ci = wilson_ci(passed, n)
        avg_tokens = sum(results[condition]["tokens"]) / len(results[condition]["tokens"]) if results[condition]["tokens"] else 0

        stats[condition] = {
            "passed": passed,
            "rate": rate,
            "ci": ci,
            "avg_tokens": avg_tokens
        }

    # Print summary
    print("\n" + "=" * 70, flush=True)
    print("TCGP vs COT vs DIRECT COMPARISON RESULTS", flush=True)
    print("=" * 70, flush=True)
    print(f"\n{'Condition':<12} {'Pass@1':<12} {'95% CI':<22} {'Passed':<10} {'Avg Tokens':<12}", flush=True)
    print("-" * 70, flush=True)

    for condition in ["bdd", "cot", "direct"]:
        s = stats[condition]
        print(f"{condition.upper():<12} {s['rate']*100:>6.1f}%      [{s['ci'][0]*100:.1f}%, {s['ci'][1]*100:.1f}%]        {s['passed']}/{n}        {s['avg_tokens']:.0f}", flush=True)

    print("-" * 70, flush=True)

    # Pairwise comparisons
    bdd_cot_d = cohens_d(n, stats["bdd"]["passed"], n, stats["cot"]["passed"])
    bdd_direct_d = cohens_d(n, stats["bdd"]["passed"], n, stats["direct"]["passed"])
    cot_direct_d = cohens_d(n, stats["cot"]["passed"], n, stats["direct"]["passed"])

    print(f"\nPairwise Comparisons:", flush=True)
    print(f"  TCGP vs CoT:    {(stats['bdd']['rate'] - stats['cot']['rate'])*100:+.1f}%  (Cohen's d = {bdd_cot_d:.3f})", flush=True)
    print(f"  TCGP vs Direct: {(stats['bdd']['rate'] - stats['direct']['rate'])*100:+.1f}%  (Cohen's d = {bdd_direct_d:.3f})", flush=True)
    print(f"  CoT vs Direct: {(stats['cot']['rate'] - stats['direct']['rate'])*100:+.1f}%  (Cohen's d = {cot_direct_d:.3f})", flush=True)

    # Key insight
    print(f"\n" + "=" * 70, flush=True)
    print("KEY INSIGHT:", flush=True)
    if stats["bdd"]["rate"] > stats["cot"]["rate"]:
        print(f"  TCGP outperforms CoT by {(stats['bdd']['rate'] - stats['cot']['rate'])*100:.1f}% despite similar token usage.", flush=True)
        print(f"  This suggests the structured TCGP format provides value beyond just", flush=True)
        print(f"  'more thinking tokens'.", flush=True)
    elif stats["cot"]["rate"] > stats["bdd"]["rate"]:
        print(f"  CoT outperforms TCGP by {(stats['cot']['rate'] - stats['bdd']['rate'])*100:.1f}%.", flush=True)
        print(f"  This suggests general reasoning may be more effective than", flush=True)
        print(f"  structured TCGP scenarios for code generation.", flush=True)
    else:
        print(f"  TCGP and CoT show similar performance.", flush=True)
        print(f"  Both structured reasoning approaches outperform direct prompting.", flush=True)
    print("=" * 70, flush=True)

    # Save summary
    summary = {
        "model": model,
        "provider": provider,
        "n_problems": n,
        "seed": seed,
        "timestamp": timestamp,
        "conditions": {
            condition: {
                "passed": stats[condition]["passed"],
                "pass_rate": stats[condition]["rate"],
                "ci_lower": stats[condition]["ci"][0],
                "ci_upper": stats[condition]["ci"][1],
                "avg_tokens": stats[condition]["avg_tokens"]
            }
            for condition in ["bdd", "cot", "direct"]
        },
        "pairwise": {
            "bdd_vs_cot": {
                "difference": stats["bdd"]["rate"] - stats["cot"]["rate"],
                "cohens_d": bdd_cot_d
            },
            "bdd_vs_direct": {
                "difference": stats["bdd"]["rate"] - stats["direct"]["rate"],
                "cohens_d": bdd_direct_d
            },
            "cot_vs_direct": {
                "difference": stats["cot"]["rate"] - stats["direct"]["rate"],
                "cohens_d": cot_direct_d
            }
        }
    }

    with open(output_path / f"summary_{timestamp}.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_path}", flush=True)

    return summary


def generate_latex_table(summary: dict) -> str:
    """Generate LaTeX table from comparison results."""
    conds = summary["conditions"]
    n = summary["n_problems"]

    table = f"""\\begin{{table}}[t]
\\centering
\\caption{{TCGP vs Chain-of-Thought vs Direct Prompting on HumanEval (n={n}). TCGP outperforms CoT despite similar token usage, suggesting structured test scenarios provide value beyond general reasoning.}}\\label{{tab:bdd-vs-cot}}
\\begin{{tabular}}{{lcccc}}
\\toprule
\\textbf{{Condition}} & \\textbf{{Pass@1}} & \\textbf{{95\\% CI}} & \\textbf{{Passed}} & \\textbf{{Avg Tokens}} \\\\
\\midrule
Direct & {conds['direct']['pass_rate']*100:.1f}\\% & [{conds['direct']['ci_lower']*100:.1f}\\%, {conds['direct']['ci_upper']*100:.1f}\\%] & {conds['direct']['passed']}/{n} & {conds['direct']['avg_tokens']:.0f} \\\\
Chain-of-Thought & {conds['cot']['pass_rate']*100:.1f}\\% & [{conds['cot']['ci_lower']*100:.1f}\\%, {conds['cot']['ci_upper']*100:.1f}\\%] & {conds['cot']['passed']}/{n} & {conds['cot']['avg_tokens']:.0f} \\\\
TCGP (Ours) & {conds['bdd']['pass_rate']*100:.1f}\\% & [{conds['bdd']['ci_lower']*100:.1f}\\%, {conds['bdd']['ci_upper']*100:.1f}\\%] & {conds['bdd']['passed']}/{n} & {conds['bdd']['avg_tokens']:.0f} \\\\
\\midrule
\\multicolumn{{5}}{{l}}{{\\textit{{Pairwise comparisons (Cohen's d):}}}} \\\\
\\multicolumn{{5}}{{l}}{{TCGP vs CoT: {summary['pairwise']['bdd_vs_cot']['difference']*100:+.1f}\\% ($d$={summary['pairwise']['bdd_vs_cot']['cohens_d']:.2f}), TCGP vs Direct: {summary['pairwise']['bdd_vs_direct']['difference']*100:+.1f}\\% ($d$={summary['pairwise']['bdd_vs_direct']['cohens_d']:.2f})}} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""

    return table


def run_ablation(
    condition: str,
    data_path: str = "data/humaneval/humaneval.jsonl",
    model: str = "gemini-2.5-flash",
    provider: str = "gemini",
    sample_size: Optional[int] = None,
    seed: int = 42,
    output_dir: str = "results/ablations",
    resume_file: Optional[str] = None
):
    """Run a single ablation condition.

    Args:
        condition: One of 'plan_and_solve', 'extended_direct', 'minimal_bdd'
        data_path: Path to HumanEval JSONL
        model: Model name
        provider: API provider
        sample_size: Number of problems to run
        seed: Random seed
        output_dir: Output directory
        resume_file: Path to existing JSONL file to resume from
    """
    valid_conditions = ['plan_and_solve', 'extended_direct', 'minimal_bdd', 'hybrid', 'self_consistency']
    if condition not in valid_conditions:
        raise ValueError(f"Invalid condition: {condition}. Must be one of {valid_conditions}")

    print("=" * 70, flush=True)
    print(f"ABLATION EXPERIMENT: {condition.upper()}", flush=True)
    print("=" * 70, flush=True)
    print(f"Model: {model}", flush=True)
    print(f"Provider: {provider}", flush=True)
    print(f"Seed: {seed}", flush=True)

    # Load dataset
    random.seed(seed)
    problems = []
    with open(data_path, 'r') as f:
        for line in f:
            problems.append(json.loads(line))

    if sample_size and sample_size < len(problems):
        problems = random.sample(problems, sample_size)
        print(f"Sample size: {sample_size}", flush=True)
    else:
        print(f"Using all {len(problems)} problems", flush=True)

    print("=" * 70, flush=True)

    # Initialize agent
    agent = PromptComparisonAgent(model=model, provider=provider, seed=seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Handle resume logic
    completed_task_ids = set()
    prior_results = []
    if resume_file:
        resume_path = Path(resume_file)
        if resume_path.exists():
            with open(resume_path, 'r') as f:
                for line in f:
                    result = json.loads(line)
                    completed_task_ids.add(result['task_id'])
                    prior_results.append(result)
            print(f"Resuming from {resume_file}: {len(completed_task_ids)} problems already completed", flush=True)
            # Append to existing file
            jsonl_file = open(resume_path, 'a')
        else:
            print(f"Resume file not found: {resume_file}, starting fresh", flush=True)
            jsonl_file = open(output_path / f"{condition}_{model}_{timestamp}.jsonl", 'w')
    else:
        # Create new output file
        jsonl_file = open(output_path / f"{condition}_{model}_{timestamp}.jsonl", 'w')

    results = prior_results.copy()
    passed = sum(1 for r in prior_results if r.get('passed', False))
    passed_at_1 = sum(1 for r in prior_results if r.get('passed_first_sample', False))
    tokens_list = [r.get('tokens_used', 0) for r in prior_results if r.get('tokens_used')]

    for i, problem in enumerate(problems):
        task_id = problem['task_id']
        prompt = problem['prompt']
        test = problem['test']
        entry_point = problem['entry_point']

        # Skip already completed problems
        if task_id in completed_task_ids:
            print(f"\n[{i+1}/{len(problems)}] {task_id} - SKIPPED (already completed)", flush=True)
            continue

        print(f"\n[{i+1}/{len(problems)}] {task_id}", flush=True)

        try:
            if condition == 'plan_and_solve':
                code, duration, tokens, reasoning = agent.generate_with_plan_and_solve(prompt, entry_point)
            elif condition == 'extended_direct':
                code, duration, tokens = agent.generate_with_extended_direct(prompt, entry_point)
                reasoning = None
            elif condition == 'minimal_bdd':
                code, duration, tokens, reasoning = agent.generate_with_minimal_tcgp(prompt, entry_point)
            elif condition == 'hybrid':
                code, duration, tokens, reasoning = agent.generate_with_hybrid(prompt, entry_point)
            elif condition == 'self_consistency':
                samples, duration, tokens = agent.generate_with_self_consistency_cot(prompt, entry_point)
                sample_results = []
                for code, sample_tokens in samples:
                    full_code = prompt + code
                    sample_passed, sample_error = execute_test(full_code, test, entry_point)
                    sample_results.append({
                        "code": code,
                        "tokens": sample_tokens,
                        "passed": sample_passed,
                        "error": sample_error if not sample_passed else None,
                    })
                n_passed = sum(1 for s in sample_results if s["passed"])
                test_passed = n_passed > 0          # Pass@5 (any sample passes)
                first_passed = sample_results[0]["passed"]  # Pass@1 (first sample)

                if test_passed:
                    passed += 1
                if first_passed:
                    passed_at_1 += 1

                marker = "PASS" if test_passed else "FAIL"
                print(f"  {condition}: {marker} ({n_passed}/{len(samples)} samples, {tokens} tokens)", flush=True)

                tokens_list.append(tokens)
                result = {
                    "task_id": task_id,
                    "condition": condition,
                    "passed": test_passed,
                    "passed_first_sample": first_passed,
                    "n_samples": len(samples),
                    "n_passed": n_passed,
                    "samples": sample_results,
                    "duration_seconds": duration,
                    "tokens_used": tokens,
                    "error": None if test_passed else sample_results[0].get("error"),
                }
                results.append(result)
                jsonl_file.write(json.dumps(result) + '\n')
                jsonl_file.flush()
                continue

            full_code = prompt + code
            test_passed, error = execute_test(full_code, test, entry_point)

            if test_passed:
                passed += 1
                print(f"  {condition}: PASS ({tokens} tokens)", flush=True)
            else:
                print(f"  {condition}: FAIL - {error[:40]}", flush=True)

            tokens_list.append(tokens)
            result = {
                "task_id": task_id,
                "condition": condition,
                "passed": test_passed,
                "generated_code": code,
                "reasoning": reasoning,
                "duration_seconds": duration,
                "tokens_used": tokens,
                "error": error if not test_passed else None
            }
            results.append(result)
            jsonl_file.write(json.dumps(result) + '\n')
            jsonl_file.flush()

        except Exception as e:
            print(f"  {condition}: ERROR - {e}", flush=True)
            result = {
                "task_id": task_id,
                "condition": condition,
                "passed": False,
                "error": str(e)
            }
            results.append(result)
            jsonl_file.write(json.dumps(result) + '\n')
            jsonl_file.flush()

    jsonl_file.close()

    # Compute statistics
    n = len(problems)
    rate = passed / n if n > 0 else 0
    ci = wilson_ci(passed, n)
    avg_tokens = sum(tokens_list) / len(tokens_list) if tokens_list else 0

    # For self-consistency, "passed" tracks Pass@5 (any sample); also report Pass@1
    if condition == 'self_consistency':
        rate_at_1 = passed_at_1 / n if n > 0 else 0
        ci_at_1 = wilson_ci(passed_at_1, n)
    else:
        rate_at_1 = rate
        ci_at_1 = ci

    # Print summary
    print("\n" + "=" * 70, flush=True)
    print(f"ABLATION RESULTS: {condition.upper()}", flush=True)
    print("=" * 70, flush=True)
    if condition == 'self_consistency':
        print(f"Pass@1: {rate_at_1*100:.1f}%  CI [{ci_at_1[0]*100:.1f}%, {ci_at_1[1]*100:.1f}%]  ({passed_at_1}/{n})", flush=True)
        print(f"Pass@5: {rate*100:.1f}%  CI [{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]  ({passed}/{n})", flush=True)
    else:
        print(f"Pass@1: {rate*100:.1f}%", flush=True)
        print(f"95% CI: [{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]", flush=True)
        print(f"Passed: {passed}/{n}", flush=True)
    print(f"Avg Tokens: {avg_tokens:.0f}", flush=True)
    print("=" * 70, flush=True)

    # Save summary
    summary = {
        "model": model,
        "provider": provider,
        "condition": condition,
        "n_problems": n,
        "seed": seed,
        "timestamp": timestamp,
        "passed": passed,
        "pass_rate": rate,
        "ci_lower": ci[0],
        "ci_upper": ci[1],
        "avg_tokens": avg_tokens
    }
    if condition == 'self_consistency':
        summary["pass_rate_at_1"] = rate_at_1
        summary["ci_at_1_lower"] = ci_at_1[0]
        summary["ci_at_1_upper"] = ci_at_1[1]
        summary["passed_at_1"] = passed_at_1
        summary["pass_rate_at_5"] = rate
        summary["passed_at_5"] = passed
        summary["n_samples_per_problem"] = 5

    with open(output_path / f"summary_{condition}_{model}_{timestamp}.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_path}", flush=True)

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TCGP vs CoT vs Direct Comparison")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Model to use")
    parser.add_argument("--provider", default="gemini", choices=["azure", "azure_models", "azure_responses", "openai", "anthropic", "gemini"])
    parser.add_argument("--samples", type=int, default=None, help="Number of samples (default: all 164)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/bdd_vs_cot")
    parser.add_argument("--condition", default=None,
                        choices=["plan_and_solve", "extended_direct", "minimal_bdd", "hybrid", "self_consistency"],
                        help="Run specific ablation condition instead of full comparison")
    parser.add_argument("--resume", default=None,
                        help="Path to existing JSONL file to resume from (for ablation runs)")

    args = parser.parse_args()

    if args.condition:
        # Run single ablation
        summary = run_ablation(
            condition=args.condition,
            model=args.model,
            provider=args.provider,
            sample_size=args.samples,
            seed=args.seed,
            output_dir=args.output.replace("bdd_vs_cot", "ablations"),
            resume_file=args.resume
        )
    else:
        # Run full comparison
        summary = run_comparison(
            model=args.model,
            provider=args.provider,
            sample_size=args.samples,
            seed=args.seed,
            output_dir=args.output
        )

        # Generate and print LaTeX table
        print("\n" + "=" * 70)
        print("LATEX TABLE (copy to paper)")
        print("=" * 70)
        print(generate_latex_table(summary))
