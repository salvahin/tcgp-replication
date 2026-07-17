#!/usr/bin/env python3
"""
LiveCodeBench evaluation: TCGP vs CoT vs Direct for competitive programming.

LiveCodeBench is a contamination-free benchmark with 500+ problems from
LeetCode, AtCoder, and Codeforces (2023-2024).
"""

import json
import os
import re
import time
import signal
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

RESULTS_DIR = Path(__file__).parent / "results" / "livecodebench"


class TimeoutError(Exception):
    pass


@contextmanager
def time_limit(seconds):
    """Simple timeout using signal."""
    def signal_handler(signum, frame):
        raise TimeoutError("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def create_tcgp_prompt(question: str, starter_code: str = "") -> str:
    """Create TCGP-style prompt for code generation."""
    prompt = f"""You are solving a competitive programming problem. First, analyze the requirements using TCGP scenarios, then implement the solution.

## Problem:
{question}

## TCGP Analysis:
Before coding, identify:
1. Given: What inputs/constraints are provided?
2. When: What operation needs to be performed?
3. Then: What output is expected?

Consider edge cases:
- Empty inputs
- Single element
- Maximum constraints
- Boundary values

## Instructions:
1. Write TCGP scenarios for the key test cases
2. Then implement the complete solution
3. Your code must read from stdin and print to stdout
4. Return ONLY the complete Python solution

{f"## Starter Code:{chr(10)}```python{chr(10)}{starter_code}{chr(10)}```{chr(10)}" if starter_code else ""}

```python
"""
    return prompt


def create_cot_prompt(question: str, starter_code: str = "") -> str:
    """Create Chain-of-Thought prompt for code generation."""
    prompt = f"""You are solving a competitive programming problem. Think step by step about the solution.

## Problem:
{question}

## Instructions:
Think through the problem step by step:
1. What is the input format?
2. What is the expected output?
3. What algorithm or approach should be used?
4. What are the edge cases to handle?
5. What is the time complexity?

After thinking through the approach, implement the complete solution.
Your code must read from stdin and print to stdout.
Return ONLY the complete Python solution.

{f"## Starter Code:{chr(10)}```python{chr(10)}{starter_code}{chr(10)}```{chr(10)}" if starter_code else ""}

Let me think step by step...

```python
"""
    return prompt


def create_direct_prompt(question: str, starter_code: str = "") -> str:
    """Create direct prompt for code generation."""
    prompt = f"""Solve the following competitive programming problem:

## Problem:
{question}

{f"## Starter Code:{chr(10)}```python{chr(10)}{starter_code}{chr(10)}```{chr(10)}" if starter_code else ""}

Your code must read from stdin and print to stdout.
Return ONLY the complete Python solution:

```python
"""
    return prompt


def extract_code(response: str) -> str:
    """Extract Python code from LLM response."""
    # Try to find code between ```python and ```
    matches = re.findall(r'```python\s*(.*?)```', response, re.DOTALL)
    if matches:
        return max(matches, key=len).strip()

    # Try without language specifier
    matches = re.findall(r'```\s*(.*?)```', response, re.DOTALL)
    if matches:
        return max(matches, key=len).strip()

    # Handle truncated responses
    if '```python' in response:
        code = response.split('```python', 1)[1]
        if '```' in code:
            code = code.split('```')[0]
        return code.strip()

    if '```' in response:
        code = response.split('```', 1)[1]
        if code.startswith(('python\n', 'Python\n')):
            code = code.split('\n', 1)[1] if '\n' in code else code
        if '```' in code:
            code = code.split('```')[0]
        return code.strip()

    return response.strip()


def run_test(code: str, test_input: str, expected_output: str, timeout: int = 10) -> tuple:
    """
    Run code with test input and compare to expected output.
    Returns (passed: bool, actual_output: str, error: str or None)
    """
    # Create temp file with the code
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_file = f.name

    try:
        # Run the code with input
        result = subprocess.run(
            ['python3', temp_file],
            input=test_input,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        if result.returncode != 0:
            return False, "", f"Runtime error: {result.stderr[:200]}"

        actual = result.stdout.strip()
        expected = expected_output.strip()

        # Compare outputs (normalize whitespace)
        actual_lines = [line.strip() for line in actual.split('\n') if line.strip()]
        expected_lines = [line.strip() for line in expected.split('\n') if line.strip()]

        passed = actual_lines == expected_lines
        return passed, actual, None if passed else f"Expected: {expected[:100]}, Got: {actual[:100]}"

    except subprocess.TimeoutExpired:
        return False, "", "timeout"
    except Exception as e:
        return False, "", f"{type(e).__name__}: {str(e)[:100]}"
    finally:
        os.unlink(temp_file)


def get_client(provider: str):
    """Get API client based on provider."""
    if provider == "azure":
        from openai import OpenAI
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
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
        # For Azure responses API (gpt-5.3-codex, Llama, codex-mini)
        return None  # We'll use requests directly
    elif provider == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        return genai
    elif provider == "anthropic":
        import anthropic
        return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    else:
        raise ValueError(f"Unknown provider: {provider}")


def call_llm(prompt: str, model: str, provider: str, max_tokens: int = 4000) -> tuple:
    """Call LLM and return (response, tokens, duration)."""
    client = get_client(provider)
    start = time.time()

    if provider in ("azure", "azure_models"):
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=max_tokens
        )
        duration = time.time() - start
        content = response.choices[0].message.content
        tokens = response.usage.total_tokens if response.usage else 0
    elif provider == "azure_responses":
        import requests
        endpoint = os.getenv("AZURE_RESPONSES_ENDPOINT", "https://chalan-resource.cognitiveservices.azure.com")
        api_key = os.getenv("AZURE_RESPONSES_KEY")
        url = f"{endpoint}/openai/responses?api-version=2025-04-01-preview"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        data = {
            "input": prompt,
            "max_output_tokens": max_tokens,
            "model": model
        }
        resp = requests.post(url, headers=headers, json=data, timeout=300)
        resp.raise_for_status()
        result = resp.json()
        duration = time.time() - start
        # Extract text from output[0].content[0].text
        output = result.get("output", [])
        if output and len(output) > 0:
            content_list = output[0].get("content", [])
            if content_list and len(content_list) > 0:
                content = content_list[0].get("text", "")
            else:
                content = ""
        else:
            content = ""
        tokens = result.get("usage", {}).get("total_tokens", 0)
    elif provider == "gemini":
        gen_model = client.GenerativeModel(model)
        response = gen_model.generate_content(
            prompt,
            generation_config=client.types.GenerationConfig(
                temperature=0.2,
                max_output_tokens=max_tokens
            )
        )
        duration = time.time() - start
        content = response.text
        tokens = response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else 0
    elif provider == "anthropic":
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        duration = time.time() - start
        content = response.content[0].text
        tokens = response.usage.input_tokens + response.usage.output_tokens
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return content, tokens, duration


def evaluate_problem(problem: dict, condition: str, model: str, provider: str) -> dict:
    """Evaluate a single LiveCodeBench problem."""
    question = problem.get('question_content', problem.get('question', ''))
    starter_code = problem.get('starter_code', '')

    # Get test cases
    public_tests = problem.get('public_test_cases', [])
    if isinstance(public_tests, str):
        try:
            public_tests = json.loads(public_tests)
        except:
            public_tests = []

    # Create prompt based on condition
    if condition == "bdd":
        prompt = create_tcgp_prompt(question, starter_code)
    elif condition == "cot":
        prompt = create_cot_prompt(question, starter_code)
    else:  # direct
        prompt = create_direct_prompt(question, starter_code)

    # Call LLM
    try:
        response, tokens, duration = call_llm(prompt, model, provider)
        code = extract_code(response)
        error = None
    except Exception as e:
        response = ""
        code = ""
        tokens = 0
        duration = 0
        error = f"API error: {str(e)}"

    # Run tests
    passed = False
    test_results = []

    if code and not error and public_tests:
        all_passed = True
        for test in public_tests[:3]:  # Limit to first 3 public tests
            test_input = test.get('input', '')
            expected_output = test.get('output', test.get('expected_output', ''))

            if test_input and expected_output:
                test_passed, actual, test_error = run_test(code, test_input, expected_output)
                test_results.append({
                    "passed": test_passed,
                    "input": test_input[:100],
                    "expected": expected_output[:100],
                    "actual": actual[:100] if actual else "",
                    "error": test_error
                })
                if not test_passed:
                    all_passed = False
                    if not error:
                        error = test_error

        passed = all_passed and len(test_results) > 0

    return {
        "question_id": problem.get('question_id', problem.get('id', 'unknown')),
        "title": problem.get('question_title', problem.get('title', 'unknown')),
        "difficulty": problem.get('difficulty', 'unknown'),
        "condition": condition,
        "passed": passed,
        "generated_code": code[:2000],  # Truncate for storage
        "tokens_used": tokens,
        "duration_seconds": duration,
        "error": error,
        "n_tests": len(test_results),
        "tests_passed": sum(1 for t in test_results if t["passed"])
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run LiveCodeBench TCGP vs CoT evaluation")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model to use")
    parser.add_argument("--provider", type=str, default="azure",
                        choices=["azure", "azure_models", "azure_responses", "gemini", "anthropic"],
                        help="API provider")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of problems")
    parser.add_argument("--start", type=int, default=0, help="Start from problem index (for resuming)")
    parser.add_argument("--conditions", nargs="+", default=["bdd", "cot", "direct"],
                        help="Conditions to evaluate")
    parser.add_argument("--version", type=str, default="release_v2",
                        help="Dataset version tag")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading LiveCodeBench dataset...")
    # Use bzantium/livecodebench which is compatible with modern datasets library
    dataset = load_dataset("bzantium/livecodebench", split="test")

    print(f"Loaded {len(dataset)} problems")

    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
        print(f"Limited to {len(dataset)} problems")

    # Apply start offset
    if args.start > 0:
        dataset = dataset.select(range(args.start, len(dataset)))
        print(f"Starting from problem {args.start}, {len(dataset)} problems remaining")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create output files (append mode if resuming)
    output_files = {}
    file_mode = 'a' if args.start > 0 else 'w'
    for condition in args.conditions:
        filepath = RESULTS_DIR / f"{condition}_{args.model}_{timestamp}.jsonl"
        output_files[condition] = open(filepath, file_mode)

    results = {cond: {"passed": 0, "total": 0} for cond in args.conditions}

    print(f"\nRunning evaluation on {args.model}...")
    print(f"Conditions: {args.conditions}")
    print("-" * 60)

    for i, problem in enumerate(dataset):
        qid = problem.get('question_id', problem.get('id', f'prob_{i}'))
        title = problem.get('question_title', problem.get('title', 'Unknown'))[:40]
        difficulty = problem.get('difficulty', '?')

        print(f"\n[{i+1}/{len(dataset)}] {qid} - {title} ({difficulty})")

        for condition in args.conditions:
            print(f"  {condition}...", end=" ", flush=True)

            result = evaluate_problem(problem, condition, args.model, args.provider)

            # Save result
            output_files[condition].write(json.dumps(result) + '\n')
            output_files[condition].flush()

            # Track stats
            results[condition]["total"] += 1
            if result["passed"]:
                results[condition]["passed"] += 1
                print("✓", end="")
            else:
                err_msg = result['error'][:25] if result['error'] else 'failed'
                print(f"✗ ({err_msg})", end="")
            print()

    # Close files
    for f in output_files.values():
        f.close()

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for condition in args.conditions:
        passed = results[condition]["passed"]
        total = results[condition]["total"]
        rate = passed / total * 100 if total > 0 else 0
        print(f"{condition:10} {passed}/{total} = {rate:.1f}%")

    # Save summary
    summary = {
        "model": args.model,
        "provider": args.provider,
        "timestamp": timestamp,
        "n_problems": len(dataset),
        "conditions": {
            cond: {
                "passed": results[cond]["passed"],
                "total": results[cond]["total"],
                "pass_rate": results[cond]["passed"] / results[cond]["total"] if results[cond]["total"] > 0 else 0
            }
            for cond in args.conditions
        }
    }

    summary_path = RESULTS_DIR / f"summary_{args.model}_{timestamp}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
