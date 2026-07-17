"""
Ablation Prompt Templates

Contains prompt templates for:
1. Plan-and-Solve CoT (improved CoT baseline)
2. Token-matched ablations (control for token budget)
3. Self-consistency CoT (optional)

These prompts address reviewer concerns about fair comparison.
"""

# =============================================================================
# PLAN-AND-SOLVE COT (Improved CoT Baseline)
# =============================================================================
# Reference: Wang et al. 2023, "Plan-and-Solve Prompting"
# This is a stronger CoT variant that explicitly structures the reasoning

PLAN_AND_SOLVE_STEP1 = """Let's approach this problem systematically using a plan-and-solve strategy.

{prompt}

## Step 1: Understand the Problem
- What are the inputs and their types?
- What is the expected output?
- What are the constraints?

## Step 2: Devise a Plan
- What algorithm or approach should we use?
- What are the key steps?

## Step 3: Consider Edge Cases
- What special inputs need handling?
- What could go wrong?

## Step 4: Plan Summary
Write a brief plan (2-3 sentences) for the implementation.

Provide your analysis (no code yet):
"""

PLAN_AND_SOLVE_STEP2 = """Complete this Python function following your plan.

{prompt}

Your plan:
{reasoning}

REQUIREMENTS:
1. Return ONLY the function body (the code that goes inside the function)
2. Do NOT include the function signature (no 'def' line)
3. Use proper 4-space indentation
4. Every code path must return a value

Return ONLY the implementation code, no explanations.
"""


# =============================================================================
# STRUCTURED COT (Alternative improved baseline)
# =============================================================================
# More structured than basic "think step by step"

STRUCTURED_COT_STEP1 = """Analyze this function systematically before implementing.

{prompt}

Provide a structured analysis:

1. INPUT ANALYSIS:
   - Parameter types and constraints
   - Example inputs from docstring

2. OUTPUT SPECIFICATION:
   - Return type
   - Expected output format

3. ALGORITHM CHOICE:
   - Best approach (brute force, divide-conquer, DP, etc.)
   - Time/space complexity consideration

4. EDGE CASES:
   - Empty/null inputs
   - Boundary values
   - Special cases

Provide your analysis:
"""

STRUCTURED_COT_STEP2 = PLAN_AND_SOLVE_STEP2  # Same code generation prompt


# =============================================================================
# TOKEN-MATCHED ABLATIONS
# =============================================================================

# Ablation 1: Extended Direct (adds "thinking space" tokens to Direct)
# Tests whether improvement comes from extra tokens, not methodology
EXTENDED_DIRECT = """Complete this Python function. Think carefully before coding.

{prompt}

Before implementing, briefly consider:
- What inputs does this function receive?
- What should it return?
- Any edge cases to handle?

[You may use up to 200 words to think through the problem]

Now provide the implementation.

REQUIREMENTS:
1. Return ONLY the function body (the code that goes inside the function)
2. Do NOT include the function signature (no 'def' line)
3. Use proper 4-space indentation
4. Every code path must return a value

Return ONLY the implementation code, no explanations.
"""


# Ablation 2: Minimal TCGP (reduces TCGP to match Direct token count)
# Tests whether TCGP's structure matters or just having test cases
MINIMAL_BDD_STEP1 = """Analyze this function and generate exactly 2 test scenarios.

{prompt}

Generate exactly 2 scenarios (one normal case, one edge case) with concrete values:

Scenario 1 (Normal):
  Input: [concrete value]
  Expected: [concrete result]

Scenario 2 (Edge):
  Input: [edge case value]
  Expected: [expected result]

Keep each scenario to one line. Use real values only.
"""

MINIMAL_BDD_STEP2 = """Complete this Python function. It must pass these test scenarios:

{prompt}

Test scenarios:
{bdd_scenarios}

REQUIREMENTS:
1. Return ONLY the function body
2. No function signature
3. Proper 4-space indentation
4. Handle the edge case

Return ONLY the implementation code.
"""


# Ablation: Hybrid (TCGP + CoT) — addresses methodology-fairness counterfactual.
# Same total token budget (1400) and same 2-step structure as TCGP and CoT,
# but step 1 asks for BOTH scenarios AND step-by-step reasoning.
HYBRID_STEP1 = """Analyze this Python function. Provide your analysis in two parts.

{prompt}

PART A: Generate 3-4 concrete test scenarios with exact input-output values.
- Use real values (not placeholders).
- Include normal cases and at least one edge case.
- Format each as: Input: <values>  ->  Expected: <result>

PART B: Reason step-by-step about the algorithm needed to satisfy the scenarios.
- Identify inputs, constraints, and the core algorithm.
- Note any edge cases that need explicit handling.

Provide PART A then PART B. Do not write code yet.
"""

HYBRID_STEP2 = """Complete this Python function. Your code must satisfy the test scenarios listed in PART A and follow the reasoning in PART B.

{prompt}

Scenarios and reasoning:
{combined}

REQUIREMENTS:
1. Return ONLY the function body (the code that goes inside the function)
2. Do NOT include the function signature (no 'def' line)
3. Use proper 4-space indentation
4. Every code path must return a value

Return ONLY the implementation code, no explanations.
"""


# Ablation 3: Truncated CoT (limits reasoning to match Direct)
# Tests whether CoT needs full token budget
TRUNCATED_COT_STEP1 = """Briefly analyze this function (max 100 words).

{prompt}

Quick analysis (be concise):
- Main goal:
- Key algorithm:
- Edge cases:
"""

TRUNCATED_COT_STEP2 = PLAN_AND_SOLVE_STEP2  # Same code generation


# =============================================================================
# SELF-CONSISTENCY COT (Optional - requires multiple samples)
# =============================================================================
# Reference: Wang et al. 2022, "Self-Consistency Improves Chain of Thought"
# Generate N solutions, pick majority answer

SELF_CONSISTENCY_COT = """Analyze this function and think step by step about how to implement it.

{prompt}

Think through:
1. What are the inputs and expected outputs?
2. What algorithm should be used?
3. What edge cases need handling?

Provide your reasoning, then output the implementation code in a fenced markdown block.

OUTPUT FORMAT:
First, write your step-by-step reasoning as plain text.
Then, output the function body inside a ```python ... ``` markdown code block.

REQUIREMENTS for the code block:
1. Wrap the code in ```python ... ``` markdown fences (mandatory)
2. Inside the fences, return ONLY the function body (no 'def' line)
3. Use proper 4-space indentation
4. Every code path must return a value

Example output structure:
[your reasoning here, multiple paragraphs]

```python
    [function body code here]
```

First reason, then provide the code in a ```python ... ``` block:
"""


# =============================================================================
# TOKEN BUDGET REFERENCE
# =============================================================================
# For fair comparison, all methods should use similar total tokens

TOKEN_BUDGETS = {
    # Original methods
    "bdd": {"step1": 600, "step2": 800, "total": 1400},
    "cot": {"step1": 600, "step2": 800, "total": 1400},
    "direct": {"total": 800},

    # Ablations
    "plan_and_solve": {"step1": 600, "step2": 800, "total": 1400},
    "structured_cot": {"step1": 600, "step2": 800, "total": 1400},
    "extended_direct": {"total": 1000},  # ~25% more than direct
    "minimal_bdd": {"step1": 200, "step2": 600, "total": 800},
    "truncated_cot": {"step1": 200, "step2": 600, "total": 800},
    "self_consistency": {"per_sample": 1000, "n_samples": 5, "total": 5000},
}


# =============================================================================
# EXPERIMENT CONFIGURATIONS
# =============================================================================

ABLATION_CONFIGS = {
    "E1_plan_and_solve": {
        "description": "Plan-and-Solve CoT (improved CoT baseline)",
        "step1_prompt": PLAN_AND_SOLVE_STEP1,
        "step2_prompt": PLAN_AND_SOLVE_STEP2,
        "step1_tokens": 600,
        "step2_tokens": 800,
        "temperature": {"step1": 0.3, "step2": 0.2},
        "models": ["gpt-4o", "gpt-35-turbo", "gemini-2.5-flash"],
        "priority": "ESSENTIAL",
    },
    "E2a_extended_direct": {
        "description": "Extended Direct (token-matched control)",
        "prompt": EXTENDED_DIRECT,
        "max_tokens": 1000,
        "temperature": 0.2,
        "models": ["gpt-4o", "gpt-35-turbo"],
        "priority": "ESSENTIAL",
    },
    "E2b_minimal_bdd": {
        "description": "Minimal TCGP (2 scenarios, token-matched)",
        "step1_prompt": MINIMAL_BDD_STEP1,
        "step2_prompt": MINIMAL_BDD_STEP2,
        "step1_tokens": 200,
        "step2_tokens": 600,
        "temperature": {"step1": 0.3, "step2": 0.2},
        "models": ["gpt-4o", "gpt-35-turbo"],
        "priority": "ESSENTIAL",
    },
    "E3_structured_cot": {
        "description": "Structured CoT (explicit categories)",
        "step1_prompt": STRUCTURED_COT_STEP1,
        "step2_prompt": STRUCTURED_COT_STEP2,
        "step1_tokens": 600,
        "step2_tokens": 800,
        "temperature": {"step1": 0.3, "step2": 0.2},
        "models": ["gpt-4o"],
        "priority": "MEDIUM",
    },
    "E4_self_consistency": {
        "description": "Self-Consistency CoT (5 samples, majority vote)",
        "prompt": SELF_CONSISTENCY_COT,
        "max_tokens": 1000,
        "n_samples": 5,
        "temperature": 0.7,  # Higher for diversity
        "models": ["gpt-4o"],
        "priority": "MEDIUM",
    },
}


def get_prompt_for_ablation(ablation_name: str, prompt: str, **kwargs) -> str:
    """Get formatted prompt for a specific ablation."""
    config = ABLATION_CONFIGS.get(ablation_name)
    if not config:
        raise ValueError(f"Unknown ablation: {ablation_name}")

    if "step1_prompt" in config:
        # Two-step method
        step = kwargs.get("step", 1)
        if step == 1:
            return config["step1_prompt"].format(prompt=prompt)
        else:
            reasoning = kwargs.get("reasoning", "")
            bdd_scenarios = kwargs.get("bdd_scenarios", "")
            return config["step2_prompt"].format(
                prompt=prompt,
                reasoning=reasoning,
                bdd_scenarios=bdd_scenarios
            )
    else:
        # Single-step method
        return config["prompt"].format(prompt=prompt)


if __name__ == "__main__":
    # Print summary of ablations
    print("Ablation Prompt Templates")
    print("=" * 60)

    for name, config in ABLATION_CONFIGS.items():
        print(f"\n{name}: {config['description']}")
        print(f"  Priority: {config['priority']}")
        print(f"  Models: {config['models']}")
        if 'step1_tokens' in config:
            total = config['step1_tokens'] + config['step2_tokens']
            print(f"  Tokens: {config['step1_tokens']} + {config['step2_tokens']} = {total}")
        else:
            print(f"  Tokens: {config.get('max_tokens', 'N/A')}")
