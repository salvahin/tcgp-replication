#!/usr/bin/env python3
"""
validate_lcb_v3.py -- Evidence that the v3 LiveCodeBench harness grades like the
official leaderboard. Run from the package root. Checks:
  1. the v3 sample is exactly the v2 180-problem sample;
  2. known-correct LeetCode solutions pass ALL public+private tests;
  3. a wrong solution fails with 'Wrong Answer';
  4. the stdin path: an old Codeforces output that passed its 3 public tests is
     graded on all tests;
  5. extraction follows the official last-fenced-block rule.
"""
import sys, json, glob, importlib.util
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from lcb_eval import grade, extract_code

def main():
    from datasets import load_dataset
    spec = importlib.util.spec_from_file_location("lcbv2", HERE / "run_livecodebench_v2.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    ds = load_dataset("bzantium/livecodebench", split="test")
    probs = {str(p["question_id"]): p for p in m.select_problems(ds, 0, 60)}
    old = set()
    for f in glob.glob(str(HERE / "results/livecodebench_v2_strat/direct_*_s42.jsonl")):
        for l in open(f):
            if l.strip(): old.add(str(json.loads(l)["question_id"]))
    print(f"[identity] v3 sample == v2 sample: {set(probs) == old} (n={len(probs)})")
    good = {
        "2825": "class Solution:\n    def minimizedStringLength(self, s: str) -> int:\n        return len(set(s))\n",
        "3226": "class Solution:\n    def numberGame(self, nums):\n        nums.sort(); out=[]\n        for i in range(0,len(nums),2): out += [nums[i+1], nums[i]]\n        return out\n",
    }
    for q, code in good.items():
        g = grade(probs[q], code)
        print(f"[correct]  {q}: passed={g['passed']} public={g['public_pass']} {g['n_passed']}/{g['n_tests']} tests")
    g = grade(probs["2825"], "class Solution:\n    def minimizedStringLength(self, s: str) -> int:\n        return len(s)\n")
    print(f"[wrong]    2825: passed={g['passed']} {g['n_passed']}/{g['n_tests']} -> {g['error_message'][:30]}")
    sf = {q for q, p in probs.items() if not (p.get("starter_code") or "").strip()}
    for f in sorted(glob.glob(str(HERE / "results/livecodebench_v2_strat/direct_gpt-5.5_s42.jsonl"))):
        for l in open(f):
            r = json.loads(l)
            if str(r["question_id"]) in sf and r.get("passed") and r.get("extracted_code"):
                g = grade(probs[str(r["question_id"])], r["extracted_code"])
                print(f"[stdin]    {r['question_id']} old v2 codeforces output: all-tests passed={g['passed']} {g['n_passed']}/{g['n_tests']}")
                break
    print("[extract]  last block:", repr(extract_code("t\n```python\nx=1\n```\nm\n```python\ny=2\n```")))

if __name__ == "__main__":
    main()
