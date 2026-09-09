"""
Official LiveCodeBench evaluator, vendored verbatim for reproducibility.

testing_util.py is an unmodified copy of
  https://github.com/LiveCodeBench/LiveCodeBench/blob/28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24/lcb_runner/evaluation/testing_util.py
(MIT License, see LICENSE-LiveCodeBench). It grades LeetCode ("functional")
problems by instantiating `Solution` and calling the method named in the
problem metadata with JSON-decoded arguments, and Codeforces/AtCoder
("stdin") problems by feeding stdin and comparing stdout, exactly as the
official leaderboard does.

`check_correctness` below mirrors the official wrapper in
compute_code_generation_metrics.py: it runs `run_test` in an isolated child
process (run_test installs a reliability guard that must not run in the
parent). Two small robustness changes: the child's stdin is /dev/null so a
stray input() cannot block, and a missing metadata entry is tolerated.
"""
import json
import multiprocessing
import os
import sys


def _child(sample, generation, timeout, q):
    sys.stdin = open(os.devnull)
    from lcb_eval.testing_util import run_test
    res, meta = run_test(sample, test=generation, debug=False, timeout=timeout)
    q.put((res, meta))


def check_correctness(sample, generation, timeout=6):
    """Return (results, metadata). results[i] is True for a passed test; the
    official grader stops at the first failure, so len(results) may be < n."""
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_child, args=(sample, generation, timeout, q))
    p.start()
    n = len(json.loads(sample["input_output"])["inputs"])
    p.join(timeout=(timeout + 1) * n + 5)
    if p.is_alive():
        p.kill()
        p.join()
    try:
        res, meta = q.get(timeout=1)
    except Exception:
        res, meta = [-1] * n, {"error_code": -1, "error_message": "global timeout"}
    return res, meta


def make_sample(problem):
    """Build the official evaluation sample (public + private tests, fn_name)."""
    import base64, pickle, zlib
    pub = problem.get("public_test_cases") or []
    if isinstance(pub, str):
        pub = json.loads(pub)
    priv = problem.get("private_test_cases") or []
    if isinstance(priv, str):
        try:
            priv = json.loads(priv)
        except Exception:
            priv = json.loads(pickle.loads(zlib.decompress(base64.b64decode(priv.encode("utf-8")))))
    meta = problem.get("metadata") or {}
    if isinstance(meta, str):
        meta = json.loads(meta)
    tests = list(pub) + list(priv)
    sample = {"input_output": json.dumps({
        "inputs": [t["input"] for t in tests],
        "outputs": [t["output"] for t in tests],
        "fn_name": meta.get("func_name", None),
    })}
    return sample, len(pub), len(tests)


def extract_code(model_output: str) -> str:
    """Official extraction (LMStyle generic): the LAST fenced block."""
    lines = model_output.split("\n")
    idx = [i for i, l in enumerate(lines) if "```" in l]
    if len(idx) < 2:
        return ""
    return "\n".join(lines[idx[-2] + 1: idx[-1]])


def grade(problem, code, timeout=6):
    """Grade `code` on all tests of `problem`. Returns a dict with the overall
    verdict, the public-only verdict, and counts."""
    sample, n_pub, n_all = make_sample(problem)
    if not code.strip():
        return {"passed": False, "public_pass": False, "n_tests": n_all, "n_public": n_pub,
                "n_passed": 0, "error_code": None, "error_message": "no code"}
    res, meta = check_correctness(sample, code, timeout)
    ok = [r is True or r == True for r in res]
    n_passed = sum(ok)
    passed = len(res) == n_all and all(ok)
    public_pass = len(res) >= n_pub and all(ok[:n_pub])
    return {"passed": passed, "public_pass": public_pass, "n_tests": n_all, "n_public": n_pub,
            "n_passed": n_passed, "error_code": (meta or {}).get("error_code"),
            "error_message": str((meta or {}).get("error_message", ""))[:200]}
