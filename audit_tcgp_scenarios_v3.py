#!/usr/bin/env python3
"""
audit_tcgp_scenarios_v3.py -- Are the test scenarios that TCGP writes in step 1
actually correct? (R1, concern 4.)

Oracle: for each problem, any v3 generation (any model/condition) that passed
ALL official tests. Scenario parsing: for functional problems we look for
"Input ... Output" pairs in the step-1 text, parse the argument literals
(`name = value` lines or a bare tuple) and the expected literal with
ast.literal_eval, call the oracle's method, and compare. Unparseable scenarios
are counted separately; stdin problems are skipped. This is a heuristic audit
and is reported as such. Writes results/scenario_audit_v3.json.
"""
import sys, json, glob, os, re, ast, importlib.util, multiprocessing
from collections import defaultdict
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from lcb_eval.testing_util import import_string

D = HERE / "results" / "livecodebench_v3_strat"

def _run_oracle(code, fn, args, q):
    try:
        ns = {}; exec(import_string + "\n\n" + code, ns)
        sol = ns["Solution"]() if "Solution" in ns else ns
        res = getattr(sol, fn)(*args)
        if isinstance(res, tuple): res = list(res)
        q.put(("ok", res))
    except Exception as e:
        q.put(("err", repr(e)[:80]))

def oracle_call(code, fn, args, timeout=6):
    ctx = multiprocessing.get_context("spawn"); q = ctx.Queue()
    p = ctx.Process(target=_run_oracle, args=(code, fn, args, q)); p.start(); p.join(timeout)
    if p.is_alive(): p.kill(); p.join(); return ("timeout", None)
    try: return q.get(timeout=1)
    except Exception: return ("err", "no result")

PAIR = re.compile(r"input\s*[:\-]?\s*(.*?)\s*(?:expected\s+output|output|expected|returns?)\s*[:\-]?\s*(.*?)(?=(?:\n\s*\n)|(?:\n\s*(?:#+|\*\*|test|scenario|example|input)\b)|\Z)", re.I | re.S)

def parse_literal(s):
    s = s.strip().strip("`*").strip()
    s = re.sub(r"^\s*[\w\[\]]+\s*=\s*", "", s)  # drop leading "name = "
    s = s.rstrip(".").strip()
    return ast.literal_eval(s)

def parse_args(block):
    """'nums = [1,2], k = 3' or multi-line 'nums = [..]\\nk = 3' or bare '[1,2], 3'."""
    block = block.strip().strip("`").strip()
    parts = [p for p in re.split(r"\n|,\s*(?=[A-Za-z_]\w*\s*=)", block) if p.strip()]
    args = []
    for p in parts:
        p = p.strip().strip("`*").strip()
        if "=" in p and re.match(r"^\s*[A-Za-z_]\w*\s*=", p):
            args.append(ast.literal_eval(p.split("=", 1)[1].strip().rstrip(",")))
        else:
            v = ast.literal_eval(p.rstrip(","))
            if isinstance(v, tuple) and len(parts) == 1: args = list(v)
            else: args.append(v)
    return args

def main():
    from datasets import load_dataset
    spec = importlib.util.spec_from_file_location("lcbv2", HERE / "run_livecodebench_v2.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    probs = {str(p["question_id"]): p for p in m.select_problems(load_dataset("bzantium/livecodebench", split="test"), 0, 60)}
    oracle = {}
    tcgp = []
    for f in glob.glob(str(D / "*_s42.jsonl")):
        cond = os.path.basename(f).split("_")[0]
        for l in open(f):
            if not l.strip(): continue
            r = json.loads(l); q = r["question_id"]
            if r.get("passed") and q not in oracle and r.get("testtype") == "functional": oracle[q] = r["extracted_code"]
            if cond == "tcgp" and r.get("testtype") == "functional": tcgp.append(r)
    fn_of = {q: (json.loads(p["metadata"]) if isinstance(p["metadata"], str) else p["metadata"]).get("func_name") for q, p in probs.items()}
    stats = defaultdict(lambda: {"scen": 0, "parsed": 0, "correct": 0, "recs": 0, "recs_with_all_correct": 0, "recs_with_any_wrong": 0,
                                 "pass_all_correct": 0, "pass_any_wrong": 0})
    for r in tcgp:
        q = r["question_id"]
        if q not in oracle: continue
        st = stats[r["model"]]; st["recs"] += 1
        pairs = PAIR.findall(r.get("raw_step1") or ""); st["scen"] += len(pairs)
        verdicts = []
        for a, e in pairs[:6]:
            try: args = parse_args(a); exp = parse_literal(e)
            except Exception: continue
            st["parsed"] += 1
            status, res = oracle_call(oracle[q], fn_of[q], args)
            if status == "ok":
                ok = (res == exp) or (isinstance(res, float) and isinstance(exp, (int, float)) and abs(res - exp) < 1e-6)
                verdicts.append(ok); st["correct"] += ok
        if verdicts:
            if all(verdicts): st["recs_with_all_correct"] += 1; st["pass_all_correct"] += bool(r["passed"])
            else: st["recs_with_any_wrong"] += 1; st["pass_any_wrong"] += bool(r["passed"])
    tot = defaultdict(int)
    for st in stats.values():
        for k, v in st.items(): tot[k] += v
    out = {"per_model": dict(stats), "total": dict(tot), "n_oracle_problems": len(oracle)}
    json.dump(out, open(HERE / "results" / "scenario_audit_v3.json", "w"), indent=1)
    print(f"oracle available for {len(oracle)} functional problems; TCGP records audited: {tot['recs']}")
    print(f"scenarios found: {tot['scen']}, parseable: {tot['parsed']} ({100*tot['parsed']/max(tot['scen'],1):.0f}%), correct among parseable: {tot['correct']} ({100*tot['correct']/max(tot['parsed'],1):.0f}%)")
    ac, aw = tot["recs_with_all_correct"], tot["recs_with_any_wrong"]
    print(f"records with all parsed scenarios correct: {ac} (TCGP pass rate {100*tot['pass_all_correct']/max(ac,1):.0f}%)")
    print(f"records with at least one wrong scenario:  {aw} (TCGP pass rate {100*tot['pass_any_wrong']/max(aw,1):.0f}%)")
    print("per model (parsed, %correct, pass|all-correct, pass|any-wrong):")
    for mname, st in sorted(stats.items()):
        print(f"  {mname:<26} {st['parsed']:4d}  {100*st['correct']/max(st['parsed'],1):3.0f}%   {100*st['pass_all_correct']/max(st['recs_with_all_correct'],1):3.0f}% (n={st['recs_with_all_correct']})   {100*st['pass_any_wrong']/max(st['recs_with_any_wrong'],1):3.0f}% (n={st['recs_with_any_wrong']})")

if __name__ == "__main__":
    main()
