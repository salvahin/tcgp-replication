#!/usr/bin/env python3
"""make_sample_manifest.py -- Pin the 180-problem LiveCodeBench sample independently
of the dataset revision: question_id, difficulty, test type, contest date, and a
SHA-256 over the public and private tests of each problem. Writes
data/livecodebench_sample_manifest.json. The dataset revision used at run time
was not recorded; this manifest lets a reader verify that a later snapshot
contains the same problems with the same tests."""
import json, hashlib, importlib.util
from pathlib import Path
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("v3", HERE / "run_livecodebench_v3.py"); v3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v3)
v3._init_heavy(); probs = v3.load_problems()
rows = []
for p in probs:
    h = hashlib.sha256((str(p.get("public_test_cases")) + "||" + str(p.get("private_test_cases"))).encode()).hexdigest()
    md = p.get("metadata") or {}
    if isinstance(md, str):
        try: md = json.loads(md)
        except Exception: md = {}
    rows.append({"question_id": str(p["question_id"]), "difficulty": p.get("difficulty"), "testtype": "functional" if md.get("func_name") else "stdin", "contest_date": str(p.get("contest_date")), "tests_sha256": h})
out = {"source": "bzantium/livecodebench, split test; first 60 problems by question_id per difficulty", "n": len(rows),
       "dataset_revision_at_run": "not recorded", "problems": rows}
(HERE / "data").mkdir(exist_ok=True); json.dump(out, open(HERE / "data" / "livecodebench_sample_manifest.json", "w"), indent=1)
print("problems:", len(rows), "difficulties:", {d: sum(1 for r in rows if r["difficulty"] == d) for d in {r["difficulty"] for r in rows}})
