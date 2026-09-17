#!/usr/bin/env python3
"""build_manifest_v3.py -- Run-level manifest rebuilt from the raw record files
(the per-run manifests written during the experiments were incomplete: workers
running in parallel overwrote each other's entries and request parameters were
not captured at run level). One entry per results file: model, provider,
condition, records, seed, first/last timestamp, request parameters and output
caps as recorded per call, truncation and API-error counts, SHA-256 of the file
and of the concatenated program hashes, evaluator and harness revisions.
Endpoints are recorded as provider names only; deployment names equal the
model field. Writes results/manifest_v3.json."""
import json, glob, os, hashlib, subprocess
from collections import Counter
from pathlib import Path
HERE = Path(__file__).resolve().parent
def sha_file(p):
    h = hashlib.sha256(); h.update(open(p, "rb").read()); return h.hexdigest()
def commit():
    try: return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, text=True).strip()
    except Exception: return None
runs = []
for f in sorted(glob.glob(str(HERE / "results/livecodebench_v3_*/*_s*.jsonl")) + glob.glob(str(HERE / "results/humaneval_v2/*_s*.jsonl"))):
    rows = [json.loads(l) for l in open(f) if l.strip()]
    if not rows: continue
    params = Counter(json.dumps(r.get("params_used"), sort_keys=True) for r in rows if r.get("params_used"))
    ts = sorted(r["timestamp"] for r in rows if r.get("timestamp"))
    caps = Counter(str((r.get("params_used") or {}).get("cap") or (r.get("params_used") or {}).get("max_output_tokens")) for r in rows)
    codes = hashlib.sha256("".join(r.get("code_sha256") or "" for r in rows).encode()).hexdigest()
    runs.append({"file": str(Path(f).relative_to(HERE)), "model": rows[0].get("model"), "provider": rows[0].get("provider"), "condition": rows[0].get("condition"),
                 "seed": rows[0].get("seed"), "records": len(rows), "first_timestamp": ts[0] if ts else None, "last_timestamp": ts[-1] if ts else None,
                 "request_parameters": [{"params": json.loads(k), "calls": v} for k, v in params.most_common(3)], "output_cap_values": dict(caps),
                 "truncated": sum(1 for r in rows if (r.get("usage_step2") or {}).get("truncated") or r.get("truncated")), "api_errors": sum(1 for r in rows if r.get("api_error")),
                 "sha256_file": sha_file(f), "sha256_program_hashes": codes})
out = {"evaluator": "official LiveCodeBench testing_util @ 28fef95 (vendored in lcb_eval/)", "dataset": "bzantium/livecodebench (test split, 1055 problems); HumanEval (164 problems)",
       "harness_commit_at_manifest_build": commit(), "n_files": len(runs), "runs": runs}
json.dump(out, open(HERE / "results/manifest_v3.json", "w"), indent=1)
print("manifest entries:", len(runs), "| files with request parameters recorded:", sum(1 for r in runs if r["request_parameters"]))
