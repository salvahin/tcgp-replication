#!/usr/bin/env python3
"""Remove v3 records that must be regenerated so the harness's per-problem
resume re-runs them: API errors, and (optionally) truncated generations.
Usage: purge_v3_records.py [--truncated] [file-substring ...]"""
import sys, json, glob
from pathlib import Path
HERE = Path(__file__).resolve().parent
args = [a for a in sys.argv[1:] if not a.startswith("--")]; trunc = "--truncated" in sys.argv
def is_trunc(r):
    u = r.get("usage_step2") or {}; fr = str(u.get("finish_reason", "")).lower()
    return bool(u.get("truncated")) or fr in ("2", "max_tokens", "length")
tot = 0
for f in sorted(glob.glob(str(HERE / "results/livecodebench_v3_strat/*_s42.jsonl"))):
    if args and not any(a in f for a in args): continue
    rows = [json.loads(l) for l in open(f) if l.strip()]
    keep = [r for r in rows if not r.get("api_error") and not (trunc and is_trunc(r))]
    if len(keep) != len(rows):
        with open(f, "w") as w:
            for r in keep: w.write(json.dumps(r) + "\n")
        print(f"{Path(f).name}: removed {len(rows)-len(keep)}"); tot += len(rows) - len(keep)
print("total removed:", tot)
