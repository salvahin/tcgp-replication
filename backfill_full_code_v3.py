#!/usr/bin/env python3
"""backfill_full_code_v3.py -- Early v3 records stored only the first 6,000
characters of the extracted program (the full program was graded, and the raw
model response was stored whole). This script re-extracts the full program from
the stored raw response with the same official rule, verifies that its first
6,000 characters equal what was stored, and rewrites `extracted_code` (and the
loop rounds' `code`) in full with a SHA-256. Idempotent."""
import json, glob, os, sys, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from lcb_eval import extract_code
sha = lambda c: hashlib.sha256(c.encode()).hexdigest()
tot = long_ = mism = 0
for f in sorted(glob.glob(str(HERE / "results/livecodebench_v3_*/*_s42.jsonl"))):
    cond = os.path.basename(f).split("_")[0]; rows = [json.loads(l) for l in open(f) if l.strip()]
    for r in rows:
        if cond == "repairR":
            for rd in r.get("rounds", []):
                full = extract_code(rd.get("raw") or ""); tot += 1
                if (rd.get("code") or "") != full[:len(rd.get("code") or "")] and len(rd.get("code") or "") == 6000: mism += 1
                long_ += len(full) > 6000; rd["code"] = full; rd["code_sha256"] = sha(full)
            continue
        raw = r.get("raw_repair") if (cond == "repair" and r.get("repaired")) else r.get("raw_step2")
        if r.get("api_error") and not raw: continue
        full = extract_code(raw or ""); old = r.get("extracted_code") or ""; tot += 1
        if old != full[:6000]: mism += 1
        long_ += len(full) > 6000; r["extracted_code"] = full; r["code_sha256"] = sha(full)
    with open(f, "w") as w:
        for r in rows: w.write(json.dumps(r) + "\n")
print(f"programs={tot}  longer than 6000 chars={long_}  stored-prefix mismatches={mism}")
