#!/usr/bin/env python3
"""
make_paper_analyses.py -- Statistical analyses for the manuscript, computed from
the logged V2 results. Part of the replication package; no API calls.

Analyses:
  1. McNemar paired tests (TCGP vs Direct, TCGP vs CoT, CoT vs Direct) per model
     on the stratified LiveCodeBench sample, with Holm correction across models.
  2. Partial-correctness breakdown on LiveCodeBench (all/some/none of the
     public tests passed), per condition.
  3. Reasoning-token usage by strategy for models that expose hidden
     reasoning tokens.
  4. Sample-robustness: pilot (first-50) vs stratified (n=180) aggregates.
  5. Cross-seed variance on HumanEval (3 seeds).

Usage:  python make_paper_analyses.py [--json out.json]
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
STRAT = os.path.join(HERE, "results", "livecodebench_v2_strat")
PILOT = os.path.join(HERE, "results", "livecodebench_v2")
HE = os.path.join(HERE, "results", "humaneval_v2")
COND = ["direct", "cot", "tcgp"]


def load_dir(path, seed="42"):
    """-> {model: {cond: {qid: record}}}"""
    out = defaultdict(lambda: defaultdict(dict))
    for f in glob.glob(os.path.join(path, f"*_s{seed}.jsonl")):
        b = os.path.basename(f)[: -len(f"_s{seed}.jsonl")]
        cond, model = b.split("_", 1)
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            qid = str(r.get("question_id", r.get("task_id")))
            out[model][cond][qid] = r
    return out


def mcnemar(data, a, b):
    """Exact McNemar via two-sided binomial test on discordant pairs."""
    n01 = n10 = 0
    for qid, ra in data[a].items():
        rb = data[b].get(qid)
        if rb is None:
            continue
        pa, pb = bool(ra.get("passed")), bool(rb.get("passed"))
        if pa and not pb:
            n01 += 1
        elif pb and not pa:
            n10 += 1
    if n01 + n10 == 0:
        return n01, n10, 1.0
    p = binomtest(n01, n01 + n10, 0.5, alternative="two-sided").pvalue
    return n01, n10, p


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj.tolist()


def analysis_mcnemar(strat, out):
    print("\n=== 1. McNemar paired tests, LiveCodeBench stratified (n=180) ===")
    res = {}
    for pair in (("tcgp", "direct"), ("tcgp", "cot"), ("cot", "direct")):
        rows = []
        for m in sorted(strat):
            n01, n10, p = mcnemar(strat[m], *pair)
            rows.append((m, n01, n10, p))
        adj = holm([r[3] for r in rows])
        key = f"{pair[0]}_vs_{pair[1]}"
        res[key] = []
        print(f"-- {pair[0].upper()} vs {pair[1].upper()} (n01 = only {pair[0]} passes; n10 = only {pair[1]} passes)")
        for (m, n01, n10, p), pa in zip(rows, adj):
            sig = "*" if pa < 0.05 else " "
            print(f"  {m:<26} n01={n01:>3} n10={n10:>3}  p={p:.4f}  holm={pa:.4f}{sig}")
            res[key].append(dict(model=m, n01=n01, n10=n10, p=p, holm=pa))
        pooled01 = sum(r[1] for r in rows)
        pooled10 = sum(r[2] for r in rows)
        pp = binomtest(pooled01, pooled01 + pooled10, 0.5).pvalue if pooled01 + pooled10 else 1.0
        print(f"  {'POOLED':<26} n01={pooled01:>3} n10={pooled10:>3}  p={pp:.2e}")
        res[key + "_pooled"] = dict(n01=pooled01, n10=pooled10, p=pp)
    out["mcnemar"] = res


def analysis_partial(strat, out):
    print("\n=== 2. Partial correctness on public tests (pooled, n=1800/cond) ===")
    res = {}
    for c in COND:
        allp = some = none = tot = 0
        for m in strat:
            for r in strat[m][c].values():
                if not r.get("n_tests"):
                    continue
                tot += 1
                if r.get("passed"):
                    allp += 1
                elif (r.get("tests_passed") or 0) > 0:
                    some += 1
                else:
                    none += 1
        res[c] = dict(all=allp, some=some, none=none, total=tot)
        print(f"  {c:<8} all={100*allp/tot:5.1f}%  some={100*some/tot:5.1f}%  none={100*none/tot:5.1f}%  (n={tot})")
    out["partial"] = res


def analysis_reasoning_tokens(strat, out):
    print("\n=== 3. Hidden reasoning tokens by strategy (models exposing them) ===")
    res = {}
    for m in sorted(strat):
        per_cond = {}
        for c in COND:
            vals = []
            for r in strat[m][c].values():
                u1, u2 = r.get("usage_step1") or {}, r.get("usage_step2") or {}
                vals.append((u1.get("reasoning_tokens", 0) or 0) + (u2.get("reasoning_tokens", 0) or 0))
            per_cond[c] = float(np.mean(vals)) if vals else 0.0
        if max(per_cond.values()) > 50:
            res[m] = per_cond
            print(f"  {m:<26} direct={per_cond['direct']:>7.0f}  cot={per_cond['cot']:>7.0f}  tcgp={per_cond['tcgp']:>7.0f}")
    out["reasoning_tokens"] = res


def analysis_sample_robustness(strat, pilot, out):
    print("\n=== 4. Sample robustness: pilot (first-50) vs stratified (n=180) ===")
    res = {}
    for name, data in (("pilot", pilot), ("stratified", strat)):
        agg = {}
        for c in COND:
            n = p = 0
            for m in data:
                for r in data[m][c].values():
                    n += 1
                    p += 1 if r.get("passed") else 0
            agg[c] = (p, n, 100 * p / n if n else 0)
        res[name] = {c: dict(passed=v[0], n=v[1], rate=v[2]) for c, v in agg.items()}
        print(f"  {name:<11} " + "  ".join(f"{c}={agg[c][2]:.0f}%" for c in COND))
    out["sample_robustness"] = res


def analysis_seed_variance(out):
    print("\n=== 5. Cross-seed variance, HumanEval (3 seeds) ===")
    rates = defaultdict(lambda: defaultdict(dict))
    for f in glob.glob(os.path.join(HE, "*_s*.jsonl")):
        b = os.path.basename(f)[:-6]
        cond, rest = b.split("_", 1)
        model, seed = rest.rsplit("_s", 1)
        n = p = 0
        for line in open(f):
            if line.strip():
                n += 1
                p += 1 if json.loads(line).get("passed") else 0
        if n:
            rates[model][cond][seed] = 100 * p / n
    res = {}
    sds = []
    for m in sorted(rates):
        res[m] = {}
        for c in COND:
            vals = list(rates[m][c].values())
            sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            res[m][c] = dict(rates=vals, sd=sd)
            sds.append(sd)
        print(f"  {m:<26} " + "  ".join(f"{c}: sd={res[m][c]['sd']:.1f}" for c in COND))
    print(f"  max cross-seed SD: {max(sds):.1f} pp;  median: {float(np.median(sds)):.1f} pp")
    out["seed_variance"] = dict(per_model=res, max_sd=max(sds), median_sd=float(np.median(sds)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(HERE, "results", "paper_analyses.json"))
    args = ap.parse_args()
    strat = load_dir(STRAT)
    pilot = load_dir(PILOT)
    out = {}
    analysis_mcnemar(strat, out)
    analysis_partial(strat, out)
    analysis_reasoning_tokens(strat, out)
    analysis_sample_robustness(strat, pilot, out)
    analysis_seed_variance(out)
    with open(args.json, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
