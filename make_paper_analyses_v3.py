#!/usr/bin/env python3
"""
make_paper_analyses_v3.py -- Analyses of the corrected (v3, official-evaluator)
LiveCodeBench results. Addresses the R1 review:
  * per-model paired McNemar tests with Holm correction (as before),
  * the pooled comparison uses a PROBLEM-LEVEL permutation test (problems are
    the clustering unit: 180 problems x 10 models are not 1,800 independent
    pairs),
  * public-vs-private verdict disagreement (are public tests independent
    evidence?),
  * the format-restatement control (extra call vs. content of the call),
  * breakdown by problem type (functional vs stdin) and difficulty,
  * data-quality accounting: API errors, no-output, truncation, timeouts.
Writes results/paper_analyses_v3.json.
"""
import json, glob, os, math, random
from collections import defaultdict
from pathlib import Path
HERE = Path(__file__).resolve().parent
D = HERE / "results" / "livecodebench_v3_strat"
COND = ["direct", "cot", "tcgp", "restate"]

def load():
    recs = defaultdict(dict)  # (model, cond) -> qid -> rec
    for f in glob.glob(str(D / "*_s42.jsonl")):
        b = os.path.basename(f)[:-len("_s42.jsonl")]; cond, model = b.split("_", 1)
        for l in open(f):
            if l.strip():
                r = json.loads(l); recs[(model, cond)][r["question_id"]] = r
    return recs

def wilson(p, n, z=1.96):
    if n == 0: return (0, 0)
    ph = p / n; d = 1 + z*z/n; c = (ph + z*z/(2*n))/d; m = z*math.sqrt((ph*(1-ph) + z*z/(4*n))/n)/d
    return 100*(c-m), 100*(c+m)

def mcnemar_p(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k+1)) / 2**n)

def holm(ps):
    idx = sorted(range(len(ps)), key=lambda i: ps[i]); out = [0]*len(ps); m = len(ps); run = 0
    for rank, i in enumerate(idx):
        run = max(run, (m - rank) * ps[i]); out[i] = min(1.0, run)
    return out

def perm_test(recs, models, a, b, qids, iters=20000, seed=0):
    """Problem-level permutation test for pooled Pass@1(a) - Pass@1(b): within
    each problem, swap the a/b labels for all models jointly (respects the
    clustering), recompute the pooled difference."""
    rng = random.Random(seed)
    da = {q: sum(bool(recs[(m, a)].get(q, {}).get("passed")) for m in models) for q in qids}
    db = {q: sum(bool(recs[(m, b)].get(q, {}).get("passed")) for m in models) for q in qids}
    obs = sum(da[q] - db[q] for q in qids)
    diffs = [da[q] - db[q] for q in qids]
    ge = 0
    for _ in range(iters):
        s = sum(d if rng.random() < 0.5 else -d for d in diffs)
        if abs(s) >= abs(obs): ge += 1
    n = len(models) * len(qids)
    return {"diff_pp": 100*obs/n, "p_two_sided": (ge + 1) / (iters + 1)}

def main():
    recs = load()
    models = sorted({m for m, _ in recs})
    out = {"n_models": len(models), "per_model": {}, "pooled": {}, "quality": {}, "by_type": {}, "by_difficulty": {}, "public_private": {}}
    # ---- per-model rates + McNemar ----
    comps = [("tcgp", "direct"), ("tcgp", "cot"), ("cot", "direct"), ("restate", "direct"), ("tcgp", "restate")]
    pvals = {c: [] for c in comps}
    for m in models:
        row = {}
        for c in COND:
            R = recs.get((m, c), {}); n = len(R); p = sum(bool(r.get("passed")) for r in R.values())
            row[c] = {"n": n, "pass": p, "rate": 100*p/n if n else None, "ci": wilson(p, n) if n else None}
        for a, b in comps:
            A, B = recs.get((m, a), {}), recs.get((m, b), {}); common = set(A) & set(B)
            n01 = sum(1 for q in common if A[q].get("passed") and not B[q].get("passed"))
            n10 = sum(1 for q in common if B[q].get("passed") and not A[q].get("passed"))
            p = mcnemar_p(n01, n10); pvals[(a, b)].append(p)
            row[f"{a}_vs_{b}"] = {"n01": n01, "n10": n10, "p": p, "n_common": len(common)}
        out["per_model"][m] = row
    for (a, b), ps in pvals.items():
        hs = holm(ps)
        for m, h in zip(models, hs): out["per_model"][m][f"{a}_vs_{b}"]["holm_p"] = h
    # ---- pooled: rates + problem-level permutation ----
    qids = sorted(set.intersection(*[set(recs[(m, c)]) for m in models for c in COND if (m, c) in recs]) or set())
    for c in COND:
        n = sum(len(recs.get((m, c), {})) for m in models); p = sum(sum(bool(r.get("passed")) for r in recs.get((m, c), {}).values()) for m in models)
        out["pooled"][c] = {"n": n, "rate": 100*p/n if n else None, "ci": wilson(p, n) if n else None}
    for a, b in comps:
        ms = [m for m in models if (m, a) in recs and (m, b) in recs]
        if qids and ms: out["pooled"][f"{a}_vs_{b}_perm"] = perm_test(recs, ms, a, b, qids)
    # ---- by problem type and difficulty (pooled) ----
    for key, field in (("by_type", "testtype"), ("by_difficulty", "difficulty")):
        agg = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        for (m, c), R in recs.items():
            for r in R.values(): agg[r.get(field)][c][0] += 1; agg[r.get(field)][c][1] += bool(r.get("passed"))
        out[key] = {k: {c: {"n": v[c][0], "rate": 100*v[c][1]/v[c][0] if v[c][0] else None} for c in COND} for k, v in agg.items()}
    # ---- public vs private disagreement ----
    for c in COND:
        both = pubonly = 0; n = 0
        for m in models:
            for r in recs.get((m, c), {}).values():
                n += 1
                if r.get("public_pass") and not r.get("passed"): pubonly += 1
                if r.get("public_pass") and r.get("passed"): both += 1
        out["public_private"][c] = {"n": n, "public_pass_but_private_fail": pubonly,
                                    "share_of_public_passes_that_fail_private": (pubonly / (pubonly + both)) if (pubonly + both) else None}
    # ---- quality ----
    for c in COND:
        rows = [r for m in models for r in recs.get((m, c), {}).values()]
        out["quality"][c] = {"n": len(rows), "api_error": sum(1 for r in rows if r.get("api_error")),
            "no_output": sum(1 for r in rows if r.get("no_output")),
            "truncated": sum(1 for r in rows if (r.get("usage_step2") or {}).get("truncated")),
            "timeout_verdicts": sum(1 for r in rows if "time" in str(r.get("error_message", "")).lower()),
            "regraded_timeouts": sum(1 for r in rows if r.get("regrade_timeout_isolated"))}
    json.dump(out, open(HERE / "results" / "paper_analyses_v3.json", "w"), indent=1)
    # ---- console report ----
    print(f"models={len(models)}  common problems across all cells={len(qids)}")
    print("\nPooled Pass@1 (%):", {c: (round(v['rate'], 1) if v['rate'] is not None else None) for c, v in out['pooled'].items() if c in COND})
    for a, b in comps:
        k = f"{a}_vs_{b}_perm"
        if k in out["pooled"]: print(f"  {a} - {b}: {out['pooled'][k]['diff_pp']:+.1f} pp, problem-level permutation p={out['pooled'][k]['p_two_sided']:.4f}")
    print("\nPer model Pass@1 (direct/cot/tcgp/restate) and Holm p for tcgp-vs-direct, tcgp-vs-cot, restate-vs-direct:")
    for m in models:
        r = out["per_model"][m]
        rates = "/".join(f"{r[c]['rate']:.0f}" if r[c]['rate'] is not None else "--" for c in COND)
        ps = "  ".join(f"{k}={r[k].get('holm_p', float('nan')):.3f}({r[k]['n01']}/{r[k]['n10']})" for k in ("tcgp_vs_direct", "tcgp_vs_cot", "restate_vs_direct"))
        print(f"  {m:<26} {rates:<16} {ps}")
    print("\nBy type:", {k: {c: round(v[c]['rate'], 1) if v[c]['rate'] is not None else None for c in COND} for k, v in out['by_type'].items()})
    print("By difficulty:", {k: {c: round(v[c]['rate'], 1) if v[c]['rate'] is not None else None for c in COND} for k, v in out['by_difficulty'].items()})
    print("\nPublic-pass but private-fail:", {c: v['public_pass_but_private_fail'] for c, v in out['public_private'].items()})
    print("Quality:", out["quality"])

if __name__ == "__main__":
    main()
