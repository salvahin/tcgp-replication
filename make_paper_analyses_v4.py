#!/usr/bin/env python3
"""
make_paper_analyses_v4.py -- Recipe study on the corrected (v3) LiveCodeBench
results: are prompting strategies distinguishable from rewording noise?

Conditions: direct, cot, tcgp, restate, para1..3 (paraphrases of direct),
persona, fewshot, stacked, plansolve, repair (one execution-feedback round).
Analyses (problems are the clustering unit throughout):
  * pooled Pass@1 with Wilson CIs, by difficulty and by problem type;
  * NOISE FLOOR: spread of Pass@1 across the four wordings of the direct
    prompt (direct, para1..3), pooled and per model;
  * each condition vs direct: pooled difference, problem-level permutation p,
    problem-level cluster-bootstrap 90% CI, and a TOST-style equivalence
    verdict (equivalent if the 90% CI lies within +/- DELTA points);
  * per-model paired McNemar with Holm, counted as wins/losses;
  * repair: gain over direct, and the fix rate among records that received
    feedback; token cost per condition.
Models: the nine with every condition (Gemini has only the first four).
Writes results/paper_analyses_v4.json.
"""
import json, glob, os, math, random
from collections import defaultdict
from pathlib import Path
HERE = Path(__file__).resolve().parent
D = HERE / "results" / "livecodebench_v3_strat"
COND = ["direct", "para1", "para2", "para3", "persona", "fewshot", "cot", "plansolve", "tcgp", "restate", "stacked", "repair"]
DELTA = 3.0  # equivalence margin, percentage points

def load():
    recs = defaultdict(dict)
    for f in glob.glob(str(D / "*_s42.jsonl")):
        b = os.path.basename(f)[:-len("_s42.jsonl")]; cond, model = b.split("_", 1)
        if cond not in COND: continue
        for l in open(f):
            if l.strip():
                r = json.loads(l); recs[(model, cond)][r["question_id"]] = r
    return recs

def wilson(p, n, z=1.96):
    if n == 0: return (0, 0)
    ph = p/n; d = 1+z*z/n; c = (ph+z*z/(2*n))/d; m = z*math.sqrt((ph*(1-ph)+z*z/(4*n))/n)/d
    return 100*(c-m), 100*(c+m)

def mcnemar_p(b, c):
    n = b+c
    if n == 0: return 1.0
    k = min(b, c); return min(1.0, 2*sum(math.comb(n, i) for i in range(k+1))/2**n)

def holm(ps):
    idx = sorted(range(len(ps)), key=lambda i: ps[i]); out = [0]*len(ps); m = len(ps); run = 0
    for rank, i in enumerate(idx):
        run = max(run, (m-rank)*ps[i]); out[i] = min(1.0, run)
    return out

def problem_diffs(recs, models, a, b, qids):
    return [sum(bool(recs[(m, a)][q].get("passed")) - bool(recs[(m, b)][q].get("passed")) for m in models) for q in qids]

def perm_p(diffs, iters=20000, seed=0):
    rng = random.Random(seed); obs = sum(diffs); ge = 0
    for _ in range(iters):
        s = sum(d if rng.random() < 0.5 else -d for d in diffs)
        if abs(s) >= abs(obs): ge += 1
    return (ge+1)/(iters+1)

def boot_ci(diffs, n_models, iters=10000, seed=1, alpha=0.10):
    rng = random.Random(seed); n = len(diffs); out = []
    for _ in range(iters):
        s = sum(diffs[rng.randrange(n)] for _ in range(n)); out.append(100*s/(n*n_models))
    out.sort(); return out[int(alpha/2*iters)], out[int((1-alpha/2)*iters)-1]

def main():
    recs = load()
    models_all = sorted({m for m, _ in recs})
    models = [m for m in models_all if all((m, c) in recs and len(recs[(m, c)]) >= 180 for c in COND)]
    qids = sorted(set.intersection(*[set(recs[(m, c)]) for m in models for c in COND]))
    out = {"models": models, "n_problems": len(qids), "delta_pp": DELTA, "pooled": {}, "vs_direct": {}, "noise_floor": {}, "per_model": {}, "by_difficulty": {}, "by_type": {}, "repair": {}, "tokens": {}}
    def rate(m, c): R = recs[(m, c)]; return 100*sum(bool(R[q].get("passed")) for q in qids)/len(qids)
    for c in COND:
        p = sum(bool(recs[(m, c)][q].get("passed")) for m in models for q in qids); n = len(models)*len(qids)
        out["pooled"][c] = {"rate": 100*p/n, "ci": wilson(p, n), "n": n}
    # noise floor
    W = ["direct", "para1", "para2", "para3"]
    pooled_w = [out["pooled"][c]["rate"] for c in W]
    per_model_range = {m: max(rate(m, c) for c in W) - min(rate(m, c) for c in W) for m in models}
    out["noise_floor"] = {"pooled_rates": dict(zip(W, pooled_w)), "pooled_range": max(pooled_w)-min(pooled_w),
                          "pooled_max_abs_dev_from_direct": max(abs(out["pooled"][c]["rate"]-out["pooled"]["direct"]["rate"]) for c in W[1:]),
                          "per_model_range": per_model_range, "median_per_model_range": sorted(per_model_range.values())[len(models)//2]}
    # each condition vs direct
    for c in COND:
        if c == "direct": continue
        diffs = problem_diffs(recs, models, c, "direct", qids); d = 100*sum(diffs)/(len(models)*len(qids))
        lo, hi = boot_ci(diffs, len(models)); p = perm_p(diffs)
        ps = []; wins = losses = 0; pm = {}
        for m in models:
            A, B = recs[(m, c)], recs[(m, "direct")]
            n01 = sum(1 for q in qids if A[q].get("passed") and not B[q].get("passed")); n10 = sum(1 for q in qids if B[q].get("passed") and not A[q].get("passed"))
            pv = mcnemar_p(n01, n10); ps.append(pv); pm[m] = {"rate": rate(m, c), "direct": rate(m, "direct"), "n01": n01, "n10": n10, "p": pv}
        for m, h in zip(models, holm(ps)):
            pm[m]["holm_p"] = h
            if h < 0.05: wins += pm[m]["n01"] > pm[m]["n10"]; losses += pm[m]["n10"] > pm[m]["n01"]
        out["vs_direct"][c] = {"diff_pp": d, "perm_p": p, "boot90_ci": [lo, hi], "equivalent_within_delta": (lo > -DELTA and hi < DELTA),
                               "sig_wins": wins, "sig_losses": losses, "per_model": pm}
    # by difficulty / type
    for key, field in (("by_difficulty", "difficulty"), ("by_type", "testtype")):
        agg = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        for m in models:
            for c in COND:
                for q in qids:
                    r = recs[(m, c)][q]; agg[r.get(field)][c][0] += 1; agg[r.get(field)][c][1] += bool(r.get("passed"))
        out[key] = {k: {c: 100*v[c][1]/v[c][0] for c in COND} for k, v in agg.items()}
    # repair detail
    fed = [recs[(m, "repair")][q] for m in models for q in qids if recs[(m, "repair")][q].get("repaired")]
    out["repair"] = {"records_with_feedback": len(fed), "fixed": sum(bool(r.get("passed")) for r in fed),
                     "fix_rate": 100*sum(bool(r.get("passed")) for r in fed)/max(len(fed), 1)}
    # tokens
    for c in COND:
        t = n = 0
        for m in models:
            for q in qids:
                r = recs[(m, c)][q]
                for k in ("usage_step1", "usage_step2", "usage_repair"):
                    u = r.get(k) or {}; t += u.get("total_tokens") or 0
                n += 1
        out["tokens"][c] = t/n
    json.dump(out, open(HERE / "results" / "paper_analyses_v4.json", "w"), indent=1)
    # report
    print(f"models={len(models)} problems={len(qids)}  (equivalence margin +/-{DELTA} pp)")
    print(f"\n{'condition':<10} {'Pass@1':>7} {'95% CI':>14} {'vs direct':>10} {'perm p':>8} {'boot90 CI':>16} {'equiv':>6} {'wins/losses':>12} {'tokens':>8}")
    for c in COND:
        P = out["pooled"][c]; v = out["vs_direct"].get(c)
        vs = f"{v['diff_pp']:+.1f}" if v else "-"; pp = f"{v['perm_p']:.4f}" if v else "-"
        ci = f"[{v['boot90_ci'][0]:+.1f},{v['boot90_ci'][1]:+.1f}]" if v else "-"; eq = ("yes" if v["equivalent_within_delta"] else "no") if v else "-"
        wl = f"{v['sig_wins']}/{v['sig_losses']}" if v else "-"
        print(f"{c:<10} {P['rate']:7.1f} [{P['ci'][0]:5.1f},{P['ci'][1]:5.1f}] {vs:>10} {pp:>8} {ci:>16} {eq:>6} {wl:>12} {out['tokens'][c]:8.0f}")
    nf = out["noise_floor"]
    print(f"\nNoise floor (four wordings of the direct prompt): pooled rates {', '.join(f'{k}={v:.1f}' for k, v in nf['pooled_rates'].items())}; pooled range {nf['pooled_range']:.1f} pp; max |dev from direct| {nf['pooled_max_abs_dev_from_direct']:.1f} pp; per-model range median {nf['median_per_model_range']:.1f} pp, max {max(nf['per_model_range'].values()):.1f} pp")
    print("\nBy difficulty:"); [print(f"  {k:<7}" + " ".join(f"{c}={v[c]:.0f}" for c in COND)) for k, v in out["by_difficulty"].items()]
    print(f"\nRepair: {out['repair']['records_with_feedback']} records received feedback, {out['repair']['fixed']} fixed ({out['repair']['fix_rate']:.0f}%)")
    print("\nPer model Pass@1:"); print(f"  {'model':<26}" + "".join(f"{c[:7]:>8}" for c in COND))
    for m in models: print(f"  {m:<26}" + "".join(f"{rate(m, c):8.0f}" for c in COND))

if __name__ == "__main__":
    main()
