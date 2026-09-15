#!/usr/bin/env python3
"""analyze_repair_rounds.py -- The feedback-loop failure mode. For each round
r = 0..R of public-test repair: hidden-test Pass@1, public-test pass rate, and
the share of solutions that pass the public tests but fail the hidden ones
(overfitting to the feedback signal). Pooled over models and per model; also
tokens per round. Writes results/repair_rounds.json and fig_rounds.pdf."""
import sys, json, glob, os, argparse
from collections import defaultdict
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent; D = HERE / "results" / "livecodebench_v3_strat"
DISP = {"gpt-4o": "GPT-4o", "gpt-4.1": "GPT-4.1", "grok-4-20-reasoning": "Grok-4-20-reas.", "grok-4-20-non-reasoning": "Grok-4-20-non-reas.",
        "gpt-5.4": "GPT-5.4", "gpt-5.4-mini-2": "GPT-5.4-mini", "gpt-5.4-nano": "GPT-5.4-nano", "gpt-5.5": "GPT-5.5", "gpt-5.3-codex": "GPT-5.3-codex"}

def state_after(rec, r):
    """(hidden_pass, public_pass) after round r (0 = initial)."""
    h, p = rec["round0"]["passed"], rec["round0"]["public_pass"]
    for rd in rec["rounds"][:r]:
        h, p = rd["passed"], rd["public_pass"]
    return h, p

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--outdir", default=str(HERE / "figures_paper")); ap.add_argument("--rounds", type=int, default=3); a = ap.parse_args()
    files = sorted(glob.glob(str(D / "repairR_*_s42.jsonl"))); R = a.rounds
    per = {}; pooled = defaultdict(lambda: [0, 0, 0, 0])  # r -> [n, hidden, public, pub_not_hidden]
    tokens = defaultdict(int); calls = defaultdict(int)
    for f in files:
        m = os.path.basename(f)[len("repairR_"):-len("_s42.jsonl")]
        rows = [json.loads(l) for l in open(f) if l.strip()]
        if len(rows) < 180: print(f"skip {m}: {len(rows)} rows"); continue
        per[m] = {}
        for r in range(R + 1):
            n = len(rows); h = p = pnh = 0
            for rec in rows:
                hh, pp = state_after(rec, r); h += hh; p += pp; pnh += (pp and not hh)
            per[m][r] = {"hidden": 100*h/n, "public": 100*p/n, "public_not_hidden": 100*pnh/n}
            q = pooled[r]; q[0] += n; q[1] += h; q[2] += p; q[3] += pnh
        for rec in rows:
            for rd in rec["rounds"]:
                tokens[rd["round"]] += (rd.get("usage") or {}).get("total_tokens") or 0; calls[rd["round"]] += 1
    out = {"per_model": per, "pooled": {r: {"n": v[0], "hidden": 100*v[1]/v[0], "public": 100*v[2]/v[0], "public_not_hidden": 100*v[3]/v[0]} for r, v in pooled.items() if v[0]},
           "calls_per_round": dict(calls), "tokens_per_round": dict(tokens), "n_models": len(per)}
    json.dump(out, open(HERE / "results" / "repair_rounds.json", "w"), indent=1)
    print(f"models={len(per)}"); print("round  hidden  public  public-not-hidden  calls")
    for r in range(R + 1):
        v = out["pooled"].get(r) or out["pooled"].get(str(r))
        if v: print(f"  {r}     {v['hidden']:5.1f}   {v['public']:5.1f}      {v['public_not_hidden']:5.1f}         {calls.get(r, 0)}")
    print("per model hidden by round:", {DISP.get(m, m): [round(per[m][r]["hidden"]) for r in range(R + 1)] for m in per})
    # figure
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e8e8e8", "axes.axisbelow": True, "figure.dpi": 150})
    fig, ax = plt.subplots(figsize=(6.0, 3.4)); xs = list(range(R + 1))
    pk = lambda r, k: (out["pooled"].get(r) or out["pooled"].get(str(r)))[k]
    ax.plot(xs, [pk(r, "public") for r in xs], "s-", color="#3b6ea5", label="passes public tests (what the loop sees)")
    ax.plot(xs, [pk(r, "hidden") for r in xs], "o-", color="#d1495b", label="passes all hidden tests (what it is judged by)")
    ax.fill_between(xs, [pk(r, "hidden") for r in xs], [pk(r, "public") for r in xs], color="#d1495b", alpha=0.12, label="passes public but fails hidden")
    ax.set_xticks(xs); ax.set_xticklabels(["initial"] + [f"round {r}" for r in xs[1:]]); ax.set_ylabel("share of problems (%)"); ax.set_ylim(75, 100)
    ax.legend(frameon=False, fontsize=8, loc="lower right"); ax.grid(axis="x")
    fig.tight_layout(); fig.savefig(Path(a.outdir) / "fig_rounds.pdf"); print("figure written")

if __name__ == "__main__":
    main()
