#!/usr/bin/env python3
"""make_paper_figures_v3.py -- Figures for the revised manuscript, from the
analysis JSONs (paper_analyses_v3/v4, harness_sensitivity_v3) and the logged
records. Usage: make_paper_figures_v3.py --outdir DIR"""
import argparse, json, glob, os
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent; R = HERE / "results"
plt.rcParams.update({"font.size": 10, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e8e8e8", "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.dpi": 150})
DISP = {"gpt-4o": "GPT-4o", "gpt-4.1": "GPT-4.1", "gemini-2.5-flash": "Gemini-2.5-Flash", "grok-4-20-reasoning": "Grok-4-20-reas.",
        "grok-4-20-non-reasoning": "Grok-4-20-non-reas.", "gpt-5.4": "GPT-5.4", "gpt-5.4-mini-2": "GPT-5.4-mini", "gpt-5.4-nano": "GPT-5.4-nano",
        "gpt-5.5": "GPT-5.5", "gpt-5.3-codex": "GPT-5.3-codex"}
LAB = {"para1": "Paraphrase 1", "para2": "Paraphrase 2", "para3": "Paraphrase 3", "persona": "Persona", "fewshot": "Few-shot",
       "plansolve": "Plan-and-solve", "tcgp": "Test-case-guided", "restate": "Restate contract", "stacked": "Stacked recipe",
       "cot": "Chain-of-thought", "repair": "One repair round"}

def rate_file(f):
    rows = [json.loads(l) for l in open(f) if l.strip()]; return 100 * sum(bool(r.get("passed")) for r in rows) / len(rows), len(rows)

def fig_harness(out, v4):
    """All values come from results/harness_budget.json (make_harness_budget.py)."""
    B = json.load(open(R / "harness_budget.json")); rows = []
    c = B["contract"]; rows.append(("I/O contract: functional problems\nserialized through stdin", abs(c["deflation_direct_pp"]), "direct prompting"))
    be = B.get("body_extraction_humaneval") or {}
    if be:
        m = min(be, key=lambda k: be[k]["diff_pp"]); rows.append(("Function-body extraction with\nre-indentation (HumanEval)", abs(be[m]["diff_pp"]), DISP.get(m, m)))
    rows.append(("Manufactured strategy effect\n(same outputs, two evaluators)", abs(c["stdin_contract"]["tcgp_vs_direct"]["diff_pp"]), "TCGP vs direct"))
    ex = B.get("extraction_rule") or {}
    if ex:
        k = max(ex, key=lambda q: abs(ex[q]["delta_first_block_pp"])); rows.append(("Extraction rule, first vs last block", abs(ex[k]["delta_first_block_pp"]), LAB.get(k, k)))
    po = B.get("public_only_inflation_pp") or {}
    if po: rows.append(("Public tests only vs hidden tests", sum(po.values()) / len(po), f"mean; {min(po.values()):.1f} to {max(po.values()):.1f}"))
    w = B.get("wording_lcb")
    if w: rows.append(("Prompt wording (4 paraphrases,\npooled range)", w["pooled_range_pp"], f"max per model: {max(w['per_model_range_pp'].values()):.0f}"))
    cp = B.get("output_cap") or {}
    if cp: rows.append(("Output cap, 4k vs 12k tokens\n(reasoning models)", max(abs(v["diff_pp"]) for v in cp.values()), "both models"))
    cc = B.get("crosscheck")
    if cc: rows.append(("Timeouts under parallel load", 100 * (1 - cc["agreement_first_pass"]), "verdicts disagreeing"))
    fl = B.get("flakiness")
    if fl: rows.append(("Grading non-determinism (isolated)", 100 * fl["share_of_all_records"], "unstable verdicts"))
    rows.sort(key=lambda r: -r[1])
    recipe = abs(B["largest_recipe_effect"]["diff_pp"]) if B.get("largest_recipe_effect") else None
    fig, ax = plt.subplots(figsize=(7.0, 4.6)); y = np.arange(len(rows))[::-1]; vals = [r[1] for r in rows]
    ax.barh(y, vals, color=["#d1495b" if v >= 10 else "#3b6ea5" if v >= 1 else "#9aa0a6" for v in vals], height=0.62)
    for yi, (lab, v, note) in zip(y, rows): ax.text(max(v, 0.06) * 1.15, yi, f"{v:.3g} pp  ({note})", va="center", fontsize=8)
    if recipe:
        ax.axvline(recipe, color="black", ls="--", lw=1); ax.text(recipe * 1.05, y[0] + 0.55, f"largest recipe effect\n({recipe:.1f} pp)", fontsize=8, va="bottom")
    ax.set_xscale("log"); ax.set_xlim(0.03, 3000); ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlabel("Effect on measured Pass@1 (percentage points, log scale)"); ax.grid(axis="y")
    fig.tight_layout(); fig.savefig(out / "fig_harness_budget.pdf"); plt.close(fig)

def fig_forest(out, v4):
    conds = ["para1", "para2", "para3", "persona", "fewshot", "plansolve", "tcgp", "restate", "stacked", "cot", "repair"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2)); y = np.arange(len(conds))[::-1]; d = v4["delta_pp"]; nf = v4["noise_floor"]["pooled_max_abs_dev_from_direct"]
    ax.axvspan(-d, d, color="#e8f0e8", zorder=0, label=f"equivalence margin (±{d:.0f} pp)")
    ax.axvspan(-nf, nf, color="#cfd8e3", zorder=0, alpha=0.7, label=f"paraphrase noise (±{nf:.1f} pp)")
    for yi, c in zip(y, conds):
        v = v4["vs_direct"][c]; lo, hi = v["boot90_ci"]; col = "#d1495b" if not v["equivalent_within_delta"] else "#3b6ea5"
        ax.plot([lo, hi], [yi, yi], color=col, lw=2); ax.plot(v["diff_pp"], yi, "o", color=col, ms=6)
    ax.axvline(0, color="black", lw=0.8); ax.set_yticks(y); ax.set_yticklabels([LAB[c] for c in conds])
    ax.set_xlabel("Pass@1 difference vs. direct prompting (pp), 90% cluster-bootstrap CI"); ax.legend(frameon=False, fontsize=8, loc="upper left"); ax.grid(axis="y")
    fig.tight_layout(); fig.savefig(out / "fig_recipe_forest.pdf"); plt.close(fig)

def fig_wording(out, v4):
    models = sorted(v4["models"], key=lambda m: v4["vs_direct"]["cot"]["per_model"][m]["direct"])
    fig, ax = plt.subplots(figsize=(6.4, 4.0)); y = np.arange(len(models))
    for yi, m in zip(y, models):
        rng = v4["noise_floor"]["per_model_range"][m]
        gains = [v4["vs_direct"][c]["per_model"][m]["rate"] - v4["vs_direct"][c]["per_model"][m]["direct"] for c in ("persona", "fewshot", "plansolve", "tcgp", "stacked", "cot")]
        ax.barh(yi, rng, color="#cfd8e3", height=0.6, zorder=1)
        ax.scatter([abs(g) for g in gains], [yi] * len(gains), color="#d1495b", s=18, zorder=2)
    ax.set_yticks(y); ax.set_yticklabels([DISP[m] for m in models]); ax.set_xlabel("percentage points")
    ax.barh([], [], color="#cfd8e3", label="wording range (4 paraphrases of the same prompt)"); ax.scatter([], [], color="#d1495b", s=18, label="|recipe effect| vs. direct (6 recipes)")
    ax.legend(frameon=False, fontsize=8, loc="upper right"); ax.grid(axis="y")
    fig.tight_layout(); fig.savefig(out / "fig_wording_vs_recipes.pdf"); plt.close(fig)

def fig_artifact(out):
    c = json.load(open(R / "harness_budget.json"))["contract"]; conds = ["direct", "cot", "tcgp"]
    series = [([c["stdin_contract"][k] for k in conds], "standard-input contract", "#d1495b"),
              ([c["same_outputs_official"][k] for k in conds], "same outputs, official evaluator", "#3b6ea5"),
              ([c["official_fresh"][k] for k in conds], "official contract, fresh run", "#5a9e6f")]
    fig, ax = plt.subplots(figsize=(6.0, 3.2)); x = np.arange(3); w = 0.26
    for i, (vals, lab, col) in enumerate(series):
        ax.bar(x + (i - 1) * w, vals, w, color=col, label=lab)
        for xi, v in zip(x + (i - 1) * w, vals): ax.text(xi, v + 1, f"{v:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(["Direct", "CoT", "TCGP"]); ax.set_ylabel("Pass@1 (%)"); ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8, loc="upper left"); ax.grid(axis="x")
    fig.tight_layout(); fig.savefig(out / "fig_artifact.pdf"); plt.close(fig)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--outdir", default=str(HERE / "figures_paper")); a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    v4 = json.load(open(R / "paper_analyses_v4.json"))
    fig_harness(out, v4); fig_forest(out, v4); fig_wording(out, v4); fig_artifact(out)
    print("figures written to", out)

if __name__ == "__main__":
    main()
