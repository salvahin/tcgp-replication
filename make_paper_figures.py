#!/usr/bin/env python3
"""
make_paper_figures.py -- Generate all figures used in the manuscript from the V2
results, for reproducibility. Part of the replication package.

Reads:
  results/humaneval_v2/*.jsonl              (HumanEval, 3 seeds)
  results/livecodebench_v2_strat/*.jsonl    (LiveCodeBench, stratified n=180)

Writes (PDF, vector) to --outdir (default: figures_paper/):
  fig_lcb_difficulty.pdf   LiveCodeBench Pass@1 by difficulty (grouped bars + CIs)
  fig_lcb_permodel.pdf     per-model Direct->TCGP dumbbell
  fig_cost_accuracy.pdf    two-panel bars: accuracy (with 95% CI) and token cost

Usage:
  python make_paper_figures.py [--outdir DIR]
  # e.g. to write straight into the manuscript repo:
  python make_paper_figures.py --outdir /path/to/paper/diagrams
"""

import argparse
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
STRAT = os.path.join(HERE, "results", "livecodebench_v2_strat")
HE = os.path.join(HERE, "results", "humaneval_v2")

COND = ["direct", "cot", "tcgp"]
LABEL = {"direct": "Direct", "cot": "CoT", "tcgp": "TCGP"}
COLOR = {"direct": "#9aa0a6", "cot": "#3b6ea5", "tcgp": "#d1495b"}
MARKER = {"direct": "o", "cot": "s", "tcgp": "^"}
DISPLAY = {
    "gpt-4o": "GPT-4o", "gpt-4.1": "GPT-4.1", "gemini-2.5-flash": "Gemini-2.5-Flash",
    "grok-4-20-reasoning": "Grok-4-20-reas.", "grok-4-20-non-reasoning": "Grok-4-20-non-reas.",
    "gpt-5.4": "GPT-5.4", "gpt-5.4-mini-2": "GPT-5.4-mini", "gpt-5.4-nano": "GPT-5.4-nano",
    "gpt-5.5": "GPT-5.5", "gpt-5.3-codex": "GPT-5.3-codex",
}


def style():
    plt.rcParams.update({
        "font.size": 11, "font.family": "serif",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e8e8e8", "grid.linewidth": 0.8,
        "axes.axisbelow": True, "figure.dpi": 150,
    })


def wilson(p, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    ph = p / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    m = z * math.sqrt((ph * (1 - ph) + z * z / (4 * n)) / n) / d
    return 100 * (c - m), 100 * (c + m)


def load_strat():
    """Return per-model, per-condition pass/total, token totals, and per-difficulty."""
    pm = defaultdict(lambda: defaultdict(lambda: [0, 0]))          # model->cond->[n,pass]
    tok = defaultdict(lambda: defaultdict(lambda: [0, 0]))         # model->cond->[tokens,count]
    diff = {c: {d: [0, 0] for d in ("easy", "medium", "hard")} for c in COND}
    for f in glob.glob(os.path.join(STRAT, "*_s42.jsonl")):
        b = os.path.basename(f)[:-len("_s42.jsonl")]
        cond, model = b.split("_", 1)
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            passed = 1 if r.get("passed") else 0
            pm[model][cond][0] += 1
            pm[model][cond][1] += passed
            u1, u2 = r.get("usage_step1") or {}, r.get("usage_step2") or {}
            tok[model][cond][0] += (u1.get("total_tokens", 0) or 0) + (u2.get("total_tokens", 0) or 0)
            tok[model][cond][1] += 1
            dd = str(r.get("difficulty", "?")).lower()
            if dd in diff[cond]:
                diff[cond][dd][0] += 1
                diff[cond][dd][1] += passed
    return pm, tok, diff


def fig_difficulty(diff, outdir):
    style()
    order = ["easy", "medium", "hard", "all"]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    x = np.arange(len(order))
    w = 0.26
    for i, c in enumerate(COND):
        vals, errs = [], []
        for d in order:
            if d == "all":
                p = sum(diff[c][k][1] for k in ("easy", "medium", "hard"))
                n = sum(diff[c][k][0] for k in ("easy", "medium", "hard"))
            else:
                p, n = diff[c][d][1], diff[c][d][0]
            rate = 100 * p / n
            lo, hi = wilson(p, n)
            vals.append(rate)
            errs.append([rate - lo, hi - rate])
        errs = np.array(errs).T
        ax.bar(x + (i - 1) * w, vals, w, yerr=errs, capsize=2, color=COLOR[c],
               label=LABEL[c], edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["Easy", "Medium", "Hard", "All"])
    ax.set_ylabel("Pass@1 (%)")
    ax.set_ylim(0, 70)
    ax.grid(axis="x")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    ax.set_title("LiveCodeBench by difficulty (n=600 per cell)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_lcb_difficulty.pdf"))
    plt.close(fig)


def fig_permodel(pm, outdir):
    style()
    def rate(m, c):
        return 100 * pm[m][c][1] / pm[m][c][0]
    rows = sorted(pm, key=lambda m: rate(m, "tcgp"))
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, m in enumerate(rows):
        d, t = rate(m, "direct"), rate(m, "tcgp")
        ax.plot([d, t], [i, i], color="#cccccc", lw=2, zorder=1)
        ax.scatter([d], [i], color=COLOR["direct"], s=45, zorder=2)
        ax.scatter([t], [i], color=COLOR["tcgp"], s=45, zorder=2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([DISPLAY.get(m, m) for m in rows])
    ax.set_xlabel("Pass@1 (%)")
    ax.set_xlim(0, 95)
    ax.grid(axis="y")
    ax.scatter([], [], color=COLOR["direct"], label="Direct")
    ax.scatter([], [], color=COLOR["tcgp"], label="TCGP")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("LiveCodeBench: per-model gain from TCGP")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_lcb_permodel.pdf"))
    plt.close(fig)


def fig_cost_accuracy(pm, tok, outdir):
    """Two panels: accuracy (pooled Pass@1 with 95% CI bars) and token cost
    (box plot of per-model tokens/problem, so the spread and reasoning-model
    outliers are visible)."""
    style()
    agg = {c: [sum(pm[m][c][0] for m in pm), sum(pm[m][c][1] for m in pm)] for c in COND}
    acc = [100 * agg[c][1] / agg[c][0] for c in COND]
    lo = [acc[i] - wilson(agg[c][1], agg[c][0])[0] for i, c in enumerate(COND)]
    hi = [wilson(agg[c][1], agg[c][0])[1] - acc[i] for i, c in enumerate(COND)]
    permodel_tokens = {c: [tok[m][c][0] / tok[m][c][1] for m in tok if tok[m][c][1]] for c in COND}
    tokmed = [int(np.median(permodel_tokens[c])) for c in COND]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 3.5))
    x = np.arange(3)
    cols = [COLOR[c] for c in COND]
    # left: accuracy bars with CI
    a1.bar(x, acc, yerr=[lo, hi], capsize=3, color=cols, width=0.62, edgecolor="white")
    a1.set_xticks(x); a1.set_xticklabels([LABEL[c] for c in COND])
    a1.set_ylabel("Pass@1 (%)"); a1.set_ylim(0, 55)
    a1.set_title("Accuracy (higher is better)", fontsize=11); a1.grid(axis="x")
    for i, v in enumerate(acc):
        a1.text(i, v + hi[i] + 1.2, f"{v:.0f}", ha="center", fontsize=10, fontweight="bold")
    # right: cost box plot (per-model tokens per problem)
    data = [permodel_tokens[c] for c in COND]
    bp = a2.boxplot(data, positions=x, widths=0.5, patch_artist=True, showfliers=True,
                    medianprops=dict(color="black", lw=1.4),
                    flierprops=dict(marker="o", markersize=4, markerfacecolor="none", markeredgecolor="#555"))
    for patch, c in zip(bp["boxes"], COND):
        patch.set_facecolor(COLOR[c]); patch.set_alpha(0.75); patch.set_edgecolor("#555")
    for elem in bp["whiskers"] + bp["caps"]:
        elem.set_color("#555")
    a2.set_xticks(x); a2.set_xticklabels([LABEL[c] for c in COND])
    a2.set_ylabel("Tokens / problem (per model)")
    a2.set_title("Cost (lower is better)", fontsize=11); a2.grid(axis="x")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_cost_accuracy.pdf"), bbox_inches="tight")
    plt.close(fig)
    return acc, tokmed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(HERE, "figures_paper"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    pm, tok, diff = load_strat()
    fig_difficulty(diff, args.outdir)
    fig_permodel(pm, args.outdir)
    acc, tokmed = fig_cost_accuracy(pm, tok, args.outdir)
    print("Wrote figures to", args.outdir)
    print("  LCB accuracy (Direct/CoT/TCGP):", [round(a) for a in acc])
    print("  LCB median tokens (Direct/CoT/TCGP):", tokmed)


if __name__ == "__main__":
    main()
