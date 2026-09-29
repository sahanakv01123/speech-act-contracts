"""Generate all 4 publication figures for the speech-act-contract paper.

Outputs (PDF + PNG) to results/figures/:
  fig1_pipeline_effect.{pdf,png}   — before/after uptake per model
  fig2_family_repair.{pdf,png}     — failure-repaired rate by speech-act family
  fig3_quality_preservation.{pdf,png} — helpfulness & naturalness preserved
  fig4_classifier_validation.{pdf,png} — classifier Cohen's κ per family

Usage:
    python scripts/generate_figures.py
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT   = Path(__file__).resolve().parents[1]
FIGS   = ROOT / "results" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)
RESULTS = ROOT / "results"

# ── colour palette ────────────────────────────────────────────────────────────
C_GPT4O  = "#4472C4"   # blue
C_CLAUDE = "#ED7D31"   # orange
C_GPT52  = "#70AD47"   # green
BEFORE   = "#BFBFBF"   # light grey
AFTER    = "#2E75B6"   # strong blue

MODELS = ["gpt-4o", "Claude\nSonnet 4.6", "GPT-5.2"]
MODEL_COLORS = [C_GPT4O, C_CLAUDE, C_GPT52]

# ── helper ────────────────────────────────────────────────────────────────────
def save(fig, name):
    for ext in ("pdf", "png"):
        p = FIGS / f"{name}.{ext}"
        fig.savefig(p, dpi=180, bbox_inches="tight")
        print(f"  saved {p.name}")
    plt.close(fig)


def load_json(name: str):
    return json.loads((RESULTS / name).read_text())


# ─────────────────────────────────────────────────────────────────────────────
# FIG 1 — Pipeline effect: before vs after uptake accuracy
# ─────────────────────────────────────────────────────────────────────────────
def fig1():
    files = [
        RESULTS / "azure_full_benchmark_mitigation_eval_v4.json",
        RESULTS / "claude_full_benchmark_mitigation_eval_v4.json",
        RESULTS / "gpt52_full_benchmark_mitigation_eval_v4.json",
    ]
    before_vals, after_vals = [], []
    for p in files:
        d = json.loads(p.read_text())
        before_vals.append(d["before"]["uptake_accuracy"])
        after_vals.append(d["after"]["uptake_accuracy"])

    x = np.arange(len(MODELS))
    width = 0.32

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    bars_b = ax.bar(x - width/2, before_vals, width, color=BEFORE, edgecolor="white", label="Before mitigation")
    bars_a = ax.bar(x + width/2, after_vals,  width, color=AFTER,  edgecolor="white", label="After mitigation")

    for bar in bars_b:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.012, f"{h:.2f}",
                ha="center", va="bottom", fontsize=8.5, color="#555")
    for bar in bars_a:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.012, f"{h:.2f}",
                ha="center", va="bottom", fontsize=8.5, color=AFTER, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=10)
    ax.set_ylabel("Uptake Accuracy", fontsize=11)
    ax.set_ylim(0, 1.14)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(
        frameon=False,
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        borderaxespad=0.0,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Speech-Act Uptake: Before vs. After Mitigation Pipeline", fontsize=11, pad=10)
    fig.subplots_adjust(top=0.88, bottom=0.22)
    save(fig, "fig1_pipeline_effect")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 — Failure-repaired rate by speech-act family
# ─────────────────────────────────────────────────────────────────────────────
def fig2():
    families = ["apology", "correction", "frustration", "offer", "preference", "thanks"]
    fam_labels = ["Apology", "Correction", "Frustration", "Offer", "Preference", "Thanks"]
    judge = load_json("judge_analysis.json")
    gpt4o_rep  = [judge["gpt-4o"]["by_family"][f]["failure_repaired_rate"] for f in families]
    claude_rep = [judge["claude-sonnet-4-6"]["by_family"][f]["failure_repaired_rate"] for f in families]
    gpt52_rep  = [judge["gpt-5.2"]["by_family"][f]["failure_repaired_rate"] for f in families]

    x     = np.arange(len(families))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8.1, 4.8))
    ax.bar(x - width, gpt4o_rep,  width, color=C_GPT4O,  label="gpt-4o",          edgecolor="white")
    ax.bar(x,         claude_rep, width, color=C_CLAUDE, label="Claude Sonnet 4.6",edgecolor="white")
    ax.bar(x + width, gpt52_rep,  width, color=C_GPT52,  label="GPT-5.2",          edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(fam_labels, fontsize=10)
    ax.set_ylabel("Failure Repaired Rate", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(
        frameon=False,
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=3,
        borderaxespad=0.0,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Failure-Repaired Rate by Speech-Act Family (LLM Judge)", fontsize=11, pad=10)
    ax.axhline(1.0, color="#999", linestyle="--", linewidth=0.8, alpha=0.6)
    fig.subplots_adjust(top=0.88, bottom=0.22)
    save(fig, "fig2_family_repair")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 — Quality preservation: helpfulness & naturalness
# ─────────────────────────────────────────────────────────────────────────────
def fig3():
    categories  = ["Helpfulness\nPreserved", "Naturalness\nPreserved", "Overall\nPreference ↑"]
    judge = load_json("judge_analysis.json")
    gpt4o_vals  = [
        judge["gpt-4o"]["overall"]["helpfulness_preserved"],
        judge["gpt-4o"]["overall"]["naturalness_preserved"],
        judge["gpt-4o"]["overall"]["overall_preference_after"],
    ]
    claude_vals = [
        judge["claude-sonnet-4-6"]["overall"]["helpfulness_preserved"],
        judge["claude-sonnet-4-6"]["overall"]["naturalness_preserved"],
        judge["claude-sonnet-4-6"]["overall"]["overall_preference_after"],
    ]
    gpt52_vals  = [
        judge["gpt-5.2"]["overall"]["helpfulness_preserved"],
        judge["gpt-5.2"]["overall"]["naturalness_preserved"],
        judge["gpt-5.2"]["overall"]["overall_preference_after"],
    ]

    x     = np.arange(len(categories))
    width = 0.25

    fig, ax = plt.subplots(figsize=(6.7, 4.2))
    ax.bar(x - width, gpt4o_vals,  width, color=C_GPT4O,  label="gpt-4o",           edgecolor="white")
    ax.bar(x,         claude_vals, width, color=C_CLAUDE, label="Claude Sonnet 4.6", edgecolor="white")
    ax.bar(x + width, gpt52_vals,  width, color=C_GPT52,  label="GPT-5.2",           edgecolor="white")

    # Annotate bars
    for bars, vals in zip(
        [ax.containers[0], ax.containers[1], ax.containers[2]],
        [gpt4o_vals, claude_vals, gpt52_vals],
    ):
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color="#444")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylabel("Rate (judge evaluation)", fontsize=11)
    ax.set_ylim(0, 1.16)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(
        frameon=False,
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=3,
        borderaxespad=0.0,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Quality Preservation After Mitigation (LLM Judge)", fontsize=11, pad=10)
    fig.subplots_adjust(top=0.88, bottom=0.22)
    save(fig, "fig3_quality_preservation")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 4 — Classifier validation: Cohen's κ per family
# ─────────────────────────────────────────────────────────────────────────────
def fig4():
    d = load_json("classifier_validation_scores.json")
    pf = d["per_family"]

    families = ["apology", "correction", "frustration", "offer", "preference", "thanks"]
    fam_labels = ["Apology", "Correction", "Frustration", "Offer", "Preference", "Thanks"]
    kappas = [pf[f]["cohens_kappa"] for f in families]
    f1s = [pf[f]["f1"] for f in families]

    # A family whose classifier gave the same label to every item has no
    # prediction variance, so Cohen's kappa is degenerate (~0) even when
    # accuracy/F1 is high. Show F1 for those, hatched and greyed, not as a
    # "poor" red bar.
    degenerate = [k == 0.0 for k in kappas]

    heights, bar_colors = [], []
    for k, f1, deg in zip(kappas, f1s, degenerate):
        if deg:
            heights.append(f1)
            bar_colors.append("#BBBBBB")            # neutral: kappa undefined
        else:
            heights.append(k)
            if k >= 0.80:   bar_colors.append("#2E75B6")   # near-perfect
            elif k >= 0.60: bar_colors.append("#70AD47")  # substantial
            elif k >= 0.40: bar_colors.append("#ED7D31")  # moderate
            else:           bar_colors.append("#FF4444")   # poor

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(fam_labels, heights, color=bar_colors, edgecolor="white", width=0.55)
    for bar, deg in zip(bars, degenerate):
        if deg:
            bar.set_hatch("//")

    for bar, k, f1, deg in zip(bars, kappas, f1s, degenerate):
        label = f"$\\kappa$ n/a\n(F1={f1:.2f})" if deg else f"κ={k:.3f}"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                label, ha="center", va="bottom", fontsize=8.5)

    # Reference lines
    for level, label, ls in [(0.80, "near-perfect (0.80)", "--"), (0.60, "substantial (0.60)", ":"), (0.40, "moderate (0.40)", "-.")]:
        ax.axhline(level, color="#888", linewidth=0.9, linestyle=ls, alpha=0.7)
        ax.text(5.6, level + 0.01, label, fontsize=7, color="#666", ha="right")

    # Overall κ annotation
    overall_k = d["overall"]["cohens_kappa"]
    ax.text(0.5, 0.93, f"Overall κ = {overall_k:.3f}  |  F1 = {d['overall']['f1']:.3f}",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=9.5, color="#222",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0F4FF", edgecolor="#AAAACC"))

    # Legend patches
    legend_patches = [
        mpatches.Patch(color="#2E75B6", label="Near-perfect (≥0.80)"),
        mpatches.Patch(color="#70AD47", label="Substantial (0.60–0.80)"),
        mpatches.Patch(color="#ED7D31", label="Moderate (0.40–0.60)"),
        mpatches.Patch(color="#FF4444", label="Poor (<0.40)"),
        mpatches.Patch(facecolor="#BBBBBB", hatch="//", label="κ undefined (bar = F1)"),
    ]
    ax.legend(
        handles=legend_patches,
        frameon=False,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=3,
        borderaxespad=0.0,
    )

    ax.set_ylabel("Cohen's κ  (grey: F1 where κ undefined)", fontsize=10.5)
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Classifier Validation: Cohen's κ per Speech-Act Family", fontsize=11, pad=10)
    fig.subplots_adjust(top=0.88, bottom=0.24)
    save(fig, "fig4_classifier_validation")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating figures …")
    fig1()
    fig2()
    fig3()
    fig4()
    print("Done.")
