"""Per-family and per-difficulty breakdown of judge results across all three models.

Reads:
  results/*_failure_judge_records.jsonl  (judge-level records)
  results/*_mitigation_eval_v4.json      (automatic eval, for baseline numbers)

Writes:
  results/judge_analysis.json   — full breakdown for programmatic use
  results/judge_analysis.txt    — human-readable report for paper writing

Usage:
    python scripts/analyze_judge_results.py
"""

from collections import defaultdict
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def rate(items, key, value=True):
    if not items:
        return None
    if value is True:
        return round(sum(1 for r in items if r.get(key) is True) / len(items), 3)
    return round(sum(1 for r in items if r.get(key) == value) / len(items), 3)


def family_breakdown(records: list[dict]) -> dict:
    by_family = defaultdict(list)
    for r in records:
        by_family[r["speech_act_family"]].append(r)

    result = {}
    for fam in sorted(by_family):
        recs = by_family[fam]
        result[fam] = {
            "n": len(recs),
            "failure_repaired_rate": rate(recs, "failure_repaired"),
            "overall_preference_after": rate(recs, "overall_preference", "after"),
            "pragmatic_better_after": rate(recs, "pragmatic_better", "after"),
            "helpfulness_preserved": rate(recs, "helpfulness_preserved"),
            "naturalness_preserved": rate(recs, "naturalness_preserved"),
        }
    return result


def difficulty_breakdown(records: list[dict]) -> dict:
    by_diff = defaultdict(list)
    for r in records:
        by_diff[r["difficulty"]].append(r)

    result = {}
    for diff in sorted(by_diff):
        recs = by_diff[diff]
        result[diff] = {
            "n": len(recs),
            "failure_repaired_rate": rate(recs, "failure_repaired"),
            "helpfulness_preserved": rate(recs, "helpfulness_preserved"),
            "naturalness_preserved": rate(recs, "naturalness_preserved"),
        }
    return result


def baseline_from_mitigation_eval(path: Path) -> dict:
    """Extract per-family baseline + after uptake from the automatic eval."""
    data = json.loads(path.read_text(encoding="utf-8"))
    before = data.get("before", {})
    after = data.get("after", {})
    return {
        "baseline_uptake": before.get("uptake_accuracy"),
        "after_uptake": after.get("uptake_accuracy"),
        "baseline_void": before.get("void_commitment_rate"),
        "after_void": after.get("void_commitment_rate"),
    }


# ---------------------------------------------------------------------------
# Model registry — maps model_name to (judge_records_file, mitigation_eval_file)
# ---------------------------------------------------------------------------

MODEL_FILES = {
    "gpt-5.2": (
        RESULTS / "gpt52_full_benchmark_mitigation_eval_v4_failure_judge_records.jsonl",
        RESULTS / "gpt52_full_benchmark_mitigation_eval_v4.json",
    ),
    "gpt-4o": (
        RESULTS / "azure_full_benchmark_mitigation_eval_v4_failure_judge_records.jsonl",
        RESULTS / "azure_full_benchmark_mitigation_eval_v4.json",
    ),
    "claude-sonnet-4-6": (
        RESULTS / "claude_full_benchmark_mitigation_eval_v4_failure_judge_records.jsonl",
        RESULTS / "claude_full_benchmark_mitigation_eval_v4.json",
    ),
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    all_models = {}

    for model_name, (judge_path, eval_path) in MODEL_FILES.items():
        if not judge_path.exists():
            print(f"WARNING: missing {judge_path.name} — skipping {model_name}")
            continue

        records = load_jsonl(judge_path)
        baseline = baseline_from_mitigation_eval(eval_path) if eval_path.exists() else {}

        all_models[model_name] = {
            "n_judged": len(records),
            "overall": {
                "failure_repaired_rate": rate(records, "failure_repaired"),
                "overall_preference_after": rate(records, "overall_preference", "after"),
                "pragmatic_better_after": rate(records, "pragmatic_better", "after"),
                "helpfulness_preserved": rate(records, "helpfulness_preserved"),
                "naturalness_preserved": rate(records, "naturalness_preserved"),
            },
            "baseline": baseline,
            "by_family": family_breakdown(records),
            "by_difficulty": difficulty_breakdown(records),
        }

    # Save JSON
    json_out = RESULTS / "judge_analysis.json"
    json_out.write_text(json.dumps(all_models, indent=2), encoding="utf-8")

    # ---------------------------------------------------------------------------
    # Build text report
    # ---------------------------------------------------------------------------
    lines = []
    models = list(all_models.keys())

    def row(label, *vals, width=28):
        parts = [f"{label:<{width}}"]
        for v in vals:
            if v is None:
                parts.append(f"{'—':>10}")
            elif isinstance(v, float):
                parts.append(f"{v:>10.3f}")
            elif isinstance(v, int):
                parts.append(f"{v:>10d}")
            else:
                parts.append(f"{str(v):>10}")
        return "  ".join(parts)

    header = row("", *models)
    sep = "-" * len(header)

    # ---- Overall summary ----
    lines += [
        "=" * len(header),
        "OVERALL RESULTS (failure cases only)",
        "=" * len(header),
        header, sep,
    ]
    for metric, key in [
        ("N judged",             "n_judged"),
        ("Baseline uptake",      ("baseline", "baseline_uptake")),
        ("After uptake",         ("baseline", "after_uptake")),
        ("Baseline void rate",   ("baseline", "baseline_void")),
        ("After void rate",      ("baseline", "after_void")),
        ("Failure repaired",     ("overall", "failure_repaired_rate")),
        ("Pref → after",         ("overall", "overall_preference_after")),
        ("Pragmatic better",     ("overall", "pragmatic_better_after")),
        ("Helpfulness preserved",("overall", "helpfulness_preserved")),
        ("Naturalness preserved",("overall", "naturalness_preserved")),
    ]:
        vals = []
        for m in models:
            d = all_models.get(m, {})
            if isinstance(key, tuple):
                v = d.get(key[0], {}).get(key[1])
            else:
                v = d.get(key)
            vals.append(v)
        lines.append(row(metric, *vals))
    lines.append("")

    # ---- Per-family breakdown ----
    all_families = sorted({
        fam
        for m in all_models.values()
        for fam in m.get("by_family", {})
    })

    for metric_label, metric_key in [
        ("FAILURE REPAIRED RATE BY FAMILY",    "failure_repaired_rate"),
        ("HELPFULNESS PRESERVED BY FAMILY",    "helpfulness_preserved"),
        ("NATURALNESS PRESERVED BY FAMILY",    "naturalness_preserved"),
        ("OVERALL PREF → AFTER BY FAMILY",     "overall_preference_after"),
    ]:
        lines += ["", "=" * len(header), metric_label, "=" * len(header), header, sep]
        for fam in all_families:
            vals = []
            for m in models:
                v = all_models.get(m, {}).get("by_family", {}).get(fam, {}).get(metric_key)
                vals.append(v)
            # include n for first metric only
            n_vals = [all_models.get(m, {}).get("by_family", {}).get(fam, {}).get("n") for m in models]
            n_str = "  ".join(f"n={v}" for v in n_vals if v is not None)
            lines.append(row(fam, *vals) + f"   [{n_str}]")

    # ---- Per-difficulty breakdown ----
    all_diffs = sorted({
        d
        for m in all_models.values()
        for d in m.get("by_difficulty", {})
    })

    for metric_label, metric_key in [
        ("FAILURE REPAIRED RATE BY DIFFICULTY", "failure_repaired_rate"),
        ("HELPFULNESS PRESERVED BY DIFFICULTY", "helpfulness_preserved"),
        ("NATURALNESS PRESERVED BY DIFFICULTY", "naturalness_preserved"),
    ]:
        lines += ["", "=" * len(header), metric_label, "=" * len(header), header, sep]
        for diff in all_diffs:
            vals = []
            for m in models:
                v = all_models.get(m, {}).get("by_difficulty", {}).get(diff, {}).get(metric_key)
                vals.append(v)
            n_vals = [all_models.get(m, {}).get("by_difficulty", {}).get(diff, {}).get("n") for m in models]
            n_str = "  ".join(f"n={v}" for v in n_vals if v is not None)
            lines.append(row(diff, *vals) + f"   [{n_str}]")

    lines.append("")

    report = "\n".join(lines)
    txt_out = RESULTS / "judge_analysis.txt"
    txt_out.write_text(report, encoding="utf-8")

    print(report)
    print(f"\nSaved JSON  → {json_out}")
    print(f"Saved report → {txt_out}")


if __name__ == "__main__":
    main()
