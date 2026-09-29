"""Compute confidence intervals and paired significance for v4 benchmark results.

Reads:
  - results/azure_full_benchmark_mitigation_eval_v4.json
  - results/claude_full_benchmark_mitigation_eval_v4.json
  - results/gpt52_full_benchmark_mitigation_eval_v4.json

Writes:
  - results/full_benchmark_significance.json
  - results/full_benchmark_significance.txt
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from speech_act_contract.stats.significance import (
    paired_binary_improvement_pvalue,
    wilson_interval,
)


INPUTS = {
    "gpt-4o": ROOT / "results" / "azure_full_benchmark_mitigation_eval_v4.json",
    "claude-sonnet-4-6": ROOT / "results" / "claude_full_benchmark_mitigation_eval_v4.json",
    "gpt-5.2": ROOT / "results" / "gpt52_full_benchmark_mitigation_eval_v4.json",
}


def summarize_model(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data["records"]
    total = len(records)

    before_uptake = sum(record["before_uptake_ok"] for record in records)
    after_uptake = sum(record["after_uptake_ok"] for record in records)
    before_void = sum(record["before_void_commitment"] for record in records)
    after_void = sum(record["after_void_commitment"] for record in records)

    uptake_improved = sum(
        (not record["before_uptake_ok"]) and record["after_uptake_ok"] for record in records
    )
    uptake_worsened = sum(
        record["before_uptake_ok"] and (not record["after_uptake_ok"]) for record in records
    )
    void_fixed = sum(
        record["before_void_commitment"] and (not record["after_void_commitment"])
        for record in records
    )
    void_added = sum(
        (not record["before_void_commitment"]) and record["after_void_commitment"]
        for record in records
    )

    return {
        "num_items": total,
        "uptake": {
            "before_rate": before_uptake / total,
            "before_count": before_uptake,
            "before_wilson_95": wilson_interval(before_uptake, total),
            "after_rate": after_uptake / total,
            "after_count": after_uptake,
            "after_wilson_95": wilson_interval(after_uptake, total),
            "improved_count": uptake_improved,
            "worsened_count": uptake_worsened,
            "paired_pvalue": paired_binary_improvement_pvalue(uptake_improved, uptake_worsened),
        },
        "void_commitment": {
            "before_rate": before_void / total,
            "before_count": before_void,
            "before_wilson_95": wilson_interval(before_void, total),
            "after_rate": after_void / total,
            "after_count": after_void,
            "after_wilson_95": wilson_interval(after_void, total),
            "fixed_count": void_fixed,
            "added_count": void_added,
            "paired_pvalue": paired_binary_improvement_pvalue(void_fixed, void_added),
        },
    }


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def format_ci(interval: tuple[float, float]) -> str:
    lo, hi = interval
    return f"[{lo * 100:.1f}, {hi * 100:.1f}]"


def main() -> None:
    summary = {model: summarize_model(path) for model, path in INPUTS.items()}

    json_path = ROOT / "results" / "full_benchmark_significance.json"
    txt_path = ROOT / "results" / "full_benchmark_significance.txt"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = ["Full benchmark significance summary"]
    for model, stats in summary.items():
        uptake = stats["uptake"]
        void = stats["void_commitment"]
        lines.extend(
            [
                "",
                model,
                (
                    f"  Uptake: {format_pct(uptake['before_rate'])} {format_ci(uptake['before_wilson_95'])}"
                    f" -> {format_pct(uptake['after_rate'])} {format_ci(uptake['after_wilson_95'])}; "
                    f"improved={uptake['improved_count']}, worsened={uptake['worsened_count']}, "
                    f"paired p={uptake['paired_pvalue']:.3g}"
                ),
                (
                    f"  Void:   {format_pct(void['before_rate'])} {format_ci(void['before_wilson_95'])}"
                    f" -> {format_pct(void['after_rate'])} {format_ci(void['after_wilson_95'])}; "
                    f"fixed={void['fixed_count']}, added={void['added_count']}, "
                    f"paired p={void['paired_pvalue']:.3g}"
                ),
            ]
        )

    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved {json_path}")
    print(f"Saved {txt_path}")


if __name__ == "__main__":
    main()
