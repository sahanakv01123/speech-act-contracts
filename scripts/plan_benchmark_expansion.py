"""Plan a 1000+ benchmark expansion from the current curated core benchmark.

Default target:
  - total items: 1008
  - 168 items per family
  - 56 items per family per difficulty

Writes:
  - results/benchmark_expansion_plan.json
  - results/benchmark_expansion_plan.txt
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from speech_act_contract.data.loader import load_full_benchmark


BENCHMARK_PATH = ROOT / "data" / "annotations" / "full_uptake_benchmark_v3.jsonl"
TARGET_TOTAL = 1008
FAMILIES = ["apology", "thanks", "offer", "correction", "frustration", "preference"]
DIFFICULTIES = ["easy", "near_miss", "mixed_act"]
TARGET_PER_FAMILY = TARGET_TOTAL // len(FAMILIES)
TARGET_PER_FAMILY_DIFFICULTY = TARGET_PER_FAMILY // len(DIFFICULTIES)

EXPANSION_SOURCE_TYPES = [
    "synthetic",
    "adapted_from_real_conversation",
    "mined_from_public_corpus",
    "human_written_naturalistic",
]


def main() -> None:
    items = load_full_benchmark(BENCHMARK_PATH)

    family_counts = Counter()
    family_difficulty_counts = defaultdict(Counter)

    for item in items:
        family_counts[item.speech_act_family] += 1
        family_difficulty_counts[item.speech_act_family][item.difficulty] += 1

    plan = {
        "benchmark_path": str(BENCHMARK_PATH),
        "current_total": len(items),
        "target_total": TARGET_TOTAL,
        "additional_items_needed": TARGET_TOTAL - len(items),
        "target_per_family": TARGET_PER_FAMILY,
        "target_per_family_difficulty": TARGET_PER_FAMILY_DIFFICULTY,
        "recommended_expansion_source_types": EXPANSION_SOURCE_TYPES,
        "by_family": {},
    }

    text_lines = [
        "Benchmark expansion plan",
        f"Current total: {len(items)}",
        f"Target total: {TARGET_TOTAL}",
        f"Additional items needed: {TARGET_TOTAL - len(items)}",
        f"Target per family: {TARGET_PER_FAMILY}",
        f"Target per family per difficulty: {TARGET_PER_FAMILY_DIFFICULTY}",
        "",
    ]

    for family in FAMILIES:
        current_family_total = family_counts[family]
        family_plan = {
            "current_total": current_family_total,
            "target_total": TARGET_PER_FAMILY,
            "additional_needed": TARGET_PER_FAMILY - current_family_total,
            "by_difficulty": {},
        }
        text_lines.append(f"{family}")
        text_lines.append(
            f"  current={current_family_total} target={TARGET_PER_FAMILY} add={TARGET_PER_FAMILY - current_family_total}"
        )
        for difficulty in DIFFICULTIES:
            current_count = family_difficulty_counts[family][difficulty]
            additional = TARGET_PER_FAMILY_DIFFICULTY - current_count
            family_plan["by_difficulty"][difficulty] = {
                "current": current_count,
                "target": TARGET_PER_FAMILY_DIFFICULTY,
                "additional_needed": additional,
            }
            text_lines.append(
                f"    {difficulty:10s} current={current_count:3d} target={TARGET_PER_FAMILY_DIFFICULTY:3d} add={additional:3d}"
            )
        text_lines.append("")
        plan["by_family"][family] = family_plan

    json_path = ROOT / "results" / "benchmark_expansion_plan.json"
    txt_path = ROOT / "results" / "benchmark_expansion_plan.txt"
    json_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    txt_path.write_text("\n".join(text_lines) + "\n", encoding="utf-8")

    print("\n".join(text_lines))
    print(f"Saved {json_path}")
    print(f"Saved {txt_path}")


if __name__ == "__main__":
    main()
