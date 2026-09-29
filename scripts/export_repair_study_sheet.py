"""Export a larger blinded repair-study sheet for external annotators.

The sheet presents response pairs in randomized A/B order so annotators do not
know which response is original and which is repaired.
"""

from __future__ import annotations

import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILES = [
    ROOT / "results" / "azure_full_benchmark_mitigation_eval_v4.json",
    ROOT / "results" / "claude_full_benchmark_mitigation_eval_v4.json",
    ROOT / "results" / "gpt52_full_benchmark_mitigation_eval_v4.json",
]
OUTPUT_PATH = ROOT / "results" / "repair_study_sheet_blinded.csv"
RANDOM_SEED = 23
TARGET_PER_MODEL = 36


def canonical_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_failure_rows(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    model_name = data["model_name"]
    rows = []
    for record in data["records"]:
        is_failure_case = (not record["before_uptake_ok"]) or record["before_void_commitment"]
        materially_changed = canonical_text(record["before_response_text"]) != canonical_text(
            record["after_response_text"]
        )
        if not is_failure_case or not materially_changed:
            continue
        rows.append(
            {
                "model_name": model_name,
                "id": record["id"],
                "speech_act_family": record["speech_act_family"],
                "difficulty": record["difficulty"],
                "source_type": record["source_type"],
                "before_response": record["before_response_text"],
                "after_response": record["after_response_text"],
            }
        )
    return rows


def stratified_sample(rows: list[dict], per_model: int, rng: random.Random) -> list[dict]:
    by_family = defaultdict(list)
    for row in rows:
        by_family[row["speech_act_family"]].append(row)

    sampled = []
    for family in sorted(by_family):
        candidates = by_family[family][:]
        rng.shuffle(candidates)
        sampled.extend(candidates[: min(6, len(candidates))])

    if len(sampled) < per_model:
        chosen = {(row["model_name"], row["id"]) for row in sampled}
        leftovers = [row for row in rows if (row["model_name"], row["id"]) not in chosen]
        rng.shuffle(leftovers)
        sampled.extend(leftovers[: max(0, per_model - len(sampled))])

    rng.shuffle(sampled)
    return sampled[:per_model]


def main() -> None:
    rng = random.Random(RANDOM_SEED)
    sampled_rows: list[dict] = []

    for input_path in INPUT_FILES:
        rows = load_failure_rows(input_path)
        sampled_rows.extend(stratified_sample(rows, TARGET_PER_MODEL, rng))

    rng.shuffle(sampled_rows)

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "row",
                "model_name",
                "id",
                "speech_act_family",
                "difficulty",
                "source_type",
                "response_a",
                "response_b",
                "a_is_repaired_hidden",
                "overall_better_a_or_b_or_tie",
                "more_helpful_a_or_b_or_tie",
                "more_natural_a_or_b_or_tie",
                "contains_void_commitment_a_yes_no",
                "contains_void_commitment_b_yes_no",
                "notes",
            ],
        )
        writer.writeheader()

        for row_num, record in enumerate(sampled_rows, start=1):
            if rng.random() < 0.5:
                response_a = record["before_response"]
                response_b = record["after_response"]
                hidden = "FALSE"
            else:
                response_a = record["after_response"]
                response_b = record["before_response"]
                hidden = "TRUE"

            writer.writerow(
                {
                    "row": row_num,
                    "model_name": record["model_name"],
                    "id": record["id"],
                    "speech_act_family": record["speech_act_family"],
                    "difficulty": record["difficulty"],
                    "source_type": record["source_type"],
                    "response_a": response_a,
                    "response_b": response_b,
                    "a_is_repaired_hidden": hidden,
                    "overall_better_a_or_b_or_tie": "",
                    "more_helpful_a_or_b_or_tie": "",
                    "more_natural_a_or_b_or_tie": "",
                    "contains_void_commitment_a_yes_no": "",
                    "contains_void_commitment_b_yes_no": "",
                    "notes": "",
                }
            )

    print(f"Exported {len(sampled_rows)} blinded repair-study rows to {OUTPUT_PATH}")
    print("Intended use: external pairwise A/B annotation with randomized presentation.")


if __name__ == "__main__":
    main()
