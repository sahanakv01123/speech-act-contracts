"""Export a larger classifier-validation sheet across multiple models.

This extends the smaller single-model validation export by sampling from all
available benchmark eval files and producing a broader human-validation sheet.

Output:
  results/classifier_validation_sheet_large.csv
"""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from speech_act_contract.data.loader import load_full_benchmark


RANDOM_SEED = 17
TARGET_PER_FAMILY_STRATUM_MODEL = 4
INPUT_FILES = [
    ROOT / "results" / "azure_full_benchmark_eval_v3.json",
    ROOT / "results" / "claude_full_benchmark_eval_v3.json",
    ROOT / "results" / "gpt52_full_benchmark_eval_v3.json",
]
OUTPUT_PATH = ROOT / "results" / "classifier_validation_sheet_large.csv"


def main() -> None:
    benchmark = {
        item.id: item
        for item in load_full_benchmark(ROOT / "data" / "annotations" / "full_uptake_benchmark_v3.jsonl")
    }
    rng = random.Random(RANDOM_SEED)

    sampled = []
    for input_path in INPUT_FILES:
        data = json.loads(input_path.read_text(encoding="utf-8"))
        model_name = data["model_name"]
        pool: dict[tuple[str, bool], list[dict]] = defaultdict(list)

        for record in data["records"]:
            pool[(record["speech_act_family"], record["uptake_ok"])].append(record)

        for family in sorted({record["speech_act_family"] for record in data["records"]}):
            for uptake_ok in [True, False]:
                candidates = pool[(family, uptake_ok)]
                n = min(TARGET_PER_FAMILY_STRATUM_MODEL, len(candidates))
                for picked in rng.sample(candidates, n):
                    sampled.append((model_name, picked))

    rng.shuffle(sampled)

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
                "user_turn",
                "model_response",
                "expected_response_act",
                "classifier_label",
                "classifier_uptake_ok",
                "human_uptake_ok",
                "notes",
            ],
        )
        writer.writeheader()
        for row_num, (model_name, record) in enumerate(sampled, start=1):
            item = benchmark[record["id"]]
            writer.writerow(
                {
                    "row": row_num,
                    "model_name": model_name,
                    "id": record["id"],
                    "speech_act_family": record["speech_act_family"],
                    "difficulty": record["difficulty"],
                    "source_type": record["source_type"],
                    "user_turn": item.user_turn,
                    "model_response": record["response_text"],
                    "expected_response_act": record["expected_response_act"],
                    "classifier_label": record["observed_response_act"],
                    "classifier_uptake_ok": record["uptake_ok"],
                    "human_uptake_ok": "",
                    "notes": "",
                }
            )

    print(f"Exported {len(sampled)} rows to {OUTPUT_PATH}")
    print("Intended use: external validation of the automatic uptake classifier across all models.")


if __name__ == "__main__":
    main()
