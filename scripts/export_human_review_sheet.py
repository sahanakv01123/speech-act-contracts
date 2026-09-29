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

OUTPUT_PATH = ROOT / "results" / "human_review_sheet_v4_failure_sample.csv"
SAMPLE_PER_MODEL = 18
SEED = 7


def canonical_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_records(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    model_name = data["model_name"]
    rows = []
    for record in data["records"]:
        is_failure_case = (not record["before_uptake_ok"]) or record["before_void_commitment"]
        materially_changed = canonical_text(record["before_response_text"]) != canonical_text(
            record["after_response_text"]
        )
        if (not is_failure_case) or (not materially_changed):
            continue
        rows.append(
            {
                "model_name": model_name,
                "id": record["id"],
                "speech_act_family": record["speech_act_family"],
                "difficulty": record["difficulty"],
                "source_type": record["source_type"],
                "expected_response_act": record["expected_response_act"],
                "before_uptake_ok": record["before_uptake_ok"],
                "before_void_commitment": record["before_void_commitment"],
                "after_uptake_ok": record["after_uptake_ok"],
                "after_void_commitment": record["after_void_commitment"],
                "before_response": record["before_response_text"],
                "after_response": record["after_response_text"],
            }
        )
    return rows


def stratified_sample(rows: list[dict], per_model: int) -> list[dict]:
    rng = random.Random(SEED)
    by_model = defaultdict(list)
    for row in rows:
        by_model[row["model_name"]].append(row)

    sampled: list[dict] = []
    for model_name, model_rows in by_model.items():
        by_family = defaultdict(list)
        for row in model_rows:
            by_family[row["speech_act_family"]].append(row)

        model_sample: list[dict] = []
        families = sorted(by_family.keys())

        for family in families:
            candidates = by_family[family][:]
            rng.shuffle(candidates)
            take = min(3, len(candidates))
            model_sample.extend(candidates[:take])

        if len(model_sample) > per_model:
            rng.shuffle(model_sample)
            model_sample = model_sample[:per_model]
        elif len(model_sample) < per_model:
            chosen_ids = {row["id"] for row in model_sample}
            leftovers = [row for row in model_rows if row["id"] not in chosen_ids]
            rng.shuffle(leftovers)
            model_sample.extend(leftovers[: max(0, per_model - len(model_sample))])

        sampled.extend(sorted(model_sample, key=lambda r: (r["speech_act_family"], r["id"])))

    return sampled


def main() -> None:
    all_rows: list[dict] = []
    for path in INPUT_FILES:
        all_rows.extend(load_records(path))

    sampled = stratified_sample(all_rows, SAMPLE_PER_MODEL)

    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model_name",
                "id",
                "speech_act_family",
                "difficulty",
                "source_type",
                "expected_response_act",
                "before_uptake_ok",
                "before_void_commitment",
                "after_uptake_ok",
                "after_void_commitment",
                "before_response",
                "after_response",
                "better_same_worse",
                "helpfulness_preserved_yes_no",
                "natural_yes_no",
                "notes",
            ]
        )
        for row in sampled:
            writer.writerow(
                [
                    row["model_name"],
                    row["id"],
                    row["speech_act_family"],
                    row["difficulty"],
                    row["source_type"],
                    row["expected_response_act"],
                    row["before_uptake_ok"],
                    row["before_void_commitment"],
                    row["after_uptake_ok"],
                    row["after_void_commitment"],
                    row["before_response"],
                    row["after_response"],
                    "",
                    "",
                    "",
                    "",
                ]
            )

    print(f"Exported {len(sampled)} failure-case review rows to {OUTPUT_PATH}")
    print(f"Sampling policy: {SAMPLE_PER_MODEL} items per model, stratified by speech-act family.")


if __name__ == "__main__":
    main()
