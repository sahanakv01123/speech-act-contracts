"""Export CSV starter batches for a 1000+ benchmark expansion.

This creates annotation-ready slot files for the additional items needed to
grow the current 300-item benchmark to 1008 items while preserving family and
difficulty balance and increasing source diversity.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from speech_act_contract.data.loader import load_full_benchmark


BENCHMARK_PATH = ROOT / "data" / "annotations" / "full_uptake_benchmark_v3.jsonl"
BATCH_DIR = ROOT / "data" / "annotations" / "expansion_batches"
MANIFEST_PATH = BATCH_DIR / "expansion_batch_manifest.json"
FAMILIES = ["apology", "thanks", "offer", "correction", "frustration", "preference"]
DIFFICULTIES = ["easy", "near_miss", "mixed_act"]
TARGET_PER_FAMILY_DIFFICULTY = 56

SOURCE_ROTATION = {
    "easy": ["adapted_from_real_conversation", "human_written_naturalistic"],
    "near_miss": ["adapted_from_real_conversation", "mined_from_public_corpus"],
    "mixed_act": ["synthetic", "human_written_naturalistic"],
}

EXPECTED_RESPONSE_BY_FAMILY = {
    "apology": "acceptance_reassurance",
    "thanks": "acknowledgment",
    "offer": "accept_or_decline_offer",
    "correction": "acknowledge_update",
    "frustration": "repair_acknowledgment",
    "preference": "bounded_acknowledgment",
}


def main() -> None:
    items = load_full_benchmark(BENCHMARK_PATH)
    current_counts = {family: {difficulty: 0 for difficulty in DIFFICULTIES} for family in FAMILIES}
    for item in items:
        current_counts[item.speech_act_family][item.difficulty] += 1

    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}
    total_slots = 0

    for family in FAMILIES:
        rows = []
        slot_number = 1
        for difficulty in DIFFICULTIES:
            need = TARGET_PER_FAMILY_DIFFICULTY - current_counts[family][difficulty]
            sources = SOURCE_ROTATION[difficulty]
            for i in range(need):
                source_type = sources[i % len(sources)]
                slot_id = f"{family}_exp_{slot_number:03d}"
                rows.append(
                    {
                        "slot_id": slot_id,
                        "speech_act_family": family,
                        "difficulty": difficulty,
                        "target_source_type": source_type,
                        "expected_response_act": EXPECTED_RESPONSE_BY_FAMILY[family],
                        "acceptable_alternative_response_acts_json": "[]",
                        "gold_void_commitment_risk": "",
                        "user_turn": "",
                        "context_notes": (
                            f"Expansion slot for {family}/{difficulty}; target source type {source_type}."
                        ),
                        "annotator_id": "",
                        "reviewer_id": "",
                        "status": "todo",
                    }
                )
                slot_number += 1
                total_slots += 1

        output_path = BATCH_DIR / f"{family}_expansion_batch.csv"
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)

        manifest[family] = {
            "output_path": str(output_path),
            "rows": len(rows),
            "current_counts": current_counts[family],
            "target_per_difficulty": TARGET_PER_FAMILY_DIFFICULTY,
        }

    manifest["summary"] = {
        "benchmark_path": str(BENCHMARK_PATH),
        "target_total_after_expansion": 1008,
        "total_new_slots": total_slots,
        "batch_dir": str(BATCH_DIR),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Created expansion batches in {BATCH_DIR}")
    print(f"Total new slots: {total_slots}")
    print(f"Saved manifest to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
