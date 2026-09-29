"""Prepare a prompt manifest for evaluating an open-weight model offline.

This script does not run a model. It creates a JSONL manifest that can be fed
to a local inference stack, then scored with the existing evaluation scripts
after responses are added.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from speech_act_contract.data.loader import load_full_benchmark


OUTPUT_PATH = ROOT / "results" / "open_model_full_benchmark_manifest.jsonl"
BENCHMARK_PATH = ROOT / "data" / "annotations" / "full_uptake_benchmark_v3.jsonl"


def main() -> None:
    items = load_full_benchmark(BENCHMARK_PATH)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        for item in items:
            record = {
                "id": item.id,
                "prompt": item.user_turn,
                "expected_output_schema": {
                    "id": item.id,
                    "model_name": "replace_with_open_model_name",
                    "response_text": "fill with model response",
                },
            }
            handle.write(json.dumps(record) + "\n")

    print(f"Wrote {len(items)} prompts to {OUTPUT_PATH}")
    print("Next step: run an open-weight instruct model and save outputs as JSONL with id/model_name/response_text.")


if __name__ == "__main__":
    main()
