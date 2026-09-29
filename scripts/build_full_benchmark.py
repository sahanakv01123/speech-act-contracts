from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from speech_act_contract.data.loader import load_jsonl_records


def main() -> None:
    batch_paths = [
        ROOT / "data" / "annotations" / "full_uptake_benchmark_v1.jsonl",
        ROOT / "data" / "annotations" / "full_uptake_benchmark_batch2.jsonl",
    ]
    output_path = ROOT / "data" / "annotations" / "full_uptake_benchmark_v2.jsonl"

    records = []
    for path in batch_paths:
        records.extend(load_jsonl_records(path))

    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            import json

            handle.write(json.dumps(record) + "\n")

    print(f"Built {len(records)} items into {output_path.name}")


if __name__ == "__main__":
    main()
