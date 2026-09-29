from pathlib import Path
import json
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from speech_act_contract.annotation.corpus_export import build_annotation_starter_record
from speech_act_contract.data.loader import load_jsonl_records


def main() -> None:
    input_path = (
        Path(sys.argv[1]).expanduser().resolve()
        if len(sys.argv) > 1
        else ROOT / "data" / "processed" / "corpus_lite_v1.jsonl"
    )
    output_path = (
        Path(sys.argv[2]).expanduser().resolve()
        if len(sys.argv) > 2
        else ROOT / "data" / "annotations" / "corpus_lite_annotation_starter_v1.jsonl"
    )

    raw_records = load_jsonl_records(input_path)
    starter_records = [
        build_annotation_starter_record(record, example_id=f"lite_{index:04d}")
        for index, record in enumerate(raw_records)
    ]

    with output_path.open("w", encoding="utf-8") as handle:
        for record in starter_records:
            handle.write(json.dumps(record) + "\n")

    print(f"Loaded {len(raw_records)} corpus-lite records.")
    print(f"Saved annotation starter file to {output_path}")


if __name__ == "__main__":
    main()
