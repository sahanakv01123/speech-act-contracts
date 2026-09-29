from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    drafts_path = ROOT / "results" / "azure_full_benchmark_drafts_v2.jsonl"
    output_path = ROOT / "data" / "processed" / "corpus_lite_v1.jsonl"

    records = []
    with drafts_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            record = json.loads(line)
            records.append(
                {
                    "conversation_id": f"corpus_lite_{index:04d}",
                    "turn_index": 1,
                    "user_turn": record["user_turn"],
                    "assistant_turn": record["response_text"],
                    "seed_id": record["id"],
                    "seed_family": record["speech_act_family"],
                    "model_name": record["model_name"],
                    "source_type": "benchmark_derived_model_output",
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    print(f"Built {len(records)} corpus-lite examples into {output_path}")


if __name__ == "__main__":
    main()
