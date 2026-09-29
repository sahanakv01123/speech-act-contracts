from collections import Counter
from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    input_path = ROOT / "data" / "processed" / "corpus_lite_v1.jsonl"
    family_counts = Counter()
    model_counts = Counter()
    total = 0

    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            family_counts[record["seed_family"]] += 1
            model_counts[record["model_name"]] += 1
            total += 1

    print(f"Loaded {total} corpus-lite examples.")
    print("Seed family counts:")
    for family, count in sorted(family_counts.items()):
        print(f"  {family}: {count}")
    print("Model counts:")
    for model_name, count in sorted(model_counts.items()):
        print(f"  {model_name}: {count}")


if __name__ == "__main__":
    main()
