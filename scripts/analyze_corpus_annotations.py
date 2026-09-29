from collections import Counter
from pathlib import Path
import json
import sys


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python3 scripts/analyze_corpus_annotations.py <annotated_corpus.jsonl>"
        )

    input_path = Path(sys.argv[1]).expanduser().resolve()
    family_counts = Counter()
    uptake_failures = Counter()
    void_commitments = Counter()
    total = 0
    completed = 0

    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            family = record["trigger_family"]
            family_counts[family] += 1
            if record.get("uptake_failure") is True:
                uptake_failures[family] += 1
            if record.get("void_commitment") is True:
                void_commitments[family] += 1
            if record.get("uptake_failure") is not None and record.get("void_commitment") is not None:
                completed += 1
            total += 1

    print(f"Loaded {total} annotated corpus examples.")
    print(f"Completed annotations: {completed}")
    print("Per-family rates:")
    for family, count in sorted(family_counts.items()):
        uptake_rate = round(uptake_failures[family] / count, 3)
        void_rate = round(void_commitments[family] / count, 3)
        print(f"  {family}: uptake_failure={uptake_rate}, void_commitment={void_rate}")


if __name__ == "__main__":
    main()
