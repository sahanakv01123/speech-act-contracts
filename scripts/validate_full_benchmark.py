from collections import Counter
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from speech_act_contract.data.loader import load_full_benchmark


VALID_FAMILIES = {
    "apology",
    "thanks",
    "offer",
    "correction",
    "frustration",
    "preference",
}
VALID_DIFFICULTIES = {"easy", "near_miss", "mixed_act"}
VALID_SOURCE_TYPES = {"synthetic", "adapted_from_real_conversation"}


def main() -> None:
    path = ROOT / "data" / "annotations" / "full_uptake_benchmark_v3.jsonl"
    if not path.exists():
        path = ROOT / "data" / "annotations" / "full_uptake_benchmark_v2.jsonl"
    if not path.exists():
        path = ROOT / "data" / "annotations" / "full_uptake_benchmark_v1.jsonl"
    items = load_full_benchmark(path)

    ids = set()
    family_counts = Counter()
    difficulty_counts = Counter()
    errors = []

    for item in items:
        if item.id in ids:
            errors.append(f"Duplicate id: {item.id}")
        ids.add(item.id)

        if item.speech_act_family not in VALID_FAMILIES:
            errors.append(f"Invalid family for {item.id}: {item.speech_act_family}")
        if item.difficulty not in VALID_DIFFICULTIES:
            errors.append(f"Invalid difficulty for {item.id}: {item.difficulty}")
        if item.source_type not in VALID_SOURCE_TYPES:
            errors.append(f"Invalid source type for {item.id}: {item.source_type}")
        if not item.acceptable_alternative_response_acts:
            errors.append(f"No alternative acts for {item.id}")
        if item.expected_response_act not in item.acceptable_alternative_response_acts:
            errors.append(f"Expected act missing from alternatives for {item.id}")

        family_counts[item.speech_act_family] += 1
        difficulty_counts[item.difficulty] += 1

    print(f"Loaded {len(items)} full benchmark items from {path.name}")
    print("Family counts:")
    for family, count in sorted(family_counts.items()):
        print(f"  {family}: {count}")
    print("Difficulty counts:")
    for difficulty, count in sorted(difficulty_counts.items()):
        print(f"  {difficulty}: {count}")

    if errors:
        print("\nValidation errors:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)

    print("\nFull benchmark validation passed.")


if __name__ == "__main__":
    main()
