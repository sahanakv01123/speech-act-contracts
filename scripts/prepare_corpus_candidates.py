from pathlib import Path
import json
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


TRIGGER_PATTERNS = {
    "apology": ["sorry", "my mistake", "my bad"],
    "thanks": ["thank you", "thanks"],
    "offer": ["i can ", "i could ", "i can send", "i can share"],
    "correction": ["actually", "not ", "no, "],
    "frustration": ["frustrating", "frustrated", "annoying", "you keep"],
    "preference": ["please keep", "please use", "please avoid", "from now on"],
}


def detect_trigger_family(text: str) -> str:
    lowered = text.lower()
    for family, patterns in TRIGGER_PATTERNS.items():
        if any(pattern in lowered for pattern in patterns):
            return family
    return "other"


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python3 scripts/prepare_corpus_candidates.py <raw_corpus.jsonl> <output_candidates.jsonl>"
        )

    input_path = Path(sys.argv[1]).expanduser().resolve()
    output_path = Path(sys.argv[2]).expanduser().resolve()

    candidates = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            record = json.loads(line)
            user_turn = record.get("user_turn", "")
            assistant_turn = record.get("assistant_turn", "")
            trigger_family = detect_trigger_family(user_turn)
            if trigger_family == "other":
                continue

            candidates.append(
                {
                    "example_id": f"cand_{line_number:06d}",
                    "conversation_id": record.get("conversation_id", f"conv_{line_number:06d}"),
                    "turn_index": record.get("turn_index", 0),
                    "user_turn": user_turn,
                    "assistant_turn": assistant_turn,
                    "trigger_family": trigger_family,
                }
            )

    with output_path.open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            handle.write(json.dumps(candidate) + "\n")

    print(f"Prepared {len(candidates)} corpus candidates into {output_path}")


if __name__ == "__main__":
    main()
