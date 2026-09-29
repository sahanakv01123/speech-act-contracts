from pathlib import Path
import json
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from speech_act_contract.data.loader import load_full_benchmark
from speech_act_contract.llm.azure_claude import generate_claude_message, load_azure_claude_config


SYSTEM_PROMPT = (
    "You are a helpful research assistant. Respond naturally to the user's message in one short reply."
)


def build_user_prompt(user_turn: str) -> str:
    return f"Reply to this user message:\n\n{user_turn}"


def main() -> None:
    config = load_azure_claude_config()
    items = load_full_benchmark(ROOT / "data" / "annotations" / "full_uptake_benchmark_v3.jsonl")
    output_path = ROOT / "results" / "claude_full_benchmark_drafts_v3.jsonl"

    with output_path.open("w", encoding="utf-8") as handle:
        for item in items:
            response_text = generate_claude_message(
                config=config,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=build_user_prompt(item.user_turn),
            )
            record = {
                "id": item.id,
                "user_turn": item.user_turn,
                "speech_act_family": item.speech_act_family,
                "model_name": config.deployment,
                "response_text": response_text,
            }
            handle.write(json.dumps(record) + "\n")
            print(f"Generated draft for {item.id}")

    print(f"Saved {len(items)} Claude full-benchmark drafts to {output_path}")


if __name__ == "__main__":
    main()
