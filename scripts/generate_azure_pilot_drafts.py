from pathlib import Path
import json
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from speech_act_contract.data.loader import load_pilot_eval_set
from speech_act_contract.llm.azure_openai import generate_chat_completion, load_azure_openai_config


SYSTEM_PROMPT = (
    "You are a helpful research assistant. Respond naturally to the user's message in one short reply."
)


def build_user_prompt(user_turn: str) -> str:
    return f"Reply to this user message:\n\n{user_turn}"


def main() -> None:
    config = load_azure_openai_config()
    items = load_pilot_eval_set(ROOT / "data" / "annotations" / "pilot_eval_set.jsonl")
    output_path = ROOT / "results" / "azure_pilot_drafts.jsonl"

    records = []
    for item in items:
        response_text = generate_chat_completion(
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
        records.append(record)
        print(f"Generated draft for {item.id}")

    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    print(f"Saved {len(records)} model drafts to {output_path}")


if __name__ == "__main__":
    main()
