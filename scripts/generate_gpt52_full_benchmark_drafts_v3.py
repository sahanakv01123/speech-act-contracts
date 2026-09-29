"""Generate full benchmark (v3, 300 items) drafts for gpt-5.2.

Reads credentials from .env.gpt52 in the project root.
Writes to results/gpt52_full_benchmark_drafts_v3.jsonl.

Usage:
    cp .env.gpt52 .env   # or set AZURE_OPENAI_DEPLOYMENT etc. directly
    python scripts/generate_gpt52_full_benchmark_drafts_v3.py
"""

from pathlib import Path
import json
import os
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Load .env.gpt52 before importing config
_env_path = ROOT / ".env.gpt52"
if _env_path.exists():
    for raw_line in _env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value
else:
    print(
        "Warning: .env.gpt52 not found. "
        "Ensure AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, "
        "AZURE_OPENAI_DEPLOYMENT, and AZURE_OPENAI_API_VERSION are set."
    )

from speech_act_contract.data.loader import load_full_benchmark
from speech_act_contract.llm.azure_openai import generate_chat_completion, load_azure_openai_config


SYSTEM_PROMPT = (
    "You are a helpful research assistant. Respond naturally to the user's message in one short reply."
)


def build_user_prompt(user_turn: str) -> str:
    return f"Reply to this user message:\n\n{user_turn}"


def main() -> None:
    config = load_azure_openai_config()
    items = load_full_benchmark(ROOT / "data" / "annotations" / "full_uptake_benchmark_v3.jsonl")
    output_path = ROOT / "results" / "gpt52_full_benchmark_drafts_v3.jsonl"

    print(f"Generating {len(items)} drafts with deployment: {config.deployment}")

    with output_path.open("w", encoding="utf-8") as handle:
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
            handle.write(json.dumps(record) + "\n")
            print(f"Generated draft for {item.id}")

    print(f"Saved {len(items)} drafts to {output_path}")


if __name__ == "__main__":
    main()
