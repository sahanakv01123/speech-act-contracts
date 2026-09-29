"""Generate full benchmark drafts with a local Ollama model.

Usage:
    python scripts/generate_ollama_full_benchmark_drafts.py llama3.1:8b

Writes:
    results/ollama_<model_slug>_full_benchmark_drafts_v3.jsonl
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from speech_act_contract.data.loader import load_full_benchmark


SYSTEM_PROMPT = (
    "You are a helpful research assistant. Respond naturally to the user's message in one short reply."
)


def build_user_prompt(user_turn: str) -> str:
    return f"Reply to this user message:\n\n{user_turn}"


def model_slug(name: str) -> str:
    return name.replace(":", "_").replace("/", "_").replace(" ", "_")


def run_ollama(model_name: str, prompt: str) -> str:
    result = subprocess.run(
        ["ollama", "run", model_name, prompt],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/generate_ollama_full_benchmark_drafts.py <ollama-model-name>")

    model_name = sys.argv[1]
    items = load_full_benchmark(ROOT / "data" / "annotations" / "full_uptake_benchmark_v3.jsonl")
    output_path = ROOT / "results" / f"ollama_{model_slug(model_name)}_full_benchmark_drafts_v3.jsonl"

    print(f"Generating {len(items)} drafts with Ollama model: {model_name}")
    with output_path.open("w", encoding="utf-8") as handle:
        for item in items:
            response_text = run_ollama(model_name, f"{SYSTEM_PROMPT}\n\n{build_user_prompt(item.user_turn)}")
            record = {
                "id": item.id,
                "user_turn": item.user_turn,
                "speech_act_family": item.speech_act_family,
                "model_name": model_name,
                "response_text": response_text,
            }
            handle.write(json.dumps(record) + "\n")
            print(f"Generated draft for {item.id}")

    print(f"Saved {len(items)} drafts to {output_path}")


if __name__ == "__main__":
    main()
