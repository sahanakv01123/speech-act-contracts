"""Baseline B1: system-prompt mitigation.

Instead of the rule-based SpeakFix pipeline, this baseline simply instructs the
model, via an augmented system prompt, to perform the correct uptake move and
to avoid unbounded memory/persistence commitments. It regenerates a fresh draft
per benchmark item and writes a standard drafts JSONL that can be scored with
the same evaluator used for the base drafts and the SpeakFix "after" condition.

Usage:
    python scripts/generate_baseline_systemprompt_drafts.py --model azure
    python scripts/generate_baseline_systemprompt_drafts.py --model claude
    python scripts/generate_baseline_systemprompt_drafts.py --model gpt52
"""

from pathlib import Path
import argparse
import json
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from speech_act_contract.data.loader import load_full_benchmark
from speech_act_contract.llm.dispatch import MODEL_KEYS, resolve_generator


# Augmented system prompt: describes the desired pragmatic behavior in plain
# language, without exposing any of SpeakFix's rules, lexicons, or phrasing.
SYSTEM_PROMPT = (
    "You are a helpful research assistant. Respond naturally to the user's "
    "message in one short reply. Before answering, make sure your reply "
    "explicitly acknowledges the user's conversational move: accept and "
    "reassure after an apology, acknowledge thanks, accept or politely decline "
    "an offer, acknowledge and incorporate a correction, address frustration "
    "with a brief repair, and confirm a stated preference. Never claim to "
    "remember information across sessions, to save preferences permanently, or "
    "to change your behavior 'going forward' or 'from now on'. If you commit to "
    "anything, bound it explicitly to the current conversation."
)


def build_user_prompt(user_turn: str) -> str:
    return f"Reply to this user message:\n\n{user_turn}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_KEYS, required=True)
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT / "data" / "annotations" / "full_uptake_benchmark_v3.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path. Defaults to results/<model>_full_benchmark_baseline_systemprompt_v1.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _config, generate, model_name = resolve_generator(args.model)
    items = load_full_benchmark(args.benchmark)

    output_path = args.output or (
        ROOT / "results" / f"{args.model}_full_benchmark_baseline_systemprompt_v1.jsonl"
    )

    print(f"[B1 system-prompt baseline] model={args.model} deployment={model_name}")
    print(f"Generating {len(items)} drafts -> {output_path}")

    with output_path.open("w", encoding="utf-8") as handle:
        for item in items:
            response_text = generate(SYSTEM_PROMPT, build_user_prompt(item.user_turn))
            record = {
                "id": item.id,
                "user_turn": item.user_turn,
                "speech_act_family": item.speech_act_family,
                "model_name": model_name,
                "baseline": "system_prompt",
                "response_text": response_text,
            }
            handle.write(json.dumps(record) + "\n")
            print(f"Generated baseline draft for {item.id}")

    print(f"Saved {len(items)} system-prompt baseline drafts to {output_path}")


if __name__ == "__main__":
    main()
