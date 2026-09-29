"""Baseline B2: lightweight LLM-rewrite mitigation.

Instead of the rule-based SpeakFix pipeline, this baseline hands each base draft
back to an LLM with a single, generic instruction to rewrite it so it responds
appropriately to the user's move and makes no unsupported memory claims. It uses
no families, lexicons, or matcher rules. It reads an existing base drafts JSONL
(the same file scored as the "before" condition) and writes a rewritten drafts
JSONL that can be scored with the same evaluator.

Usage:
    python scripts/generate_baseline_llm_rewrite_drafts.py --model azure
    python scripts/generate_baseline_llm_rewrite_drafts.py --model claude
    python scripts/generate_baseline_llm_rewrite_drafts.py --model gpt52
"""

from pathlib import Path
import argparse
import json
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from speech_act_contract.data.loader import load_jsonl_records
from speech_act_contract.llm.dispatch import MODEL_KEYS, resolve_generator


REWRITE_SYSTEM_PROMPT = (
    "You are an editor that improves the pragmatics of a chat assistant's reply. "
    "Given the user's message and a draft reply, rewrite the reply so it clearly "
    "responds to what the user is doing (for example, accepting an apology, "
    "acknowledging thanks or a correction, responding to an offer, addressing "
    "frustration, or confirming a preference) while preserving all helpful "
    "content. Do not claim to remember anything across sessions or to store "
    "preferences permanently. Keep it to one short reply. Return only the "
    "rewritten reply, with no preamble or explanation."
)

DEFAULT_INPUTS = {
    "azure": "azure_full_benchmark_drafts_v3.jsonl",
    "claude": "claude_full_benchmark_drafts_v3.jsonl",
    "gpt52": "gpt52_full_benchmark_drafts_v3.jsonl",
}


def build_rewrite_prompt(user_turn: str, draft_response: str) -> str:
    return (
        f"User message:\n{user_turn}\n\n"
        f"Draft reply:\n{draft_response}\n\n"
        "Rewritten reply:"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_KEYS, required=True)
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Base drafts JSONL to rewrite. Defaults to results/<model>_full_benchmark_drafts_v3.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path. Defaults to results/<model>_full_benchmark_baseline_llmrewrite_v1.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _config, generate, model_name = resolve_generator(args.model)

    input_path = args.input or (ROOT / "results" / DEFAULT_INPUTS[args.model])
    output_path = args.output or (
        ROOT / "results" / f"{args.model}_full_benchmark_baseline_llmrewrite_v1.jsonl"
    )

    draft_records = load_jsonl_records(input_path)
    print(f"[B2 LLM-rewrite baseline] model={args.model} deployment={model_name}")
    print(f"Rewriting {len(draft_records)} drafts from {input_path} -> {output_path}")

    with output_path.open("w", encoding="utf-8") as handle:
        for record in draft_records:
            rewritten = generate(
                REWRITE_SYSTEM_PROMPT,
                build_rewrite_prompt(record["user_turn"], record["response_text"]),
            )
            out = {
                "id": record["id"],
                "user_turn": record["user_turn"],
                "speech_act_family": record["speech_act_family"],
                "model_name": model_name,
                "baseline": "llm_rewrite",
                "base_response_text": record["response_text"],
                "response_text": rewritten,
            }
            handle.write(json.dumps(out) + "\n")
            print(f"Rewrote draft for {record['id']}")

    print(f"Saved {len(draft_records)} LLM-rewrite baseline drafts to {output_path}")


if __name__ == "__main__":
    main()
