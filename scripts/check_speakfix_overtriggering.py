"""C3: SpeakFix over-triggering check on already-fine responses.

Circularity/over-triggering concern: does SpeakFix ever "repair" responses that
were already correct? This script isolates the base drafts that were already
fine under the automatic evaluator (uptake_ok AND no void commitment), runs each
through SpeakFix, and measures:

  - over-trigger rate: fraction of already-fine drafts that SpeakFix modifies
  - harmful edits: already-fine drafts that SpeakFix turned into a uptake miss
    or introduced a void commitment
  - no-op correctness: fraction of already-fine drafts left untouched

It also dumps every over-triggered example for manual/independent inspection.

Usage:
    python scripts/check_speakfix_overtriggering.py --model gpt52
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from speech_act_contract.data.loader import load_full_benchmark, load_jsonl_records
from speech_act_contract.mitigation.pipeline import (
    classify_response_act,
    detect_void_commitment,
    run_pipeline,
)
from speech_act_contract.stats.significance import wilson_interval


def uptake_ok(item, text: str) -> bool:
    return classify_response_act(text) in item.acceptable_alternative_response_acts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("azure", "claude", "gpt52"), required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gold_items = load_full_benchmark(ROOT / "data" / "annotations" / "full_uptake_benchmark_v3.jsonl")
    gold_by_id = {item.id: item for item in gold_items}

    base_path = ROOT / "results" / f"{args.model}_full_benchmark_drafts_v3.jsonl"
    base_records = load_jsonl_records(base_path)
    model_name = base_records[0]["model_name"] if base_records else args.model

    already_fine = 0
    over_triggered = 0
    harmful = 0
    family_fine: Counter = Counter()
    family_over: Counter = Counter()
    over_examples = []

    for record in base_records:
        item = gold_by_id[record["id"]]
        before_text = record["response_text"]

        # "Already fine" = correct uptake and no void commitment before repair.
        if not (uptake_ok(item, before_text) and not detect_void_commitment(before_text)):
            continue

        already_fine += 1
        family_fine[item.speech_act_family] += 1

        after_text = run_pipeline(item.user_turn, before_text).repaired_response
        changed = after_text.strip() != before_text.strip()
        if not changed:
            continue

        over_triggered += 1
        family_over[item.speech_act_family] += 1

        after_ok = uptake_ok(item, after_text)
        after_void = detect_void_commitment(after_text)
        is_harmful = (not after_ok) or after_void
        if is_harmful:
            harmful += 1

        over_examples.append(
            {
                "id": item.id,
                "speech_act_family": item.speech_act_family,
                "harmful": is_harmful,
                "after_uptake_ok": after_ok,
                "after_void_commitment": after_void,
                "before_response_text": before_text,
                "after_response_text": after_text,
            }
        )

    over_rate = over_triggered / already_fine if already_fine else 0.0
    result = {
        "model": args.model,
        "model_name": model_name,
        "num_items": len(base_records),
        "already_fine_count": already_fine,
        "over_triggered_count": over_triggered,
        "over_trigger_rate": round(over_rate, 4),
        "over_trigger_rate_wilson_95": wilson_interval(over_triggered, already_fine)
        if already_fine
        else (0.0, 0.0),
        "harmful_edit_count": harmful,
        "no_op_correctness_rate": round(1.0 - over_rate, 4),
        "per_family_already_fine": dict(sorted(family_fine.items())),
        "per_family_over_triggered": dict(sorted(family_over.items())),
        "over_triggered_examples": over_examples,
    }

    output_path = args.output or (ROOT / "results" / f"{args.model}_overtriggering_check.json")
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Over-triggering check -- {args.model} ({model_name})")
    print(f"  already-fine drafts: {already_fine}/{len(base_records)}")
    print(
        f"  over-triggered: {over_triggered} "
        f"({over_rate * 100:.1f}%, Wilson 95% "
        f"[{result['over_trigger_rate_wilson_95'][0] * 100:.1f}, "
        f"{result['over_trigger_rate_wilson_95'][1] * 100:.1f}])"
    )
    print(f"  harmful edits among those: {harmful}")
    print(f"  no-op correctness: {(1.0 - over_rate) * 100:.1f}%")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
