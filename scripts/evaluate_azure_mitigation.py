from collections import Counter
from pathlib import Path
import json
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from speech_act_contract.data.loader import load_jsonl_records, load_pilot_eval_set
from speech_act_contract.mitigation.pipeline import (
    classify_response_act,
    detect_void_commitment,
    run_pipeline,
)


def summarize_records(records: list[dict]) -> dict:
    family_counts = Counter()
    family_uptake_hits = Counter()
    family_void_counts = Counter()
    difficulty_counts = Counter()
    difficulty_uptake_hits = Counter()

    for record in records:
        family = record["speech_act_family"]
        difficulty = record["difficulty"]
        family_counts[family] += 1
        difficulty_counts[difficulty] += 1

        if record["uptake_ok"]:
            family_uptake_hits[family] += 1
            difficulty_uptake_hits[difficulty] += 1
        if record["void_commitment"]:
            family_void_counts[family] += 1

    total = len(records)
    return {
        "uptake_accuracy": round(sum(family_uptake_hits.values()) / total, 3),
        "void_commitment_rate": round(sum(family_void_counts.values()) / total, 3),
        "per_family_uptake_accuracy": {
            family: round(family_uptake_hits[family] / count, 3)
            for family, count in sorted(family_counts.items())
        },
        "per_family_void_commitment_rate": {
            family: round(family_void_counts[family] / count, 3)
            for family, count in sorted(family_counts.items())
        },
        "per_difficulty_uptake_accuracy": {
            difficulty: round(difficulty_uptake_hits[difficulty] / count, 3)
            for difficulty, count in sorted(difficulty_counts.items())
        },
    }


def main() -> None:
    gold_items = load_pilot_eval_set(ROOT / "data" / "annotations" / "pilot_eval_set.jsonl")
    draft_records = load_jsonl_records(ROOT / "results" / "azure_pilot_drafts.jsonl")
    output_path = ROOT / "results" / "azure_mitigation_eval_results.json"

    gold_by_id = {item.id: item for item in gold_items}
    before_records = []
    after_records = []
    paired_records = []

    for record in draft_records:
        item = gold_by_id[record["id"]]
        before_text = record["response_text"]
        before_observed = classify_response_act(before_text)
        before_void = detect_void_commitment(before_text)
        before_uptake = before_observed == item.expected_response_act

        pipeline_result = run_pipeline(item.user_turn, before_text)
        after_text = pipeline_result.repaired_response
        after_observed = classify_response_act(after_text)
        after_void = detect_void_commitment(after_text)
        after_uptake = after_observed == item.expected_response_act

        before_record = {
            "id": item.id,
            "speech_act_family": item.speech_act_family,
            "difficulty": item.difficulty,
            "uptake_ok": before_uptake,
            "void_commitment": before_void,
        }
        after_record = {
            "id": item.id,
            "speech_act_family": item.speech_act_family,
            "difficulty": item.difficulty,
            "uptake_ok": after_uptake,
            "void_commitment": after_void,
        }
        before_records.append(before_record)
        after_records.append(after_record)

        paired_records.append(
            {
                "id": item.id,
                "speech_act_family": item.speech_act_family,
                "difficulty": item.difficulty,
                "expected_response_act": item.expected_response_act,
                "before_observed_response_act": before_observed,
                "before_uptake_ok": before_uptake,
                "before_void_commitment": before_void,
                "after_observed_response_act": after_observed,
                "after_uptake_ok": after_uptake,
                "after_void_commitment": after_void,
                "response_changed": before_text != after_text,
                "before_response_text": before_text,
                "after_response_text": after_text,
            }
        )

    before_summary = summarize_records(before_records)
    after_summary = summarize_records(after_records)

    changed_count = sum(1 for record in paired_records if record["response_changed"])
    uptake_improved_count = sum(
        1
        for record in paired_records
        if (not record["before_uptake_ok"]) and record["after_uptake_ok"]
    )
    void_fixed_count = sum(
        1
        for record in paired_records
        if record["before_void_commitment"] and (not record["after_void_commitment"])
    )

    summary = {
        "num_items": len(paired_records),
        "model_name": draft_records[0]["model_name"] if draft_records else "unknown",
        "before": before_summary,
        "after": after_summary,
        "delta": {
            "uptake_accuracy": round(
                after_summary["uptake_accuracy"] - before_summary["uptake_accuracy"],
                3,
            ),
            "void_commitment_rate": round(
                after_summary["void_commitment_rate"] - before_summary["void_commitment_rate"],
                3,
            ),
        },
        "changed_count": changed_count,
        "uptake_improved_count": uptake_improved_count,
        "void_fixed_count": void_fixed_count,
        "records": paired_records,
    }

    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Loaded {len(paired_records)} Azure model drafts for mitigation evaluation.")
    print(f"Model: {summary['model_name']}")
    print(f"Before uptake accuracy: {before_summary['uptake_accuracy']}")
    print(f"After uptake accuracy: {after_summary['uptake_accuracy']}")
    print(f"Uptake delta: {summary['delta']['uptake_accuracy']}")
    print(f"Before void commitment rate: {before_summary['void_commitment_rate']}")
    print(f"After void commitment rate: {after_summary['void_commitment_rate']}")
    print(f"Void commitment delta: {summary['delta']['void_commitment_rate']}")
    print(f"Responses changed: {changed_count}")
    print(f"Uptake improved on {uptake_improved_count} items.")
    print(f"Void commitments fixed on {void_fixed_count} items.")
    print(f"Saved detailed results to {output_path}")


if __name__ == "__main__":
    main()
