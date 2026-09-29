"""Apply the v4 mitigation pipeline to gpt-5.2 full benchmark drafts.

Reads from results/gpt52_full_benchmark_drafts_v3.jsonl (300-item v3 benchmark).
Writes to results/gpt52_full_benchmark_mitigation_eval_v4.json.

The v4 pipeline uses prepend-not-replace repair, which preserves original
content while adding the required uptake prefix. Run this before judging.

Run generate_gpt52_full_benchmark_drafts_v3.py first.
"""

from pathlib import Path
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


def main() -> None:
    gold_items = load_full_benchmark(ROOT / "data" / "annotations" / "full_uptake_benchmark_v3.jsonl")
    draft_records = load_jsonl_records(ROOT / "results" / "gpt52_full_benchmark_drafts_v3.jsonl")
    output_path = ROOT / "results" / "gpt52_full_benchmark_mitigation_eval_v4.json"

    gold_by_id = {item.id: item for item in gold_items}
    before_records, after_records, paired_records = [], [], []

    for record in draft_records:
        item = gold_by_id[record["id"]]

        before_text = record["response_text"]
        before_observed = classify_response_act(before_text)
        before_void = detect_void_commitment(before_text)
        before_uptake = before_observed in item.acceptable_alternative_response_acts

        pipeline_result = run_pipeline(item.user_turn, before_text)
        after_text = pipeline_result.repaired_response
        after_observed = classify_response_act(after_text)
        after_void = detect_void_commitment(after_text)
        after_uptake = after_observed in item.acceptable_alternative_response_acts

        before_records.append({"id": item.id, "speech_act_family": item.speech_act_family, "difficulty": item.difficulty, "source_type": item.source_type, "uptake_ok": before_uptake, "void_commitment": before_void})
        after_records.append({"id": item.id, "speech_act_family": item.speech_act_family, "difficulty": item.difficulty, "source_type": item.source_type, "uptake_ok": after_uptake, "void_commitment": after_void})
        paired_records.append({
            "id": item.id,
            "speech_act_family": item.speech_act_family,
            "difficulty": item.difficulty,
            "source_type": item.source_type,
            "expected_response_act": item.expected_response_act,
            "acceptable_alternative_response_acts": item.acceptable_alternative_response_acts,
            "before_observed_response_act": before_observed,
            "before_uptake_ok": before_uptake,
            "before_void_commitment": before_void,
            "after_observed_response_act": after_observed,
            "after_uptake_ok": after_uptake,
            "after_void_commitment": after_void,
            "response_changed": before_text != after_text,
            "before_response_text": before_text,
            "after_response_text": after_text,
        })

    n = len(paired_records)
    before_up = sum(1 for r in before_records if r["uptake_ok"])
    after_up = sum(1 for r in after_records if r["uptake_ok"])
    before_void = sum(1 for r in before_records if r["void_commitment"])
    after_void = sum(1 for r in after_records if r["void_commitment"])
    changed = sum(1 for r in paired_records if r["response_changed"])
    uptake_improved = sum(1 for r in paired_records if not r["before_uptake_ok"] and r["after_uptake_ok"])
    void_fixed = sum(1 for r in paired_records if r["before_void_commitment"] and not r["after_void_commitment"])

    summary = {
        "num_items": n,
        "model_name": draft_records[0]["model_name"] if draft_records else "unknown",
        "before": {
            "uptake_accuracy": round(before_up / n, 3),
            "void_commitment_rate": round(before_void / n, 3),
        },
        "after": {
            "uptake_accuracy": round(after_up / n, 3),
            "void_commitment_rate": round(after_void / n, 3),
        },
        "delta": {
            "uptake_accuracy": round((after_up - before_up) / n, 3),
            "void_commitment_rate": round((after_void - before_void) / n, 3),
        },
        "changed_count": changed,
        "uptake_improved_count": uptake_improved,
        "void_fixed_count": void_fixed,
        "records": paired_records,
    }

    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Model: {summary['model_name']}")
    print(f"Before uptake: {summary['before']['uptake_accuracy']}  ->  After: {summary['after']['uptake_accuracy']}")
    print(f"Before void:   {summary['before']['void_commitment_rate']}  ->  After: {summary['after']['void_commitment_rate']}")
    print(f"Responses changed: {changed}")
    print(f"Uptake improved on {uptake_improved} items, void fixed on {void_fixed} items.")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
