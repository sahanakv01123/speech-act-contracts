from collections import Counter
from pathlib import Path
import json
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from speech_act_contract.data.loader import load_full_benchmark, load_jsonl_records
from speech_act_contract.mitigation.pipeline import classify_response_act, detect_void_commitment


def main() -> None:
    gold_items = load_full_benchmark(ROOT / "data" / "annotations" / "full_uptake_benchmark_v3.jsonl")
    draft_records = load_jsonl_records(ROOT / "results" / "azure_full_benchmark_drafts_v3.jsonl")
    output_path = ROOT / "results" / "azure_full_benchmark_eval_v3.json"

    gold_by_id = {item.id: item for item in gold_items}
    family_counts = Counter()
    family_uptake_hits = Counter()
    family_void_counts = Counter()
    difficulty_counts = Counter()
    difficulty_uptake_hits = Counter()
    source_counts = Counter()
    source_uptake_hits = Counter()
    records = []

    for record in draft_records:
        item = gold_by_id[record["id"]]
        response_text = record["response_text"]
        observed_response_act = classify_response_act(response_text)
        void_commitment = detect_void_commitment(response_text)
        uptake_ok = observed_response_act in item.acceptable_alternative_response_acts

        family_counts[item.speech_act_family] += 1
        difficulty_counts[item.difficulty] += 1
        source_counts[item.source_type] += 1

        if uptake_ok:
            family_uptake_hits[item.speech_act_family] += 1
            difficulty_uptake_hits[item.difficulty] += 1
            source_uptake_hits[item.source_type] += 1
        if void_commitment:
            family_void_counts[item.speech_act_family] += 1

        records.append(
            {
                "id": item.id,
                "speech_act_family": item.speech_act_family,
                "difficulty": item.difficulty,
                "source_type": item.source_type,
                "expected_response_act": item.expected_response_act,
                "acceptable_alternative_response_acts": item.acceptable_alternative_response_acts,
                "observed_response_act": observed_response_act,
                "uptake_ok": uptake_ok,
                "void_commitment": void_commitment,
                "response_text": response_text,
            }
        )

    summary = {
        "num_items": len(draft_records),
        "model_name": draft_records[0]["model_name"] if draft_records else "unknown",
        "uptake_accuracy": round(sum(family_uptake_hits.values()) / len(draft_records), 3),
        "void_commitment_rate": round(sum(family_void_counts.values()) / len(draft_records), 3),
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
        "per_source_uptake_accuracy": {
            source_type: round(source_uptake_hits[source_type] / count, 3)
            for source_type, count in sorted(source_counts.items())
        },
        "records": records,
    }

    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Loaded {len(draft_records)} Azure full-benchmark drafts.")
    print(f"Model: {summary['model_name']}")
    print(f"Uptake accuracy: {summary['uptake_accuracy']}")
    print(f"Void commitment rate: {summary['void_commitment_rate']}")
    print("Per-family uptake accuracy:")
    for family, score in summary["per_family_uptake_accuracy"].items():
        print(f"  {family}: {score}")
    print("Per-family void commitment rate:")
    for family, score in summary["per_family_void_commitment_rate"].items():
        print(f"  {family}: {score}")
    print("Per-difficulty uptake accuracy:")
    for difficulty, score in summary["per_difficulty_uptake_accuracy"].items():
        print(f"  {difficulty}: {score}")
    print("Per-source uptake accuracy:")
    for source_type, score in summary["per_source_uptake_accuracy"].items():
        print(f"  {source_type}: {score}")
    print(f"Saved detailed results to {output_path}")


if __name__ == "__main__":
    main()
