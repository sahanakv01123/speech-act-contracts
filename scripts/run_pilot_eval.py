from collections import Counter
from pathlib import Path
import json
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from speech_act_contract.data.loader import load_pilot_eval_set
from speech_act_contract.mitigation.pipeline import detect_void_commitment, run_pipeline


def main() -> None:
    eval_path = ROOT / "data" / "annotations" / "pilot_eval_set.jsonl"
    output_path = ROOT / "results" / "pilot_eval_results.json"
    items = load_pilot_eval_set(eval_path)

    family_counts = Counter()
    family_uptake_hits = Counter()
    family_void_hits = Counter()
    difficulty_counts = Counter()
    difficulty_repair_proxy_hits = Counter()
    repair_proxy_hits = 0
    records = []

    for item in items:
        result = run_pipeline(item.user_turn, item.bad_draft_response)
        family_counts[item.speech_act_family] += 1
        difficulty_counts[item.difficulty] += 1

        if result.user_act == item.speech_act_family:
            family_uptake_hits[item.speech_act_family] += 1

        if result.void_commitment == item.gold_void_commitment:
            family_void_hits[item.speech_act_family] += 1

        repair_changed = result.repaired_response != item.bad_draft_response
        repaired_observed_act = result.observed_response_act
        if repair_changed:
            repaired_observed_act = result.expected_response_act

        repair_success_proxy = (
            repair_changed
            and result.expected_response_act == item.expected_response_act
            and not detect_void_commitment(result.repaired_response)
        )

        if repair_success_proxy:
            repair_proxy_hits += 1
            difficulty_repair_proxy_hits[item.difficulty] += 1

        records.append(
            {
                "id": item.id,
                "speech_act_family": item.speech_act_family,
                "difficulty": item.difficulty,
                "gold_expected_response_act": item.expected_response_act,
                "predicted_user_act": result.user_act,
                "predicted_expected_response_act": result.expected_response_act,
                "observed_response_act": result.observed_response_act,
                "uptake_ok_on_bad_draft": result.uptake_ok,
                "gold_void_commitment": item.gold_void_commitment,
                "predicted_void_commitment": result.void_commitment,
                "gold_repair_should_improve": item.gold_repair_should_improve,
                "repair_changed": repair_changed,
                "repair_success_proxy": repair_success_proxy,
                "repaired_response": result.repaired_response,
            }
        )

    summary = {
        "num_items": len(items),
        "user_act_accuracy": round(
            sum(family_uptake_hits.values()) / len(items),
            3,
        ),
        "void_commitment_detection_accuracy": round(
            sum(family_void_hits.values()) / len(items),
            3,
        ),
        "repair_success_proxy_rate": round(repair_proxy_hits / len(items), 3),
        "per_family_user_act_accuracy": {
            family: round(family_uptake_hits[family] / count, 3)
            for family, count in sorted(family_counts.items())
        },
        "per_family_void_commitment_accuracy": {
            family: round(family_void_hits[family] / count, 3)
            for family, count in sorted(family_counts.items())
        },
        "per_difficulty_repair_success_proxy_rate": {
            difficulty: round(difficulty_repair_proxy_hits[difficulty] / count, 3)
            for difficulty, count in sorted(difficulty_counts.items())
        },
        "records": records,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Loaded {len(items)} pilot items.")
    print(f"User-act accuracy: {summary['user_act_accuracy']}")
    print(f"Void-commitment detection accuracy: {summary['void_commitment_detection_accuracy']}")
    print(f"Repair success proxy rate: {summary['repair_success_proxy_rate']}")
    print("Per-family user-act accuracy:")
    for family, score in summary["per_family_user_act_accuracy"].items():
        print(f"  {family}: {score}")
    print("Per-family void-commitment accuracy:")
    for family, score in summary["per_family_void_commitment_accuracy"].items():
        print(f"  {family}: {score}")
    print("Per-difficulty repair success proxy rate:")
    for difficulty, score in summary["per_difficulty_repair_success_proxy_rate"].items():
        print(f"  {difficulty}: {score}")
    print(f"Saved detailed results to {output_path}")


if __name__ == "__main__":
    main()
