"""Judge mitigation quality on failure cases only.

This script is the paper-ready replacement for judge_mitigation_changes.py.
Key differences:
  1. Reads from the v4 mitigation eval (prepend-not-replace strategy).
  2. Evaluates ONLY cases where the original response had an identified failure
     (before_uptake_ok=False OR before_void_commitment=True).
  3. Uses the updated judge prompt that describes the specific failure to the judge,
     so it evaluates repair quality rather than generic preference.
  4. Reports failure_repaired_rate as the primary metric alongside helpfulness/
     naturalness preservation.

Usage:
    python scripts/judge_mitigation_failures.py [mitigation_eval_path] [limit]

Defaults to azure_full_benchmark_mitigation_eval_v4.json for gpt-4o.
"""

from pathlib import Path
import json
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from speech_act_contract.eval.llm_judge import (
    JUDGE_SYSTEM_PROMPT,
    build_judge_user_prompt,
    parse_judge_response,
    summarize_judgments,
)
from speech_act_contract.data.loader import load_full_benchmark
from speech_act_contract.llm.azure_openai import generate_chat_completion, load_azure_openai_config


def default_output_paths(input_path: Path) -> tuple[Path, Path]:
    stem = input_path.stem.replace(".json", "")
    return (
        ROOT / "results" / f"{stem}_failure_judge_records.jsonl",
        ROOT / "results" / f"{stem}_failure_judge_summary.json",
    )


def is_failure_case(record: dict) -> bool:
    """Return True if the original response had an identified speech-act failure."""
    return (not record["before_uptake_ok"]) or record["before_void_commitment"]


def main() -> None:
    input_path = (
        Path(sys.argv[1]).expanduser().resolve()
        if len(sys.argv) > 1
        else ROOT / "results" / "azure_full_benchmark_mitigation_eval_v4.json"
    )
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    output_jsonl, output_summary = default_output_paths(input_path)

    summary = json.loads(input_path.read_text(encoding="utf-8"))
    benchmark_items = load_full_benchmark(ROOT / "data" / "annotations" / "full_uptake_benchmark_v3.jsonl")
    benchmark_by_id = {item.id: item for item in benchmark_items}

    # Filter to failure cases only — this is the key change vs judge_mitigation_changes.py
    all_records = summary["records"]
    failure_records = [r for r in all_records if is_failure_case(r)]
    non_failure_count = len(all_records) - len(failure_records)

    print(f"Total records in input:          {len(all_records)}")
    print(f"Failure cases (will be judged):  {len(failure_records)}")
    print(f"Non-failure cases (skipped):     {non_failure_count}")

    if limit > 0:
        failure_records = failure_records[:limit]
        print(f"Limit applied: judging {len(failure_records)} cases.")

    config = load_azure_openai_config()
    judged_records = []
    filtered_ids: list[str] = []

    for index, record in enumerate(failure_records, start=1):
        benchmark_item = benchmark_by_id.get(record["id"])
        if benchmark_item is None:
            raise KeyError(f"Could not find benchmark item for id={record['id']}")

        enriched_record = {
            **record,
            "user_turn": benchmark_item.user_turn,
            "acceptable_alternative_response_acts": benchmark_item.acceptable_alternative_response_acts,
            "expected_response_act": benchmark_item.expected_response_act,
        }

        judge_prompt = build_judge_user_prompt(enriched_record)
        try:
            raw_response = generate_chat_completion(
                config=config,
                system_prompt=JUDGE_SYSTEM_PROMPT,
                user_prompt=judge_prompt,
                temperature=0.0,
                max_tokens=300,
            )
        except Exception as exc:
            # Azure content filter false positives on some benchmark items
            # (typically assertive correction/frustration items misidentified as jailbreaks).
            # Skip and log rather than crash — reported in summary for transparency.
            error_str = str(exc)
            if "content_filter" in error_str or "content management" in error_str or "ResponsibleAI" in error_str:
                print(f"[{index}/{len(failure_records)}] {record['id']:30s} SKIPPED (content filter)")
                filtered_ids.append(record["id"])
                continue
            raise  # re-raise unexpected errors

        parsed = parse_judge_response(raw_response)
        judged = {
            "id": record["id"],
            "speech_act_family": record["speech_act_family"],
            "difficulty": record["difficulty"],
            "source_type": record["source_type"],
            "before_uptake_ok": record["before_uptake_ok"],
            "after_uptake_ok": record["after_uptake_ok"],
            "before_void_commitment": record["before_void_commitment"],
            "after_void_commitment": record["after_void_commitment"],
            "failure_repaired": parsed["failure_repaired"],
            "overall_preference": parsed["overall_preference"],
            "pragmatic_better": parsed["pragmatic_better"],
            "truthfulness_better": parsed["truthfulness_better"],
            "helpfulness_preserved": parsed["helpfulness_preserved"],
            "naturalness_preserved": parsed["naturalness_preserved"],
            "confidence": parsed["confidence"],
            "rationale": parsed["rationale"],
        }
        judged_records.append(judged)
        print(
            f"[{index}/{len(failure_records)}] {record['id']:30s} "
            f"repaired={parsed['failure_repaired']}  "
            f"pref={parsed['overall_preference']}  "
            f"help={parsed['helpfulness_preserved']}"
        )

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for record in judged_records:
            handle.write(json.dumps(record) + "\n")

    aggregate = summarize_judgments(judged_records)
    report = {
        "input_path": str(input_path),
        "model_name": summary.get("model_name", "unknown"),
        "total_records_in_input": len(all_records),
        "num_failure_cases": len(failure_records),
        "num_non_failure_cases_skipped": non_failure_count,
        "num_content_filtered": len(filtered_ids),
        "content_filtered_ids": filtered_ids,
        "num_judged_records": len(judged_records),
        "aggregate": aggregate,
    }
    output_summary.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print(f"=== Results (failure cases only, n={len(judged_records)}) ===")
    print(f"Model under evaluation:   {report['model_name']}")
    print(f"Failure repaired rate:    {aggregate['failure_repaired_rate']}")
    print(f"Overall preference after: {aggregate['overall_preference_rate'].get('after', 0.0)}")
    print(f"Pragmatic better (after): {aggregate['pragmatic_better_rate'].get('after', 0.0)}")
    print(f"Helpfulness preserved:    {aggregate['helpfulness_preserved_rate']}")
    print(f"Naturalness preserved:    {aggregate['naturalness_preserved_rate']}")
    if filtered_ids:
        print(f"Content-filtered (skipped): {len(filtered_ids)} — {filtered_ids}")
    print(f"Saved records to {output_jsonl}")
    print(f"Saved summary to {output_summary}")


if __name__ == "__main__":
    main()
