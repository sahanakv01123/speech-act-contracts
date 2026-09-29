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
        ROOT / "results" / f"{stem}_judge_records.jsonl",
        ROOT / "results" / f"{stem}_judge_summary.json",
    )


def main() -> None:
    input_path = (
        Path(sys.argv[1]).expanduser().resolve()
        if len(sys.argv) > 1
        else ROOT / "results" / "azure_full_benchmark_mitigation_eval_v3.json"
    )
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    output_jsonl, output_summary = default_output_paths(input_path)

    summary = json.loads(input_path.read_text(encoding="utf-8"))
    benchmark_items = load_full_benchmark(ROOT / "data" / "annotations" / "full_uptake_benchmark_v3.jsonl")
    benchmark_by_id = {item.id: item for item in benchmark_items}
    changed_records = [record for record in summary["records"] if record["response_changed"]]
    if limit > 0:
        changed_records = changed_records[:limit]

    config = load_azure_openai_config()
    judged_records = []

    for index, record in enumerate(changed_records, start=1):
        benchmark_item = benchmark_by_id.get(record["id"])
        if benchmark_item is None:
            raise KeyError(f"Could not find benchmark item for mitigation record id={record['id']}")

        enriched_record = {
            **record,
            "user_turn": benchmark_item.user_turn,
            "acceptable_alternative_response_acts": benchmark_item.acceptable_alternative_response_acts,
            "expected_response_act": benchmark_item.expected_response_act,
        }

        judge_prompt = build_judge_user_prompt(enriched_record)
        raw_response = generate_chat_completion(
            config=config,
            system_prompt=JUDGE_SYSTEM_PROMPT,
            user_prompt=judge_prompt,
            temperature=0.0,
            max_tokens=260,
        )
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
            "overall_preference": parsed["overall_preference"],
            "pragmatic_better": parsed["pragmatic_better"],
            "truthfulness_better": parsed["truthfulness_better"],
            "helpfulness_preserved": parsed["helpfulness_preserved"],
            "naturalness_preserved": parsed["naturalness_preserved"],
            "confidence": parsed["confidence"],
            "rationale": parsed["rationale"],
        }
        judged_records.append(judged)
        print(f"Judged {index}/{len(changed_records)} changed responses: {record['id']}")

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for record in judged_records:
            handle.write(json.dumps(record) + "\n")

    aggregate = summarize_judgments(judged_records)
    report = {
        "input_path": str(input_path),
        "model_name": summary.get("model_name", "unknown"),
        "num_changed_records_in_input": len([record for record in summary["records"] if record["response_changed"]]),
        "num_judged_records": len(judged_records),
        "aggregate": aggregate,
    }
    output_summary.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Judged {len(judged_records)} changed mitigation outputs.")
    print(f"Model under evaluation: {report['model_name']}")
    print(f"Overall preference for after: {aggregate['overall_preference_rate'].get('after', 0.0)}")
    print(f"Pragmatic improvement rate: {aggregate['pragmatic_better_rate'].get('after', 0.0)}")
    print(f"Truthfulness improvement rate: {aggregate['truthfulness_better_rate'].get('after', 0.0)}")
    print(f"Helpfulness preserved rate: {aggregate['helpfulness_preserved_rate']}")
    print(f"Naturalness preserved rate: {aggregate['naturalness_preserved_rate']}")
    print(f"Saved judged records to {output_jsonl}")
    print(f"Saved summary to {output_summary}")


if __name__ == "__main__":
    main()
