from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    path = ROOT / "results" / "azure_full_benchmark_mitigation_eval_v2.json"
    output_path = ROOT / "results" / "azure_full_benchmark_mitigation_sample.md"
    data = json.loads(path.read_text(encoding="utf-8"))

    changed = [record for record in data["records"] if record["response_changed"]]
    sample = changed[:20]

    lines = ["# Mitigation Sample Review", ""]
    for record in sample:
        lines.extend(
            [
                f"## {record['id']}",
                f"- family: `{record['speech_act_family']}`",
                f"- difficulty: `{record['difficulty']}`",
                f"- expected: `{record['expected_response_act']}`",
                f"- before act: `{record['before_observed_response_act']}`",
                f"- after act: `{record['after_observed_response_act']}`",
                "",
                "### Before",
                record["before_response_text"],
                "",
                "### After",
                record["after_response_text"],
                "",
            ]
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(sample)} changed examples to {output_path}")


if __name__ == "__main__":
    main()
