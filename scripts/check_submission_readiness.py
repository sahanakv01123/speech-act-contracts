from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


REQUIRED_PATHS = [
    ROOT / "paper" / "submission_ready_experiments.md",
    ROOT / "paper" / "submission_checklist.md",
    ROOT / "data" / "annotations" / "annotation_guide.md",
    ROOT / "data" / "annotations" / "full_benchmark_schema.md",
    ROOT / "data" / "annotations" / "void_commitment_guide.md",
    ROOT / "results" / "azure_pilot_eval_results.json",
    ROOT / "results" / "azure_mitigation_eval_results.json",
    ROOT / "results" / "azure_full_benchmark_mitigation_eval_v4.json",
    ROOT / "results" / "claude_full_benchmark_mitigation_eval_v4.json",
    ROOT / "results" / "gpt52_full_benchmark_mitigation_eval_v4.json",
    ROOT / "results" / "classifier_validation_scores.json",
    ROOT / "results" / "judge_analysis.json",
    ROOT / "results" / "full_benchmark_significance.json",
]


def main() -> None:
    missing = [path for path in REQUIRED_PATHS if not path.exists()]

    print("Submission readiness artifact check")
    for path in REQUIRED_PATHS:
        status = "OK" if path.exists() else "MISSING"
        print(f"{status:8} {path}")

    if missing:
        print(f"\nMissing {len(missing)} required artifacts.")
    else:
        print("\nAll currently tracked required artifacts are present.")


if __name__ == "__main__":
    main()
