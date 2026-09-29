"""C2: export a blinded A/B human-study sheet for an independent (non-author) panel.

Improves on export_repair_study_sheet.py in two ways so the study is suitable
for external raters and directly addresses the circularity concern:

  1. The answer key is written to a SEPARATE private file, so the annotator
     sheet contains no leak of which response is the SpeakFix repair.
  2. The sheet mixes two strata the panel cannot tell apart:
       - repaired_failure: cases SpeakFix genuinely fixed (base vs. repaired)
       - over_trigger_control: already-fine responses SpeakFix nonetheless
         edited (base vs. redundant edit), which tests whether over-triggering
         costs perceived quality.

Each row shows the user message and two responses (A/B) in randomized order.

Usage:
    python scripts/export_ab_human_study.py
    python scripts/export_ab_human_study.py --failures-per-model 12 --controls-total 24
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import argparse
import csv
import json
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from speech_act_contract.data.loader import load_full_benchmark

V4_FILES = {
    "gpt-4o": ROOT / "results" / "azure_full_benchmark_mitigation_eval_v4.json",
    "claude-sonnet-4-6": ROOT / "results" / "claude_full_benchmark_mitigation_eval_v4.json",
    "gpt-5.2": ROOT / "results" / "gpt52_full_benchmark_mitigation_eval_v4.json",
}
OVERTRIGGER_FILES = {
    "gpt-4o": ROOT / "results" / "azure_overtriggering_check.json",
    "claude-sonnet-4-6": ROOT / "results" / "claude_overtriggering_check.json",
    "gpt-5.2": ROOT / "results" / "gpt52_overtriggering_check.json",
}


def canonical(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_repaired_failures(path: Path, model_name: str) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for record in data["records"]:
        is_failure = (not record["before_uptake_ok"]) or record["before_void_commitment"]
        changed = canonical(record["before_response_text"]) != canonical(record["after_response_text"])
        if is_failure and changed:
            rows.append(
                {
                    "model_name": model_name,
                    "id": record["id"],
                    "speech_act_family": record["speech_act_family"],
                    "stratum": "repaired_failure",
                    "base_response": record["before_response_text"],
                    "speakfix_response": record["after_response_text"],
                }
            )
    return rows


def load_over_trigger_controls(path: Path, model_name: str) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for ex in data["over_triggered_examples"]:
        rows.append(
            {
                "model_name": model_name,
                "id": ex["id"],
                "speech_act_family": ex["speech_act_family"],
                "stratum": "over_trigger_control",
                "base_response": ex["before_response_text"],
                "speakfix_response": ex["after_response_text"],
            }
        )
    return rows


def stratified_family_sample(rows: list[dict], target: int, rng: random.Random) -> list[dict]:
    by_family = defaultdict(list)
    for row in rows:
        by_family[row["speech_act_family"]].append(row)
    per_family = max(1, target // max(1, len(by_family)))
    sampled = []
    for family in sorted(by_family):
        cands = by_family[family][:]
        rng.shuffle(cands)
        sampled.extend(cands[:per_family])
    rng.shuffle(sampled)
    if len(sampled) < target:
        chosen = {(r["model_name"], r["id"]) for r in sampled}
        leftovers = [r for r in rows if (r["model_name"], r["id"]) not in chosen]
        rng.shuffle(leftovers)
        sampled.extend(leftovers[: target - len(sampled)])
    return sampled[:target]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failures-per-model", type=int, default=12)
    parser.add_argument("--controls-total", type=int, default=24)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--sheet", type=Path, default=ROOT / "results" / "human_ab_study_sheet.csv")
    parser.add_argument("--key", type=Path, default=ROOT / "results" / "human_ab_study_key.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    user_turn_by_id = {item.id: item.user_turn for item in load_full_benchmark(
        ROOT / "data" / "annotations" / "full_uptake_benchmark_v3.jsonl"
    )}

    pairs: list[dict] = []
    for model_name, path in V4_FILES.items():
        failures = load_repaired_failures(path, model_name)
        pairs.extend(stratified_family_sample(failures, args.failures_per_model, rng))

    all_controls: list[dict] = []
    for model_name, path in OVERTRIGGER_FILES.items():
        all_controls.extend(load_over_trigger_controls(path, model_name))
    rng.shuffle(all_controls)
    pairs.extend(all_controls[: args.controls_total])

    rng.shuffle(pairs)

    with args.sheet.open("w", newline="", encoding="utf-8") as sheet_handle, \
            args.key.open("w", newline="", encoding="utf-8") as key_handle:
        sheet_writer = csv.DictWriter(
            sheet_handle,
            fieldnames=[
                "row",
                "user_message",
                "response_a",
                "response_b",
                "overall_better_A_B_Tie",
                "more_helpful_A_B_Tie",
                "more_natural_A_B_Tie",
                "a_claims_memory_Yes_No",
                "b_claims_memory_Yes_No",
                "notes",
            ],
        )
        sheet_writer.writeheader()

        key_writer = csv.DictWriter(
            key_handle,
            fieldnames=["row", "model_name", "id", "speech_act_family", "stratum", "a_is_speakfix"],
        )
        key_writer.writeheader()

        for row_num, pair in enumerate(pairs, start=1):
            a_is_speakfix = rng.random() < 0.5
            if a_is_speakfix:
                response_a, response_b = pair["speakfix_response"], pair["base_response"]
            else:
                response_a, response_b = pair["base_response"], pair["speakfix_response"]

            sheet_writer.writerow(
                {
                    "row": row_num,
                    "user_message": user_turn_by_id.get(pair["id"], ""),
                    "response_a": response_a,
                    "response_b": response_b,
                    "overall_better_A_B_Tie": "",
                    "more_helpful_A_B_Tie": "",
                    "more_natural_A_B_Tie": "",
                    "a_claims_memory_Yes_No": "",
                    "b_claims_memory_Yes_No": "",
                    "notes": "",
                }
            )
            key_writer.writerow(
                {
                    "row": row_num,
                    "model_name": pair["model_name"],
                    "id": pair["id"],
                    "speech_act_family": pair["speech_act_family"],
                    "stratum": pair["stratum"],
                    "a_is_speakfix": "TRUE" if a_is_speakfix else "FALSE",
                }
            )

    n_fail = sum(1 for p in pairs if p["stratum"] == "repaired_failure")
    n_ctrl = sum(1 for p in pairs if p["stratum"] == "over_trigger_control")
    print(f"Exported {len(pairs)} blinded A/B rows ({n_fail} repaired failures, {n_ctrl} over-trigger controls).")
    print(f"  Annotator sheet (no key): {args.sheet}")
    print(f"  Private answer key:       {args.key}")
    print("Give raters only the sheet; keep the key for scoring.")


if __name__ == "__main__":
    main()
