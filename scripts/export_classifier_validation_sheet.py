"""Export a stratified sample for human validation of the automatic uptake classifier.

Samples ~100 items from gpt-4o responses (one model, to keep annotation scope
manageable), stratified 50/50 between classifier-ok and classifier-fail cases,
balanced across speech-act families.

This tests both:
  - Precision: when classifier says uptake_ok, is it right?
  - Recall:    when classifier says uptake_fail, did it actually fail?

Output: results/classifier_validation_sheet.csv
  Open in Excel or Google Sheets. Fill in the 'human_uptake_ok' column only.
  Leave other columns untouched.

After annotating, run:
    python scripts/score_classifier_validation.py
"""

import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from speech_act_contract.data.loader import load_full_benchmark

RANDOM_SEED = 42
TARGET_PER_STRATUM = 8   # 6 families × 2 strata (ok/fail) × ~8 = ~96 items


def main() -> None:
    benchmark = {
        item.id: item
        for item in load_full_benchmark(ROOT / "data" / "annotations" / "full_uptake_benchmark_v3.jsonl")
    }

    # Load gpt-4o mitigation eval (has before_response_text and classifier labels)
    data = json.loads(
        (ROOT / "results" / "azure_full_benchmark_mitigation_eval_v4.json").read_text()
    )
    records = data["records"]

    # Build pool: (family, classifier_ok) -> list of records
    pool: dict[tuple, list] = defaultdict(list)
    for r in records:
        ok = r["before_uptake_ok"]
        pool[(r["speech_act_family"], ok)].append(r)

    rng = random.Random(RANDOM_SEED)
    sampled = []
    for family in sorted({r["speech_act_family"] for r in records}):
        for ok in [True, False]:
            candidates = pool[(family, ok)]
            n = min(TARGET_PER_STRATUM, len(candidates))
            sampled.extend(rng.sample(candidates, n))

    # Shuffle so annotator doesn't see family clusters
    rng.shuffle(sampled)

    # Add a row number so annotator can reference items
    output_path = ROOT / "results" / "classifier_validation_sheet.csv"
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "row",
            "id",
            "speech_act_family",
            "difficulty",
            "user_turn",
            "model_response",
            "expected_response_act",
            "classifier_label",
            "classifier_uptake_ok",
            # --- ANNOTATOR FILLS THIS IN ---
            "human_uptake_ok",
            # optional: free-text notes
            "notes",
        ])
        writer.writeheader()
        for i, r in enumerate(sampled, start=1):
            bm = benchmark[r["id"]]
            writer.writerow({
                "row": i,
                "id": r["id"],
                "speech_act_family": r["speech_act_family"],
                "difficulty": r["difficulty"],
                "user_turn": bm.user_turn,
                "model_response": r["before_response_text"],
                "expected_response_act": r["expected_response_act"],
                "classifier_label": r["before_observed_response_act"],
                "classifier_uptake_ok": r["before_uptake_ok"],
                "human_uptake_ok": "",   # <-- annotator fills this: TRUE or FALSE
                "notes": "",
            })

    print(f"Exported {len(sampled)} items to {output_path}")
    print()
    print("ANNOTATION INSTRUCTIONS")
    print("=" * 60)
    print("For each row, read the 'user_turn' and 'model_response'.")
    print("In the 'human_uptake_ok' column, write TRUE or FALSE:")
    print()
    print("  TRUE  — the response adequately satisfies the speech-act")
    print("          obligation described in 'expected_response_act'.")
    print("          The model made the right conversational move.")
    print()
    print("  FALSE — the response fails to satisfy the obligation.")
    print("          The model ignored, deflected, or made the wrong")
    print("          conversational move.")
    print()
    print("Label guide:")
    print("  acceptance_reassurance  — for apology: reassures the user")
    print("  acknowledgment          — for thanks: you're welcome / glad to help")
    print("  repair_acknowledgment   — for frustration: acknowledges + commits to fix")
    print("  bounded_acknowledgment  — for preference: confirms the preference in-session")
    print("  accept_or_decline_offer — for offer: accepts or declines clearly")
    print("  acknowledge_update      — for correction: confirms the correction")
    print()
    print("You do NOT need to identify the label — just TRUE/FALSE for uptake_ok.")
    print("Use 'notes' column for anything borderline.")
    print()
    print(f"Total items: {len(sampled)}")
    from collections import Counter
    strat = Counter((r["speech_act_family"], r["before_uptake_ok"]) for r in sampled)
    for (fam, ok), cnt in sorted(strat.items()):
        print(f"  {fam:20s}  classifier={'ok  ' if ok else 'fail'}  n={cnt}")


if __name__ == "__main__":
    main()
