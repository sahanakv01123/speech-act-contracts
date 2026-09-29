"""Export a stratified IAA annotation sheet from the gold benchmark.

Samples 50 items stratified across speech-act families (≈8–9 per family),
balanced across difficulty levels. Strips all gold labels so the second
annotator sees only the user turn.

Output: results/iaa_annotation_sheet.csv
  Give this to a second annotator. They fill in:
    annotator_family         — one of: apology, thanks, offer, correction,
                               frustration, preference, other
    annotator_response_act   — one of: acceptance_reassurance, acknowledgment,
                               repair_acknowledgment, bounded_acknowledgment,
                               accept_or_decline_offer, acknowledge_update, other

After annotation, run:
    python scripts/score_iaa.py
"""

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RANDOM_SEED = 123
TARGET_PER_FAMILY = 9   # 6 families × 9 ≈ 54, trimmed to ~50

FAMILIES = ["apology", "thanks", "offer", "correction", "frustration", "preference"]


def main() -> None:
    items = []
    with (ROOT / "data" / "annotations" / "full_uptake_benchmark_v3.jsonl").open() as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    # Stratify by family × difficulty
    by_family: dict[str, list] = defaultdict(list)
    for item in items:
        by_family[item["speech_act_family"]].append(item)

    rng = random.Random(RANDOM_SEED)
    sampled = []
    for fam in FAMILIES:
        pool = by_family[fam]
        n = min(TARGET_PER_FAMILY, len(pool))
        sampled.extend(rng.sample(pool, n))

    # Shuffle so annotator doesn't see family clusters
    rng.shuffle(sampled)

    out_path = ROOT / "results" / "iaa_annotation_sheet.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "row",
            "id",
            "difficulty",
            "user_turn",
            # Annotator fills these two columns:
            "annotator_family",
            "annotator_response_act",
            # Hidden gold (for scoring later — annotator should not see):
            "_gold_family",
            "_gold_response_act",
        ])
        writer.writeheader()
        for i, item in enumerate(sampled, start=1):
            writer.writerow({
                "row": i,
                "id": item["id"],
                "difficulty": item["difficulty"],
                "user_turn": item["user_turn"],
                "annotator_family": "",
                "annotator_response_act": "",
                "_gold_family": item["speech_act_family"],
                "_gold_response_act": item["expected_response_act"],
            })

    print(f"Exported {len(sampled)} items to {out_path}")
    print()
    print("ANNOTATOR INSTRUCTIONS")
    print("=" * 60)
    print("For each row, read the 'user_turn' and fill in two columns:")
    print()
    print("  annotator_family: the type of speech act the user is making.")
    print("    apology      — user apologises or expresses regret")
    print("    thanks       — user expresses gratitude")
    print("    offer        — user offers help/share something")
    print("    correction   — user corrects a factual/formatting error")
    print("    frustration  — user expresses frustration at the assistant")
    print("    preference   — user states a conversational preference")
    print("    other        — doesn't fit any category above")
    print()
    print("  annotator_response_act: the ideal conversational move in reply.")
    print("    acceptance_reassurance — reassure / accept the apology")
    print("    acknowledgment         — 'you're welcome', glad to help")
    print("    repair_acknowledgment  — acknowledge + commit to fix the error")
    print("    bounded_acknowledgment — confirm preference for THIS conversation")
    print("    accept_or_decline_offer— clearly accept or decline the offer")
    print("    acknowledge_update     — confirm the correction was received")
    print("    other                  — none of the above")
    print()
    print("Do NOT look at the _gold_* columns — those are for scoring only.")
    print(f"\nTotal items: {len(sampled)}")
    from collections import Counter
    fam_counts = Counter(item["speech_act_family"] for item in sampled)
    for fam in FAMILIES:
        print(f"  {fam:<14}  n={fam_counts[fam]}")


if __name__ == "__main__":
    main()
