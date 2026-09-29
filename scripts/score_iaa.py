"""Score inter-annotator agreement on the IAA annotation sheet.

Reads: results/iaa_annotation_sheet.csv  (after second annotator fills it in)
Writes: results/iaa_scores.json
        results/iaa_report.txt

Computes:
  - Cohen's κ for speech_act_family (annotator vs. gold)
  - Cohen's κ for expected_response_act (annotator vs. gold)
  - Per-family accuracy
  - Confusion matrices

Usage:
    python scripts/score_iaa.py
    python scripts/score_iaa.py path/to/sheet.csv
"""

import csv
import json
import sys
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JSON_OUT = ROOT / "results" / "iaa_scores.json"
TXT_OUT  = ROOT / "results" / "iaa_report.txt"


def cohens_kappa_multiclass(gold_labels: list, pred_labels: list) -> float:
    """Cohen's kappa for multi-class labels."""
    assert len(gold_labels) == len(pred_labels)
    n = len(gold_labels)
    if n == 0:
        return 0.0

    # All unique labels
    labels = sorted(set(gold_labels) | set(pred_labels))
    label_idx = {l: i for i, l in enumerate(labels)}
    k = len(labels)

    # Confusion matrix
    cm = [[0] * k for _ in range(k)]
    for g, p in zip(gold_labels, pred_labels):
        cm[label_idx[g]][label_idx[p]] += 1

    p_o = sum(cm[i][i] for i in range(k)) / n

    gold_counts = Counter(gold_labels)
    pred_counts = Counter(pred_labels)
    p_e = sum(
        (gold_counts[l] / n) * (pred_counts[l] / n)
        for l in labels
    )

    if p_e >= 1.0:
        return 1.0
    return round((p_o - p_e) / (1 - p_e), 3)


def per_family_accuracy(rows: list[dict], gold_key: str, pred_key: str) -> dict:
    by_family: dict[str, list] = defaultdict(list)
    for r in rows:
        by_family[r["_gold_family"]].append(r)
    result = {}
    for fam in sorted(by_family):
        recs = by_family[fam]
        correct = sum(1 for r in recs if r[gold_key] == r[pred_key])
        result[fam] = {
            "n": len(recs),
            "correct": correct,
            "accuracy": round(correct / len(recs), 3),
        }
    return result


def load_csv(path: Path) -> tuple[list[dict], int]:
    rows, skipped = [], 0
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fam = row.get("annotator_family", "").strip()
            act = row.get("annotator_response_act", "").strip()
            if not fam or not act:
                skipped += 1
                continue
            rows.append({
                "_gold_family": row["_gold_family"].strip().lower(),
                "_gold_response_act": row["_gold_response_act"].strip().lower(),
                "annotator_family": fam.lower(),
                "annotator_response_act": act.lower(),
            })
    return rows, skipped


def load_numbers(path: Path) -> tuple[list[dict], int]:
    try:
        from numbers_parser import Document
    except ImportError:
        raise RuntimeError("Install numbers-parser: pip install numbers-parser --break-system-packages")
    doc = Document(str(path))
    table = doc.sheets[0].tables[0]
    raw_rows = list(table.iter_rows())
    headers = [cell.value for cell in raw_rows[0]]
    col = {h: i for i, h in enumerate(headers)}
    rows, skipped = [], 0
    for row in raw_rows[1:]:
        fam = str(row[col["annotator_family"]].value or "").strip()
        act = str(row[col["annotator_response_act"]].value or "").strip()
        if not fam or not act or fam == "None" or act == "None":
            skipped += 1
            continue
        rows.append({
            "_gold_family": str(row[col["_gold_family"]].value or "").strip().lower(),
            "_gold_response_act": str(row[col["_gold_response_act"]].value or "").strip().lower(),
            "annotator_family": fam.lower(),
            "annotator_response_act": act.lower(),
        })
    return rows, skipped


def main() -> None:
    if len(sys.argv) > 1:
        sheet_path = Path(sys.argv[1]).expanduser().resolve()
    else:
        # Check root dir first (user may have saved there), then results/
        candidates = [
            ROOT / "iaa_annotation_sheet.numbers",
            ROOT / "results" / "iaa_annotation_sheet.numbers",
            ROOT / "results" / "iaa_annotation_sheet.csv",
        ]
        sheet_path = next((p for p in candidates if p.exists()), None)
        if sheet_path is None:
            raise FileNotFoundError("IAA sheet not found. Run export_iaa_sheet.py first.")

    if not sheet_path.exists():
        raise FileNotFoundError(f"IAA sheet not found: {sheet_path}")

    print(f"Reading: {sheet_path}")
    if sheet_path.suffix == ".numbers":
        rows, skipped = load_numbers(sheet_path)
    else:
        rows, skipped = load_csv(sheet_path)

    if not rows:
        print("No annotated rows found. Fill in 'annotator_family' and 'annotator_response_act' first.")
        return

    print(f"Annotated: {len(rows)}  |  Skipped (empty): {skipped}")

    # Kappa for family label
    gold_fam  = [r["_gold_family"] for r in rows]
    pred_fam  = [r["annotator_family"] for r in rows]
    kappa_fam = cohens_kappa_multiclass(gold_fam, pred_fam)
    acc_fam   = round(sum(g == p for g, p in zip(gold_fam, pred_fam)) / len(rows), 3)

    # Kappa for response-act label
    gold_act  = [r["_gold_response_act"] for r in rows]
    pred_act  = [r["annotator_response_act"] for r in rows]
    kappa_act = cohens_kappa_multiclass(gold_act, pred_act)
    acc_act   = round(sum(g == p for g, p in zip(gold_act, pred_act)) / len(rows), 3)

    # Per-family accuracy for both tasks
    fam_acc = per_family_accuracy(rows, "_gold_family", "annotator_family")
    act_acc = per_family_accuracy(rows, "_gold_response_act", "annotator_response_act")

    # Confusion matrix (family)
    fam_labels = sorted(set(gold_fam) | set(pred_fam))
    act_labels = sorted(set(gold_act) | set(pred_act))

    results = {
        "n": len(rows),
        "family_kappa": kappa_fam,
        "family_accuracy": acc_fam,
        "response_act_kappa": kappa_act,
        "response_act_accuracy": acc_act,
        "per_family": fam_acc,
        "per_act": act_acc,
    }
    JSON_OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # ---- Text report ----
    W = 60
    lines = [
        "=" * W,
        "INTER-ANNOTATOR AGREEMENT REPORT",
        f"N annotated: {len(rows)}  |  N skipped: {skipped}",
        "=" * W,
        "",
        "TASK 1: Speech-Act Family Label (annotator vs. gold)",
        "-" * W,
        f"  Cohen's κ:   {kappa_fam}",
        f"  Accuracy:    {acc_fam}",
        "",
        "TASK 2: Expected Response Act (annotator vs. gold)",
        "-" * W,
        f"  Cohen's κ:   {kappa_act}",
        f"  Accuracy:    {acc_act}",
        "",
        "Kappa interpretation: <0.40 poor, 0.40-0.60 moderate,",
        "                       0.60-0.80 substantial, >0.80 near-perfect",
        "",
        "=" * W,
        "PER-FAMILY ACCURACY (response-act label)",
        "=" * W,
        f"{'Family':<22}  {'n':>4}  {'Correct':>7}  {'Accuracy':>8}",
        "-" * W,
    ]
    for fam in sorted(act_acc):
        s = act_acc[fam]
        lines.append(f"{fam:<22}  {s['n']:>4}  {s['correct']:>7}  {s['accuracy']:>8.3f}")

    # Confusion matrix for family
    lines += [
        "",
        "=" * W,
        "CONFUSION MATRIX: Family label (rows=gold, cols=annotator)",
        "=" * W,
    ]
    header_cols = "  ".join(f"{l[:5]:>6}" for l in fam_labels)
    lines.append(f"{'':>14}  {header_cols}")
    cm_fam: dict[tuple, int] = Counter(zip(gold_fam, pred_fam))
    for g in fam_labels:
        row_vals = "  ".join(f"{cm_fam.get((g, p), 0):>6}" for p in fam_labels)
        lines.append(f"{g:<14}  {row_vals}")

    report = "\n".join(lines)
    TXT_OUT.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved JSON   → {JSON_OUT}")
    print(f"Saved report → {TXT_OUT}")


if __name__ == "__main__":
    main()
