"""Score human annotations against the automatic classifier.

Reads: results/classifier_validation_sheet.csv   (CSV version)
    or any .numbers file passed as argument       (Apple Numbers version)
Writes: results/classifier_validation_scores.json
        results/classifier_validation_report.txt

Computes:
  - Overall precision, recall, F1 on the binary uptake_ok judgment
  - Cohen's kappa (human vs classifier)
  - Per-family breakdown
  - Confusion matrix

Usage:
    python scripts/score_classifier_validation.py
    python scripts/score_classifier_validation.py path/to/sheet.numbers
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JSON_OUT = ROOT / "results" / "classifier_validation_scores.json"
TXT_OUT  = ROOT / "results" / "classifier_validation_report.txt"


def parse_bool(val) -> bool | None:
    if isinstance(val, bool):
        return val
    if val is None:
        return None
    v = str(val).strip().upper()
    if v in ("TRUE", "YES", "1", "T", "Y"):
        return True
    if v in ("FALSE", "NO", "0", "F", "N"):
        return False
    return None


def load_csv(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            human = parse_bool(row.get("human_uptake_ok", ""))
            if human is None:
                continue
            rows.append({
                "id": row["id"],
                "family": row["speech_act_family"],
                "classifier_ok": row["classifier_uptake_ok"].strip().upper() in ("TRUE", "1"),
                "human_ok": human,
            })
    return rows


def load_numbers(path: Path) -> list[dict]:
    try:
        from numbers_parser import Document
    except ImportError:
        raise RuntimeError("Install numbers-parser: pip install numbers-parser --break-system-packages")
    doc = Document(str(path))
    table = doc.sheets[0].tables[0]
    raw_rows = list(table.iter_rows())
    headers = [cell.value for cell in raw_rows[0]]
    col = {h: i for i, h in enumerate(headers)}
    rows = []
    for row in raw_rows[1:]:
        human = parse_bool(row[col["human_uptake_ok"]].value)
        if human is None:
            continue
        classifier_raw = row[col["classifier_uptake_ok"]].value
        classifier_ok = str(classifier_raw).strip().upper() in ("TRUE", "1", "TRUE") if classifier_raw is not None else False
        rows.append({
            "id": row[col["id"]].value,
            "family": row[col["speech_act_family"]].value,
            "classifier_ok": classifier_ok,
            "human_ok": human,
        })
    return rows


def cohens_kappa(tp, fp, fn, tn) -> float:
    n = tp + fp + fn + tn
    if n == 0:
        return 0.0
    p_o = (tp + tn) / n
    p_yes = ((tp + fp) / n) * ((tp + fn) / n)
    p_no  = ((fn + tn) / n) * ((fp + tn) / n)
    p_e = p_yes + p_no
    if p_e == 1.0:
        return 1.0
    return round((p_o - p_e) / (1 - p_e), 3)


def safe_div(a, b):
    return round(a / b, 3) if b else None


def main() -> None:
    if len(sys.argv) > 1:
        sheet_path = Path(sys.argv[1]).expanduser().resolve()
    else:
        sheet_path = ROOT / "results" / "classifier_validation_sheet.csv"

    if not sheet_path.exists():
        raise FileNotFoundError(f"Annotation sheet not found: {sheet_path}")

    if sheet_path.suffix == ".numbers":
        rows = load_numbers(sheet_path)
    else:
        rows = load_csv(sheet_path)

    skipped = 90 - len(rows)  # approximate — 90 is the total exported

    if not rows:
        print("No annotated rows found. Fill in the 'human_uptake_ok' column first.")
        return

    print(f"Annotated rows: {len(rows)}  |  Skipped (empty): {skipped}")

    # Overall confusion matrix
    tp = sum(1 for r in rows if r["classifier_ok"] and r["human_ok"])
    fp = sum(1 for r in rows if r["classifier_ok"] and not r["human_ok"])
    fn = sum(1 for r in rows if not r["classifier_ok"] and r["human_ok"])
    tn = sum(1 for r in rows if not r["classifier_ok"] and not r["human_ok"])

    precision   = safe_div(tp, tp + fp)
    recall      = safe_div(tp, tp + fn)
    f1          = safe_div(2 * tp, 2 * tp + fp + fn)
    accuracy    = safe_div(tp + tn, len(rows))
    kappa       = cohens_kappa(tp, fp, fn, tn)

    overall = {
        "n": len(rows),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "cohens_kappa": kappa,
    }

    # Per-family breakdown
    by_family: dict[str, list] = defaultdict(list)
    for r in rows:
        by_family[r["family"]].append(r)

    per_family = {}
    for fam in sorted(by_family):
        recs = by_family[fam]
        ftp = sum(1 for r in recs if r["classifier_ok"] and r["human_ok"])
        ffp = sum(1 for r in recs if r["classifier_ok"] and not r["human_ok"])
        ffn = sum(1 for r in recs if not r["classifier_ok"] and r["human_ok"])
        ftn = sum(1 for r in recs if not r["classifier_ok"] and not r["human_ok"])
        per_family[fam] = {
            "n": len(recs),
            "precision": safe_div(ftp, ftp + ffp),
            "recall": safe_div(ftp, ftp + ffn),
            "f1": safe_div(2 * ftp, 2 * ftp + ffp + ffn),
            "cohens_kappa": cohens_kappa(ftp, ffp, ffn, ftn),
            "classifier_ok_rate": safe_div(ftp + ffp, len(recs)),
            "human_ok_rate": safe_div(ftp + ffn, len(recs)),
        }

    results = {"overall": overall, "per_family": per_family}
    JSON_OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Text report
    lines = []
    W = 60

    lines += [
        "=" * W,
        "CLASSIFIER VALIDATION REPORT",
        f"N annotated: {len(rows)}  |  N skipped: {skipped}",
        "=" * W,
        "",
        "OVERALL (classifier vs human, binary uptake_ok)",
        "-" * W,
        f"  Precision:      {precision}   (when classifier says ok, it's right)",
        f"  Recall:         {recall}   (of true ok cases, classifier catches)",
        f"  F1:             {f1}",
        f"  Accuracy:       {accuracy}",
        f"  Cohen's kappa:  {kappa}",
        "",
        "  Confusion matrix:",
        f"                   Human TRUE   Human FALSE",
        f"  Classifier TRUE    {tp:>5}        {fp:>5}    (TP / FP)",
        f"  Classifier FALSE   {fn:>5}        {tn:>5}    (FN / TN)",
        "",
        "Kappa interpretation: <0.40 poor, 0.40-0.60 moderate,",
        "                       0.60-0.80 substantial, >0.80 near-perfect",
        "",
        "=" * W,
        "PER-FAMILY BREAKDOWN",
        "=" * W,
        f"{'Family':<22}  {'n':>4}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'κ':>6}",
        "-" * W,
    ]
    for fam, s in per_family.items():
        def fmt(v):
            return f"{v:.3f}" if v is not None else "  —  "
        lines.append(
            f"{fam:<22}  {s['n']:>4}  {fmt(s['precision'])}  {fmt(s['recall'])}  "
            f"{fmt(s['f1'])}  {fmt(s['cohens_kappa'])}"
        )

    lines += [
        "",
        "Note: precision = classifier correctly identified true uptake success.",
        "      recall    = classifier correctly identified true uptake failure.",
        "      A high-precision / lower-recall pattern means the classifier",
        "      under-counts failures (conservative) — typical for keyword matchers.",
    ]

    report = "\n".join(lines)
    TXT_OUT.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved JSON   → {JSON_OUT}")
    print(f"Saved report → {TXT_OUT}")


if __name__ == "__main__":
    main()
