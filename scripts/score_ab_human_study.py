"""C2 scorer: aggregate completed blinded A/B human-study sheets against the key.

Reads the private key (which side was SpeakFix per row) and one or more completed
annotator sheets, converts each A/B judgment into a base-vs-SpeakFix preference,
and reports preference/helpfulness/naturalness rates split by stratum
(repaired_failure vs. over_trigger_control), plus raw inter-rater agreement when
multiple sheets are supplied.

Usage:
    python scripts/score_ab_human_study.py --key results/human_ab_study_key.csv \
        --sheet rater1.csv --sheet rater2.csv
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import argparse
import csv
import json

ROOT = Path(__file__).resolve().parents[1]

CHOICE = {"a": "A", "b": "B", "tie": "Tie"}


def norm_choice(value: str) -> str | None:
    v = (value or "").strip().lower()
    return CHOICE.get(v)


def resolve_preference(choice: str, a_is_speakfix: bool) -> str:
    """Map an A/B/Tie choice to speakfix/base/tie using the hidden key."""
    if choice == "Tie":
        return "tie"
    a_side = "speakfix" if a_is_speakfix else "base"
    b_side = "base" if a_is_speakfix else "speakfix"
    return a_side if choice == "A" else b_side


def load_key(path: Path) -> dict[int, dict]:
    key = {}
    for row in csv.DictReader(path.open(encoding="utf-8-sig")):
        key[int(row["row"])] = {
            "stratum": row["stratum"],
            "model_name": row["model_name"],
            "a_is_speakfix": row["a_is_speakfix"].strip().upper() == "TRUE",
        }
    return key


def score_sheet(path: Path, key: dict[int, dict]) -> dict[int, dict]:
    """Return {row: {overall, helpful, natural}} in speakfix/base/tie terms."""
    out = {}
    for row in csv.DictReader(path.open(encoding="utf-8-sig")):
        rnum = int(row["row"])
        info = key.get(rnum)
        if info is None:
            continue
        judgments = {}
        for field, label in (
            ("overall_better_A_B_Tie", "overall"),
            ("more_helpful_A_B_Tie", "helpful"),
            ("more_natural_A_B_Tie", "natural"),
        ):
            choice = norm_choice(row.get(field, ""))
            if choice is not None:
                judgments[label] = resolve_preference(choice, info["a_is_speakfix"])
        if judgments:
            out[rnum] = judgments
    return out


def rate_table(rows: list[str]) -> dict:
    c = Counter(rows)
    n = len(rows)
    return {
        "n": n,
        "speakfix_pct": round(100 * c["speakfix"] / n, 1) if n else 0.0,
        "base_pct": round(100 * c["base"] / n, 1) if n else 0.0,
        "tie_pct": round(100 * c["tie"] / n, 1) if n else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", type=Path, default=ROOT / "results" / "human_ab_study_key.csv")
    parser.add_argument("--sheet", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "human_ab_study_scores.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    key = load_key(args.key)
    scored = {str(p): score_sheet(p, key) for p in args.sheet}

    # Pool all raters for rate tables, split by stratum and dimension.
    by_stratum_dim: dict[tuple[str, str], list[str]] = defaultdict(list)
    for judgments in scored.values():
        for rnum, dims in judgments.items():
            stratum = key[rnum]["stratum"]
            for dim, pref in dims.items():
                by_stratum_dim[(stratum, dim)].append(pref)

    summary: dict[str, dict] = {"num_raters": len(args.sheet), "by_stratum": {}}
    for (stratum, dim), prefs in sorted(by_stratum_dim.items()):
        summary["by_stratum"].setdefault(stratum, {})[dim] = rate_table(prefs)

    # Raw inter-rater agreement on 'overall' when >=2 raters.
    if len(args.sheet) >= 2:
        sheets = list(scored.values())
        common = set(sheets[0])
        for s in sheets[1:]:
            common &= set(s)
        agree = 0
        total = 0
        for rnum in common:
            vals = [s[rnum].get("overall") for s in sheets if "overall" in s[rnum]]
            if len(vals) == len(sheets):
                total += 1
                if len(set(vals)) == 1:
                    agree += 1
        summary["overall_agreement"] = {
            "compared_rows": total,
            "raw_agreement_pct": round(100 * agree / total, 1) if total else 0.0,
        }

    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Raters: {summary['num_raters']}")
    for stratum, dims in summary["by_stratum"].items():
        print(f"\n[{stratum}]")
        for dim, t in dims.items():
            print(
                f"  {dim:<8} n={t['n']:>3}  SpeakFix {t['speakfix_pct']}%  "
                f"Base {t['base_pct']}%  Tie {t['tie_pct']}%"
            )
    if "overall_agreement" in summary:
        a = summary["overall_agreement"]
        print(f"\nInter-rater raw agreement (overall): {a['raw_agreement_pct']}% on {a['compared_rows']} rows")
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
