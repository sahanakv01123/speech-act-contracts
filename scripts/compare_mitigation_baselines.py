"""B3: unified comparison of mitigation conditions on the full benchmark.

For one model, scores four conditions with the SAME automatic evaluator
(response-act matcher + void-commitment detector) so they are directly
comparable:

  1. base            -- raw model drafts (the "before" condition)
  2. speakfix        -- SpeakFix pipeline applied to the base drafts
  3. system_prompt   -- B1 system-prompt baseline drafts
  4. llm_rewrite     -- B2 lightweight LLM-rewrite baseline drafts

Reports, per condition: uptake accuracy and void-commitment rate with Wilson
95% CIs and per-family uptake. Against the base condition it also reports
paired improved/worsened counts and an exact sign-test p-value for both uptake
and void commitment.

Usage:
    python scripts/compare_mitigation_baselines.py --model gpt52
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from speech_act_contract.data.loader import load_full_benchmark, load_jsonl_records
from speech_act_contract.mitigation.pipeline import (
    classify_response_act,
    detect_void_commitment,
    run_pipeline,
)
from speech_act_contract.stats.significance import (
    paired_binary_improvement_pvalue,
    wilson_interval,
)


DEFAULT_PATHS = {
    "base": "{model}_full_benchmark_drafts_v3.jsonl",
    "system_prompt": "{model}_full_benchmark_baseline_systemprompt_v1.jsonl",
    "llm_rewrite": "{model}_full_benchmark_baseline_llmrewrite_v1.jsonl",
}

CONDITION_ORDER = ["base", "speakfix", "system_prompt", "llm_rewrite"]


def score_texts(gold_by_id, texts_by_id: dict[str, str]) -> dict[str, dict]:
    """Return per-item {id: {family, uptake_ok, void_commitment}}."""
    scored = {}
    for item_id, text in texts_by_id.items():
        item = gold_by_id[item_id]
        observed = classify_response_act(text)
        scored[item_id] = {
            "speech_act_family": item.speech_act_family,
            "uptake_ok": observed in item.acceptable_alternative_response_acts,
            "void_commitment": detect_void_commitment(text),
        }
    return scored


def summarize_condition(scored: dict[str, dict]) -> dict:
    n = len(scored)
    uptake_hits = sum(1 for r in scored.values() if r["uptake_ok"])
    void_hits = sum(1 for r in scored.values() if r["void_commitment"])

    family_counts: Counter = Counter()
    family_uptake: Counter = Counter()
    for r in scored.values():
        family_counts[r["speech_act_family"]] += 1
        if r["uptake_ok"]:
            family_uptake[r["speech_act_family"]] += 1

    return {
        "num_items": n,
        "uptake_accuracy": round(uptake_hits / n, 3),
        "uptake_count": uptake_hits,
        "uptake_wilson_95": wilson_interval(uptake_hits, n),
        "void_commitment_rate": round(void_hits / n, 3),
        "void_count": void_hits,
        "void_wilson_95": wilson_interval(void_hits, n),
        "per_family_uptake_accuracy": {
            fam: round(family_uptake[fam] / count, 3)
            for fam, count in sorted(family_counts.items())
        },
    }


def paired_vs_base(base: dict[str, dict], cond: dict[str, dict]) -> dict:
    ids = [i for i in base if i in cond]
    uptake_improved = sum(
        1 for i in ids if (not base[i]["uptake_ok"]) and cond[i]["uptake_ok"]
    )
    uptake_worsened = sum(
        1 for i in ids if base[i]["uptake_ok"] and (not cond[i]["uptake_ok"])
    )
    void_fixed = sum(
        1 for i in ids if base[i]["void_commitment"] and (not cond[i]["void_commitment"])
    )
    void_added = sum(
        1 for i in ids if (not base[i]["void_commitment"]) and cond[i]["void_commitment"]
    )
    return {
        "uptake_improved": uptake_improved,
        "uptake_worsened": uptake_worsened,
        "uptake_paired_pvalue": paired_binary_improvement_pvalue(uptake_improved, uptake_worsened),
        "void_fixed": void_fixed,
        "void_added": void_added,
        "void_paired_pvalue": paired_binary_improvement_pvalue(void_fixed, void_added),
    }


def load_texts(path: Path) -> dict[str, str]:
    return {r["id"]: r["response_text"] for r in load_jsonl_records(path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("azure", "claude", "gpt52"), required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def fmt_ci(interval: tuple[float, float]) -> str:
    lo, hi = interval
    return f"[{lo * 100:.1f}, {hi * 100:.1f}]"


def main() -> None:
    args = parse_args()
    gold_items = load_full_benchmark(ROOT / "data" / "annotations" / "full_uptake_benchmark_v3.jsonl")
    gold_by_id = {item.id: item for item in gold_items}

    base_path = ROOT / "results" / DEFAULT_PATHS["base"].format(model=args.model)
    base_records = load_jsonl_records(base_path)
    model_name = base_records[0]["model_name"] if base_records else args.model
    base_texts = {r["id"]: r["response_text"] for r in base_records}

    # SpeakFix "after" is derived from the base drafts via the pipeline.
    speakfix_texts = {
        r["id"]: run_pipeline(gold_by_id[r["id"]].user_turn, r["response_text"]).repaired_response
        for r in base_records
    }

    texts_by_condition = {
        "base": base_texts,
        "speakfix": speakfix_texts,
        "system_prompt": load_texts(ROOT / "results" / DEFAULT_PATHS["system_prompt"].format(model=args.model)),
        "llm_rewrite": load_texts(ROOT / "results" / DEFAULT_PATHS["llm_rewrite"].format(model=args.model)),
    }

    scored = {cond: score_texts(gold_by_id, texts) for cond, texts in texts_by_condition.items()}
    summaries = {cond: summarize_condition(scored[cond]) for cond in CONDITION_ORDER}
    comparisons = {
        cond: paired_vs_base(scored["base"], scored[cond])
        for cond in CONDITION_ORDER
        if cond != "base"
    }

    result = {
        "model": args.model,
        "model_name": model_name,
        "conditions": summaries,
        "paired_vs_base": comparisons,
    }

    output_path = args.output or (ROOT / "results" / f"{args.model}_mitigation_baseline_comparison.json")
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [f"Mitigation baseline comparison -- {args.model} ({model_name})", ""]
    header = f"{'condition':<15}{'uptake':>10}{'  95% CI':>16}{'void':>9}{'  95% CI':>16}"
    lines.append(header)
    for cond in CONDITION_ORDER:
        s = summaries[cond]
        lines.append(
            f"{cond:<15}"
            f"{fmt_pct(s['uptake_accuracy']):>10}"
            f"{fmt_ci(s['uptake_wilson_95']):>16}"
            f"{fmt_pct(s['void_commitment_rate']):>9}"
            f"{fmt_ci(s['void_wilson_95']):>16}"
        )
    lines.append("")
    lines.append("Paired vs. base (exact sign test):")
    for cond in CONDITION_ORDER:
        if cond == "base":
            continue
        c = comparisons[cond]
        lines.append(
            f"  {cond:<14} uptake +{c['uptake_improved']}/-{c['uptake_worsened']} "
            f"(p={c['uptake_paired_pvalue']:.3g}); "
            f"void fixed {c['void_fixed']}/added {c['void_added']} "
            f"(p={c['void_paired_pvalue']:.3g})"
        )

    txt_path = output_path.with_suffix(".txt")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved {output_path}")
    print(f"Saved {txt_path}")


if __name__ == "__main__":
    main()
