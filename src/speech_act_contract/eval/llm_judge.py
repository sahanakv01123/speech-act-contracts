import json
from typing import Any


JUDGE_SYSTEM_PROMPT = (
    "You are a careful evaluator for dialogue pragmatics research. "
    "You will be shown two assistant responses to the same user turn. "
    "The BEFORE response has been identified as having a specific speech-act failure. "
    "The AFTER response is a repaired version that attempts to fix that failure. "
    "Your task is to assess whether the repair is an improvement, a regression, or neutral. "
    "Return only valid JSON."
)


def _describe_failure(record: dict[str, Any]) -> str:
    parts = []
    if not record.get("before_uptake_ok", True):
        expected = record.get("expected_response_act", "unknown")
        observed = record.get("before_observed_response_act", "unknown")
        parts.append(
            f"Uptake failure: the user's speech act required a '{expected}' response "
            f"but the model produced '{observed}'."
        )
    if record.get("before_void_commitment", False):
        parts.append(
            "Void commitment: the response made a memory or future-capability claim "
            "the model cannot actually guarantee (e.g. 'I'll remember this for future sessions')."
        )
    return " ".join(parts) if parts else "Unspecified failure."


def build_judge_user_prompt(record: dict[str, Any]) -> str:
    failure_description = _describe_failure(record)
    return f"""
Evaluate the two assistant responses below.

User turn:
{record["user_turn"]}

Speech-act family:
{record["speech_act_family"]}

Expected response act:
{record["expected_response_act"]}

Acceptable alternative response acts:
{", ".join(record.get("acceptable_alternative_response_acts", []))}

Identified failure in the BEFORE response:
{failure_description}

BEFORE response (contains the identified failure):
{record["before_response_text"]}

AFTER response (attempted repair):
{record["after_response_text"]}

Evaluate whether the AFTER response successfully repairs the identified failure
while preserving helpfulness and naturalness. The AFTER need not be more helpful
than the BEFORE — it only needs to fix the specific failure without making things worse.

Return JSON with exactly these keys:
- failure_repaired: boolean — did the AFTER fix the identified failure?
- overall_preference: one of ["before", "after", "tie"]
- pragmatic_better: one of ["before", "after", "tie"]
- truthfulness_better: one of ["before", "after", "tie"]
- helpfulness_preserved: boolean
- naturalness_preserved: boolean
- confidence: integer from 1 to 5
- rationale: short string, at most 2 sentences
""".strip()


def extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in judge response: {text}")
    return stripped[start : end + 1]


def parse_judge_response(text: str) -> dict[str, Any]:
    payload = json.loads(extract_json_object(text))

    allowed_label = {"before", "after", "tie"}
    for key in ("overall_preference", "pragmatic_better", "truthfulness_better"):
        if payload.get(key) not in allowed_label:
            raise ValueError(f"Unexpected value for {key}: {payload.get(key)}")

    for key in ("helpfulness_preserved", "naturalness_preserved", "failure_repaired"):
        if not isinstance(payload.get(key), bool):
            raise ValueError(f"{key} must be a boolean: {payload.get(key)}")

    confidence = payload.get("confidence")
    if not isinstance(confidence, int) or not (1 <= confidence <= 5):
        raise ValueError(f"confidence must be an integer from 1 to 5: {confidence}")

    rationale = payload.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("rationale must be a non-empty string")

    return payload


def summarize_judgments(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    if total == 0:
        return {
            "num_judged": 0,
            "failure_repaired_rate": 0.0,
            "overall_preference_rate": {},
            "pragmatic_better_rate": {},
            "truthfulness_better_rate": {},
            "helpfulness_preserved_rate": 0.0,
            "naturalness_preserved_rate": 0.0,
            "average_confidence": 0.0,
        }

    def distribution(key: str) -> dict[str, float]:
        counts = {"before": 0, "after": 0, "tie": 0}
        for record in records:
            counts[record[key]] += 1
        return {label: round(count / total, 3) for label, count in counts.items()}

    failure_repaired_hits = sum(1 for record in records if record.get("failure_repaired", False))
    helpfulness_hits = sum(1 for record in records if record["helpfulness_preserved"])
    naturalness_hits = sum(1 for record in records if record["naturalness_preserved"])
    average_confidence = round(sum(record["confidence"] for record in records) / total, 3)

    return {
        "num_judged": total,
        "failure_repaired_rate": round(failure_repaired_hits / total, 3),
        "overall_preference_rate": distribution("overall_preference"),
        "pragmatic_better_rate": distribution("pragmatic_better"),
        "truthfulness_better_rate": distribution("truthfulness_better"),
        "helpfulness_preserved_rate": round(helpfulness_hits / total, 3),
        "naturalness_preserved_rate": round(naturalness_hits / total, 3),
        "average_confidence": average_confidence,
    }
