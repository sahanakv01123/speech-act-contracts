from typing import Any

from speech_act_contract.mitigation.pipeline import (
    classify_response_act,
    classify_user_act,
    detect_void_commitment,
    expected_response_act,
)


def build_annotation_starter_record(raw_record: dict[str, Any], example_id: str) -> dict[str, Any]:
    user_turn = raw_record.get("user_turn", "")
    assistant_turn = raw_record.get("assistant_turn", "")
    trigger_family = raw_record.get("seed_family") or classify_user_act(user_turn)

    return {
        "example_id": example_id,
        "conversation_id": raw_record.get("conversation_id", example_id),
        "turn_index": raw_record.get("turn_index", 0),
        "user_turn": user_turn,
        "assistant_turn": assistant_turn,
        "trigger_family": trigger_family,
        "expected_response_act": "",
        "observed_response_act": "",
        "uptake_failure": None,
        "void_commitment": None,
        "annotator_notes": "",
        "annotation_status": "needs_review",
        "suggested_expected_response_act": expected_response_act(trigger_family),
        "suggested_observed_response_act": classify_response_act(assistant_turn),
        "suggested_void_commitment": detect_void_commitment(assistant_turn),
        "source_type": raw_record.get("source_type", "unknown"),
        "model_name": raw_record.get("model_name", "unknown"),
        "seed_id": raw_record.get("seed_id", ""),
    }
