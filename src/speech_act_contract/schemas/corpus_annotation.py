from dataclasses import dataclass


@dataclass(frozen=True)
class CorpusAnnotationItem:
    example_id: str
    conversation_id: str
    turn_index: int
    user_turn: str
    assistant_turn: str
    trigger_family: str
    expected_response_act: str
    observed_response_act: str
    uptake_failure: bool
    void_commitment: bool
    annotator_notes: str
