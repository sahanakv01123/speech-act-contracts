from dataclasses import dataclass


@dataclass(frozen=True)
class PilotEvalItem:
    id: str
    user_turn: str
    speech_act_family: str
    expected_response_act: str
    bad_draft_response: str
    notes_on_why_it_fails: str
    gold_void_commitment: bool
    difficulty: str
    gold_repair_should_improve: bool
