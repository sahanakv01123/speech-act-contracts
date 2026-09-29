from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class FullBenchmarkItem:
    id: str
    user_turn: str
    speech_act_family: str
    expected_response_act: str
    acceptable_alternative_response_acts: List[str]
    difficulty: str
    context_notes: str
    gold_void_commitment_risk: bool
    source_type: str
