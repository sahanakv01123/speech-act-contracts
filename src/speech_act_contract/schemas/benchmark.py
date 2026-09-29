from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class BenchmarkItem:
    id: str
    speech_act_family: str
    context: List[str]
    expected_response_act: str
    acceptable_response_notes: str
    bad_response_notes: str
