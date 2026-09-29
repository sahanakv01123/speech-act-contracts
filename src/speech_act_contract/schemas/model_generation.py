from dataclasses import dataclass


@dataclass(frozen=True)
class ModelDraftRecord:
    id: str
    user_turn: str
    speech_act_family: str
    model_name: str
    response_text: str
