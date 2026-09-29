import json
from pathlib import Path
from typing import Any, List

from speech_act_contract.schemas.benchmark import BenchmarkItem
from speech_act_contract.schemas.full_benchmark import FullBenchmarkItem
from speech_act_contract.schemas.pilot_eval import PilotEvalItem


def load_benchmark(path: str | Path) -> List[BenchmarkItem]:
    items: List[BenchmarkItem] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            items.append(BenchmarkItem(**record))
    return items


def load_full_benchmark(path: str | Path) -> List[FullBenchmarkItem]:
    items: List[FullBenchmarkItem] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            items.append(FullBenchmarkItem(**record))
    return items


def load_pilot_eval_set(path: str | Path) -> List[PilotEvalItem]:
    items: List[PilotEvalItem] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            items.append(PilotEvalItem(**record))
    return items


def load_jsonl_records(path: str | Path) -> List[dict[str, Any]]:
    records: List[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
    return records
