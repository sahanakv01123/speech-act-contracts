"""Audit coverage statistics for any full benchmark JSONL file.

Usage:
    python scripts/audit_full_benchmark.py
    python scripts/audit_full_benchmark.py data/annotations/full_uptake_benchmark_v3.jsonl
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from speech_act_contract.data.loader import load_full_benchmark


DEFAULT_PATH = ROOT / "data" / "annotations" / "full_uptake_benchmark_v3.jsonl"


def main() -> None:
    benchmark_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    items = load_full_benchmark(benchmark_path)

    family = Counter()
    difficulty = Counter()
    source = Counter()
    family_difficulty = defaultdict(Counter)
    family_source = defaultdict(Counter)

    for item in items:
        family[item.speech_act_family] += 1
        difficulty[item.difficulty] += 1
        source[item.source_type] += 1
        family_difficulty[item.speech_act_family][item.difficulty] += 1
        family_source[item.speech_act_family][item.source_type] += 1

    print(f"Benchmark audit: {benchmark_path}")
    print(f"Total items: {len(items)}")
    print()
    print("By family")
    for key, count in sorted(family.items()):
        print(f"  {key:16s} {count:4d}")
    print()
    print("By difficulty")
    for key, count in sorted(difficulty.items()):
        print(f"  {key:16s} {count:4d}")
    print()
    print("By source_type")
    for key, count in sorted(source.items()):
        print(f"  {key:28s} {count:4d}")
    print()
    print("Family x difficulty")
    for fam in sorted(family_difficulty):
        cells = "  ".join(
            f"{diff}={family_difficulty[fam][diff]}"
            for diff in sorted(family_difficulty[fam])
        )
        print(f"  {fam:16s} {cells}")
    print()
    print("Family x source_type")
    for fam in sorted(family_source):
        cells = "  ".join(
            f"{src}={family_source[fam][src]}"
            for src in sorted(family_source[fam])
        )
        print(f"  {fam:16s} {cells}")


if __name__ == "__main__":
    main()
