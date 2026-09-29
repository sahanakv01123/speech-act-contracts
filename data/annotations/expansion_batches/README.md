# Expansion batches (future/extra data — NOT part of the paper's benchmark)

The CSV files in this directory are **candidate items for a planned future
expansion** of the benchmark. They are **not** part of the 300-item curated core
benchmark used for the experiments reported in the paper.

The evaluated dataset for all reported results is the 300-item core benchmark:

- [`../full_uptake_benchmark_v3.jsonl`](../full_uptake_benchmark_v3.jsonl)

These expansion batches are unannotated draft candidates only. They have **not**
been through the full annotation, adjudication, or quality-control process
applied to the core benchmark, and they were **not** used to produce any number,
table, or figure in the paper. They are included for transparency and to support
later expansion work.

## Contents

- `*_expansion_batch.csv` — draft candidate items per speech-act family
  (apology, thanks, offer, correction, frustration, preference).
- `expansion_batch_manifest.json` — planning manifest (repository-relative
  paths, current per-difficulty counts, and target counts for the expansion).

Do not use these files to reproduce the paper's results.
