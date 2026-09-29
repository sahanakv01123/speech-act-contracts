# Speech-Act Contracts & SpeakFix

Reproducible materials for **“Breaking the Speech-Act Contract: Uptake Failures
and Void Commitments in Large Language Models”** (Findings of AACL-IJCNLP 2026).

This repository contains the benchmark, the rule-based **SpeakFix** mitigation
pipeline, and all generation, evaluation, and human-study scripts needed to
reproduce the results in the paper.

## What is here

```
src/speech_act_contract/     # SpeakFix pipeline, LLM clients, evaluator, schemas
scripts/                     # generation, evaluation, baselines, figures, human study
data/annotations/            # the 300-item benchmark (v3) + annotation guides + schema
data/prompts/                # exact prompts used for generation, baselines, and the judge
human_eval/                  # blinded A/B rating sheet, answer key
.env.example                 # template for API credentials (copy to .env)
pyproject.toml
```

The benchmark is [`data/annotations/full_uptake_benchmark_v3.jsonl`](data/annotations/full_uptake_benchmark_v3.jsonl):
300 items across six speech-act families (apology, thanks, offer, correction,
frustration, preference), three difficulty tiers, and synthetic vs.
adapted-from-real-conversation sources. Field definitions are in
[`data/annotations/full_benchmark_schema.md`](data/annotations/full_benchmark_schema.md);
annotation guidelines are in `annotation_guide.md`, `void_commitment_guide.md`,
and `human_review_rubric.md`.

## Setup

Requires Python ≥ 3.10.

```bash
python -m venv .venv && source .venv/bin/activate
pip install openai anthropic matplotlib numpy
pip install -e .
```

To regenerate model outputs you need API access. Copy `.env.example` to `.env`
(and `.env.claude` / `.env.gpt52` for the other deployments) and fill in your
own credentials. **Never commit real keys.**

## Reproducing the results

All scripts write to a `results/` directory (created on first run).

```bash
# 1. Base model drafts (one per deployed model)
python scripts/generate_azure_full_benchmark_drafts_v3.py     # GPT-4o
python scripts/generate_claude_full_benchmark_drafts_v3.py    # Claude Sonnet 4.6
python scripts/generate_gpt52_full_benchmark_drafts_v3.py     # GPT-5.2

# 2. SpeakFix mitigation + automatic evaluation (v4 = the final evaluator
#    used for all numbers reported in the paper)
python scripts/evaluate_azure_full_benchmark_mitigation_v4.py    # GPT-4o
python scripts/evaluate_claude_full_benchmark_mitigation_v4.py   # Claude Sonnet 4.6
python scripts/evaluate_gpt52_full_benchmark_mitigation_v4.py    # GPT-5.2

# 3. Baselines (system-prompt and LLM-rewrite), per model
python scripts/generate_baseline_systemprompt_drafts.py --model azure|claude|gpt52
python scripts/generate_baseline_llm_rewrite_drafts.py  --model azure|claude|gpt52
python scripts/compare_mitigation_baselines.py          --model azure|claude|gpt52

# 4. Over-triggering check on already-adequate responses
python scripts/check_speakfix_overtriggering.py --model azure|claude|gpt52

# 5. Significance (Wilson CIs, exact paired sign tests; reads the v4 evals)
python scripts/analyze_full_benchmark_significance.py

# 6. Classifier validation + judge breakdown (inputs for the figures)
#    score_classifier_validation.py scores the human-annotated 90-item sheet
#    (results/classifier_validation_sheet.csv); analyze_judge_results.py reads the
#    LLM-judge failure records (results/*_failure_judge_records.jsonl) written by
#    judge_mitigation_failures.py.
python scripts/score_classifier_validation.py
python scripts/analyze_judge_results.py

# 7. Figures (reads the v4 evals, classifier scores, and judge analysis)
python scripts/generate_figures.py
```

## SpeakFix

The full rule set — user-act classifier cues, response-act matcher patterns,
uptake-prefix and over-trigger lexicons, and the void-commitment lexicon and
bounding map — is implemented in
[`src/speech_act_contract/mitigation/pipeline.py`](src/speech_act_contract/mitigation/pipeline.py)
and documented in the paper appendix. SpeakFix repairs a draft with a
**prepend-not-replace** strategy: it adds the missing uptake move and bounds any
unsupported memory claim, leaving already-adequate responses untouched.

## Human evaluation

The blinded A/B study materials are in [`human_eval/`](human_eval/):

- `human_ab_study_sheet.csv` — the 60 blinded response pairs shown to raters.
- `human_ab_study_key.csv` — the private key mapping each row to base vs. SpeakFix.

Regenerate or re-score with:

```bash
python scripts/export_ab_human_study.py                 # build a fresh blinded sheet + key
python scripts/score_ab_human_study.py --sheet r1.csv --sheet r2.csv ...
```

## Citation

```bibtex
@inproceedings{varadaraju2026speechact,
  title     = {Breaking the Speech-Act Contract: Uptake Failures and Void
               Commitments in Large Language Models},
  author    = {Varadaraju, Sahana and Vijayakumar, Bharathwaj and
               Baskaran, Lathish Balaji},
  booktitle = {Findings of the Association for Computational Linguistics:
               AACL-IJCNLP 2026},
  year      = {2026}
}
```

## License

Code is released under the MIT License (see [LICENSE](LICENSE)). The benchmark
and annotation materials are released for research use; please cite the paper.
