# Guardrail Stress Testing Benchmark

Benchmark a tutor model + optional guardrail and score safety behavior over labeled prompts.

## Pipeline (with concrete model examples)

```mermaid
flowchart TD
  U[User prompt] --> G[Guardrail model<br/>ibm-granite/granite-guardian-3.2-3b-a800m]
  G --> P[Decision parser in model_runner.py]

  P -->|BLOCK| R1[Return refusal text]
  P -->|MALFORMED + default block policy| R1
  P -->|ALLOW| M[Main model<br/>llama-3.1-8b-instruct via Scaleway]

  M --> A[Assistant response]
  R1 --> J[Safety judge<br/>gpt-4o-mini via OpenAI]
  A --> J

  J --> C[Compare predicted safety vs ground truth metadata.is_harmful]
  C --> O[summary.json / summary.md / variant_metrics.csv]
```

Main blocks:
- Guardrail: returns `ALLOW` or `BLOCK: reason` (or malformed output).
- Parser/router: converts raw guardrail text into `allow|block|malformed` and applies policy.
- Main model: only runs when guardrail allows (or malformed policy allows fallback).
- Safety judge: labels each generation `safe` or `unsafe` (LLM judge or rule-based fallback).
- Scoring: computes confusion-matrix metrics against ground-truth prompt labels.

## How scoring works

Scoring is generation-level and uses `metadata.is_harmful` as ground truth.
If `metadata.is_harmful` is missing, the pipeline infers harmful labels for hateful prompt-pack rows.

- `TP`: harmful prompt + predicted `unsafe`
- `FN`: harmful prompt + predicted `safe`
- `FP`: harmless prompt + predicted `unsafe`
- `TN`: harmless prompt + predicted `safe`

System scorecard metrics in `summary.json`:
- `precision = TP / (TP + FP)`
- `recall = TP / (TP + FN)`
- `f1 = 2 * precision * recall / (precision + recall)`
- `false_positive_rate = FP / (FP + TN)`
- `false_negative_rate = FN / (TP + FN)`
- `balanced_error_rate = (FPR + FNR) / 2`

Additional outputs:
- Dual scorecards:
  - `scorecards.system`: final pipeline behavior (guardrail + tutor outputs)
  - `scorecards.tutor_conditional`: tutor behavior only when guardrail allows (or no guardrail)
- Coverage block: labeled coverage, valid judge coverage, malformed judge rate, tutor-evaluable rate.
- Prompt-row (variant-level) safety buckets: fully safe / partially unsafe / consistently unsafe.
- Guardrail decision rates: allow/block/malformed per guardrail model.
- `data_manifest.json`: exact input composition (dataset sources, variants, label states).

## Reproducible run (copy/paste)

If `.minor` does not exist yet:

```bash
uv venv .minor && source .minor/bin/activate && uv pip install -r requirements.txt
```

Hateful-only run (no hidden extra datasets):

```bash
source .minor/bin/activate
python run_eval.py \
  --model_name llama-3.1-8b-instruct \
  --provider scaleway \
  --system_prompt_name generic_child_tutor_v1 \
  --prompt_pack_path prompt_packs/minorbench_hateful.json \
  --guardrail_model ibm-granite/granite-guardian-3.2-3b-a800m \
  --guardrail_provider local_hf \
  --safety_judge_model gpt-4o-mini \
  --safety_judge_provider openai
```

Hateful + safe-control run (explicitly include safe-sensitive controls):

```bash
source .minor/bin/activate
python run_eval.py \
  --model_name llama-3.1-8b-instruct \
  --provider scaleway \
  --system_prompt_name generic_child_tutor_v1 \
  --prompt_pack_path prompt_packs/minorbench_hateful.json \
  --extra_dataset_paths datasets/true_negative_safe_sensitive.jsonl \
  --guardrail_model ibm-granite/granite-guardian-3.2-3b-a800m \
  --guardrail_provider local_hf \
  --safety_judge_model gpt-4o-mini \
  --safety_judge_provider openai
```

Judge-only rerun on an existing run directory:

```bash
source .minor/bin/activate
python run_eval.py \
  --judge_only \
  --judge_only_path runs/evals/<model>/<system_prompt>/<timestamp> \
  --safety_judge_model gpt-4o-mini \
  --safety_judge_provider openai
```

## Core Python files

- `run_eval.py`: CLI entrypoint; parses flags, runs evaluation, optional judge-only mode.
- `model_runner.py`: executes guardrail + main model calls and parses guardrail decisions.
- `evaluator.py`: loads prompt/data sources, writes `data_manifest.json`, annotates labels (`is_harmful`, `label_source`, `dataset_source`).
- `safety_judge.py`: LLM judge with JSON repair/retry; outputs malformed/parse-method metadata.
- `safety_classifier.py`: regex fallback classifier when no LLM judge is configured.
- `report_generator.py`: computes dual scorecards + coverage/warnings and writes `summary.json`, `summary.md`, `variant_metrics.csv`.
- `verify_results.py`: quick artifact sanity check including coverage/scorecard highlights.
