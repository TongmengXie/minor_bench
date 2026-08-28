# MinorBench Repo Structure And Usage Map

This document is the entry point for reconstructing the current benchmark state on a new machine.

## Current Canonical State

- Git remote: `origin git@github.com:TongmengXie/minor_bench.git`
- Synced commit checked before this document was added: `663cdca4168872648eb95369c8494c33e011ead2`
- Current evaluation policy: `assistant_policy_v2`
- Current metric definition: `assistant_policy_v2.1`
- Dependency lockfile: `requirements.lock`
- Canonical main model: `mistral-small-3.2-24b-instruct-2506` via Scaleway
- Canonical local judge: `meta-llama/Llama-3.2-1B-Instruct` via `local_hf`
- Canonical guardrail anchor: `ibm-granite/granite-guardian-3.2-3b-a800m` via `local_hf`
- Canonical result summary: `docs/current_granite_vs_baseline_tables.md`

## What Is And Is Not In Git

In Git:

- Source code, tests, prompt packs, safe-control datasets, policy docs, matrix configs.
- `requirements.lock` for pinned Python package reproduction with `uv pip sync`.
- Canonical corrected run outputs under `runs/evals/.../20260610_*`.
- Canonical controlled replay outputs under `runs/replay/mistral_granite_3_2_hateful_policy_v2_1_canonical_20260610`.
- Historical small probe/matrix manifests needed to understand previous guardrail debugging.

Not in Git:

- `.env` and provider credentials.
- `.minor` Python environment.
- Hugging Face model weights and token-gated model access.
- Local cache directories.

## Reconstructing On A New Instance

```bash
git clone git@github.com:TongmengXie/minor_bench.git
cd minor_bench

uv venv .minor
source .minor/bin/activate
uv pip sync requirements.lock
```

Create `.env` manually with provider credentials. At minimum, current canonical API-backed runs require the Scaleway credentials used by `model_runner.py`; OpenAI is not required for the current local-HF judge path.

Download local judge and guardrail models:

```bash
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct
huggingface-cli download ibm-granite/granite-guardian-3.2-3b-a800m
```

For the next guardrail-selection round, additionally download whichever candidates are selected:

```bash
huggingface-cli download Qwen/Qwen3Guard-Gen-4B
huggingface-cli download allenai/wildguard
huggingface-cli download nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3
```

Some model repos may require Hugging Face terms acceptance before download.

## Script Roles

| Script | Main purpose | Primary inputs | Primary outputs |
| --- | --- | --- | --- |
| `run_eval.py` | Run one assistant or assistant+guardrail eval and judge assistant-generated responses. | Prompt pack, optional safe dataset, model/provider config, guardrail profile. | `runs/evals/<model>/<prompt>/<timestamp>/{results.jsonl,summary.json,summary.md,variant_metrics.csv,meta.json,data_manifest.json}` |
| `probe_guardrail_profiles.py` | Validate guardrail rows before expensive runs. | Matrix config with `guardrail_probe` settings. | `guardrail_probe_manifest.json`, `guardrail_probe_report.md`, `guardrail_probe_samples.json` |
| `run_guardrail_replay.py` | Paired counterfactual guardrail replay over fixed baseline generations. | Baseline run dir, guardrail id, optional recorded decision run. | `runs/replay/<name>/{results.jsonl,summary.*,variant_metrics.csv,guardrail_effect_metrics.csv,guardrail_transition_report.*}` |
| `rescore_policy_outcomes.py` | Recompute `assistant_policy_v2.1` deterministic block outcomes for historical runs. | Existing run dir. | New derived run dir with regenerated summaries. |
| `aggregate_matrix.py` | Aggregate row/mutation summaries into matrix tables. | Matrix manifest. | `matrix_metrics.csv`, pivot CSVs, `matrix_report.md` |
| `run_matrix_eval.py` | Run end-to-end row-by-mutation matrix; useful for operational validation. | Matrix YAML. | `runs/matrix/<name>/manifest.json` plus aggregate outputs. |
| `analyze_guardrail_failures.py` | Inspect operational/integration failure modes in guardrail rows. | Matrix/run artifacts. | Failure analysis markdown and sampled JSON. |
| `evaluate_judge_quality.py` | Sidecar judge-quality checks against existing human audit and heuristics. | Run dir, optional safe subset. | `judge_quality.json`, `judge_quality.md`, `judge_disagreements.csv` |
| `label_results_cli.py` | Manual response-safety annotation for sampled outputs. | Existing `results.jsonl`. | Incremental audit CSV. |
| `build_policy_assets.py` | Build policy v1 calibration assets from already adjudicated rows. | Human audit CSV, `policy_v1_selection.json`. | `docs/safe_policy_v1*`, `datasets/judge_policy_safe_subset_v1.jsonl` |

## Core Data And Policy Files

| File | Role | Consumed by |
| --- | --- | --- |
| `assistant_policy.py` | Canonical policy constants and deterministic outcome resolver. | Evaluator, report generator, replay/rescore scripts. |
| `system_prompts/system_prompts.yaml` | Assistant, guardrail, and judge prompts. | `run_eval.py`, `model_runner.py`, `safety_judge.py`. |
| `guardrails.py` | Guardrail profile registry and output parsers. | Probe, matrix, eval, replay scripts. |
| `prompt_packs/minorbench_hateful.json` | Current harmful prompt slice. | Eval, probe, matrix scripts. |
| `datasets/true_negative_safe_sensitive_variants.jsonl` | Current matched safe-control prompts. | Eval, probe, matrix scripts. |
| `matrix_configs/*.yaml` | Reproducible matrix settings. | Probe and matrix scripts. |
| `docs/assistant_policy_v2.md` | Human-readable policy constitution. | Documentation and policy consistency checks. |
| `docs/guardrail_inventory.md` | Guardrail profile notes and validity context. | Human inspection. |
| `docs/current_granite_vs_baseline_tables.md` | Audience-facing current result report. | Human inspection. |

## Artifact Lineage

| Artifact | Produced by | Depends on | Notes |
| --- | --- | --- | --- |
| `runs/evals/.../20260610_baseline_policy_v2_1` | `rescore_policy_outcomes.py` over canonical baseline. | Historical Mistral baseline run. | Assistant-only baseline under `assistant_policy_v2.1`. |
| `runs/evals/.../20260610_derived_policy_v2_1` | `rescore_policy_outcomes.py` over independent Granite run. | Historical Mistral+Granite run. | End-to-end validation, not primary causal estimate. |
| `runs/replay/...canonical_20260610` | `run_guardrail_replay.py`. | `20260610_baseline_policy_v2_1` plus validated Granite decisions. | Primary paired counterfactual estimate. |
| `docs/current_granite_vs_baseline_tables.md` | Manual/report collation from replay outputs. | Canonical baseline, replay, independent validation. | Best single document for current claims. |

## Raw Output Pointers

| Setup | Raw output file | What it contains |
| --- | --- | --- |
| Assistant-only baseline | `runs/evals/mistral-small-32-24b-instruct-2506/assistant_child_safety_v2/20260610_baseline_policy_v2_1/results.jsonl` | Prompt rows, assistant generations, effective labels, judge diagnostics. |
| Granite controlled replay | `runs/replay/mistral_granite_3_2_hateful_policy_v2_1_canonical_20260610/results.jsonl` | Baseline generations plus replayed final responses and guardrail decisions. |
| Independent Granite validation | `runs/evals/mistral-small-32-24b-instruct-2506/assistant_child_safety_v2/20260610_derived_policy_v2_1/results.jsonl` | End-to-end Granite run, deterministically rescored under `assistant_policy_v2.1`. |

## Current Main Claims

- Controlled replay is the primary estimate of guardrail effect because it fixes assistant generations and varies only guardrail routing.
- Granite Guardian 3.2 improves harmful-prompt handling and reduces harmful unsafe answers on the hateful slice.
- The improvement has collateral safe-prompt cost, concentrated in camouflage safe controls.
- Older `assistant_policy_v2` reports that judge-scored guardrail blocks are legacy and should not be mixed with `assistant_policy_v2.1`.

## Next Guardrail Selection Workflow

1. Confirm baseline exists and is `assistant_policy_v2.1`.
2. Add or verify guardrail profile/parser in `guardrails.py`.
3. Run probe-first validation.
4. Run controlled replay only for operationally valid guardrails.
5. Rank guardrails by residual harmful unsafe-answer risk versus safe-prompt incorrect handling.
6. Scale the selected 1-2 guardrails to broader MinorBench categories only after matched safe controls exist.

## Hardware Notes

- Current checked-in outputs are small and do not require large disk.
- Local model execution does require substantial disk for model weights.
- Recommended new instance for the next guardrail round: at least 48GB VRAM and 150GB disk; 80GB VRAM and 250GB disk is safer if running multiple 7B-8B guardrails without repeated cache cleanup.
