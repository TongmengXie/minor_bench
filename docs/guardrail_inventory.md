# Guardrail Inventory

This file separates the current legacy guardrail setup from the new fair comparison setup.

## Current Legacy Setup

The currently checked-in historical Granite row uses:

- Input contract: `legacy_exchange_binary`
- Input fields: full guardrail prompt, full assistant system prompt, user prompt
- Output contract: `ALLOW` or `BLOCK: <brief reason>`
- Control flow: `BLOCK` skips the main model and returns the fixed refusal text
- Status: reproducible legacy baseline, but not ideal for cross-model guardrail comparison

Reason this is legacy:

- It tests whether a guardrail can parse our internal assistant prompt stack.
- It does not isolate the external guardrail stage cleanly enough for model-vs-model guardrail comparison.

## Fair Comparison Contract

New comparison rows use:

- Input contract: `shared_policy_native_adapter`
- Input fields: shared child-safety policy summary plus user prompt only
- Output normalization: adapter-specific parsing to `allow | block | malformed`
- Control flow: unchanged; `block` still returns the fixed refusal text and skips tutor generation

Reason this is the preferred contract:

- Every external guardrail sees the same policy objective.
- The comparison varies the external guardrail stage, not our hidden assistant prompt.
- Matrix rows become interpretable as `assistant-only` versus `assistant + this specific guardrail`.

## Guardrail Profiles

| Profile ID | Label | Model | Access | Adapter | Intended status | Current environment status |
| --- | --- | --- | --- | --- | --- | --- |
| `granite_guardian_3_2_compare_v2` | Granite Guardian 3.2 3B | `ibm-granite/granite-guardian-3.2-3b-a800m` | `local_open` | `allow_block_first_line` | Keep as current external anchor under fair contract | Cached locally and resolvable offline |
| `granite_guardian_3_3_compare_v2` | Granite Guardian 3.3 8B | `ibm-granite/granite-guardian-3.3-8b` | `local_open` | `allow_block_first_line` | Stronger same-family Granite comparison | Cached locally and resolvable offline |
| `nemotron_safety_guard_8b_compare_v2` | Nemotron Safety Guard 8B | `nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3` | `local_open` | `nemotron_json` | Strong local moderation baseline | Not currently runnable here as configured; this environment has a different cached NeMoGuard asset and the profile likely needs adapter-specific loading rather than plain causal LM loading |
| `wildguard_7b_compare_v2` | WildGuard 7B | `allenai/wildguard` | `local_open` | `wildguard_prompt_harm` | Strong refusal-aware moderation baseline | Not cached locally in this environment |
| `llama_guard_3_8b_compare_v2` | Llama Guard 3 8B | `meta-llama/Llama-Guard-3-8B` | `local_gated` | `llama_guard_text` | Good second-wave baseline if access is available | Gated and not cached locally here |
| `shieldgemma_9b_compare_v2` | ShieldGemma 9B | `google/shieldgemma-9b` | `local_gated` | `yes_no_policy` | Good second-wave baseline if access is available | Gated and not cached locally here |

## What To Compare Next

Given the current machine state, the next meaningful local comparison is:

- assistant-only baseline
- Granite Guardian 3.2 3B
- Granite Guardian 3.3 8B

That comparison isolates a same-family strength upgrade while avoiding rows that are not actually runnable here.

## Matrix To Run

For the meaningful external-guardrail comparison, use:

- [hateful_guardrail_compare_matrix_assistant_policy_v2.yaml](/root/minor_bench/matrix_configs/hateful_guardrail_compare_matrix_assistant_policy_v2.yaml)

For the GPU-ready comparison that only uses rows known to be locally available here, use:

- [hateful_guardrail_compare_granite_only.yaml](/root/minor_bench/matrix_configs/hateful_guardrail_compare_granite_only.yaml)

That matrix keeps fixed:

- assistant prompt: `assistant_child_safety_v2`
- judge: local `meta-llama/Llama-3.2-1B-Instruct`
- dataset mix: MinorBench hateful plus safe controls
- policy version: `assistant_policy_v2`
- metric definition: `assistant_policy_v2.1`

Only the external guardrail stage changes across rows.

Canonical guardrail-effect comparisons use controlled replay over fixed assistant generations.
Independent end-to-end rows remain operational validation because generation randomness otherwise
confounds row differences.

## Probe-First Execution

The Granite-only matrix is now probe-gated:

- probe script: `probe_guardrail_profiles.py`
- runtime prompt asset for Granite rows: `guardrail_granite_binary_v1`
- malformed policy for comparison rows: `fallback_to_main`
- invalid row policy: `exclude_main`

This means:

- rows with high malformed rate, runtime errors, or zero allow rate are excluded from headline BER/F1 tables
- invalid rows still appear in `guardrail_operational_metrics.csv` and `guardrail_operational_report.md`
- occasional malformed guardrail outputs fall back to the assistant path instead of being silently converted into blocked outputs

## Judge Throughput Caveat

On 2026-04-29 in this environment:

- assistant generation was fast enough for iteration
- the local `meta-llama/Llama-3.2-1B-Instruct` judge on CPU was the bottleneck
- one local judge batch of 4 outputs took about 14 minutes

So:

- full fair-comparison matrix runs are realistic on GPU
- CPU-only local judging is best treated as resumable overnight work, not interactive iteration
