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

| Profile ID | Label | Model | Access | Adapter | Intended status |
| --- | --- | --- | --- | --- | --- |
| `granite_guardian_3_2_compare_v2` | Granite Guardian 3.2 3B | `ibm-granite/granite-guardian-3.2-3b-a800m` | `local_open` | `allow_block_text` | Keep as current external anchor under fair contract |
| `granite_guardian_3_3_compare_v2` | Granite Guardian 3.3 8B | `ibm-granite/granite-guardian-3.3-8b` | `local_open` | `allow_block_text` | Stronger same-family Granite comparison |
| `nemotron_safety_guard_8b_compare_v2` | Nemotron Safety Guard 8B | `nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3` | `local_open` | `nemotron_json` | Strong local moderation baseline |
| `wildguard_7b_compare_v2` | WildGuard 7B | `allenai/wildguard` | `local_open` | `wildguard_prompt_harm` | Strong refusal-aware moderation baseline |
| `llama_guard_3_8b_compare_v2` | Llama Guard 3 8B | `meta-llama/Llama-Guard-3-8B` | `local_gated` | `llama_guard_text` | Good second-wave baseline if access is available |
| `shieldgemma_9b_compare_v2` | ShieldGemma 9B | `google/shieldgemma-9b` | `local_gated` | `yes_no_policy` | Good second-wave baseline if access is available |

## Matrix To Run

For the meaningful external-guardrail comparison, use:

- [hateful_guardrail_compare_matrix_assistant_policy_v2.yaml](/root/minor_bench/matrix_configs/hateful_guardrail_compare_matrix_assistant_policy_v2.yaml)

That matrix keeps fixed:

- assistant prompt: `assistant_child_safety_v2`
- judge: local `meta-llama/Llama-3.2-1B-Instruct`
- dataset mix: MinorBench hateful plus safe controls
- metric definition: `assistant_policy_v2`

Only the external guardrail stage changes across rows.
