from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from guardrails import (
    LEGACY_GUARDRAIL_INPUT_CONTRACT,
    SHARED_POLICY_GUARDRAIL_INPUT_CONTRACT,
    build_guardrail_request,
    merge_guardrail_profile,
    parse_guardrail_output,
    resolve_guardrail_profile,
)


def test_merge_guardrail_profile_sets_compare_defaults() -> None:
    merged = merge_guardrail_profile({}, "granite_guardian_3_3_compare_v2")
    assert merged["guardrail_model"] == "ibm-granite/granite-guardian-3.3-8b"
    assert merged["guardrail_provider"] == "local_hf"
    assert merged["guardrail_prompt_name"] == "guardrail_policy_summary_v2"
    assert merged["guardrail_input_contract"] == SHARED_POLICY_GUARDRAIL_INPUT_CONTRACT
    assert merged["guardrail_adapter"] == "allow_block_text"


def test_build_guardrail_request_shared_contract_excludes_assistant_prompt() -> None:
    payload = build_guardrail_request(
        input_contract=SHARED_POLICY_GUARDRAIL_INPUT_CONTRACT,
        adapter_type="allow_block_text",
        policy_text="policy summary",
        system_prompt="assistant system prompt should not leak",
        user_prompt="hello",
    )
    assert "assistant system prompt should not leak" not in payload["user_message"]
    assert "assistant system prompt should not leak" not in payload["local_prompt"]
    assert "policy summary" in payload["user_message"]
    assert "User prompt:\nhello" in payload["user_message"]


def test_build_guardrail_request_legacy_contract_keeps_system_prompt() -> None:
    payload = build_guardrail_request(
        input_contract=LEGACY_GUARDRAIL_INPUT_CONTRACT,
        adapter_type="allow_block_text",
        policy_text="legacy prompt",
        system_prompt="assistant system prompt",
        user_prompt="hello",
    )
    assert "System prompt:\nassistant system prompt" in payload["user_message"]


def test_parse_nemotron_json_output() -> None:
    status, reason, mode = parse_guardrail_output(
        "nemotron_json",
        '```json\n{"User Safety":"unsafe","Violated Categories":["hate"],"Reason":"hateful"}\n```',
    )
    assert status == "block"
    assert "hate" in (reason or "")
    assert mode == "json"


def test_parse_wildguard_output() -> None:
    status, reason, mode = parse_guardrail_output(
        "wildguard_prompt_harm",
        "Harmful request: yes\nResponse refusal: yes",
    )
    assert status == "block"
    assert "response refusal=yes" in (reason or "")
    assert mode == "wildguard"


def test_parse_shieldgemma_yes_no_output() -> None:
    status, reason, mode = parse_guardrail_output("yes_no_policy", "No. Benign question.")
    assert status == "allow"
    assert reason is None
    assert mode == "yes_no"


def test_resolve_guardrail_profile_unknown_raises() -> None:
    try:
        resolve_guardrail_profile("missing")
    except KeyError as exc:
        assert "Known profiles" in str(exc)
    else:
        raise AssertionError("Expected KeyError")
