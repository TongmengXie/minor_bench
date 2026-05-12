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
from model_runner import resolve_local_hf_model_reference


def test_merge_guardrail_profile_sets_compare_defaults() -> None:
    merged = merge_guardrail_profile({}, "granite_guardian_3_3_compare_v2")
    assert merged["guardrail_model"] == "ibm-granite/granite-guardian-3.3-8b"
    assert merged["guardrail_provider"] == "local_hf"
    assert merged["guardrail_prompt_name"] == "guardrail_granite_native_v1"
    assert merged["guardrail_input_contract"] == SHARED_POLICY_GUARDRAIL_INPUT_CONTRACT
    assert merged["guardrail_adapter"] == "granite_guardian_yes_no"
    assert merged["guardrail_malformed_policy"] == "fallback_to_main"
    assert merged["guardrail_prompt_normalizer"] == "strip_known_jailbreak_prefix"
    assert merged["guardrail_risk_names"] == [
        "harm",
        "social_bias",
        "profanity",
        "jailbreak",
        "sexual_content",
        "violence",
        "unethical_behavior",
    ]
    assert merged["guardrail_block_risk_names"] == [
        "social_bias",
        "jailbreak",
        "sexual_content",
        "violence",
        "unethical_behavior",
    ]


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


def test_build_guardrail_request_granite_native_returns_structured_metadata() -> None:
    payload = build_guardrail_request(
        input_contract=SHARED_POLICY_GUARDRAIL_INPUT_CONTRACT,
        adapter_type="granite_guardian_yes_no",
        policy_text="policy summary",
        system_prompt="assistant system prompt should not leak",
        user_prompt="hello",
    )
    assert payload["requires_chat_template"] is True
    assert payload["expected_output"] == "yes_no"
    assert payload["chat_messages"] == [{"role": "user", "content": "hello"}]
    assert "assistant system prompt should not leak" not in payload["user_message"]


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


def test_parse_allow_block_first_line_rejects_embedded_tokens() -> None:
    status, reason, mode = parse_guardrail_output(
        "allow_block_first_line",
        "Explanation first\nALLOW",
    )
    assert status == "malformed"
    assert "did not start" in (reason or "")
    assert mode == "text"


def test_parse_allow_block_first_line_accepts_block_reason_on_next_line() -> None:
    status, reason, mode = parse_guardrail_output(
        "allow_block_first_line",
        "BLOCK\nBrief reason here",
    )
    assert status == "block"
    assert reason == "Brief reason here"
    assert mode == "binary_first_line"


def test_parse_granite_guardian_yes_no_output() -> None:
    status, reason, mode = parse_guardrail_output(
        "granite_guardian_yes_no",
        "Yes\n<confidence> High </confidence>",
    )
    assert status == "block"
    assert "Granite guardian" in (reason or "")
    assert mode == "yes_no"

    status, reason, mode = parse_guardrail_output(
        "granite_guardian_yes_no",
        "No\n<confidence> Low </confidence>",
    )
    assert status == "allow"
    assert reason is None
    assert mode == "yes_no"


def test_parse_granite_guardian_yes_no_rejects_scaffold_echo() -> None:
    status, reason, mode = parse_guardrail_output(
        "granite_guardian_yes_no",
        "<end_of_conversation>\nOur safety risk definition is defined below:",
    )
    assert status == "malformed"
    assert "scaffold" in (reason or "")
    assert mode == "native_template_mismatch"


def test_resolve_guardrail_profile_unknown_raises() -> None:
    try:
        resolve_guardrail_profile("missing")
    except KeyError as exc:
        assert "Known profiles" in str(exc)
    else:
        raise AssertionError("Expected KeyError")


def test_resolve_local_hf_model_reference_prefers_existing_path(tmp_path) -> None:
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()

    ref, kwargs = resolve_local_hf_model_reference(str(model_dir))

    assert ref == str(model_dir)
    assert kwargs == {"local_files_only": True}


def test_resolve_local_hf_model_reference_prefers_cached_snapshot(tmp_path, monkeypatch) -> None:
    cache_root = tmp_path / "hub"
    snapshot = (
        cache_root
        / "models--meta-llama--Llama-3.2-1B-Instruct"
        / "snapshots"
        / "snapshot-a"
    )
    snapshot.mkdir(parents=True)
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(cache_root))

    ref, kwargs = resolve_local_hf_model_reference("meta-llama/Llama-3.2-1B-Instruct")

    assert ref == str(snapshot)
    assert kwargs == {"local_files_only": True}


def test_resolve_local_hf_model_reference_offline_without_cache(monkeypatch) -> None:
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", "/tmp/nonexistent_hf_cache_for_test")
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    ref, kwargs = resolve_local_hf_model_reference("meta-llama/Llama-3.2-1B-Instruct")

    assert ref == "meta-llama/Llama-3.2-1B-Instruct"
    assert kwargs == {"local_files_only": True}
