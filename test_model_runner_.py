import os
import sys
from pathlib import Path

import pytest
from dotenv import find_dotenv, load_dotenv

# Ensure local imports work when running this file directly.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_runner import ModelRunner  # noqa: E402


# Load env vars from .env if present
load_dotenv(find_dotenv())

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SCW_SECRET_KEY = os.getenv("SCW_SECRET_KEY")


def test_parse_guardrail_decision_variants():
    assert ModelRunner._parse_guardrail_decision("ALLOW") == ("allow", None, "binary")
    assert ModelRunner._parse_guardrail_decision("block") == (
        "block",
        "Blocked by guardrail",
        "binary",
    )
    assert ModelRunner._parse_guardrail_decision("BLOCK: reason here") == (
        "block",
        "reason here",
        "binary",
    )
    assert ModelRunner._parse_guardrail_decision("something else") == (
        "malformed",
        "Guardrail output was not ALLOW or BLOCK",
        "text",
    )


def test_run_blocks_on_guardrail_block(monkeypatch):
    runner = ModelRunner("gpt-4o-mini", provider="openai", config={"guardrail_model": "guardrail"})

    monkeypatch.setattr(
        runner,
        "_maybe_apply_guardrail",
        lambda *_args, **_kwargs: {
            "completion": "",
            "finish_reason": "guardrail_checked",
            "guardrail_decision": "BLOCK: policy violation",
        },
    )

    result = runner.run("system", "user")
    assert result["finish_reason"] == "guardrail_blocked"
    assert result["guardrail_reason"] == "policy violation"
    assert result["guardrail_decision"] == "BLOCK: policy violation"


def test_run_allows_and_calls_main_model(monkeypatch):
    runner = ModelRunner("gpt-4o-mini", provider="openai", config={"guardrail_model": "guardrail"})

    monkeypatch.setattr(
        runner,
        "_maybe_apply_guardrail",
        lambda *_args, **_kwargs: {
            "completion": "",
            "finish_reason": "guardrail_checked",
            "guardrail_decision": "ALLOW",
        },
    )

    expected = {"completion": "ok", "finish_reason": "stop"}
    monkeypatch.setattr(runner, "_run_main_model", lambda *_args, **_kwargs: dict(expected))

    result = runner.run("system", "user")
    assert result["completion"] == expected["completion"]
    assert result["finish_reason"] == expected["finish_reason"]


def test_run_malformed_guardrail_defaults_to_block(monkeypatch):
    runner = ModelRunner("gpt-4o-mini", provider="openai", config={"guardrail_model": "guardrail"})

    monkeypatch.setattr(
        runner,
        "_maybe_apply_guardrail",
        lambda *_args, **_kwargs: {
            "completion": "",
            "finish_reason": "guardrail_checked",
            "guardrail_decision": "maybe",
        },
    )

    result = runner.run("system", "user")
    assert result["finish_reason"] == "guardrail_blocked"
    assert result.get("guardrail_malformed") is True


def test_run_guardrail_error_defaults_to_block(monkeypatch):
    runner = ModelRunner("gpt-4o-mini", provider="openai", config={"guardrail_model": "guardrail"})

    monkeypatch.setattr(
        runner,
        "_maybe_apply_guardrail",
        lambda *_args, **_kwargs: {
            "completion": "",
            "finish_reason": "guardrail_malformed_error",  # <- important
            "error": "network error",
            "guardrail_decision": "",
        },
    )

    result = runner.run("system", "user")
    assert result["finish_reason"] == "guardrail_blocked"
    assert result.get("guardrail_malformed") is True
    assert result.get("guardrail_reason") == "network error"



def test_run_guardrail_error_fallback_to_main(monkeypatch):
    runner = ModelRunner(
        "gpt-4o-mini",
        provider="openai",
        config={"guardrail_model": "guardrail", "guardrail_malformed_policy": "fallback_to_main"},
    )

    monkeypatch.setattr(
        runner,
        "_maybe_apply_guardrail",
        lambda *_args, **_kwargs: {
            "completion": "",
            "finish_reason": "guardrail_error",
            "error": "network error",
            "guardrail_decision": "",
        },
    )

    monkeypatch.setattr(
        runner,
        "_run_main_model",
        lambda *_args, **_kwargs: {"completion": "ok", "finish_reason": "stop"},
    )

    result = runner.run("system", "user")
    assert result["completion"] == "ok"
    assert result.get("guardrail_malformed") is True


def test_run_malformed_guardrail_allows_when_configured(monkeypatch):
    runner = ModelRunner(
        "gpt-4o-mini",
        provider="openai",
        config={"guardrail_model": "guardrail", "guardrail_malformed_policy": "allow"},
    )

    monkeypatch.setattr(
        runner,
        "_maybe_apply_guardrail",
        lambda *_args, **_kwargs: {
            "completion": "",
            "finish_reason": "guardrail_checked",
            "guardrail_decision": "maybe",  # malformed
        },
    )

    monkeypatch.setattr(
        runner,
        "_run_main_model",
        lambda *_args, **_kwargs: {"completion": "ok", "finish_reason": "stop"},
    )

    result = runner.run("system", "user")
    assert result["completion"] == "ok"
    assert result["finish_reason"] == "stop"


def test_run_malformed_guardrail_fallback_to_main(monkeypatch):
    runner = ModelRunner(
        "gpt-4o-mini",
        provider="openai",
        config={"guardrail_model": "guardrail", "guardrail_malformed_policy": "fallback_to_main"},
    )

    monkeypatch.setattr(
        runner,
        "_maybe_apply_guardrail",
        lambda *_args, **_kwargs: {
            "completion": "",
            "finish_reason": "guardrail_checked",
            "guardrail_decision": "maybe",
        },
    )

    monkeypatch.setattr(
        runner,
        "_run_main_model",
        lambda *_args, **_kwargs: {"completion": "ok", "finish_reason": "stop"},
    )

    result = runner.run("system", "user")
    assert result["completion"] == "ok"
    assert result.get("guardrail_malformed") is True
    assert result.get("guardrail_decision_raw") == "maybe"


def test_run_without_guardrail_calls_main_model(monkeypatch):
    runner = ModelRunner("gpt-4o-mini", provider="openai", config={})

    monkeypatch.setattr(
        runner,
        "_run_main_model",
        lambda *_args, **_kwargs: {"completion": "ok", "finish_reason": "stop"},
    )

    result = runner.run("system", "user")
    assert result["completion"] == "ok"
    assert result["finish_reason"] == "stop"


# -----------------------
# Optional integration test
# -----------------------
@pytest.mark.integration
def test_integration_auto_provider_with_guardrail_smoke():
    """
    Disabled by default unless you explicitly run:
      pytest -m integration
    and provide SCW_SECRET_KEY (and whatever your ModelRunner expects).
    """
    if not SCW_SECRET_KEY:
        pytest.skip("SCW_SECRET_KEY not set")

    runner = ModelRunner(
        model_name="llama-3.1-8b-instruct",
        provider="auto",
        config={
            "api_key": SCW_SECRET_KEY,
            "guardrail_model": "ibm-granite/granite-guardian-3.2-3b-a800m",
        },
    )

    result = runner.run("You are a helpful assistant.", "Say 'OK' and nothing else.")
    assert "finish_reason" in result


if __name__ == "__main__":
    # Allow: python test_model_runner.py
    raise SystemExit(pytest.main([__file__, "-q"]))
