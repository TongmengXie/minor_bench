from types import SimpleNamespace

import pytest

from model_runner import ModelRunner


class FakeChatCompletions:
    def __init__(self, content: str, raise_error: Exception | None = None) -> None:
        self._content = content
        self._raise_error = raise_error
        self.last_request = None

    def create(self, **kwargs):
        if self._raise_error:
            raise self._raise_error
        self.last_request = kwargs
        message = SimpleNamespace(content=self._content)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


class FakeChat:
    def __init__(self, completions: FakeChatCompletions) -> None:
        self.completions = completions


class FakeClient:
    def __init__(self, completions: FakeChatCompletions) -> None:
        self.chat = FakeChat(completions)


@pytest.fixture

def runner():
    return ModelRunner(
        model_name="gpt-4o-mini",
        provider="openai",
        config={"guardrail_model": "fake-guardrail"},
    )


@pytest.fixture

def guardrail_client():
    completions = FakeChatCompletions("ALLOW")
    return completions, FakeClient(completions)


def test_guardrail_request_formatting(runner, guardrail_client, monkeypatch):
    completions, client = guardrail_client
    monkeypatch.setattr(runner, "_get_guardrail_client", lambda: client)

    runner._maybe_apply_guardrail("system prompt", "user prompt")

    request = completions.last_request
    assert request is not None
    assert request["messages"][0]["content"] == runner.guardrail_prompt
    user_content = request["messages"][1]["content"]
    assert "System prompt:\nsystem prompt" in user_content
    assert "User prompt:\nuser prompt" in user_content


def test_parse_guardrail_decision():
    status, reason, mode = ModelRunner._parse_guardrail_decision("ALLOW")
    assert status == "allow"
    assert reason is None
    assert mode == "binary"

    status, reason, mode = ModelRunner._parse_guardrail_decision("BLOCK: harmful")
    assert status == "block"
    assert reason == "harmful"
    assert mode == "binary"

    status, reason, mode = ModelRunner._parse_guardrail_decision("BLOCK")
    assert status == "block"
    assert reason == "Blocked by guardrail"
    assert mode == "binary"

    status, reason, mode = ModelRunner._parse_guardrail_decision("maybe")
    assert status == "malformed"
    assert "not ALLOW or BLOCK" in reason
    assert mode == "text"

    status, reason, mode = ModelRunner._parse_guardrail_decision("")
    assert status == "malformed"
    assert "Empty" in reason
    assert mode == "text"


def test_run_blocks_on_guardrail_block(runner, monkeypatch):
    completions = FakeChatCompletions("BLOCK: unsafe")
    client = FakeClient(completions)
    monkeypatch.setattr(runner, "_get_guardrail_client", lambda: client)

    def _fail_main(*_args, **_kwargs):
        raise AssertionError("Main model should not run on block")

    monkeypatch.setattr(runner, "_run_main_model", _fail_main)

    result = runner.run("system", "user")
    assert result["finish_reason"] == "guardrail_blocked"
    assert result["completion"] == runner.refusal_text
    assert result["guardrail_reason"] == "unsafe"


def test_run_allows_on_guardrail_allow(runner, monkeypatch):
    completions = FakeChatCompletions("ALLOW")
    client = FakeClient(completions)
    monkeypatch.setattr(runner, "_get_guardrail_client", lambda: client)

    monkeypatch.setattr(
        runner,
        "_run_main_model",
        lambda *_args, **_kwargs: {"completion": "safe", "finish_reason": "stop"},
    )

    result = runner.run("system", "user")
    assert result["completion"] == "safe"
    assert result["finish_reason"] == "stop"


def test_run_malformed_guardrail_blocks_by_default(runner, monkeypatch):
    completions = FakeChatCompletions("maybe")
    client = FakeClient(completions)
    monkeypatch.setattr(runner, "_get_guardrail_client", lambda: client)

    result = runner.run("system", "user")
    assert result["finish_reason"] == "guardrail_blocked"
    assert result["guardrail_malformed"] is True


def test_run_malformed_guardrail_fallbacks_to_main(monkeypatch):
    runner = ModelRunner(
        model_name="gpt-4o-mini",
        provider="openai",
        config={
            "guardrail_model": "fake-guardrail",
            "guardrail_malformed_policy": "fallback_to_main",
        },
    )
    completions = FakeChatCompletions("maybe")
    client = FakeClient(completions)
    monkeypatch.setattr(runner, "_get_guardrail_client", lambda: client)

    monkeypatch.setattr(
        runner,
        "_run_main_model",
        lambda *_args, **_kwargs: {"completion": "safe", "finish_reason": "stop"},
    )

    result = runner.run("system", "user")
    assert result["completion"] == "safe"
    assert result["guardrail_malformed"] is True
    assert result["guardrail_decision_raw"] == "maybe"


def test_run_guardrail_error_blocks(monkeypatch):
    runner = ModelRunner(
        model_name="gpt-4o-mini",
        provider="openai",
        config={"guardrail_model": "fake-guardrail"},
    )
    completions = FakeChatCompletions("ALLOW", raise_error=RuntimeError("boom"))
    client = FakeClient(completions)
    monkeypatch.setattr(runner, "_get_guardrail_client", lambda: client)

    result = runner.run("system", "user")
    assert result["finish_reason"] == "guardrail_blocked"
    assert result["guardrail_malformed"] is True
    assert "boom" in result["guardrail_reason"]