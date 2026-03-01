from typing import Any

import pytest

from safety_judge import LLMJudge


class _FailingClient:
    class chat:  # noqa: N801 - match OpenAI client shape
        class completions:  # noqa: N801
            @staticmethod
            def create(**_kwargs: Any) -> Any:
                raise RuntimeError("model not found")


def test_llm_judge_preflight_fails_on_invalid_model(monkeypatch: pytest.MonkeyPatch) -> None:
    judge = LLMJudge(model_name="not-a-real-model", provider="openai")

    def _fake_get_client() -> _FailingClient:
        return _FailingClient()

    monkeypatch.setattr(judge, "_get_client", _fake_get_client)

    with pytest.raises(RuntimeError, match="preflight failed"):
        judge.preflight()


def test_parse_response_repairs_markdown_fences() -> None:
    judge = LLMJudge(model_name="fake", provider="local_hf")
    text = '```json\n{"label":"safe","reasons":[]}\n```'
    parsed = judge._parse_response(text)
    assert parsed["label"] == "safe"
    assert parsed["judge_malformed"] is False
    assert parsed["judge_parse_method"] in {"substring", "direct"}


def test_parse_response_repairs_trailing_comma_and_extra_brace() -> None:
    judge = LLMJudge(model_name="fake", provider="local_hf")
    text = '{"label":"unsafe","reasons":["x",],}}'
    parsed = judge._parse_response(text)
    assert parsed["label"] == "unsafe"
    assert parsed["judge_malformed"] is False
    assert parsed["judge_parse_method"] == "heuristic_repair"


def test_parse_response_extracts_prefixed_suffixed_json() -> None:
    judge = LLMJudge(model_name="fake", provider="local_hf")
    text = 'note: {"label":"unsafe","reasons":["x"]} trailing'
    parsed = judge._parse_response(text)
    assert parsed["label"] == "unsafe"
    assert parsed["judge_malformed"] is False
    assert parsed["judge_parse_method"] == "substring"


def test_parse_response_retry_path_can_recover(monkeypatch: pytest.MonkeyPatch) -> None:
    judge = LLMJudge(
        model_name="fake",
        provider="local_hf",
        config={"judge_retry_on_malformed": True, "judge_retry_max_attempts": 1},
    )

    monkeypatch.setattr(
        judge,
        "_attempt_repair_with_model",
        lambda _raw: '{"label":"safe","reasons":[]}',
    )

    parsed = judge._parse_response("not-json")
    assert parsed["label"] == "safe"
    assert parsed["judge_malformed"] is False
    assert parsed["judge_parse_method"] == "retry_repair"
