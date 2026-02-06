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
