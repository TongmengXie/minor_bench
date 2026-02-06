from typing import Any, List

from model_runner import ModelRunner


class _FakeGenerator:
    def __init__(self, suffix: str) -> None:
        self.suffix = suffix

    def __call__(self, prompts: List[str], **_kwargs: Any) -> List[dict]:
        outputs: List[dict] = []
        for prompt in prompts:
            outputs.append({"generated_text": f"{prompt}{self.suffix}"})
        return outputs


def test_run_local_guardrail_hf_batch_parses_decisions(monkeypatch) -> None:
    runner = ModelRunner(
        model_name="fake",
        provider="local_hf",
        config={"guardrail_model": "fake-guardrail", "guardrail_provider": "local_hf"},
    )

    monkeypatch.setattr(runner, "_init_local_guardrail_hf", lambda: None)
    runner._guardrail_generator = _FakeGenerator("ALLOW")

    outputs = runner._run_local_guardrail_hf_batch("system", ["a", "b"])
    assert outputs[0]["guardrail_decision"] == "ALLOW"
    assert outputs[1]["guardrail_decision"] == "ALLOW"


def test_run_local_hf_batch_strips_prompt(monkeypatch) -> None:
    runner = ModelRunner(model_name="fake", provider="local_hf")

    monkeypatch.setattr(runner, "_init_local_hf", lambda: None)
    monkeypatch.setattr(runner, "_build_dataset", lambda prompts: prompts)
    runner._generator = _FakeGenerator(" completion")

    prompts = ["P1", "P2"]
    outputs = runner._run_local_hf_batch(prompts)
    assert outputs[0]["completion"] == "completion"
    assert outputs[1]["completion"] == "completion"
