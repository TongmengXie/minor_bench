from typing import Any, List

from safety_judge import LLMJudge


class _FakeJudgeGenerator:
    def __call__(self, prompts: List[str], **_kwargs: Any) -> List[dict]:
        outputs: List[dict] = []
        for prompt in prompts:
            outputs.append({"generated_text": f"{prompt}{{\"label\":\"safe\",\"reasons\":[]}}"})
        return outputs


def test_judge_batch_local_hf(monkeypatch) -> None:
    judge = LLMJudge(model_name="fake", provider="local_hf")

    monkeypatch.setattr(judge, "_init_local_hf", lambda: None)
    monkeypatch.setattr(judge, "_build_dataset", lambda prompts: prompts)
    judge._generator = _FakeJudgeGenerator()

    outputs = judge.judge_batch(["sys1", "sys2"], ["resp1", "resp2"])
    assert outputs[0]["label"] == "safe"
    assert outputs[1]["label"] == "safe"
