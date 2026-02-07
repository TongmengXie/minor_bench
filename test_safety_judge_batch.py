from typing import Any, List
import sys
from pathlib import Path

# Ensure local imports work when running this file directly.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    judge._generator = _FakeJudgeGenerator()

    outputs = judge.judge_batch(["sys1", "sys2"], ["resp1", "resp2"])
    assert outputs[0]["label"] == "safe"
    assert outputs[1]["label"] == "safe"


class _FakeTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = None


class _FakeModelConfig:
    def __init__(self) -> None:
        self.eos_token_id = 123


class _FakeModel:
    def __init__(self) -> None:
        self.config = _FakeModelConfig()


class _FakePipeline:
    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()
        self.model = _FakeModel()


def test_judge_init_sets_pad_token_id(monkeypatch) -> None:
    judge = LLMJudge(model_name="fake", provider="local_hf")
    judge._generator = _FakePipeline()
    judge._ensure_pad_token()
    assert judge._generator.tokenizer.pad_token_id == 123
