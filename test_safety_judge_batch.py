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
    def __init__(self, eos_token_id: Any = 123) -> None:
        self.eos_token_id = eos_token_id
        self.pad_token_id = None


class _FakeModel:
    def __init__(self, eos_token_id: Any = 123) -> None:
        self.config = _FakeModelConfig(eos_token_id=eos_token_id)
        self.generation_config = _FakeModelConfig(eos_token_id=eos_token_id)


class _FakePipeline:
    def __init__(self, eos_token_id: Any = 123) -> None:
        self.tokenizer = _FakeTokenizer()
        self.model = _FakeModel(eos_token_id=eos_token_id)


def test_judge_init_sets_pad_token_id() -> None:
    judge = LLMJudge(model_name="fake", provider="local_hf")
    judge._generator = _FakePipeline()
    judge._ensure_pad_token()
    assert judge._generator.tokenizer.pad_token_id == 123


def test_judge_init_sets_pad_token_id_from_list_eos() -> None:
    judge = LLMJudge(model_name="fake", provider="local_hf")
    judge._generator = _FakePipeline(eos_token_id=[128001, 128009])
    judge._ensure_pad_token()
    assert judge._generator.tokenizer.pad_token_id == 128001


class _FakeBatchPadErrorGenerator:
    def __call__(self, prompts: Any, **_kwargs: Any) -> List[dict]:
        if isinstance(prompts, list):
            raise ValueError(
                "Pipeline with tokenizer without pad_token cannot do batching."
            )
        return [{"generated_text": f"{prompts}{{\"label\":\"safe\",\"reasons\":[]}}"}]


def test_run_local_hf_batch_falls_back_to_unbatched_on_pad_error(monkeypatch) -> None:
    judge = LLMJudge(model_name="fake", provider="local_hf")

    monkeypatch.setattr(judge, "_init_local_hf", lambda: None)
    judge._generator = _FakeBatchPadErrorGenerator()

    outputs = judge._run_local_hf_batch(["p1", "p2"])
    assert outputs == ['{"label":"safe","reasons":[]}', '{"label":"safe","reasons":[]}']
