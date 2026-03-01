import json
from pathlib import Path

from evaluator import Evaluator, EvaluationConfig


class _AlwaysSafeClassifier:
    def classify(self, _text: str):
        return {"label": "safe", "reasons": []}


class _DummyRunner:
    def __init__(self) -> None:
        self.model_name = "dummy"
        self.provider = "openai"
        self.guardrail_model = None
        self.guardrail_provider = None

    def run(self, system_prompt: str, prompt_text: str):
        _ = system_prompt
        return {"completion": f"ok:{prompt_text}", "finish_reason": "stop"}


def test_annotate_with_safety_infers_hateful_and_unknown() -> None:
    evaluator = Evaluator(config=EvaluationConfig())
    results = [
        {
            "generations": [{"text": "x"}],
            "metadata": {"variant": "original", "original_category": "Hateful"},
            "dataset_source": "prompt_pack:minorbench_hateful_v1",
            "prompt_pack_name": "minorbench_hateful_v1",
            "prompt_pack_subset": "Hateful",
        },
        {
            "generations": [{"text": "x"}],
            "metadata": {"variant": "original"},
            "dataset_source": "custom_dataset",
        },
    ]

    annotated = evaluator.annotate_with_safety(results, classifier=_AlwaysSafeClassifier())
    assert annotated[0]["is_harmful"] is True
    assert annotated[0]["label_source"] == "inferred_hateful"
    assert annotated[1]["is_harmful"] is None
    assert annotated[1]["label_source"] == "unknown"


def test_evaluate_pack_writes_data_manifest(tmp_path: Path) -> None:
    prompt_pack = {
        "name": "minorbench_hateful_v1",
        "subset": "Hateful",
        "items": [
            {
                "id": "a",
                "prompt_text": "p1",
                "metadata": {"variant": "original", "original_category": "Hateful"},
            }
        ],
    }
    prompt_pack_path = tmp_path / "prompt_pack.json"
    prompt_pack_path.write_text(json.dumps(prompt_pack), encoding="utf-8")

    evaluator = Evaluator(config=EvaluationConfig(output_root=tmp_path))
    runner = _DummyRunner()

    _results, output_dir = evaluator.evaluate_pack(
        runner,
        system_prompt_name="generic_child_tutor_v1",
        prompt_pack_path=str(prompt_pack_path),
        k=1,
        batch_size=1,
    )

    manifest_path = output_dir / "data_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["prompt_pack_subset"] == "Hateful"
    assert manifest["counts_by_dataset_source"]["prompt_pack:minorbench_hateful_v1"] == 1
    assert manifest["counts_by_harmful_label_state"]["harmful"] == 1
