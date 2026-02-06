import json
from pathlib import Path
from typing import Dict, Any, List

from evaluator import Evaluator, EvaluationConfig
from report_generator import ReportGenerator


class _FakeJudge:
    def judge(self, system_prompt: str, response_text: str) -> Dict[str, Any]:
        label = "unsafe" if "unsafe" in response_text else "safe"
        return {"label": label, "reasons": ["test"] if label == "unsafe" else [], "raw": "ok"}


def test_llm_judge_results_written_and_summary_uses_judge(tmp_path: Path) -> None:
    results = [
        {
            "generations": [{"text": "safe response"}],
            "system_prompt": "system",
            "metadata": {"variant": "original", "is_harmful": False},
        },
        {
            "generations": [{"text": "unsafe response"}],
            "system_prompt": "system",
            "metadata": {"variant": "original", "is_harmful": True},
        },
    ]

    evaluator = Evaluator(config=EvaluationConfig(output_root=tmp_path))
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"

    annotated = evaluator.annotate_with_llm_judge(
        results=results,
        judge=_FakeJudge(),
        results_path=results_path,
    )

    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["judge_classifications"] == ["safe"]
    assert rows[1]["judge_classifications"] == ["unsafe"]

    summary = ReportGenerator(output_dir=output_dir, use_judge_results=True).generate(annotated)
    assert summary["safety_source"] == "llm_judge"
