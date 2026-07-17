import json
from pathlib import Path
from typing import Dict, Any, List
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator import Evaluator, EvaluationConfig
from report_generator import ReportGenerator


class _FakeJudge:
    def judge(self, system_prompt: str, response_text: str) -> Dict[str, Any]:
        label = "unsafe" if "unsafe" in response_text else "safe"
        return {"label": label, "reasons": ["test"] if label == "unsafe" else [], "raw": "ok"}


class _FakeBatchJudge:
    provider = "local_hf"

    def __init__(self) -> None:
        self.config = {"judge_batch_size": 8}
        self.calls: List[List[str]] = []

    def judge_batch(
        self,
        user_prompts: List[str],
        response_texts: List[str],
        prompt_labels: List[str],
        final_response_sources: List[str | None],
    ) -> List[Dict[str, Any]]:
        self.calls.append(list(response_texts))
        outcomes: List[Dict[str, Any]] = []
        for text in response_texts:
            label = "unsafe" if "unsafe" in text else "safe"
            outcomes.append(
                {
                    "response_safety_label": label,
                    "assistant_policy_label": "incorrect" if label == "unsafe" else "correct",
                    "answer_mode": "answered",
                    "reasons": ["test"] if label == "unsafe" else [],
                    "raw": "ok",
                    "judge_malformed": False,
                    "judge_parse_method": "direct",
                }
            )
        return outcomes


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


def test_llm_judge_resumes_from_partial_results(tmp_path: Path) -> None:
    results = [
        {
            "prompt_text": "p1",
            "generations": [{"text": "safe response"}, {"text": "unsafe response"}],
            "system_prompt": "system",
            "metadata": {"variant": "original", "is_harmful": False},
            "assistant_policy_labels": ["correct"],
            "response_safety_labels": ["safe"],
            "answer_mode_labels": ["answered"],
            "assistant_policy_reasons": [[]],
            "failure_reasons": [[]],
            "judge_raw_outputs": ["ok"],
            "judge_malformed": [False],
            "judge_parse_method": ["direct"],
        },
        {
            "prompt_text": "p2",
            "generations": [{"text": "safe response"}],
            "system_prompt": "system",
            "metadata": {"variant": "original", "is_harmful": True},
        },
    ]

    evaluator = Evaluator(config=EvaluationConfig(output_root=tmp_path))
    results_path = tmp_path / "results.jsonl"
    judge = _FakeBatchJudge()

    annotated = evaluator.annotate_with_llm_judge(
        results=results,
        judge=judge,
        results_path=results_path,
    )

    assert judge.calls == [["unsafe response", "safe response"]]
    assert annotated[0]["response_safety_labels"] == ["safe", "unsafe"]
    assert annotated[1]["response_safety_labels"] == ["safe"]

    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["response_safety_labels"] == ["safe", "unsafe"]
    assert rows[1]["response_safety_labels"] == ["safe"]


def test_llm_judge_skips_infrastructure_failures(tmp_path: Path) -> None:
    results = [
        {
            "prompt_text": "p1",
            "generations": [
                {
                    "text": "",
                    "raw": {
                        "completion": "",
                        "finish_reason": "error",
                        "error": "Connection error.",
                    },
                },
                {"text": "safe response", "raw": {"finish_reason": "stop"}},
            ],
            "system_prompt": "system",
            "metadata": {"variant": "original", "is_harmful": False},
        }
    ]

    evaluator = Evaluator(config=EvaluationConfig(output_root=tmp_path))
    results_path = tmp_path / "results.jsonl"
    judge = _FakeBatchJudge()

    annotated = evaluator.annotate_with_llm_judge(
        results=results,
        judge=judge,
        results_path=results_path,
    )

    assert judge.calls == [["safe response"]]
    assert annotated[0]["response_safety_labels"] == ["unknown", "safe"]
    assert annotated[0]["assistant_policy_labels"] == ["unknown", "correct"]
    assert annotated[0]["judge_parse_method"][0] == "skipped_infrastructure_failure"


def test_llm_judge_skips_and_overwrites_guardrail_blocks(tmp_path: Path) -> None:
    results = [
        {
            "prompt_text": "harmful prompt",
            "generations": [
                {
                    "text": "Sorry, I can't help with that.",
                    "raw": {
                        "finish_reason": "guardrail_blocked",
                        "guardrail_decision_status": "block",
                    },
                },
                {
                    "text": "safe response",
                    "raw": {"finish_reason": "stop", "guardrail_decision_status": "allow"},
                },
            ],
            "metadata": {"variant": "original", "is_harmful": True},
            "assistant_policy_labels": ["incorrect"],
            "response_safety_labels": ["unsafe"],
            "answer_mode_labels": ["answered"],
            "judge_raw_outputs": ["stale"],
        }
    ]
    evaluator = Evaluator(config=EvaluationConfig(output_root=tmp_path))
    judge = _FakeBatchJudge()

    annotated = evaluator.annotate_with_llm_judge(results=results, judge=judge)

    assert judge.calls == [["safe response"]]
    assert annotated[0]["assistant_policy_labels"] == ["correct", "correct"]
    assert annotated[0]["response_safety_labels"] == ["safe", "safe"]
    assert annotated[0]["answer_mode_labels"][0] == "refused_or_blocked"
    assert annotated[0]["judge_parse_method"][0] == "deterministic_guardrail_block"
    assert annotated[0]["policy_outcome_source"][0] == "deterministic_guardrail_block"
