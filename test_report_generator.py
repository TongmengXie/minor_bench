from pathlib import Path

import pytest

from report_generator import ReportGenerator

def test_report_generator_f1_scores(tmp_path: Path) -> None:
    results = [
        {
            "n_generations": 2,
            "safe_count": 1,
            "unsafe_count": 1,
            "is_harmful": True,
            "all_classifications": ["safe", "unsafe"],
            "failure_reasons": [["reason_a"]],
            "metadata": {"variant": "v1"},
        },
        {
            "n_generations": 1,
            "safe_count": 1,
            "unsafe_count": 0,
            "is_harmful": False,
            "all_classifications": ["safe"],
            "failure_reasons": [[]],
            "metadata": {"variant": "v1"},
        },
        {
            "n_generations": 1,
            "safe_count": 0,
            "unsafe_count": 1,
            "is_harmful": False,
            "all_classifications": ["unsafe"],
            "failure_reasons": [["reason_b"]],
            "metadata": {"variant": "v2"},
        },
    ]

    summary = ReportGenerator(output_dir=tmp_path).generate(results)

    assert summary["f1_score"] == pytest.approx(0.5)
    assert summary["scorecards"]["system"]["f1_score"] == pytest.approx(0.5)
    assert summary["per_variant_metrics"]["v1"]["f1_score"] == pytest.approx(2 / 3)
    assert summary["per_variant_metrics"]["v2"]["f1_score"] is None

    csv_path = tmp_path / "variant_metrics.csv"
    assert csv_path.exists()
    header = csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert "f1_score" in header


def test_report_generator_costs(tmp_path: Path) -> None:
    results = [
        {
            "n_generations": 1,
            "safe_count": 1,
            "unsafe_count": 0,
            "is_harmful": False,
            "all_classifications": ["safe"],
            "failure_reasons": [[]],
            "metadata": {"variant": "v1"},
            "generations": [
                {
                    "raw": {
                        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
                        "cost_usd": 0.01,
                        "guardrail_usage": {
                            "prompt_tokens": 2,
                            "completion_tokens": 1,
                            "total_tokens": 3,
                        },
                        "guardrail_cost_usd": 0.005,
                    }
                }
            ],
            "judge_cost_usd": [0.02],
            "judge_usage": [{"prompt_tokens": 4, "completion_tokens": 4, "total_tokens": 8}],
        }
    ]

    summary = ReportGenerator(output_dir=tmp_path).generate(results)
    costs = summary.get("costs") or {}
    assert costs["tutor_cost_usd"] == pytest.approx(0.01)
    assert costs["guardrail_cost_usd"] == pytest.approx(0.005)
    assert costs["judge_cost_usd"] == pytest.approx(0.02)
    assert costs["tutor_tokens"] == 10
    assert costs["guardrail_tokens"] == 3
    assert costs["judge_tokens"] == 8


def test_report_generator_excludes_malformed_unknown_from_confusion(tmp_path: Path) -> None:
    results = [
        {
            "n_generations": 2,
            "safe_count": 1,
            "unsafe_count": 1,
            "is_harmful": False,
            "judge_labels": ["unknown", "safe"],
            "judge_malformed": [True, False],
            "judge_classifications": ["unknown", "safe"],
            "judge_failure_reasons": [["Malformed judge response"], []],
            "metadata": {"variant": "v1"},
            "generations": [
                {"raw": {"guardrail_decision_status": "allow"}},
                {"raw": {"guardrail_decision_status": "allow"}},
            ],
        }
    ]

    summary = ReportGenerator(output_dir=tmp_path, use_judge_results=True).generate(results)
    system = summary["scorecards"]["system"]
    assert system["evaluated_generation_count"] == 1
    assert system["tn"] == 1
    assert system["fp"] == 0
    assert summary["coverage"]["judge_malformed_generation_count"] == 1


def test_report_generator_dual_scorecards_differ_when_blocked(tmp_path: Path) -> None:
    results = [
        {
            "n_generations": 2,
            "safe_count": 0,
            "unsafe_count": 2,
            "is_harmful": False,
            "judge_labels": ["unsafe", "unsafe"],
            "judge_malformed": [False, False],
            "judge_classifications": ["unsafe", "unsafe"],
            "judge_failure_reasons": [["x"], ["y"]],
            "metadata": {"variant": "v1"},
            "generations": [
                {"raw": {"guardrail_decision_status": "block", "finish_reason": "guardrail_blocked"}},
                {"raw": {"guardrail_decision_status": "allow", "finish_reason": "stop"}},
            ],
            "guardrail_decisions": [{"status": "block"}, {"status": "allow"}],
            "guardrail_model": "g1",
            "guardrail_provider": "local_hf",
        }
    ]

    summary = ReportGenerator(output_dir=tmp_path, use_judge_results=True).generate(results)
    system = summary["scorecards"]["system"]
    tutor = summary["scorecards"]["tutor_conditional"]
    assert system["fp"] == 2
    assert tutor["fp"] == 1


def test_report_generator_warnings_for_zero_allow_rate(tmp_path: Path) -> None:
    results = [
        {
            "n_generations": 1,
            "safe_count": 0,
            "unsafe_count": 1,
            "is_harmful": False,
            "judge_labels": ["unsafe"],
            "judge_malformed": [False],
            "judge_classifications": ["unsafe"],
            "judge_failure_reasons": [["x"]],
            "metadata": {"variant": "v1"},
            "generations": [
                {"raw": {"guardrail_decision_status": "block", "finish_reason": "guardrail_blocked"}},
            ],
            "guardrail_decisions": [{"status": "block"}],
            "guardrail_model": "g1",
            "guardrail_provider": "local_hf",
        }
    ]

    summary = ReportGenerator(output_dir=tmp_path, use_judge_results=True).generate(results)
    warnings = summary.get("warnings") or []
    assert any("allow_rate is 0.0" in warning for warning in warnings)
