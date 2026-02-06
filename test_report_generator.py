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
    assert summary["per_variant_metrics"]["v1"]["f1_score"] == pytest.approx(2 / 3)
    assert summary["per_variant_metrics"]["v2"]["f1_score"] == pytest.approx(0.0)

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
