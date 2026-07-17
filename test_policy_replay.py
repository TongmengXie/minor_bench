from copy import deepcopy
import csv
from pathlib import Path

from policy_replay import apply_effective_policy_outcomes, paired_units, write_paired_reports
from run_guardrail_replay import _apply_decision


class _Runner:
    refusal_text = "Sorry, I can't help with that."
    guardrail_model = "guard"
    guardrail_provider = "local_hf"
    guardrail_profile_id = "guard_profile"
    guardrail_adapter = "test"
    guardrail_input_contract = "test"
    guardrail_access_mode = "local"


def _baseline_row(is_harmful: bool, label: str, text: str = "baseline answer"):
    return {
        "prompt_text": "prompt",
        "is_harmful": is_harmful,
        "dataset_source": "dataset",
        "metadata": {"variant": "original", "base_prompt_id": "1"},
        "generations": [{"text": text, "raw": {"finish_reason": "stop"}}],
        "assistant_policy_labels": [label],
        "response_safety_labels": ["safe" if label == "correct" else "unsafe"],
        "answer_mode_labels": ["answered"],
    }


def test_replay_allow_preserves_generation_and_block_replaces_it() -> None:
    baseline = _baseline_row(True, "incorrect", "unsafe baseline answer")
    allowed = _apply_decision(
        deepcopy(baseline),
        _Runner(),
        {"status": "allow", "reason": None, "mode": "test", "guardrail_latency": 0.1},
    )
    blocked = _apply_decision(
        deepcopy(baseline),
        _Runner(),
        {"status": "block", "reason": "risk", "mode": "test", "guardrail_latency": 0.1},
    )

    assert allowed["generations"][0]["text"] == "unsafe baseline answer"
    assert allowed["assistant_policy_labels"] == ["incorrect"]
    assert blocked["generations"][0]["text"] == _Runner.refusal_text
    assert blocked["assistant_policy_labels"] == ["correct"]
    assert blocked["baseline_generations"][0]["text"] == "unsafe baseline answer"


def test_paired_reports_are_deterministic(tmp_path: Path) -> None:
    rows = [
        _apply_decision(
            apply_effective_policy_outcomes(_baseline_row(True, "incorrect")),
            _Runner(),
            {"status": "block", "reason": "risk", "mode": "test", "guardrail_latency": 0.1},
        ),
        _apply_decision(
            apply_effective_policy_outcomes(_baseline_row(False, "correct")),
            _Runner(),
            {"status": "allow", "reason": None, "mode": "test", "guardrail_latency": 0.1},
        ),
    ]
    assert len(paired_units(rows)) == 2
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    write_paired_reports(rows, first, bootstrap_replicates=50, bootstrap_seed=1)
    write_paired_reports(rows, second, bootstrap_replicates=50, bootstrap_seed=1)
    assert (first / "guardrail_effect_metrics.csv").read_text() == (
        second / "guardrail_effect_metrics.csv"
    ).read_text()
    with (first / "guardrail_transition_report.csv").open(newline="") as handle:
        overall = next(csv.DictReader(handle))
    assert overall["harmful_baseline_successes_regressed"] == "0"
    assert overall["safe_failures_rescued"] == "0"
