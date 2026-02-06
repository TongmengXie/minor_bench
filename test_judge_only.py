import json
from pathlib import Path

import pytest

import argparse

import run_eval


class _FakeJudge:
    def preflight(self) -> None:
        return None

    def judge(self, system_prompt: str, response_text: str) -> dict:
        label = "unsafe" if "unsafe" in response_text else "safe"
        return {"label": label, "reasons": [], "raw": "ok"}


def _write_results(path: Path) -> None:
    rows = [
        {
            "prompt_text": "p1",
            "metadata": {"variant": "original", "is_harmful": False},
            "system_prompt": "system",
            "generations": [{"text": "safe response"}],
        },
        {
            "prompt_text": "p2",
            "metadata": {"variant": "original", "is_harmful": True},
            "system_prompt": "system",
            "generations": [{"text": "unsafe response"}],
        },
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def _base_args() -> argparse.Namespace:
    return argparse.Namespace(
        judge_only=True,
        judge_only_path=None,
        safety_judge_model="fake-judge",
        safety_judge_provider="openai",
        safety_judge_prompt=None,
        config=None,
        model_name=None,
        system_prompt_name=None,
        prompt_pack_path=None,
        provider="auto",
        extra_dataset_paths=None,
        k=1,
        batch_size=1,
        include_variants=None,
        exclude_variants=None,
        guardrail_model=None,
        guardrail_provider="auto",
        guardrail_prompt=None,
    )


def test_judge_only_requires_path(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _base_args()
    args.judge_only_path = None
    monkeypatch.setattr(run_eval, "parse_args", lambda: args)
    with pytest.raises(SystemExit, match="--judge_only requires --judge_only_path"):
        run_eval.main()


def test_judge_only_rewrites_in_place(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    results_path = run_dir / "results.jsonl"
    _write_results(results_path)

    args = _base_args()
    args.judge_only_path = str(run_dir)

    monkeypatch.setattr(run_eval, "parse_args", lambda: args)
    monkeypatch.setattr(run_eval, "LLMJudge", lambda **_kwargs: _FakeJudge())

    run_eval.main()

    rewritten = results_path.read_text(encoding="utf-8").splitlines()
    assert "judge_classifications" in rewritten[0]
    assert (run_dir / "summary.json").exists()
