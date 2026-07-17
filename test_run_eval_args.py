import sys
from pathlib import Path
import argparse

import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_eval


def test_parse_args_default_extra_dataset_paths_is_none(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_eval.py"])
    args = run_eval.parse_args()
    assert args.extra_dataset_paths is None


def test_parse_args_accepts_named_prompts(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_eval.py",
            "--guardrail_prompt_name",
            "guardrail_child_safety_v1",
            "--safety_judge_prompt_name",
            "judge_child_safety_v1",
        ],
    )
    args = run_eval.parse_args()
    assert args.guardrail_prompt_name == "guardrail_child_safety_v1"
    assert args.safety_judge_prompt_name == "judge_child_safety_v1"


def test_main_aborts_on_main_model_preflight_failure(monkeypatch) -> None:
    args = argparse.Namespace(
        judge_only=False,
        judge_only_path=None,
        safety_judge_model=None,
        safety_judge_provider="local_hf",
        safety_judge_prompt=None,
        safety_judge_prompt_name=None,
        config=None,
        model_name="mistral-small-3.2-24b-instruct-2506",
        system_prompt_name="assistant_child_safety_v2",
        prompt_pack_path="prompt_packs/minorbench_hateful.json",
        provider="scaleway",
        extra_dataset_paths=None,
        k=1,
        batch_size=1,
        include_variants=None,
        exclude_variants=None,
        guardrail_model=None,
        guardrail_id=None,
        guardrail_provider=None,
        guardrail_prompt=None,
        guardrail_prompt_name=None,
    )

    class _FailingRunner:
        def __init__(self, *_args, **_kwargs) -> None:
            self.guardrail_prompt_name = None
            self.guardrail_input_contract = None
            self.guardrail_adapter = None
            self.judge_meta = {}

        def preflight(self) -> dict:
            raise RuntimeError("probe failed")

    monkeypatch.setattr(run_eval, "parse_args", lambda: args)
    monkeypatch.setattr(run_eval, "ModelRunner", _FailingRunner)

    with pytest.raises(RuntimeError, match="probe failed"):
        run_eval.main()
