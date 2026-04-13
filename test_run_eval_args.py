import sys
from pathlib import Path

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
