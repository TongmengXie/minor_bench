import sys

import run_eval


def test_parse_args_default_extra_dataset_paths_is_none(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_eval.py"])
    args = run_eval.parse_args()
    assert args.extra_dataset_paths is None
