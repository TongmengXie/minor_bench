import json
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_matrix_eval


def test_build_run_eval_command_forces_local_hf_judge() -> None:
    row_cfg = {
        "model_name": "mistral-small-3.2-24b-instruct-2506",
        "provider": "scaleway",
        "system_prompt_name": "generic_child_tutor_v1",
        "prompt_pack_path": "prompt_packs/minorbench_hateful.json",
        "safety_judge_provider": "openai",
        "safety_judge_model": "custom/judge",
        "extra_dataset_paths": ["datasets/a.jsonl", "datasets/b.jsonl"],
        "include_variants": ["original", "camouflage"],
        "guardrail_model": "ibm-granite/granite-guardian-3.2-3b-a800m",
        "guardrail_provider": "local_hf",
        "config": {"judge_retry_on_malformed": True},
    }

    cmd = run_matrix_eval._build_run_eval_command(
        python_exe="python",
        run_eval_path=Path("run_eval.py"),
        row_cfg=row_cfg,
    )

    assert "--safety_judge_provider" in cmd
    provider_idx = cmd.index("--safety_judge_provider")
    assert cmd[provider_idx + 1] == "local_hf"

    model_idx = cmd.index("--safety_judge_model")
    assert cmd[model_idx + 1] == "custom/judge"

    extra_idx = cmd.index("--extra_dataset_paths")
    assert cmd[extra_idx + 1] == "datasets/a.jsonl,datasets/b.jsonl"


def test_build_run_eval_command_supports_guardrail_profile() -> None:
    row_cfg = {
        "model_name": "mistral-small-3.2-24b-instruct-2506",
        "provider": "scaleway",
        "system_prompt_name": "assistant_child_safety_v2",
        "prompt_pack_path": "prompt_packs/minorbench_hateful.json",
        "safety_judge_model": "meta-llama/Llama-3.2-1B-Instruct",
        "guardrail_profile_id": "granite_guardian_3_3_compare_v2",
        "guardrail_model": "ibm-granite/granite-guardian-3.3-8b",
        "guardrail_provider": "local_hf",
        "guardrail_prompt_name": "guardrail_granite_native_v1",
        "guardrail_input_contract": "shared_policy_native_adapter",
        "guardrail_adapter": "granite_guardian_yes_no",
        "guardrail_malformed_policy": "fallback_to_main",
        "guardrail_prompt_normalizer": "strip_known_jailbreak_prefix",
        "guardrail_risk_names": [
            "harm",
            "social_bias",
            "profanity",
            "jailbreak",
            "sexual_content",
            "violence",
            "unethical_behavior",
        ],
        "guardrail_block_risk_names": [
            "social_bias",
            "jailbreak",
            "sexual_content",
            "violence",
            "unethical_behavior",
        ],
    }

    cmd = run_matrix_eval._build_run_eval_command(
        python_exe="python",
        run_eval_path=Path("run_eval.py"),
        row_cfg=row_cfg,
    )

    assert "--guardrail_id" in cmd
    guardrail_id_idx = cmd.index("--guardrail_id")
    assert cmd[guardrail_id_idx + 1] == "granite_guardian_3_3_compare_v2"
    prompt_name_idx = cmd.index("--guardrail_prompt_name")
    assert cmd[prompt_name_idx + 1] == "guardrail_granite_native_v1"
    config_idx = cmd.index("--config")
    payload = json.loads(cmd[config_idx + 1])
    assert payload["guardrail_input_contract"] == "shared_policy_native_adapter"
    assert payload["guardrail_adapter"] == "granite_guardian_yes_no"
    assert payload["guardrail_malformed_policy"] == "fallback_to_main"
    assert payload["guardrail_prompt_normalizer"] == "strip_known_jailbreak_prefix"
    assert payload["guardrail_risk_names"] == [
        "harm",
        "social_bias",
        "profanity",
        "jailbreak",
        "sexual_content",
        "violence",
        "unethical_behavior",
    ]
    assert payload["guardrail_block_risk_names"] == [
        "social_bias",
        "jailbreak",
        "sexual_content",
        "violence",
        "unethical_behavior",
    ]


def test_extract_run_dir() -> None:
    stdout = "foo\nEvaluation completed. Output: runs/evals/a/b/c\nbar"
    assert run_matrix_eval._extract_run_dir(stdout, "") == "runs/evals/a/b/c"


def test_main_writes_manifest_and_uses_local_hf_judge(tmp_path: Path, monkeypatch) -> None:
    matrix_cfg = {
        "global": {
            "model_name": "mistral-small-3.2-24b-instruct-2506",
            "provider": "scaleway",
            "prompt_pack_path": "prompt_packs/minorbench_hateful.json",
            "safety_judge_provider": "openai",
            "safety_judge_model": "meta-llama/Llama-3.2-1B-Instruct",
        },
        "rows": [
            {"id": "row_a", "system_prompt_name": "generic_child_tutor_v1"},
            {
                "id": "row_b",
                "system_prompt_name": "duolingo_tutor_v1",
                "guardrail_model": "ibm-granite/granite-guardian-3.2-3b-a800m",
                "guardrail_provider": "local_hf",
            },
        ],
    }
    cfg_path = tmp_path / "matrix.yaml"
    cfg_path.write_text(yaml.safe_dump(matrix_cfg), encoding="utf-8")

    recorded_cmds = []

    def fake_run(cmd, cwd, text, capture_output):
        recorded_cmds.append(cmd)
        idx = len(recorded_cmds)
        out = f"Evaluation completed. Output: {tmp_path}/eval_{idx}\n"
        return SimpleNamespace(returncode=0, stdout=out, stderr="")

    monkeypatch.setattr(run_matrix_eval.subprocess, "run", fake_run)
    monkeypatch.setattr(
        run_matrix_eval,
        "aggregate_from_manifest",
        lambda manifest_path: {"matrix_report_md": str(Path(manifest_path).with_name("matrix_report.md"))},
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_matrix_eval.py",
            "--matrix_config",
            str(cfg_path),
            "--output_root",
            str(tmp_path / "matrix_out"),
            "--name",
            "test_matrix",
        ],
    )

    run_matrix_eval.main()

    manifest_path = tmp_path / "matrix_out" / "test_matrix" / "manifest.json"
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["local_hf_judge_enforced"] is True
    assert len(manifest["runs"]) == 2
    assert all(run["status"] == "success" for run in manifest["runs"])
    assert "aggregation" in manifest

    for cmd in recorded_cmds:
        idx = cmd.index("--safety_judge_provider")
        assert cmd[idx + 1] == "local_hf"


def test_main_skips_invalid_probe_rows(tmp_path: Path, monkeypatch) -> None:
    matrix_cfg = {
        "global": {
            "model_name": "mistral-small-3.2-24b-instruct-2506",
            "provider": "scaleway",
            "prompt_pack_path": "prompt_packs/minorbench_hateful.json",
            "system_prompt_name": "assistant_child_safety_v2",
            "safety_judge_model": "meta-llama/Llama-3.2-1B-Instruct",
            "guardrail_probe": {
                "enabled": True,
                "harmful_rows_per_variant": 2,
                "safe_rows_per_variant": 1,
                "seed": 1,
            },
            "invalid_row_policy": "exclude_main",
        },
        "rows": [
            {"id": "baseline", "system_prompt_name": "assistant_child_safety_v2"},
            {
                "id": "granite_bad",
                "system_prompt_name": "assistant_child_safety_v2",
                "guardrail_profile_id": "granite_guardian_3_3_compare_v2",
            },
        ],
    }
    cfg_path = tmp_path / "matrix.yaml"
    cfg_path.write_text(yaml.safe_dump(matrix_cfg), encoding="utf-8")

    recorded_cmds = []

    def fake_run(cmd, cwd, text, capture_output):
        recorded_cmds.append(cmd)
        return SimpleNamespace(returncode=0, stdout="Evaluation completed. Output: runs/evals/x/y/z\n", stderr="")

    monkeypatch.setattr(run_matrix_eval.subprocess, "run", fake_run)
    monkeypatch.setattr(
        run_matrix_eval,
        "run_probe",
        lambda **_kwargs: {
            "manifest_path": str(tmp_path / "probe_manifest.json"),
            "samples_path": str(tmp_path / "probe_samples.json"),
            "report_path": str(tmp_path / "probe_report.md"),
            "payload": {
                "thresholds": {},
                "rows": [
                    {
                        "row_id": "granite_bad",
                        "status": "invalid",
                        "invalid_reasons": ["allow_rate=0%"],
                        "allow_rate": 0.0,
                        "block_rate": 0.0,
                        "malformed_rate": 1.0,
                        "runtime_error_rate": 1.0,
                        "safe_block_rate": 0.0,
                        "harmful_allow_rate": 0.0,
                    }
                ],
            },
        },
    )
    monkeypatch.setattr(
        run_matrix_eval,
        "aggregate_from_manifest",
        lambda manifest_path: {"matrix_report_md": str(Path(manifest_path).with_name("matrix_report.md"))},
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_matrix_eval.py",
            "--matrix_config",
            str(cfg_path),
            "--output_root",
            str(tmp_path / "matrix_out"),
            "--name",
            "test_probe_skip",
        ],
    )

    run_matrix_eval.main()

    manifest = json.loads(
        (tmp_path / "matrix_out" / "test_probe_skip" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["runs"][0]["status"] == "success"
    assert manifest["runs"][1]["status"] == "skipped_after_probe"
    assert len(recorded_cmds) == 1


def test_canonical_assistant_policy_v2_matrix_configs_use_mistral_small() -> None:
    repo_root = Path(__file__).resolve().parent
    expected = "mistral-small-3.2-24b-instruct-2506"

    for rel_path in [
        "matrix_configs/hateful_guardrail_matrix_assistant_policy_v2.yaml",
        "matrix_configs/hateful_guardrail_compare_matrix_assistant_policy_v2.yaml",
        "matrix_configs/hateful_guardrail_compare_granite_only.yaml",
    ]:
        cfg = run_matrix_eval._load_matrix_config(repo_root / rel_path)
        assert cfg["global"]["model_name"] == expected
