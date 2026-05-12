import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import probe_guardrail_profiles


def test_parse_args_supports_boolean_optional_allow_rate(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe_guardrail_profiles.py",
            "--matrix_config",
            "matrix.yaml",
            "--output_dir",
            "out",
            "--no-require_nonzero_allow_rate",
        ],
    )
    args = probe_guardrail_profiles.parse_args()
    assert args.require_nonzero_allow_rate is False


def test_run_probe_writes_manifest_report_and_samples(tmp_path: Path, monkeypatch) -> None:
    matrix_config = {
        "global": {
            "model_name": "mistral-small-3.2-24b-instruct-2506",
            "provider": "scaleway",
            "system_prompt_name": "assistant_child_safety_v2",
            "prompt_pack_path": "prompt_packs/minorbench_hateful.json",
            "extra_dataset_paths": ["datasets/true_negative_safe_sensitive_variants.jsonl"],
        },
        "rows": [
            {
                "id": "granite_guardrail",
                "label": "Granite guardrail",
                "guardrail_profile_id": "granite_guardian_3_2_compare_v2",
            }
        ],
    }
    matrix_config_path = tmp_path / "matrix.yaml"
    matrix_config_path.write_text(yaml.safe_dump(matrix_config), encoding="utf-8")

    sampled_items = [
        {
            "prompt_text": "safe prompt",
            "metadata": {"variant": "original"},
            "is_harmful": False,
        },
        {
            "prompt_text": "harmful prompt",
            "metadata": {"variant": "adversarial"},
            "is_harmful": True,
        },
    ]

    monkeypatch.setattr(probe_guardrail_profiles, "_load_items", lambda **_kwargs: list(sampled_items))
    monkeypatch.setattr(
        probe_guardrail_profiles,
        "_sample_probe_items",
        lambda items, **_kwargs: list(items),
    )
    monkeypatch.setattr(
        probe_guardrail_profiles,
        "_row_probe_result",
        lambda row_cfg, sampled_items, **_kwargs: {
            "row_id": row_cfg["id"],
            "row_label": row_cfg["label"],
            "guardrail_profile_id": row_cfg["guardrail_profile_id"],
            "guardrail_model": "ibm-granite/granite-guardian-3.2-3b-a800m",
            "guardrail_provider": "local_hf",
            "guardrail_prompt_name": "guardrail_granite_native_v1",
            "guardrail_input_contract": "shared_policy_native_adapter",
            "guardrail_adapter": "granite_guardian_yes_no",
            "guardrail_access_mode": "local",
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
            "guardrail_prompt_normalizer": "strip_known_jailbreak_prefix",
            "status": "invalid",
            "invalid_reasons": ["allow_rate=0%"],
            "total_prompts": len(sampled_items),
            "counts": {"block": 2},
            "allow_rate": 0.0,
            "block_rate": 1.0,
            "parser_malformed_rate": 0.0,
            "native_template_mismatch_rate": 0.0,
            "runtime_error_rate": 0.0,
            "malformed_rate": 0.0,
            "safe_block_rate": 1.0,
            "harmful_allow_rate": 0.0,
            "samples": {
                "runtime_error": [],
                "parser_malformed": [],
                "native_template_mismatch": [],
                "safe_blocked": [{"prompt_text": "safe prompt"}],
                "harmful_allowed": [],
            },
        },
    )

    outputs = probe_guardrail_profiles.run_probe(
        matrix_config_path=matrix_config_path,
        output_dir=tmp_path / "probe",
        seed=1,
        harmful_rows_per_variant=12,
        safe_rows_per_variant=6,
        max_malformed_rate=0.05,
        max_runtime_error_rate=0.0,
        require_nonzero_allow_rate=True,
    )

    manifest_path = Path(outputs["manifest_path"])
    samples_path = Path(outputs["samples_path"])
    report_path = Path(outputs["report_path"])

    assert manifest_path.exists()
    assert samples_path.exists()
    assert report_path.exists()

    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_payload["rows"][0]["row_id"] == "granite_guardrail"
    assert manifest_payload["rows"][0]["status"] == "invalid"

    samples_payload = json.loads(samples_path.read_text(encoding="utf-8"))
    assert samples_payload["granite_guardrail"]["safe_blocked"][0]["prompt_text"] == "safe prompt"

    report_text = report_path.read_text(encoding="utf-8")
    assert "Guardrail Probe Report" in report_text
    assert "Granite guardrail" in report_text
    assert "allow_rate=0%" in report_text
