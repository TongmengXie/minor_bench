import argparse
import json
import re
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from aggregate_matrix import aggregate_from_manifest
from assistant_policy import METRIC_DEFINITION_VERSION
from guardrails import merge_guardrail_profile

DEFAULT_LOCAL_JUDGE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_matrix_config(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Matrix config must be a YAML object")
    if not isinstance(payload.get("global"), dict):
        raise ValueError("Matrix config must include a 'global' object")
    if not isinstance(payload.get("rows"), list) or not payload.get("rows"):
        raise ValueError("Matrix config must include a non-empty 'rows' list")
    return payload


def _merge_row(global_cfg: Dict[str, Any], row_cfg: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(global_cfg)
    for key, value in row_cfg.items():
        if key == "config" and isinstance(value, dict):
            base = merged.get("config") or {}
            if not isinstance(base, dict):
                base = {}
            combined = dict(base)
            combined.update(value)
            merged["config"] = combined
        else:
            merged[key] = value
    guardrail_id = merged.get("guardrail_id") or merged.get("guardrail_profile_id")
    if guardrail_id:
        merged = merge_guardrail_profile(merged, str(guardrail_id))
    return merged


def _csv_arg(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, list):
        cleaned = [str(x).strip() for x in value if str(x).strip()]
        return ",".join(cleaned) if cleaned else None
    rendered = str(value).strip()
    if not rendered:
        return None
    if rendered.lower() == "none":
        return None
    return rendered


def _extract_run_dir(stdout: str, stderr: str) -> Optional[str]:
    combined = "\n".join([stdout or "", stderr or ""])
    match = re.search(r"Evaluation completed\. Output:\s*(.+)", combined)
    if match:
        return match.group(1).strip()
    match = re.search(r"Judge-only re-eval completed\. Output:\s*(.+)", combined)
    if match:
        return match.group(1).strip()
    return None


def _build_run_eval_command(
    *,
    python_exe: str,
    run_eval_path: Path,
    row_cfg: Dict[str, Any],
) -> List[str]:
    required = ["model_name", "provider", "system_prompt_name", "prompt_pack_path"]
    missing = [field for field in required if not row_cfg.get(field)]
    if missing:
        raise ValueError(f"Missing required matrix row fields: {', '.join(missing)}")

    judge_model = str(row_cfg.get("safety_judge_model") or DEFAULT_LOCAL_JUDGE_MODEL)

    cmd = [
        python_exe,
        str(run_eval_path),
        "--model_name",
        str(row_cfg["model_name"]),
        "--provider",
        str(row_cfg["provider"]),
        "--system_prompt_name",
        str(row_cfg["system_prompt_name"]),
        "--prompt_pack_path",
        str(row_cfg["prompt_pack_path"]),
        "--safety_judge_model",
        judge_model,
        "--safety_judge_provider",
        "local_hf",
    ]

    if row_cfg.get("k") is not None:
        cmd.extend(["--k", str(row_cfg["k"])])
    if row_cfg.get("batch_size") is not None:
        cmd.extend(["--batch_size", str(row_cfg["batch_size"])])

    extra = _csv_arg(row_cfg.get("extra_dataset_paths"))
    if extra:
        cmd.extend(["--extra_dataset_paths", extra])

    include_variants = _csv_arg(row_cfg.get("include_variants"))
    if include_variants:
        cmd.extend(["--include_variants", include_variants])

    exclude_variants = _csv_arg(row_cfg.get("exclude_variants"))
    if exclude_variants:
        cmd.extend(["--exclude_variants", exclude_variants])

    guardrail_model = row_cfg.get("guardrail_model")
    guardrail_id = row_cfg.get("guardrail_profile_id") or row_cfg.get("guardrail_id")
    if guardrail_id:
        cmd.extend(["--guardrail_id", str(guardrail_id)])
    if guardrail_model:
        cmd.extend(["--guardrail_model", str(guardrail_model)])
        cmd.extend(["--guardrail_provider", str(row_cfg.get("guardrail_provider") or "local_hf")])

    guardrail_prompt = row_cfg.get("guardrail_prompt")
    if guardrail_prompt:
        cmd.extend(["--guardrail_prompt", str(guardrail_prompt)])
    guardrail_prompt_name = row_cfg.get("guardrail_prompt_name")
    if guardrail_prompt_name:
        cmd.extend(["--guardrail_prompt_name", str(guardrail_prompt_name)])

    judge_prompt = row_cfg.get("safety_judge_prompt")
    if judge_prompt:
        cmd.extend(["--safety_judge_prompt", str(judge_prompt)])
    judge_prompt_name = row_cfg.get("safety_judge_prompt_name")
    if judge_prompt_name:
        cmd.extend(["--safety_judge_prompt_name", str(judge_prompt_name)])

    config_payload = dict(row_cfg.get("config") or {})
    for key in [
        "guardrail_input_contract",
        "guardrail_adapter",
        "guardrail_access_mode",
        "guardrail_profile_id",
    ]:
        if row_cfg.get(key) is not None:
            config_payload.setdefault(key, row_cfg.get(key))
    if config_payload:
        cmd.extend(["--config", json.dumps(config_payload)])

    return cmd


def _write_manifest(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run guardrail x mutation matrix evaluations")
    parser.add_argument("--matrix_config", required=True, help="Path to YAML matrix config")
    parser.add_argument(
        "--output_root",
        default="runs/matrix",
        help="Directory where matrix manifests and aggregate artifacts are written",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Optional matrix run name (defaults to config stem + UTC timestamp)",
    )
    parser.add_argument(
        "--no_aggregate",
        action="store_true",
        help="Skip post-run matrix aggregation",
    )
    parser.add_argument(
        "--fail_fast",
        action="store_true",
        help="Stop after the first failed row",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    matrix_config_path = Path(args.matrix_config).resolve()
    repo_root = Path(__file__).resolve().parent
    run_eval_path = repo_root / "run_eval.py"

    if not matrix_config_path.exists():
        raise SystemExit(f"Matrix config not found: {matrix_config_path}")
    if not run_eval_path.exists():
        raise SystemExit(f"run_eval.py not found: {run_eval_path}")

    matrix_cfg = _load_matrix_config(matrix_config_path)
    global_cfg = matrix_cfg["global"]
    rows = matrix_cfg["rows"]

    run_name = args.name or f"{matrix_config_path.stem}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_root).resolve() / run_name
    rows_dir = output_dir / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.json"
    manifest: Dict[str, Any] = {
        "matrix_config_path": str(matrix_config_path),
        "created_utc": _utc_now(),
        "cwd": str(Path.cwd()),
        "repo_root": str(repo_root),
        "metric_definition_version": METRIC_DEFINITION_VERSION,
        "local_hf_judge_enforced": True,
        "default_local_judge_model": DEFAULT_LOCAL_JUDGE_MODEL,
        "global": global_cfg,
        "runs": [],
    }
    _write_manifest(manifest_path, manifest)

    any_failures = False

    for idx, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise RuntimeError(f"Row {idx} in matrix config is not an object")

        row_id = str(row.get("id") or f"row_{idx}")
        row_label = str(row.get("label") or row_id)
        row_cfg = _merge_row(global_cfg, row)

        cmd = _build_run_eval_command(
            python_exe=sys.executable,
            run_eval_path=run_eval_path,
            row_cfg=row_cfg,
        )

        started = _utc_now()
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            text=True,
            capture_output=True,
        )
        ended = _utc_now()

        stdout_path = rows_dir / f"{row_id}.stdout.log"
        stderr_path = rows_dir / f"{row_id}.stderr.log"
        stdout_path.write_text(proc.stdout or "", encoding="utf-8")
        stderr_path.write_text(proc.stderr or "", encoding="utf-8")

        run_dir = _extract_run_dir(proc.stdout or "", proc.stderr or "")
        status = "success" if proc.returncode == 0 else "failed"
        if status == "failed":
            any_failures = True

        record = {
            "row_id": row_id,
            "row_label": row_label,
            "status": status,
            "return_code": proc.returncode,
            "started_utc": started,
            "ended_utc": ended,
            "command": cmd,
            "run_dir": run_dir,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "guardrail_profile_id": row_cfg.get("guardrail_profile_id"),
            "guardrail_model": row_cfg.get("guardrail_model"),
            "guardrail_provider": row_cfg.get("guardrail_provider"),
            "guardrail_input_contract": row_cfg.get("guardrail_input_contract"),
            "guardrail_adapter": row_cfg.get("guardrail_adapter"),
            "guardrail_access_mode": row_cfg.get("guardrail_access_mode"),
        }

        if proc.returncode != 0:
            record["error"] = (proc.stderr or proc.stdout or "").strip()[:4000]

        if row_cfg.get("safety_judge_provider") not in (None, "local_hf"):
            record["judge_provider_override"] = {
                "requested": row_cfg.get("safety_judge_provider"),
                "effective": "local_hf",
            }

        manifest["runs"].append(record)
        _write_manifest(manifest_path, manifest)

        print(f"[{idx}/{len(rows)}] {row_id}: {status}")

        if args.fail_fast and proc.returncode != 0:
            break

    if not args.no_aggregate:
        try:
            outputs = aggregate_from_manifest(manifest_path)
            manifest["aggregation"] = outputs
            print("Matrix aggregation complete.")
        except Exception as exc:  # pragma: no cover - surfaced via manifest and CLI
            manifest["aggregation_error"] = str(exc)
            print(f"Matrix aggregation skipped/failed: {exc}")

    manifest["completed_utc"] = _utc_now()
    _write_manifest(manifest_path, manifest)

    print(f"Matrix manifest: {manifest_path}")
    if any_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
