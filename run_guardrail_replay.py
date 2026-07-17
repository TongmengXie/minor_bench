from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

from assistant_policy import METRIC_DEFINITION_VERSION
from guardrails import merge_guardrail_profile
from model_runner import ModelRunner
from policy_replay import (
    apply_effective_policy_outcomes,
    copy_provenance,
    generate_run_artifacts,
    load_results,
    write_paired_reports,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a guardrail over existing baseline generations")
    parser.add_argument("--baseline_run_dir", required=True)
    parser.add_argument("--guardrail_id", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--decision_run_dir",
        default=None,
        help="Optional validated guardrail run whose prompt-level decisions should be reused.",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--bootstrap_replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap_seed", type=int, default=1)
    return parser.parse_args()


def _baseline_meta(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "meta.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _row_key(row: Dict[str, Any]) -> tuple[str, str, str, str]:
    metadata = row.get("metadata") or {}
    return (
        str(row.get("dataset_source") or metadata.get("dataset") or "unknown"),
        str(metadata.get("base_prompt_id") or metadata.get("original_id") or ""),
        str(metadata.get("variant") or "unknown"),
        str(row.get("prompt_text") or ""),
    )


def _load_recorded_decisions(decision_run_dir: Path, baseline_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    recorded: Dict[tuple[str, str, str, str], Dict[str, Any]] = {}
    for row in load_results(decision_run_dir):
        decisions = row.get("guardrail_decisions") or []
        statuses = {
            str(decision.get("status") or "").strip().lower()
            for decision in decisions
            if isinstance(decision, dict)
        }
        if len(statuses) != 1:
            raise ValueError(f"Recorded guardrail row does not have one stable decision: {_row_key(row)}")
        status = next(iter(statuses))
        generation = (row.get("generations") or [{}])[0]
        raw = generation.get("raw") if isinstance(generation, dict) else {}
        raw = raw if isinstance(raw, dict) else {}
        recorded[_row_key(row)] = {
            "status": status,
            "reason": raw.get("guardrail_reason"),
            "mode": raw.get("guardrail_decision_mode") or raw.get("guardrail_parser_mode"),
            "guardrail_latency": raw.get("guardrail_latency"),
        }
    missing = [_row_key(row) for row in baseline_results if _row_key(row) not in recorded]
    if missing:
        raise ValueError(f"Decision run is missing {len(missing)} baseline prompt rows")
    return [recorded[_row_key(row)] for row in baseline_results]


def _decision_metadata(runner: ModelRunner, decision: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "guardrail_decision": (
            f"BLOCK: {decision.get('reason') or 'Blocked by guardrail'}"
            if decision["status"] == "block"
            else str(decision["status"]).upper()
        ),
        "guardrail_reason": decision.get("reason"),
        "guardrail_model": runner.guardrail_model,
        "guardrail_provider": runner.guardrail_provider,
        "guardrail_profile_id": runner.guardrail_profile_id,
        "guardrail_adapter": runner.guardrail_adapter,
        "guardrail_input_contract": runner.guardrail_input_contract,
        "guardrail_access_mode": runner.guardrail_access_mode,
        "guardrail_decision_status": decision["status"],
        "guardrail_decision_mode": decision.get("mode"),
        "guardrail_latency": decision.get("guardrail_latency"),
        "guardrail_replay": True,
    }


def _apply_decision(row: Dict[str, Any], runner: ModelRunner, decision: Dict[str, Any]) -> Dict[str, Any]:
    replay = deepcopy(row)
    replay["baseline_assistant_policy_labels"] = list(row.get("assistant_policy_labels") or [])
    replay["baseline_response_safety_labels"] = list(row.get("response_safety_labels") or [])
    replay["baseline_answer_mode_labels"] = list(row.get("answer_mode_labels") or [])
    replay["baseline_generations"] = deepcopy(row.get("generations") or [])
    replay["guardrail_model"] = runner.guardrail_model
    replay["guardrail_provider"] = runner.guardrail_provider
    replay["guardrail_profile_id"] = runner.guardrail_profile_id
    replay["guardrail_adapter"] = runner.guardrail_adapter
    replay["guardrail_input_contract"] = runner.guardrail_input_contract
    replay["guardrail_access_mode"] = runner.guardrail_access_mode

    metadata = _decision_metadata(runner, decision)
    replay["guardrail_decisions"] = []
    replay_generations: List[Dict[str, Any]] = []
    for index, generation in enumerate(row.get("generations") or []):
        generation_copy = deepcopy(generation)
        raw = generation_copy.get("raw") if isinstance(generation_copy, dict) else {}
        raw = dict(raw) if isinstance(raw, dict) else {}
        raw.update(metadata)
        if decision["status"] == "block":
            generation_copy["text"] = runner.refusal_text
            raw["completion"] = runner.refusal_text
            raw["finish_reason"] = "guardrail_blocked"
        generation_copy["raw"] = raw
        replay_generations.append(generation_copy)
        replay["guardrail_decisions"].append(
            {
                "index": index,
                "status": decision["status"],
                "guardrail_model": runner.guardrail_model,
                "guardrail_provider": runner.guardrail_provider,
                "guardrail_profile_id": runner.guardrail_profile_id,
                "guardrail_input_contract": runner.guardrail_input_contract,
                "guardrail_adapter": runner.guardrail_adapter,
                "guardrail_access_mode": runner.guardrail_access_mode,
            }
        )
    replay["generations"] = replay_generations
    return apply_effective_policy_outcomes(replay)


def main() -> None:
    args = parse_args()
    baseline_dir = Path(args.baseline_run_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Output directory must be empty or absent: {output_dir}")

    baseline_results = load_results(baseline_dir)
    meta = _baseline_meta(baseline_dir)
    model_name = str(meta.get("model_name") or baseline_results[0].get("model_name") or "replay-only")
    provider = str((meta.get("main_model_preflight") or {}).get("provider") or "auto")
    config = merge_guardrail_profile({}, args.guardrail_id)
    runner = ModelRunner(model_name, provider, config=config)

    decisions: List[Dict[str, Any]] = []
    decision_run_dir = Path(args.decision_run_dir).resolve() if args.decision_run_dir else None
    if decision_run_dir:
        decisions = _load_recorded_decisions(decision_run_dir, baseline_results)
    else:
        for start in range(0, len(baseline_results), args.batch_size):
            chunk = baseline_results[start : start + args.batch_size]
            prompts = [str(row.get("prompt_text") or "") for row in chunk]
            system_prompt = str(chunk[0].get("system_prompt") or "")
            decisions.extend(runner.probe_guardrail_batch(system_prompt, prompts))
    if len(decisions) != len(baseline_results):
        raise RuntimeError("Guardrail replay did not return one decision per prompt row")

    replay_results = [
        _apply_decision(row, runner, decision)
        for row, decision in zip(baseline_results, decisions)
    ]
    copy_provenance(
        baseline_dir,
        output_dir,
        {
            "derived_from_run_dir": str(baseline_dir),
            "derivation": "controlled_guardrail_replay",
            "guardrail_profile_id": runner.guardrail_profile_id,
            "guardrail_model": runner.guardrail_model,
            "guardrail_provider": runner.guardrail_provider,
            "guardrail_prompt_name": runner.guardrail_prompt_name,
            "guardrail_input_contract": runner.guardrail_input_contract,
            "guardrail_adapter": runner.guardrail_adapter,
            "guardrail_access_mode": runner.guardrail_access_mode,
            "metric_definition_version": METRIC_DEFINITION_VERSION,
            "bootstrap_replicates": args.bootstrap_replicates,
            "bootstrap_seed": args.bootstrap_seed,
            "guardrail_decision_run_dir": str(decision_run_dir) if decision_run_dir else None,
        },
    )
    generate_run_artifacts(replay_results, output_dir)
    write_paired_reports(
        replay_results,
        output_dir,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(f"Controlled guardrail replay completed. Output: {output_dir}")


if __name__ == "__main__":
    main()
