import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List


def load_results(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify guardrail / assistant-policy judge artifacts")
    parser.add_argument("path", help="Path to results.jsonl or run directory")
    args = parser.parse_args()

    input_path = Path(args.path)
    if input_path.is_dir():
        results_path = input_path / "results.jsonl"
        summary_path = input_path / "summary.json"
    else:
        results_path = input_path
        summary_path = input_path.parent / "summary.json"

    if not results_path.exists():
        raise SystemExit(f"Missing results.jsonl at {results_path}")

    rows = load_results(results_path)
    status_counts = Counter()
    finish_counts = Counter()
    judge_seen = False

    for row in rows:
        if (
            "assistant_policy_labels" in row
            or "response_safety_labels" in row
            or "judge_failure_reasons" in row
        ):
            judge_seen = True
        for gen in row.get("generations", []) or []:
            raw = gen.get("raw") if isinstance(gen, dict) else {}
            if isinstance(raw, dict):
                status_counts.update([raw.get("guardrail_decision_status")])
                finish_counts.update([raw.get("finish_reason")])

    print("guardrail_decision_status:", dict(status_counts))
    print("finish_reason:", dict(finish_counts))
    print("structured_judge_fields_present:", judge_seen)

    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        print("summary.policy_version:", summary.get("policy_version"))
        print("summary.metric_definition_version:", summary.get("metric_definition_version"))
        print("summary.aggregation_unit:", summary.get("aggregation_unit"))
        print("summary.overall_success_rate:", summary.get("overall_success_rate"))
        print("summary.harmful_prompt_success_rate:", summary.get("harmful_prompt_success_rate"))
        print("summary.safe_prompt_success_rate:", summary.get("safe_prompt_success_rate"))
        print("summary.safe_prompt_overrefusal_rate:", summary.get("safe_prompt_overrefusal_rate"))
        print(
            "summary.harmful_prompt_unsafe_answer_rate:",
            summary.get("harmful_prompt_unsafe_answer_rate"),
        )

        coverage = summary.get("coverage") or {}
        if coverage:
            print(
                "coverage:",
                {
                    "labeled_generation_count": coverage.get("labeled_generation_count"),
                    "labeled_generation_rate": coverage.get("labeled_generation_rate"),
                    "judge_valid_generation_count": coverage.get("judge_valid_generation_count"),
                    "judge_valid_generation_rate": coverage.get("judge_valid_generation_rate"),
                    "judge_malformed_generation_count": coverage.get("judge_malformed_generation_count"),
                    "judge_malformed_generation_rate": coverage.get("judge_malformed_generation_rate"),
                    "tutor_evaluable_generation_count": coverage.get("tutor_evaluable_generation_count"),
                    "tutor_evaluable_generation_rate": coverage.get("tutor_evaluable_generation_rate"),
                },
            )

        scorecards = summary.get("scorecards") or {}
        if scorecards:
            print("scorecards.system:", scorecards.get("system"))
            print("scorecards.tutor_conditional:", scorecards.get("tutor_conditional"))

        warnings = summary.get("warnings") or []
        if warnings:
            print("warnings:", warnings)


if __name__ == "__main__":
    main()
