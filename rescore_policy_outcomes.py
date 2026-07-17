from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from assistant_policy import METRIC_DEFINITION_VERSION
from policy_replay import (
    apply_effective_policy_outcomes,
    copy_provenance,
    generate_run_artifacts,
    load_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rescore an existing run under effective policy outcomes")
    parser.add_argument("--input_run_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_run_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir == input_dir:
        raise SystemExit("Refusing to overwrite the historical input run")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Output directory must be empty or absent: {output_dir}")

    results = [
        apply_effective_policy_outcomes(deepcopy(row))
        for row in load_results(input_dir)
    ]
    copy_provenance(
        input_dir,
        output_dir,
        {
            "derived_from_run_dir": str(input_dir),
            "derivation": "deterministic_policy_rescore",
            "metric_definition_version": METRIC_DEFINITION_VERSION,
        },
    )
    summary = generate_run_artifacts(results, output_dir)
    print(
        "Policy rescore completed. "
        f"Output: {output_dir} | "
        f"TP={summary['scorecards']['system']['tp']} "
        f"FN={summary['scorecards']['system']['fn']} "
        f"TN={summary['scorecards']['system']['tn']} "
        f"FP={summary['scorecards']['system']['fp']}"
    )


if __name__ == "__main__":
    main()
