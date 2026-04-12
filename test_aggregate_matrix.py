import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aggregate_matrix import aggregate_from_manifest


def _write_variant_metrics(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "variant",
                "false_positive_rate",
                "false_negative_rate",
                "f1_score",
                "precision",
                "recall",
                "false_positives",
                "false_negatives",
                "true_positives",
                "true_negatives",
                "safe_support",
                "harmful_support",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "variant": "original",
                "false_positive_rate": "0.2",
                "false_negative_rate": "0.4",
                "f1_score": "0.5",
                "precision": "0.5",
                "recall": "0.5",
                "false_positives": "1",
                "false_negatives": "2",
                "true_positives": "2",
                "true_negatives": "3",
                "safe_support": "4",
                "harmful_support": "4",
            }
        )


def test_aggregate_from_manifest_writes_expected_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "eval_run"
    run_dir.mkdir(parents=True)

    summary = {
        "scorecards": {
            "system": {"f1_score": 0.5},
            "tutor_conditional": {"f1_score": 0.6},
        },
        "guardrail_metrics": [{"allow_rate": 0.75}],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    _write_variant_metrics(run_dir / "variant_metrics.csv")

    manifest = {
        "runs": [
            {
                "row_id": "no_guardrail_generic",
                "row_label": "No guardrail + generic",
                "status": "success",
                "run_dir": str(run_dir),
            }
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    outputs = aggregate_from_manifest(manifest_path)

    for output_path in outputs.values():
        assert Path(output_path).exists()

    ber_csv = Path(outputs["matrix_ber_csv"]).read_text(encoding="utf-8")
    assert "no_guardrail_generic" in ber_csv
    assert "0.300000" in ber_csv

    report_md = Path(outputs["matrix_report_md"]).read_text(encoding="utf-8")
    assert "BER by Row x Mutation" in report_md
    assert "F1 by Row x Mutation" in report_md
    assert "BER (Balanced Error Rate) = (false positive rate + false negative rate) / 2." in report_md
