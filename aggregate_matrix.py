import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _parse_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_summary(run_dir: Path) -> Dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.json in {run_dir}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _load_variant_metrics(run_dir: Path, summary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    csv_path = run_dir / "variant_metrics.csv"
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            out: Dict[str, Dict[str, Any]] = {}
            for row in reader:
                variant = str(row.get("variant") or "unknown")
                out[variant] = {
                    "false_positive_rate": _parse_optional_float(row.get("false_positive_rate")),
                    "false_negative_rate": _parse_optional_float(row.get("false_negative_rate")),
                    "f1_score": _parse_optional_float(row.get("f1_score")),
                    "precision": _parse_optional_float(row.get("precision")),
                    "recall": _parse_optional_float(row.get("recall")),
                    "false_positives": _parse_optional_int(row.get("false_positives")) or 0,
                    "false_negatives": _parse_optional_int(row.get("false_negatives")) or 0,
                    "true_positives": _parse_optional_int(row.get("true_positives")) or 0,
                    "true_negatives": _parse_optional_int(row.get("true_negatives")) or 0,
                    "safe_support": _parse_optional_int(row.get("safe_support")) or 0,
                    "harmful_support": _parse_optional_int(row.get("harmful_support")) or 0,
                }
            return out

    # Fallback if CSV is missing.
    fallback = summary.get("per_variant_metrics") or {}
    out = {}
    for variant, metrics in fallback.items():
        out[str(variant)] = dict(metrics)
    return out


def _compute_ber(false_positive_rate: Optional[float], false_negative_rate: Optional[float]) -> Optional[float]:
    if false_positive_rate is None or false_negative_rate is None:
        return None
    return (false_positive_rate + false_negative_rate) / 2.0


def _format_value(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def _format_md_value(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2%}"


def _allow_rate(summary: Dict[str, Any]) -> Optional[float]:
    guardrail_metrics = summary.get("guardrail_metrics") or []
    if not guardrail_metrics:
        return None
    values = [
        _parse_optional_float(row.get("allow_rate"))
        for row in guardrail_metrics
        if _parse_optional_float(row.get("allow_rate")) is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)


def _collect_records(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for run in manifest.get("runs", []):
        if run.get("status") != "success":
            continue

        run_dir = Path(str(run.get("run_dir") or ""))
        if not run_dir.exists():
            continue

        summary = _load_summary(run_dir)
        variant_metrics = _load_variant_metrics(run_dir, summary)
        allow_rate = _allow_rate(summary)
        system_card = (summary.get("scorecards") or {}).get("system") or {}
        tutor_card = (summary.get("scorecards") or {}).get("tutor_conditional") or {}

        for variant, metrics in variant_metrics.items():
            fpr = _parse_optional_float(metrics.get("false_positive_rate"))
            fnr = _parse_optional_float(metrics.get("false_negative_rate"))
            record = {
                "row_id": run.get("row_id"),
                "row_label": run.get("row_label") or run.get("row_id"),
                "variant": variant,
                "ber": _compute_ber(fpr, fnr),
                "f1_score": _parse_optional_float(metrics.get("f1_score")),
                "precision": _parse_optional_float(metrics.get("precision")),
                "recall": _parse_optional_float(metrics.get("recall")),
                "false_positive_rate": fpr,
                "false_negative_rate": fnr,
                "false_positives": _parse_optional_int(metrics.get("false_positives")) or 0,
                "false_negatives": _parse_optional_int(metrics.get("false_negatives")) or 0,
                "true_positives": _parse_optional_int(metrics.get("true_positives")) or 0,
                "true_negatives": _parse_optional_int(metrics.get("true_negatives")) or 0,
                "safe_support": _parse_optional_int(metrics.get("safe_support")) or 0,
                "harmful_support": _parse_optional_int(metrics.get("harmful_support")) or 0,
                "allow_rate": allow_rate,
                "system_f1": _parse_optional_float(system_card.get("f1_score")),
                "tutor_conditional_f1": _parse_optional_float(tutor_card.get("f1_score")),
                "run_dir": str(run_dir),
            }
            records.append(record)

    records.sort(key=lambda row: (str(row["row_id"]), str(row["variant"])))
    return records


def _write_long_csv(records: List[Dict[str, Any]], output_dir: Path) -> Path:
    path = output_dir / "matrix_metrics.csv"
    headers = [
        "row_id",
        "row_label",
        "variant",
        "ber",
        "f1_score",
        "precision",
        "recall",
        "false_positive_rate",
        "false_negative_rate",
        "false_positives",
        "false_negatives",
        "true_positives",
        "true_negatives",
        "safe_support",
        "harmful_support",
        "allow_rate",
        "system_f1",
        "tutor_conditional_f1",
        "run_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for record in records:
            row = dict(record)
            for key in [
                "ber",
                "f1_score",
                "precision",
                "recall",
                "false_positive_rate",
                "false_negative_rate",
                "allow_rate",
                "system_f1",
                "tutor_conditional_f1",
            ]:
                row[key] = _format_value(_parse_optional_float(row.get(key)))
            writer.writerow(row)
    return path


def _pivot(records: List[Dict[str, Any]], metric: str) -> Tuple[List[str], List[str], Dict[Tuple[str, str], Optional[float]]]:
    row_ids = []
    variants = set()
    values: Dict[Tuple[str, str], Optional[float]] = {}

    seen = set()
    for record in records:
        row_id = str(record["row_id"])
        variant = str(record["variant"])
        if row_id not in seen:
            seen.add(row_id)
            row_ids.append(row_id)
        variants.add(variant)
        values[(row_id, variant)] = _parse_optional_float(record.get(metric))

    return row_ids, sorted(variants), values


def _write_pivot_csv(records: List[Dict[str, Any]], metric: str, output_dir: Path) -> Path:
    row_ids, variants, values = _pivot(records, metric)
    path = output_dir / f"matrix_{metric}.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", *variants])
        for row_id in row_ids:
            writer.writerow([row_id, *[_format_value(values.get((row_id, v))) for v in variants]])
    return path


def _write_markdown(records: List[Dict[str, Any]], output_dir: Path) -> Path:
    lines = [
        "# Matrix Metrics",
        "",
        "## Metric notes",
        "- BER (Balanced Error Rate) = (false positive rate + false negative rate) / 2.",
        "- Lower BER is better. `0%` is perfect; `50%` means the system is wrong half the time on average across the safe and harmful classes.",
        "- F1 is the harmonic mean of precision and recall for the `unsafe` class. Higher is better.",
        "- Rows are evaluation configurations (for example system prompts or guardrails). Columns are prompt mutations.",
        "",
    ]

    for metric, title in [("ber", "BER"), ("f1_score", "F1")]:
        row_ids, variants, values = _pivot(records, metric)
        lines.append(f"## {title} by Row x Mutation")
        lines.append("| Row | " + " | ".join(variants) + " |")
        lines.append("| --- | " + " | ".join(["---"] * len(variants)) + " |")
        for row_id in row_ids:
            cells = [_format_md_value(values.get((row_id, variant))) for variant in variants]
            lines.append(f"| {row_id} | " + " | ".join(cells) + " |")
        lines.append("")

    path = output_dir / "matrix_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def aggregate_from_manifest(manifest_path: Path, output_dir: Optional[Path] = None) -> Dict[str, str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out_dir = output_dir or manifest_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    records = _collect_records(manifest)
    if not records:
        raise RuntimeError("No successful run records available for matrix aggregation.")

    long_csv = _write_long_csv(records, out_dir)
    ber_csv = _write_pivot_csv(records, "ber", out_dir)
    f1_csv = _write_pivot_csv(records, "f1_score", out_dir)
    report_md = _write_markdown(records, out_dir)

    return {
        "matrix_metrics_csv": str(long_csv),
        "matrix_ber_csv": str(ber_csv),
        "matrix_f1_csv": str(f1_csv),
        "matrix_report_md": str(report_md),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate row x mutation eval matrix metrics")
    parser.add_argument("--manifest", required=True, help="Path to matrix manifest.json")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional output directory for matrix artifacts (defaults to manifest dir)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir) if args.output_dir else None
    outputs = aggregate_from_manifest(manifest_path, output_dir)
    print("Matrix aggregation complete:")
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
