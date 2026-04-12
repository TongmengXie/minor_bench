import argparse
import csv
import json
import math
import random
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml


VARIANT_SUFFIXES = ("-original", "-typo", "-camouflage", "-adversarial")
T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
}


def _parse_csv_ints(text: str) -> List[int]:
    values = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(int(chunk))
    if not values:
        raise ValueError("At least one seed is required")
    return values


def _load_prompt_pack(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _prompt_family_key(item: Dict[str, Any]) -> str:
    metadata = item.get("metadata", {}) or {}
    base_prompt_id = metadata.get("base_prompt_id")
    if base_prompt_id is not None:
        return str(base_prompt_id)
    item_id = str(item.get("id") or "")
    if "-" in item_id:
        return item_id.split("-", 1)[0]
    return item_id or str(item.get("prompt_text") or "")


def _safe_family_key(item: Dict[str, Any]) -> str:
    item_id = str(item.get("id") or "")
    for suffix in VARIANT_SUFFIXES:
        if item_id.endswith(suffix):
            return item_id[: -len(suffix)]
    if "-" in item_id:
        return item_id.rsplit("-", 1)[0]
    return item_id or str(item.get("prompt_text") or "")


def _group_by_family(items: Iterable[Dict[str, Any]], key_fn) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for item in items:
        key = key_fn(item)
        groups.setdefault(key, []).append(item)
    return groups


def _sample_grouped_items(
    items: List[Dict[str, Any]],
    *,
    count: int,
    seed: int,
    key_fn,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    groups = _group_by_family(items, key_fn)
    keys = sorted(groups)
    if count > len(keys):
        raise ValueError(f"Requested {count} groups, but only {len(keys)} are available")
    rng = random.Random(seed)
    selected_keys = sorted(rng.sample(keys, count))
    sampled_items: List[Dict[str, Any]] = []
    for key in selected_keys:
        sampled_items.extend(groups[key])
    return selected_keys, sampled_items


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _build_matrix_config(
    *,
    prompt_pack_path: Path,
    safe_dataset_path: Path,
    model_name: str,
    provider: str,
    k: int,
    batch_size: int,
    system_prompt_names: List[str],
) -> Dict[str, Any]:
    rows = []
    for system_prompt_name in system_prompt_names:
        row_id = f"system_prompt_{system_prompt_name.replace('_tutor_v1', '').replace('_child', '')}"
        rows.append(
            {
                "id": row_id,
                "system_prompt_name": system_prompt_name,
            }
        )

    return {
        "global": {
            "model_name": model_name,
            "provider": provider,
            "prompt_pack_path": str(prompt_pack_path),
            "extra_dataset_paths": [str(safe_dataset_path)],
            "k": k,
            "batch_size": batch_size,
            "include_variants": [],
            "exclude_variants": [],
            "safety_judge_model": "meta-llama/Llama-3.2-1B-Instruct",
            "safety_judge_provider": "local_hf",
            "config": {
                "temperature": 0,
                "max_new_tokens": 96,
                "judge_temperature": 0,
                "judge_max_new_tokens": 96,
                "judge_batch_size": 2,
                "judge_retry_on_malformed": True,
                "judge_retry_max_attempts": 1,
            },
        },
        "rows": rows,
    }


def _run_matrix(config_path: Path, output_root: Path, run_name: str) -> Path:
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("run_matrix_eval.py")),
        "--matrix_config",
        str(config_path),
        "--output_root",
        str(output_root),
        "--name",
        run_name,
    ]
    subprocess.run(cmd, check=True)
    return output_root / run_name / "manifest.json"


def _parse_optional_float(value: str) -> float:
    return float(value) if value else math.nan


def _load_matrix_metrics(path: Path) -> Dict[Tuple[str, str], Dict[str, float]]:
    out: Dict[Tuple[str, str], Dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            out[(row["row_id"], row["variant"])] = {
                "ber": _parse_optional_float(row["ber"]),
                "f1_score": _parse_optional_float(row["f1_score"]),
            }
    return out


def _critical_value_95(n: int) -> float:
    if n <= 1:
        return 0.0
    df = n - 1
    if df in T_CRITICAL_95:
        return T_CRITICAL_95[df]
    return 1.96


def _summarize_metric(values: List[float]) -> Dict[str, float]:
    clean = [value for value in values if not math.isnan(value)]
    if not clean:
        return {"mean": math.nan, "half_width": math.nan, "n": 0}
    if len(clean) == 1:
        return {"mean": clean[0], "half_width": 0.0, "n": 1}
    mean = statistics.mean(clean)
    std = statistics.stdev(clean)
    half_width = _critical_value_95(len(clean)) * std / math.sqrt(len(clean))
    return {"mean": mean, "half_width": half_width, "n": len(clean)}


def _format_pct(mean: float, half_width: float, n: int, total_n: int) -> str:
    if math.isnan(mean):
        return "n/a"
    rendered = f"{mean:.2%} +/- {half_width * 100:.2f}pp"
    if n < total_n:
        rendered += f" (n={n}/{total_n})"
    return rendered


def _write_summary(
    *,
    output_path: Path,
    seeds: List[int],
    harmful_base_count: int,
    safe_base_count: int,
    k: int,
    by_cell: Dict[Tuple[str, str], Dict[str, Dict[str, float]]],
) -> None:
    row_ids = sorted({row_id for row_id, _ in by_cell})
    variants = sorted({variant for _, variant in by_cell})
    total_n = len(seeds)

    lines = [
        "# Small-Slice Stability Summary",
        "",
        f"- Seeds: {', '.join(str(seed) for seed in seeds)}",
        f"- Harmful base prompt families per seed: {harmful_base_count}",
        f"- Safe control families per seed: {safe_base_count}",
        f"- Generations per prompt row (`k`): {k}",
        "- Note: intervals below are approximate 95% t-intervals across slice seeds, not full benchmark confidence intervals.",
        "- If a cell shows `(n=x/3)`, one or more seeds produced an undefined metric for that cell and were excluded from that interval.",
        "",
        "## BER mean +/- interval by Row x Mutation",
        "| Row | " + " | ".join(variants) + " |",
        "| --- | " + " | ".join(["---"] * len(variants)) + " |",
    ]
    for row_id in row_ids:
        cells = []
        for variant in variants:
            stats = by_cell[(row_id, variant)]["ber"]
            cells.append(_format_pct(stats["mean"], stats["half_width"], stats["n"], total_n))
        lines.append(f"| {row_id} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## F1 mean +/- interval by Row x Mutation",
            "| Row | " + " | ".join(variants) + " |",
            "| --- | " + " | ".join(["---"] * len(variants)) + " |",
        ]
    )
    for row_id in row_ids:
        cells = []
        for variant in variants:
            stats = by_cell[(row_id, variant)]["f1_score"]
            cells.append(_format_pct(stats["mean"], stats["half_width"], stats["n"], total_n))
        lines.append(f"| {row_id} | " + " | ".join(cells) + " |")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_stats_csv(
    path: Path,
    by_cell: Dict[Tuple[str, str], Dict[str, Dict[str, float]]],
) -> None:
    fieldnames = [
        "row_id",
        "variant",
        "ber_mean",
        "ber_half_width",
        "ber_n",
        "f1_mean",
        "f1_half_width",
        "f1_n",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for (row_id, variant), metrics in sorted(by_cell.items()):
            writer.writerow(
                {
                    "row_id": row_id,
                    "variant": variant,
                    "ber_mean": metrics["ber"]["mean"],
                    "ber_half_width": metrics["ber"]["half_width"],
                    "ber_n": metrics["ber"]["n"],
                    "f1_mean": metrics["f1_score"]["mean"],
                    "f1_half_width": metrics["f1_score"]["half_width"],
                    "f1_n": metrics["f1_score"]["n"],
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small-slice 3-seed stability experiment")
    parser.add_argument("--prompt_pack_path", default="prompt_packs/minorbench_hateful.json")
    parser.add_argument("--safe_dataset_path", default="datasets/true_negative_safe_sensitive_variants.jsonl")
    parser.add_argument("--model_name", default="llama-3.1-8b-instruct")
    parser.add_argument("--provider", default="scaleway")
    parser.add_argument("--system_prompt_names", default="generic_child_tutor_v1,duolingo_tutor_v1")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--harmful_base_count", type=int, default=6)
    parser.add_argument("--safe_base_count", type=int, default=4)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_root", default="runs/stability")
    parser.add_argument("--name", default="hateful_smallslice_system_prompt_stability")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt_pack_path = Path(args.prompt_pack_path).resolve()
    safe_dataset_path = Path(args.safe_dataset_path).resolve()
    output_root = Path(args.output_root).resolve() / args.name
    inputs_root = output_root / "inputs"
    matrix_root = output_root / "matrix_runs"
    inputs_root.mkdir(parents=True, exist_ok=True)
    matrix_root.mkdir(parents=True, exist_ok=True)

    seeds = _parse_csv_ints(args.seeds)
    system_prompt_names = [name.strip() for name in args.system_prompt_names.split(",") if name.strip()]

    prompt_pack = _load_prompt_pack(prompt_pack_path)
    harmful_items = list(prompt_pack.get("items") or [])
    safe_rows = _load_jsonl(safe_dataset_path)

    per_seed_metrics: Dict[int, Dict[Tuple[str, str], Dict[str, float]]] = {}
    experiment_manifest: Dict[str, Any] = {
        "prompt_pack_path": str(prompt_pack_path),
        "safe_dataset_path": str(safe_dataset_path),
        "seeds": seeds,
        "harmful_base_count": args.harmful_base_count,
        "safe_base_count": args.safe_base_count,
        "k": args.k,
        "batch_size": args.batch_size,
        "system_prompt_names": system_prompt_names,
        "runs": [],
    }

    for seed in seeds:
        seed_dir = inputs_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        harmful_keys, sampled_harmful = _sample_grouped_items(
            harmful_items,
            count=args.harmful_base_count,
            seed=seed,
            key_fn=_prompt_family_key,
        )
        safe_keys, sampled_safe = _sample_grouped_items(
            safe_rows,
            count=args.safe_base_count,
            seed=seed + 10_000,
            key_fn=_safe_family_key,
        )

        slice_prompt_pack = dict(prompt_pack)
        slice_prompt_pack["name"] = f"{prompt_pack.get('name', 'prompt_pack')}_slice_seed_{seed}"
        slice_prompt_pack["items"] = sampled_harmful

        prompt_pack_out = seed_dir / "prompt_pack.json"
        safe_dataset_out = seed_dir / "safe_controls.jsonl"
        matrix_config_out = seed_dir / "matrix_config.yaml"

        _write_json(prompt_pack_out, slice_prompt_pack)
        _write_jsonl(safe_dataset_out, sampled_safe)
        matrix_config = _build_matrix_config(
            prompt_pack_path=prompt_pack_out,
            safe_dataset_path=safe_dataset_out,
            model_name=args.model_name,
            provider=args.provider,
            k=args.k,
            batch_size=args.batch_size,
            system_prompt_names=system_prompt_names,
        )
        matrix_config_out.write_text(yaml.safe_dump(matrix_config, sort_keys=False), encoding="utf-8")

        manifest_path = _run_matrix(matrix_config_out, matrix_root, f"seed_{seed}")
        matrix_metrics_path = manifest_path.parent / "matrix_metrics.csv"
        per_seed_metrics[seed] = _load_matrix_metrics(matrix_metrics_path)
        experiment_manifest["runs"].append(
            {
                "seed": seed,
                "harmful_family_keys": harmful_keys,
                "safe_family_keys": safe_keys,
                "matrix_manifest": str(manifest_path),
                "matrix_metrics_csv": str(matrix_metrics_path),
            }
        )

    all_keys = sorted({key for metrics in per_seed_metrics.values() for key in metrics})
    by_cell: Dict[Tuple[str, str], Dict[str, Dict[str, float]]] = {}
    for key in all_keys:
        metric_values: Dict[str, List[float]] = {"ber": [], "f1_score": []}
        for seed in seeds:
            row = per_seed_metrics.get(seed, {}).get(key)
            if row is None:
                metric_values["ber"].append(math.nan)
                metric_values["f1_score"].append(math.nan)
                continue
            metric_values["ber"].append(row["ber"])
            metric_values["f1_score"].append(row["f1_score"])
        by_cell[key] = {
            "ber": _summarize_metric(metric_values["ber"]),
            "f1_score": _summarize_metric(metric_values["f1_score"]),
        }

    summary_md_path = output_root / "stability_summary.md"
    summary_csv_path = output_root / "stability_stats.csv"
    experiment_manifest_path = output_root / "manifest.json"
    _write_summary(
        output_path=summary_md_path,
        seeds=seeds,
        harmful_base_count=args.harmful_base_count,
        safe_base_count=args.safe_base_count,
        k=args.k,
        by_cell=by_cell,
    )
    _write_stats_csv(summary_csv_path, by_cell)
    experiment_manifest["stability_summary_md"] = str(summary_md_path)
    experiment_manifest["stability_stats_csv"] = str(summary_csv_path)
    _write_json(experiment_manifest_path, experiment_manifest)

    print(json.dumps({"manifest": str(experiment_manifest_path), "summary_md": str(summary_md_path)}, indent=2))


if __name__ == "__main__":
    main()
