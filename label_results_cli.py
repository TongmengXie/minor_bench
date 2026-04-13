import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


CSV_COLUMNS = [
    "audit_id",
    "source_results_path",
    "row_index",
    "dataset_source",
    "prompt_pack_name",
    "prompt_pack_subset",
    "base_prompt_id",
    "variant",
    "generation_index",
    "is_harmful",
    "label_source",
    "assistant_policy_label",
    "response_safety_label",
    "answer_mode",
    "judge_label",
    "judge_malformed",
    "judge_parse_method",
    "judge_failure_reasons",
    "prompt_text",
    "assistant_response",
    "system_prompt",
    "human_label",
    "human_notes",
    "reviewed_at_utc",
]

LABEL_ALIASES = {
    "s": "safe",
    "safe": "safe",
    "u": "unsafe",
    "unsafe": "unsafe",
    "c": "unclear",
    "unclear": "unclear",
}


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def _stringify_reasons(reasons: object) -> str:
    if isinstance(reasons, list):
        return " | ".join(str(item) for item in reasons)
    if reasons is None:
        return ""
    return str(reasons)


def _make_audit_id(
    source_results_path: str,
    row_index: int,
    base_prompt_id: str,
    variant: str,
    generation_index: int,
    prompt_text: str,
    assistant_response: str,
) -> str:
    digest = hashlib.sha1(
        f"{source_results_path}|{row_index}|{base_prompt_id}|{variant}|{generation_index}|{prompt_text}|{assistant_response}".encode(
            "utf-8"
        )
    ).hexdigest()[:12]
    return f"{base_prompt_id or 'row'}:{variant or 'unknown'}:g{generation_index}:{digest}"


def build_audit_rows(results_path: Path) -> list[dict]:
    rows = _load_jsonl(results_path)
    audit_rows: list[dict] = []
    source_results_path = str(results_path)
    for row_index, row in enumerate(rows):
        metadata = row.get("metadata") or {}
        generations = row.get("generations") or []
        judge_labels = row.get("judge_labels") or []
        response_safety_labels = row.get("response_safety_labels") or judge_labels
        assistant_policy_labels = row.get("assistant_policy_labels") or []
        answer_mode_labels = row.get("answer_mode_labels") or []
        judge_malformed = row.get("judge_malformed") or []
        judge_parse_method = row.get("judge_parse_method") or []
        judge_failure_reasons = row.get("judge_failure_reasons") or []
        base_prompt_id = str(metadata.get("base_prompt_id") or "")
        variant = str(metadata.get("variant") or "")
        for generation_index, generation in enumerate(generations):
            assistant_response = generation.get("text") or ""
            audit_id = _make_audit_id(
                source_results_path=source_results_path,
                row_index=row_index,
                base_prompt_id=base_prompt_id,
                variant=variant,
                generation_index=generation_index,
                prompt_text=row.get("prompt_text") or "",
                assistant_response=assistant_response,
            )
            audit_rows.append(
                {
                    "audit_id": audit_id,
                    "source_results_path": source_results_path,
                    "row_index": row_index,
                    "dataset_source": row.get("dataset_source") or "",
                    "prompt_pack_name": row.get("prompt_pack_name") or "",
                    "prompt_pack_subset": row.get("prompt_pack_subset") or "",
                    "base_prompt_id": base_prompt_id,
                    "variant": variant,
                    "generation_index": generation_index,
                    "is_harmful": row.get("is_harmful"),
                    "label_source": row.get("label_source") or "",
                    "assistant_policy_label": assistant_policy_labels[generation_index]
                    if generation_index < len(assistant_policy_labels)
                    else "",
                    "response_safety_label": response_safety_labels[generation_index]
                    if generation_index < len(response_safety_labels)
                    else "",
                    "answer_mode": answer_mode_labels[generation_index]
                    if generation_index < len(answer_mode_labels)
                    else "",
                    "judge_label": judge_labels[generation_index] if generation_index < len(judge_labels) else "",
                    "judge_malformed": judge_malformed[generation_index]
                    if generation_index < len(judge_malformed)
                    else "",
                    "judge_parse_method": judge_parse_method[generation_index]
                    if generation_index < len(judge_parse_method)
                    else "",
                    "judge_failure_reasons": _stringify_reasons(
                        judge_failure_reasons[generation_index]
                        if generation_index < len(judge_failure_reasons)
                        else ""
                    ),
                    "prompt_text": row.get("prompt_text") or "",
                    "assistant_response": assistant_response,
                    "system_prompt": row.get("system_prompt") or "",
                    "human_label": "",
                    "human_notes": "",
                    "reviewed_at_utc": "",
                }
            )
    return audit_rows


def load_existing_labels(csv_path: Path) -> dict[str, dict]:
    if not csv_path.exists():
        return {}
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return {row["audit_id"]: row for row in reader if row.get("audit_id")}


def merge_existing_labels(audit_rows: list[dict], existing_rows: dict[str, dict]) -> list[dict]:
    merged: list[dict] = []
    for row in audit_rows:
        existing = existing_rows.get(row["audit_id"])
        if existing:
            row["human_label"] = existing.get("human_label", "")
            row["human_notes"] = existing.get("human_notes", "")
            row["reviewed_at_utc"] = existing.get("reviewed_at_utc", "")
        merged.append(row)
    return merged


def write_audit_csv(csv_path: Path, audit_rows: list[dict]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in audit_rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})


def _default_output_csv(results_path: Path) -> Path:
    return results_path.with_name(f"{results_path.stem}_human_audit.csv")


def _print_record(record: dict, current: int, total: int) -> None:
    print("")
    print("=" * 80)
    print(f"Item {current}/{total}")
    print(f"audit_id: {record['audit_id']}")
    print(
        "base_prompt_id: {base_prompt_id} | variant: {variant} | generation: {generation_index}".format(
            **record
        )
    )
    print(
        "dataset_source: {dataset_source} | is_harmful: {is_harmful} | assistant_policy_label: {assistant_policy_label} | response_safety_label: {response_safety_label}".format(
            **record
        )
    )
    if record.get("judge_failure_reasons"):
        print(f"judge_reasons: {record['judge_failure_reasons']}")
    print("")
    print("Prompt:")
    print(record["prompt_text"])
    print("")
    print("Assistant response:")
    print(record["assistant_response"])


def run_labeling_loop(csv_path: Path, audit_rows: list[dict]) -> None:
    pending = [row for row in audit_rows if not row.get("human_label")]
    total = len(pending)
    if total == 0:
        print(f"All rows already labeled in {csv_path}.")
        return
    for index, record in enumerate(pending, start=1):
        _print_record(record, current=index, total=total)
        while True:
            raw = input("\nLabel [s]afe / [u]nsafe / [c]unclear / [k]skip / [q]quit: ").strip().lower()
            if raw in {"k", "skip"}:
                break
            if raw in {"q", "quit"}:
                write_audit_csv(csv_path, audit_rows)
                print(f"Saved progress to {csv_path}.")
                return
            label = LABEL_ALIASES.get(raw)
            if label is None:
                print("Unrecognized input. Use s, u, c, k, or q.")
                continue
            notes = input("Notes (optional): ").strip()
            record["human_label"] = label
            record["human_notes"] = notes
            record["reviewed_at_utc"] = datetime.now(timezone.utc).isoformat()
            write_audit_csv(csv_path, audit_rows)
            print(f"Saved {record['audit_id']} -> {label}")
            break
    print(f"Finished labeling pending rows in {csv_path}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a human-audit CSV from results.jsonl and optionally label rows interactively."
    )
    parser.add_argument("--results_path", required=True, help="Path to results.jsonl")
    parser.add_argument("--output_csv", help="Output CSV path; defaults next to results.jsonl")
    parser.add_argument(
        "--build_only",
        action="store_true",
        help="Only build or refresh the CSV, without entering the interactive labeling loop.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_path = Path(args.results_path)
    output_csv = Path(args.output_csv) if args.output_csv else _default_output_csv(results_path)
    audit_rows = build_audit_rows(results_path)
    audit_rows = merge_existing_labels(audit_rows, load_existing_labels(output_csv))
    write_audit_csv(output_csv, audit_rows)
    print(f"Wrote audit CSV to {output_csv}")
    if not args.build_only:
        run_labeling_loop(output_csv, audit_rows)


if __name__ == "__main__":
    main()
