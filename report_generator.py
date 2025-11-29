import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


class ReportGenerator:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _write_markdown(self, summary: Dict[str, Any]) -> None:
        lines = ["# Evaluation Summary", ""]
        lines.append(f"Overall success rate: {summary['overall_success_rate']:.2%}")
        lines.append(f"Fully safe prompts: {summary['percent_fully_safe']:.2%}")
        lines.append(f"Partially unsafe prompts: {summary['percent_partially_unsafe']:.2%}")
        lines.append(f"Consistently unsafe prompts: {summary['percent_consistently_unsafe']:.2%}")
        lines.append("")
        lines.append("## Failure reasons histogram")
        for reason, count in summary["failure_reason_histogram"].items():
            lines.append(f"- {reason}: {count}")
        (self.output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    def generate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_prompts = len(results)
        total_generations = sum(r.get("n_generations", 0) for r in results)
        total_safe = sum(r.get("safe_count", 0) for r in results)

        fully_safe = sum(1 for r in results if r.get("safe_count", 0) == r.get("n_generations", 0) and r.get("n_generations", 0) > 0)
        consistently_unsafe = sum(1 for r in results if r.get("unsafe_count", 0) == r.get("n_generations", 0) and r.get("n_generations", 0) > 0)
        partially_unsafe = sum(
            1
            for r in results
            if r.get("n_generations", 0) > 0
            and 0 < r.get("safe_count", 0) < r.get("n_generations", 0)
        )

        histogram: Counter[str] = Counter()
        for row in results:
            for reasons in row.get("failure_reasons", []):
                histogram.update(reasons)

        summary = {
            "overall_success_rate": total_safe / total_generations if total_generations else 0.0,
            "percent_fully_safe": fully_safe / total_prompts if total_prompts else 0.0,
            "percent_partially_unsafe": partially_unsafe / total_prompts if total_prompts else 0.0,
            "percent_consistently_unsafe": consistently_unsafe / total_prompts if total_prompts else 0.0,
            "failure_reason_histogram": dict(histogram),
            "total_prompts": total_prompts,
            "total_generations": total_generations,
        }

        (self.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self._write_markdown(summary)
        return summary