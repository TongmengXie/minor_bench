import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from safety_classifier import SafetyClassifier
from system_prompts import load_system_prompt

class Evaluator:
    def __init__(self, results_root: Path | None = None) -> None:
        self.classifier = SafetyClassifier()
        self.results_root = results_root or Path(__file__).resolve().parent / "results"
        self.last_output_dir: Path | None = None

    def evaluate_prompt(
        self, model_runner, system_prompt: str, prompt_item: Dict[str, Any], k: int = 5
    ) -> Dict[str, Any]:
        completions: List[str] = []
        classifications: List[str] = []
        failure_reasons: List[List[str]] = []
        safe_count = 0
        unsafe_count = 0
        borderline_count = 0
        total_latency = 0.0

        for _ in range(k):
            try:
                result = model_runner.run(system_prompt, prompt_item["prompt_text"])
                completion = result.get("completion", "")
                latency = float(result.get("latency", 0.0))
            except Exception as exc:
                completion = ""
                latency = 0.0
                classification = {"label": "unsafe", "reasons": [f"runner_error:{exc}"]}
            else:
                classification = self.classifier.classify(completion)

            total_latency += latency
            label = classification["label"]
            reasons = classification["reasons"]

            completions.append(completion)
            classifications.append(label)
            failure_reasons.append(list(reasons))

            if label == "safe":
                safe_count += 1
            elif label == "borderline":
                borderline_count += 1
            else:
                unsafe_count += 1

        success_rate = safe_count / k if k else 0.0
        avg_latency = total_latency / k if k else 0.0

        return {
            "prompt_id": prompt_item.get("id", ""),
            "prompt_text": prompt_item.get("prompt_text", ""),
            "system_prompt_used": system_prompt,
            "model_name": getattr(model_runner, "model_name", "unknown"),
            "n_generations": k,
            "safe_count": safe_count,
            "unsafe_count": unsafe_count,
            "borderline_count": borderline_count,
            "success_rate": success_rate,
            "avg_latency": avg_latency,
            "all_completions": completions,
            "all_classifications": classifications,
            "failure_reasons": failure_reasons,
        }

    def _prepare_output_dir(self, model_name: str, system_prompt_name: str) -> Path:
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        output_dir = self.results_root / date_str / model_name / system_prompt_name
        output_dir.mkdir(parents=True, exist_ok=True)
        self.last_output_dir = output_dir
        return output_dir

    def evaluate_pack(
        self, model_runner, system_prompt_name: str, prompt_pack_path: str, k: int = 5
    ) -> List[Dict[str, Any]]:
        system_prompt = load_system_prompt(system_prompt_name)
        prompt_pack = json.loads(Path(prompt_pack_path).read_text())
        items = prompt_pack.get("items", [])
        output_dir = self._prepare_output_dir(getattr(model_runner, "model_name", "model"), system_prompt_name)

        results: List[Dict[str, Any]] = []
        for item in items:
            results.append(self.evaluate_prompt(model_runner, system_prompt, item, k=k))

        raw_jsonl_path = output_dir / "raw_results.jsonl"
        with raw_jsonl_path.open("w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        csv_path = output_dir / "raw_results.csv"
        fieldnames = [
            "prompt_id",
            "prompt_text",
            "system_prompt_used",
            "model_name",
            "n_generations",
            "safe_count",
            "unsafe_count",
            "borderline_count",
            "success_rate",
            "avg_latency",
            "all_completions",
            "all_classifications",
            "failure_reasons",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)

        return results
