from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from tqdm import tqdm
from safety_classifier import SafetyClassifier
from safety_judge import LLMJudge
from system_prompts import load_system_prompt


@dataclass
class EvaluationConfig:
    output_root: Path = Path("runs")


class Evaluator:
    """
    Evaluates a prompt pack (JSON with {"items": [...]}) and optionally extra JSONL datasets.
    Writes per-batch results to results.jsonl in an auto-created output dir.
    """

    def __init__(self, config: Optional[EvaluationConfig] = None) -> None:
        self.config = config or EvaluationConfig()

    def _slug(self, s: str) -> str:
        s = (s or "").strip().lower()
        keep = []
        for ch in s:
            if ch.isalnum() or ch in ("-", "_"):
                keep.append(ch)
            elif ch.isspace():
                keep.append("-")
        out = "".join(keep).strip("-")
        return out or "unknown"

    def _prepare_output_dir(
        self,
        model_name: str,
        system_prompt_name: str,
        judge_meta: Optional[Dict[str, Any]] = None,
    ) -> Path:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        model_part = self._slug(model_name)
        prompt_part = self._slug(system_prompt_name)

        output_dir = self.config.output_root / "evals" / model_part / prompt_part / ts
        output_dir.mkdir(parents=True, exist_ok=True)

        # Helpful metadata file
        meta = {
            "model_name": model_name,
            "system_prompt_name": system_prompt_name,
            "created_utc": ts,
            "cwd": os.getcwd(),
        }
        if judge_meta:
            meta.update(judge_meta)
        (output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return output_dir

    def _load_jsonl_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """
        Expected JSONL rows are dict-like and include either:
          - prompt_text, or
          - prompt
        Optionally:
          - metadata.label in {"safe","unsafe"} which will map to metadata.is_harmful
          - metadata.variant (defaults to "original")
        """
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        items: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue

                payload = json.loads(line)

                prompt_text = payload.get("prompt_text") or payload.get("prompt")
                if not prompt_text:
                    continue

                metadata = payload.get("metadata", {}) or {}
                label = (metadata.get("label") or "").strip().lower()
                if label == "safe":
                    metadata.setdefault("is_harmful", False)
                elif label == "unsafe":
                    metadata.setdefault("is_harmful", True)

                metadata.setdefault("variant", "original")
                payload["prompt_text"] = prompt_text
                payload["metadata"] = metadata
                items.append(payload)

        return items

    def _call_model_runner(self, model_runner: Any, system_prompt: str, prompt_text: str) -> Any:
        """
        Tries a few common model runner interfaces.
        Adjust this if your ModelRunner uses something else.
        """
        if hasattr(model_runner, "generate"):
            return model_runner.generate(prompt_text=prompt_text, system_prompt=system_prompt)
        if hasattr(model_runner, "run"):
            try:
                return model_runner.run(prompt_text=prompt_text, system_prompt=system_prompt)
            except TypeError:
                return model_runner.run(system_prompt, prompt_text)
        if callable(model_runner):
            # best-effort for runners that are callable
            try:
                return model_runner(prompt_text=prompt_text, system_prompt=system_prompt)
            except TypeError:
                return model_runner(system_prompt, prompt_text)
        raise TypeError("model_runner must expose .generate(), .run(), or be callable")

    @staticmethod
    def _extract_generation_text(generation: Any) -> str:
        if isinstance(generation, dict):
            return (
                generation.get("text")
                or generation.get("output")
                or generation.get("content")
                or generation.get("completion")
                or ""
            )
        return str(generation)

    def evaluate_prompt(
        self,
        model_runner: Any,
        system_prompt: str,
        prompt_item: Dict[str, Any],
        k: int = 5,
    ) -> Dict[str, Any]:
        prompt_text = prompt_item.get("prompt_text") or prompt_item.get("prompt")
        if not prompt_text:
            raise ValueError("prompt_item missing prompt_text/prompt")

        metadata = prompt_item.get("metadata", {}) or {}
        variant = metadata.get("variant") or prompt_item.get("variant") or "original"
        metadata.setdefault("variant", variant)

        generations: List[Dict[str, Any]] = []
        guardrail_decisions: List[Dict[str, Any]] = []
        guardrail_model = getattr(model_runner, "guardrail_model", None)
        guardrail_provider = getattr(model_runner, "guardrail_provider", None)
        model_name = getattr(model_runner, "model_name", None)
        for i in range(k):
            raw = self._call_model_runner(model_runner, system_prompt, prompt_text)

            # Normalize common response formats into a string
            if isinstance(raw, dict):
                text = (
                    raw.get("text")
                    or raw.get("output")
                    or raw.get("content")
                    or raw.get("completion")
                    or str(raw)
                )
                
                generations.append({"index": i, "text": text, "raw": raw})
                decision_status = raw.get("guardrail_decision_status")
                guardrail_decisions.append(
                    {
                        "index": i,
                        "status": decision_status,
                        "guardrail_model": raw.get("guardrail_model", guardrail_model),
                        "guardrail_provider": raw.get("guardrail_provider", guardrail_provider),
                    }
                )
            else:
                generations.append({"index": i, "text": str(raw), "raw": raw})
                guardrail_decisions.append(
                    {
                        "index": i,
                        "status": None,
                        "guardrail_model": guardrail_model,
                        "guardrail_provider": guardrail_provider,
                    }
                )

        return {
            "prompt_text": prompt_text,
            "metadata": metadata,
            "generations": generations,
            "model_name": model_name,
            "system_prompt": system_prompt,
            "guardrail_model": guardrail_model,
            "guardrail_provider": guardrail_provider,
            "guardrail_decisions": guardrail_decisions,
        }

    def evaluate_pack(
        self,
        model_runner,
        system_prompt_name: str,
        prompt_pack_path: str,
        k: int = 5,
        batch_size: int = 10,
        include_variants: List[str] | None = None,
        exclude_variants: List[str] | None = None,
        extra_dataset_paths: List[str] | None = None,
    ) -> Tuple[List[Dict[str, Any]], Path]:
        system_prompt = load_system_prompt(system_prompt_name)
        model_name = getattr(model_runner, "model_name", "unknown")
        output_dir = self._prepare_output_dir(
            model_name,
            system_prompt_name,
            judge_meta=getattr(model_runner, "judge_meta", None),
        )
        results_path = output_dir / "results.jsonl"
        if results_path.exists():
            results_path.unlink()

        prompt_pack = json.loads(Path(prompt_pack_path).read_text(encoding="utf-8"))
        items = prompt_pack.get("items", []) or []

        if extra_dataset_paths:
            for dataset_path in extra_dataset_paths:
                items.extend(self._load_jsonl_dataset(dataset_path))

        include_set = set(include_variants) if include_variants else None
        exclude_set = set(exclude_variants) if exclude_variants else None

        filtered_items: List[Dict[str, Any]] = []
        for item in items:
            metadata = item.get("metadata", {}) or {}
            variant = metadata.get("variant") or item.get("variant") or "original"
            if include_set and variant not in include_set:
                continue
            if exclude_set and variant in exclude_set:
                continue
            filtered_items.append(item)

        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        results: List[Dict[str, Any]] = []
        total_batches = (len(filtered_items) + batch_size - 1) // batch_size if filtered_items else 0

        batch_iter = range(total_batches)
        if total_batches:
            batch_iter = tqdm(batch_iter, desc="eval batches")

        for batch_index in batch_iter:
            start = batch_index * batch_size
            batch_items = filtered_items[start : start + batch_size]
            batch_results: List[Dict[str, Any]] = []

            use_batching = bool(
                getattr(model_runner, "enable_batching", False)
                and hasattr(model_runner, "run_batch")
                and (
                    getattr(model_runner, "provider", "") == "local_hf"
                    or getattr(model_runner, "guardrail_provider", "") == "local_hf"
                )
            )

            if use_batching:
                prompt_texts: List[str] = []
                base_rows: List[Dict[str, Any]] = []
                guardrail_model = getattr(model_runner, "guardrail_model", None)
                guardrail_provider = getattr(model_runner, "guardrail_provider", None)
                model_name = getattr(model_runner, "model_name", None)

                for prompt_item in batch_items:
                    prompt_text = prompt_item.get("prompt_text") or prompt_item.get("prompt")
                    if not prompt_text:
                        raise ValueError("prompt_item missing prompt_text/prompt")

                    metadata = prompt_item.get("metadata", {}) or {}
                    variant = metadata.get("variant") or prompt_item.get("variant") or "original"
                    metadata.setdefault("variant", variant)

                    prompt_texts.append(prompt_text)
                    base_rows.append(
                        {
                            "prompt_text": prompt_text,
                            "metadata": metadata,
                            "generations": [],
                            "model_name": model_name,
                            "system_prompt": system_prompt,
                            "guardrail_model": guardrail_model,
                            "guardrail_provider": guardrail_provider,
                            "guardrail_decisions": [],
                        }
                    )

                for i in range(k):
                    outputs = model_runner.run_batch(system_prompt, prompt_texts)
                    for idx, output in enumerate(outputs):
                        text = self._extract_generation_text(output)
                        base_rows[idx]["generations"].append(
                            {"index": i, "text": text, "raw": output}
                        )
                        decision_status = None
                        if isinstance(output, dict):
                            decision_status = output.get("guardrail_decision_status")
                        base_rows[idx]["guardrail_decisions"].append(
                            {
                                "index": i,
                                "status": decision_status,
                                "guardrail_model": output.get("guardrail_model", guardrail_model)
                                if isinstance(output, dict)
                                else guardrail_model,
                                "guardrail_provider": output.get(
                                    "guardrail_provider", guardrail_provider
                                )
                                if isinstance(output, dict)
                                else guardrail_provider,
                            }
                        )

                batch_results.extend(base_rows)
            else:
                for prompt_item in batch_items:
                    batch_results.append(
                        self.evaluate_prompt(
                            model_runner,
                            system_prompt,
                            prompt_item,
                            k=k,
                        )
                    )

            if batch_results:
                with results_path.open("a", encoding="utf-8") as handle:
                    for row in batch_results:
                        handle.write(json.dumps(row) + "\n")

            results.extend(batch_results)

            if total_batches:
                print(
                    f"Batch {batch_index + 1}/{total_batches} written "
                    f"({len(batch_results)} prompts)."
                )

        return results, output_dir

    def annotate_with_safety(
        self,
        results: List[Dict[str, Any]],
        classifier: Optional[SafetyClassifier] = None,
        results_path: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        classifier = classifier or SafetyClassifier()

        for row in results:
            generations = row.get("generations", []) or []
            classifications: List[str] = []
            failure_reasons: List[List[str]] = []
            safe_count = 0
            unsafe_count = 0

            for generation in generations:
                text = self._extract_generation_text(generation)
                outcome = classifier.classify(text)
                label = str(outcome.get("label") or "unknown")
                classifications.append(label)

                reasons = outcome.get("reasons") or []
                if not isinstance(reasons, list):
                    reasons = [str(reasons)]
                failure_reasons.append([str(reason) for reason in reasons])

                if label.lower() == "safe":
                    safe_count += 1
                else:
                    unsafe_count += 1

            row["n_generations"] = len(generations)
            row["safe_count"] = safe_count
            row["unsafe_count"] = unsafe_count
            row["all_classifications"] = classifications
            row["failure_reasons"] = failure_reasons

            metadata = row.get("metadata", {}) or {}
            is_harmful = metadata.get("is_harmful")
            if is_harmful is None:
                label = str(metadata.get("label") or "").lower()
                if label == "safe":
                    is_harmful = False
                elif label == "unsafe":
                    is_harmful = True
            row["is_harmful"] = is_harmful

        if results_path:
            with results_path.open("w", encoding="utf-8") as handle:
                for row in results:
                    handle.write(json.dumps(row) + "\n")

        return results

    def annotate_with_llm_judge(
        self,
        results: List[Dict[str, Any]],
        judge: LLMJudge,
        results_path: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        row_iter = results
        if results:
            row_iter = tqdm(results, desc="judge rows")

        judge_batch_size = int(judge.config.get("judge_batch_size", 8))

        for row in row_iter:
            generations = row.get("generations", []) or []
            classifications: List[str] = []
            failure_reasons: List[List[str]] = []
            raw_outputs: List[str] = []
            judge_usages: List[Optional[Dict[str, Any]]] = []
            judge_costs: List[Optional[float]] = []
            safe_count = 0
            unsafe_count = 0

            system_prompt = row.get("system_prompt") or ""

            if (
                getattr(judge, "provider", None) == "local_hf"
                and hasattr(judge, "judge_batch")
                and generations
            ):
                texts = [self._extract_generation_text(g) for g in generations]
                system_prompts = [system_prompt] * len(texts)
                outcomes: List[Dict[str, Any]] = []
                for start in range(0, len(texts), judge_batch_size):
                    chunk_texts = texts[start : start + judge_batch_size]
                    chunk_systems = system_prompts[start : start + judge_batch_size]
                    outcomes.extend(judge.judge_batch(chunk_systems, chunk_texts))
            else:
                outcomes = [
                    judge.judge(system_prompt=system_prompt, response_text=self._extract_generation_text(g))
                    for g in generations
                ]

            for outcome in outcomes:
                label = str(outcome.get("label") or "unknown")
                classifications.append(label)

                reasons = outcome.get("reasons") or []
                if not isinstance(reasons, list):
                    reasons = [str(reasons)]
                failure_reasons.append([str(reason) for reason in reasons])
                raw_outputs.append(str(outcome.get("raw") or ""))
                judge_usages.append(outcome.get("usage"))
                judge_costs.append(outcome.get("cost_usd"))

                if label.lower() == "safe":
                    safe_count += 1
                else:
                    unsafe_count += 1

            row["n_generations"] = len(generations)
            row["safe_count"] = safe_count
            row["unsafe_count"] = unsafe_count
            row["judge_classifications"] = classifications
            row["judge_failure_reasons"] = failure_reasons
            row["judge_raw_outputs"] = raw_outputs
            row["judge_usage"] = judge_usages
            row["judge_cost_usd"] = judge_costs
            row["all_classifications"] = classifications
            row["failure_reasons"] = failure_reasons
            row["safety_source"] = "llm_judge"

            metadata = row.get("metadata", {}) or {}
            is_harmful = metadata.get("is_harmful")
            if is_harmful is None:
                label = str(metadata.get("label") or "").lower()
                if label == "safe":
                    is_harmful = False
                elif label == "unsafe":
                    is_harmful = True
            row["is_harmful"] = is_harmful

        if results_path:
            with results_path.open("w", encoding="utf-8") as handle:
                for row in results:
                    handle.write(json.dumps(row) + "\n")

        return results
