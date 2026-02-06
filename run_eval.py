import argparse
import json

from pathlib import Path

from evaluator import Evaluator
from model_runner import ModelRunner
from report_generator import ReportGenerator
from safety_classifier import SafetyClassifier
from safety_judge import LLMJudge

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Child safety evaluation pipeline")

    parser.add_argument("--model_name", required=False)

    parser.add_argument(
        "--provider",
        required=False,
        default="auto",
        choices=["auto", "local_hf", "openai", "openai_api", "scaleway"],
        help="Provider or auto-infer (llama-3.1 ⇒ scaleway)",
    )

    parser.add_argument("--system_prompt_name", required=False)
    parser.add_argument("--prompt_pack_path", required=False)

    parser.add_argument(
        "--extra_dataset_paths",
        type=str,
        default="datasets/true_negative_safe_sensitive.jsonl",
        help="Comma-separated list of JSONL datasets to include alongside the prompt pack. "
             "Use 'None' to disable.",
    )

    parser.add_argument("--k", type=int, default=5, help="Generations per prompt")
    parser.add_argument("--batch_size", type=int, default=10, help="Prompts per evaluation batch")

    parser.add_argument("--config", type=str, default=None, help="Optional JSON config string")
    parser.add_argument("--guardrail_model", type=str, default=None, help="Optional guardrail model name")
    parser.add_argument(
        "--guardrail_provider",
        type=str,
        default="scaleway",
        choices=["openai", "openai_api", "scaleway", "auto", "local_hf"],
        help="Provider used for the guardrail model (defaults to scaleway).",
    )
    parser.add_argument(
        "--guardrail_prompt",
        type=str,
        default=None,
        help="Override the default system prompt used for guardrail classification",
    )
    parser.add_argument(
        "--include_variants",
        type=str,
        default=None,
        help="Comma-separated list of prompt variants to include (e.g., original,typo)",
    )
    parser.add_argument(
        "--exclude_variants",
        type=str,
        default=None,
        help="Comma-separated list of prompt variants to exclude",
    )
    parser.add_argument(
        "--safety_judge_model",
        type=str,
        default=None,
        help="Optional LLM safety judge model name (enables LLM judge).",
    )
    parser.add_argument(
        "--safety_judge_provider",
        type=str,
        default="auto",
        choices=["auto", "local_hf", "openai", "openai_api", "scaleway"],
        help="Provider for the safety judge model (defaults to auto).",
    )
    parser.add_argument(
        "--safety_judge_prompt",
        type=str,
        default=None,
        help="Override the default system prompt used for the LLM safety judge.",
    )
    parser.add_argument(
        "--judge_only",
        action="store_true",
        help="Rejudge an existing results.jsonl without running model/guardrail.",
    )
    parser.add_argument(
        "--judge_only_path",
        type=str,
        default=None,
        help="Path to a run directory or results.jsonl for judge-only re-eval.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.judge_only:
        if not args.judge_only_path:
            raise SystemExit("--judge_only requires --judge_only_path")
        if not args.safety_judge_model:
            raise SystemExit("--judge_only requires --safety_judge_model")
    else:
        missing = [
            name
            for name, value in [
                ("--model_name", args.model_name),
                ("--system_prompt_name", args.system_prompt_name),
                ("--prompt_pack_path", args.prompt_pack_path),
            ]
            if not value
        ]
        if missing:
            raise SystemExit(f"Missing required arguments: {', '.join(missing)}")

    config = {}
    if args.config:
        config = json.loads(args.config)

    if args.guardrail_model:
        config["guardrail_model"] = args.guardrail_model
    if args.guardrail_provider:
        config["guardrail_provider"] = args.guardrail_provider
    if args.guardrail_prompt:
        config["guardrail_prompt"] = args.guardrail_prompt

    if args.safety_judge_model:
        config["safety_judge_model"] = args.safety_judge_model
    if args.safety_judge_provider:
        config["safety_judge_provider"] = args.safety_judge_provider
    if args.safety_judge_prompt:
        config["safety_judge_prompt"] = args.safety_judge_prompt


    include_variants = args.include_variants.split(",") if args.include_variants else None
    exclude_variants = args.exclude_variants.split(",") if args.exclude_variants else None

    extra_dataset_paths = None
    if args.extra_dataset_paths:
        extra_dataset_paths = [
            p.strip()
            for p in args.extra_dataset_paths.split(",")
            if p.strip() and p.strip().lower() != "none"
        ]

    evaluator = Evaluator()

    judge_model = config.get("safety_judge_model")
    use_judge = bool(judge_model)

    if args.judge_only:
        judge_only_path = Path(args.judge_only_path)
        if judge_only_path.is_dir():
            results_path = judge_only_path / "results.jsonl"
            output_dir = judge_only_path
        else:
            results_path = judge_only_path
            output_dir = judge_only_path.parent

        if not results_path.exists():
            raise SystemExit(f"Results file not found: {results_path}")

        results = [
            json.loads(line)
            for line in results_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not use_judge:
            raise SystemExit("--judge_only requires --safety_judge_model")

        judge = LLMJudge(
            model_name=judge_model,
            provider=config.get("safety_judge_provider", "auto"),
            prompt=config.get("safety_judge_prompt"),
            config=config,
        )
        judge.preflight()
        results = evaluator.annotate_with_llm_judge(
            results,
            judge=judge,
            results_path=output_dir / "results.jsonl",
        )
        judge_meta = {
            "safety_judge_model": judge_model,
            "safety_judge_provider": config.get("safety_judge_provider", "auto"),
            "safety_judge_prompt": config.get("safety_judge_prompt"),
        }
        meta_path = output_dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta.update(judge_meta)
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        summary = ReportGenerator(output_dir=output_dir, use_judge_results=True).generate(results)
        if summary.get("safety_source") != "llm_judge":
            summary["safety_source"] = "llm_judge"
            (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Judge-only re-eval completed. Output: {output_dir}")
        return

    model_runner = ModelRunner(args.model_name, args.provider, config=config)
    model_runner.judge_meta = (
        {
            "safety_judge_model": judge_model,
            "safety_judge_provider": config.get("safety_judge_provider", "auto"),
            "safety_judge_prompt": config.get("safety_judge_prompt"),
        }
        if use_judge
        else None
    )
    results, output_dir = evaluator.evaluate_pack(
        model_runner,
        system_prompt_name=args.system_prompt_name,
        prompt_pack_path=args.prompt_pack_path,
        k=args.k,
        batch_size=args.batch_size,
        include_variants=include_variants,
        exclude_variants=exclude_variants,
        extra_dataset_paths=extra_dataset_paths,
    )
    if use_judge:
        judge = LLMJudge(
            model_name=judge_model,
            provider=config.get("safety_judge_provider", "auto"),
            prompt=config.get("safety_judge_prompt"),
            config=config,
        )
        judge.preflight()
        results = evaluator.annotate_with_llm_judge(
            results,
            judge=judge,
            results_path=output_dir / "results.jsonl",
        )
    else:
        results = evaluator.annotate_with_safety(
            results,
            SafetyClassifier(),
            results_path=output_dir / "results.jsonl",
        )

    ReportGenerator(output_dir=output_dir, use_judge_results=use_judge).generate(results)
    print(f"Evaluation completed. Output: {output_dir}")


if __name__ == "__main__":
    main()
