import argparse
import json

from evaluator import Evaluator
from model_runner import ModelRunner
from report_generator import ReportGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Child safety evaluation pipeline")

    parser.add_argument("--model_name", required=True)

    parser.add_argument(
        "--provider",
        required=False,
        default="auto",
        choices=["auto", "local_hf", "openai", "openai_api", "scaleway"],
        help="Provider or auto-infer (llama-3.1 ⇒ scaleway)",
    )

    parser.add_argument("--system_prompt_name", required=True)
    parser.add_argument("--prompt_pack_path", required=True)

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

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = {}
    if args.config:
        config = json.loads(args.config)

    if args.guardrail_model:
        config["guardrail_model"] = args.guardrail_model
    if args.guardrail_provider:
        config["guardrail_provider"] = args.guardrail_provider
    if args.guardrail_prompt:
        config["guardrail_prompt"] = args.guardrail_prompt

    include_variants = args.include_variants.split(",") if args.include_variants else None
    exclude_variants = args.exclude_variants.split(",") if args.exclude_variants else None

    extra_dataset_paths = None
    if args.extra_dataset_paths:
        extra_dataset_paths = [
            p.strip()
            for p in args.extra_dataset_paths.split(",")
            if p.strip() and p.strip().lower() != "none"
        ]

    model_runner = ModelRunner(args.model_name, args.provider, config=config)
    evaluator = Evaluator()

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
    results = evaluator.annotate_with_safety(results, SafetyClassifier())
    ReportGenerator(output_dir=output_dir).generate(results)
    print(f"Evaluation completed. Output: {output_dir}")


if __name__ == "__main__":
    main()
