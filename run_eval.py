import argparse
import sys
from pathlib import Path

# Ensure local imports work correctly
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

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

    parser.add_argument("--k", type=int, default=5, help="Generations per prompt")

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional JSON config string",
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
        help="Comma-separated list of prompt variants to exclude (e.g., camouflage,adversarial)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = {}
    if args.config:
        import json
        config = json.loads(args.config)
    include_variants = args.include_variants.split(",") if args.include_variants else None
    exclude_variants = args.exclude_variants.split(",") if args.exclude_variants else None
    model_runner = ModelRunner(args.model_name, args.provider, config=config)
    evaluator = Evaluator()

    results, output_dir = evaluator.evaluate_pack(
        model_runner,
        system_prompt_name=args.system_prompt_name,
        prompt_pack_path=args.prompt_pack_path,
        k=args.k,
        include_variants=include_variants,
        exclude_variants=exclude_variants
    )

    ReportGenerator(output_dir=output_dir).generate(results)
    print("Evaluation completed.")


if __name__ == "__main__":
    main()
