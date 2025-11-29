import argparse
import sys
from pathlib import Path

# Ensure local modules are importable when run as a script
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from evaluator import Evaluator
from model_runner import ModelRunner
from report_generator import ReportGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Child safety evaluation pipeline")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--provider", required=True, choices=["local_hf", "openai_api"])
    parser.add_argument("--system_prompt_name", required=True)
    parser.add_argument("--prompt_pack_path", required=True)
    parser.add_argument("--k", type=int, default=5, help="Number of generations per prompt")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional JSON string for provider-specific configuration",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = {}
    if args.config:
        import json

        config = json.loads(args.config)

    model_runner = ModelRunner(args.model_name, args.provider, config=config)
    evaluator = Evaluator()
    results = evaluator.evaluate_pack(
        model_runner,
        system_prompt_name=args.system_prompt_name,
        prompt_pack_path=args.prompt_pack_path,
        k=args.k,
    )

    if evaluator.last_output_dir is None:
        raise RuntimeError("No output directory was created during evaluation.")

    report_generator = ReportGenerator(evaluator.last_output_dir)
    summary = report_generator.generate(results)
    print(f"Saved results to {evaluator.last_output_dir}")
    print(f"Overall success rate: {summary['overall_success_rate']:.2%}")


if __name__ == "__main__":
    main()