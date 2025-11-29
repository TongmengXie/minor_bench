import json
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset


OUTPUT_FILENAME = "minorbench_hateful.json"


def build_minorbench_hateful(output_path: str = OUTPUT_FILENAME) -> Path:
    """
    Loads MinorBench dataset, filters entries where Category == 'Hateful',
    and writes them in standardized prompt pack format.

    Returns the path to the generated file.
    """

    # load govtech's MinorBench dataset
    dataset = load_dataset("govtech/MinorBench")["test"]

    # filter to Hateful entries
    filtered: List[Dict[str, Any]] = [
        row for row in dataset
        if row.get("Category") == "Hateful"
    ]

    prompt_pack: Dict[str, Any] = {
        "name": "minorbench_hateful_v1",
        "source": "MinorBench",
        "subset": "Hateful",
        "items": [],
    }

    for row in filtered:
        prompt_pack["items"].append(
            {
                "id": str(row.get("PromptID", "")),
                "prompt_text": row.get("Prompt", ""),
                "metadata": {
                    "original_category": row.get("Category", "unknown"),
                    "original_subcategory": "Hateful",
                },
            }
        )

    output_file = Path(output_path)
    output_file.write_text(json.dumps(prompt_pack, ensure_ascii=False, indent=2))
    return output_file


def main() -> None:
    output_path = Path(__file__).with_name(OUTPUT_FILENAME)
    path = build_minorbench_hateful(str(output_path))
    print(f"Wrote MinorBench Hateful prompt pack to {path}")


if __name__ == "__main__":
    main()
