import yaml
from pathlib import Path
from typing import Dict


SYSTEM_PROMPTS_FILE = Path(__file__).with_name("system_prompts.yaml")


def load_system_prompt(name: str) -> str:
    """
    Returns the system prompt string from system_prompts.yaml
    """

    if not SYSTEM_PROMPTS_FILE.exists():
        raise FileNotFoundError(f"System prompts file not found at {SYSTEM_PROMPTS_FILE}")

    data: Dict[str, str] = yaml.safe_load(SYSTEM_PROMPTS_FILE.read_text())
    if name not in data:
        raise KeyError(f"System prompt '{name}' not found in {SYSTEM_PROMPTS_FILE}")
    return data[name]