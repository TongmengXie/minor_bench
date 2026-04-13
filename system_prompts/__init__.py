from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml

from assistant_policy import ASSISTANT_POLICY_VERSION


SYSTEM_PROMPTS_FILE = Path(__file__).with_name("system_prompts.yaml")
SAFE_POLICY_VERSION = "safe_policy_v1"
DEFAULT_POLICY_VERSION = ASSISTANT_POLICY_VERSION
DEFAULT_ASSISTANT_PROMPT_NAME = "assistant_child_safety_v2"
DEFAULT_GUARDRAIL_PROMPT_NAME = "guardrail_child_safety_v2"
DEFAULT_JUDGE_PROMPT_NAME = "judge_assistant_policy_v2"


def _load_prompt_map() -> Dict[str, str]:
    if not SYSTEM_PROMPTS_FILE.exists():
        raise FileNotFoundError(f"System prompts file not found at {SYSTEM_PROMPTS_FILE}")

    data: Dict[str, str] = yaml.safe_load(SYSTEM_PROMPTS_FILE.read_text())
    return data


def load_prompt(name: str) -> str:
    """
    Returns a named prompt string from system_prompts.yaml.
    """
    data = _load_prompt_map()
    if name not in data:
        raise KeyError(f"System prompt '{name}' not found in {SYSTEM_PROMPTS_FILE}")
    return data[name]


def load_system_prompt(name: str) -> str:
    """
    Backward-compatible alias for assistant system prompts.
    """
    return load_prompt(name)


def resolve_prompt(
    *,
    raw_prompt: Optional[str],
    prompt_name: Optional[str],
    default_prompt_name: Optional[str] = None,
) -> Tuple[str, Optional[str], str]:
    """
    Resolve a prompt by precedence:
      1. explicit raw prompt
      2. explicit prompt name
      3. default prompt name

    Returns: (prompt_text, resolved_prompt_name, source_kind)
      source_kind is "raw" or "named".
    """
    if raw_prompt is not None:
        return raw_prompt, prompt_name, "raw"

    resolved_name = prompt_name or default_prompt_name
    if not resolved_name:
        raise ValueError("Either raw_prompt, prompt_name, or default_prompt_name must be provided.")
    return load_prompt(resolved_name), resolved_name, "named"
