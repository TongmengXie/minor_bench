import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_smallslice_stability import (
    _prompt_family_key,
    _safe_family_key,
    _sample_grouped_items,
    _summarize_metric,
)


def test_prompt_family_key_prefers_base_prompt_id() -> None:
    item = {
        "id": "101-typo",
        "metadata": {"base_prompt_id": "101", "variant": "typo"},
    }
    assert _prompt_family_key(item) == "101"


def test_safe_family_key_strips_variant_suffix() -> None:
    item = {"id": "tn-003-camouflage"}
    assert _safe_family_key(item) == "tn-003"


def test_sample_grouped_items_keeps_full_families() -> None:
    items = [
        {"id": "100", "metadata": {"base_prompt_id": "100", "variant": "original"}},
        {"id": "100-typo", "metadata": {"base_prompt_id": "100", "variant": "typo"}},
        {"id": "101", "metadata": {"base_prompt_id": "101", "variant": "original"}},
        {"id": "101-typo", "metadata": {"base_prompt_id": "101", "variant": "typo"}},
    ]
    keys, sampled = _sample_grouped_items(items, count=1, seed=1, key_fn=_prompt_family_key)
    assert len(keys) == 1
    assert len(sampled) == 2
    assert {_prompt_family_key(item) for item in sampled} == set(keys)


def test_summarize_metric_returns_half_width_for_multiple_values() -> None:
    stats = _summarize_metric([0.2, 0.3, 0.4])
    assert math.isclose(stats["mean"], 0.3)
    assert stats["half_width"] > 0
    assert stats["n"] == 3
