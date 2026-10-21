"""Regression tests for the rule-based reef health verdict.

The verdict is a combined label: the live-coral-cover tier (Reef Check /
Gomez) capped by coral condition (bleaching and mortality). These tests pin the
two properties that matter:

1. condition can only lower the tier, never raise it;
2. the unmodified cover tier stays available as ``live_cover_tier``.

Run directly (no test dependency):

    python tests/test_health_indicators.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.pipeline import _TIER_ORDER, compute_health_indicators


def health(**counts: int) -> dict:
    return compute_health_indicators(counts, sum(counts.values()))


def tier(indicators: dict) -> str:
    return indicators["reef_health_category"].split(" [")[0]


def test_healthy_high_cover_is_excellent() -> None:
    result = health(coral_alive=600, sand=400)
    assert result["live_cover_tier"] == "Excellent"
    assert result["reef_health_category"] == "Excellent"
    assert result["coral_condition"] == "Stable"
    assert result["condition_cap"] is None


def test_extensive_bleaching_caps_excellent_cover() -> None:
    # 37.5% live cover (Good) and 50% of all coral bleached.
    result = health(coral_alive=600, coral_bleached=600, sand=400)
    assert result["live_cover_tier"] == "Good"
    assert result["reef_health_category"] == "Poor / Degraded [Extensive Bleaching]"
    assert result["condition_cap"] == "Poor / Degraded"


def test_extensive_mortality_caps_excellent_cover() -> None:
    result = health(coral_alive=500, dead_coral=500)
    assert result["live_cover_tier"] == "Excellent"
    assert tier(result) == "Poor / Degraded"
    assert result["coral_condition"] == "Extensive Mortality"


def test_bleaching_thresholds() -> None:
    severe = health(coral_alive=300, coral_bleached=100, sand=600)  # 25% of coral
    moderate = health(coral_alive=300, coral_bleached=40, sand=660)  # 11.8% of coral
    assert severe["condition_cap"] == "Fair"
    assert severe["coral_condition"] == "Severe Bleaching Alert"
    assert moderate["condition_cap"] == "Good"
    assert moderate["coral_condition"] == "Moderate Bleaching Warning"


def test_condition_never_raises_the_tier() -> None:
    # Poor cover stays poor even with a healthy assemblage.
    poor = health(coral_alive=50, sand=950)
    assert poor["live_cover_tier"] == "Poor / Degraded"
    assert tier(poor) == "Poor / Degraded"
    assert poor["condition_cap"] is None


def test_combined_condition_labels() -> None:
    result = health(coral_alive=400, coral_bleached=300, dead_coral=300)
    assert result["coral_condition"] == "Severe Bleaching Alert + High Mortality"
    assert result["reef_health_category"] == "Fair [Severe Bleaching Alert; High Mortality]"


def test_no_reef_area_is_indeterminate() -> None:
    result = health(non_reef=1000)
    assert tier(result) == "Indeterminate (No reef area detected)"
    assert result["reef_area_pixels"] == 0
    assert result["coral_condition"] is None


def test_algae_warning_does_not_change_condition() -> None:
    result = health(coral_alive=100, algae=500, sand=400)
    assert result["reef_health_category"] == "Fair [Algae-Covered Area Warning]"
    assert result["coral_condition"] == "Stable"


def test_legacy_taxonomy_names_still_work() -> None:
    result = health(branching_coral_alive=400, coral_dead=400, sand=200)
    assert result["live_cover_tier"] == "Good"
    assert result["coral_condition"] == "Extensive Mortality"
    assert tier(result) == "Poor / Degraded"


def test_condition_only_lowers_tier_over_grid() -> None:
    """Property: combined severity >= cover severity for a grid of compositions."""
    for alive in range(0, 1001, 100):
        for bleached in range(0, 1001 - alive, 100):
            for dead in range(0, 1001 - alive - bleached, 100):
                counts = {"coral_alive": alive, "coral_bleached": bleached,
                          "dead_coral": dead, "sand": 1000 - alive - bleached - dead}
                result = compute_health_indicators(counts, 1000)
                cover = result["live_cover_tier"]
                combined = tier(result)
                assert _TIER_ORDER.index(combined) >= _TIER_ORDER.index(cover), counts


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} health-indicator tests passed")
