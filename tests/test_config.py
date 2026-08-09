"""config/default.yaml must actually be read, and be the only source of truth.

Before the fix nothing in the repository imported yaml: the file was inert and
its values were re-hardcoded in maaa_core/risk.py, maaa_core/models.py and
layers/l2_cognition.py — three copies that could drift apart.
"""

from __future__ import annotations

import maaa_config
from layers.l2_cognition import RISK_THRESHOLDS
from layers.l4_regulation import BrevityFilter, UrgencyFilter
from maaa_core.models import AutopoieticStatus, SceneGraph
from maaa_core.risk import RiskEstimationEngine


def test_config_file_is_found_and_parsed():
    path = maaa_config.config_path()
    assert path is not None and path.is_file(), "config/default.yaml not found"
    assert maaa_config.get("maaa.max_words") == 9


def test_fallback_matches_the_shipped_yaml():
    """Keeps the embedded fallback from drifting away from the file."""
    from_file = maaa_config.load_config(maaa_config.config_path())
    assert from_file == maaa_config.FALLBACK, (
        "config/default.yaml and maaa_config.FALLBACK disagree")


def test_layers_read_the_shared_risk_bands():
    bands = dict((name, (lo, hi)) for name, lo, hi in maaa_config.risk_bands())
    for level, (lo, hi) in RISK_THRESHOLDS.items():
        assert (lo, hi) == bands[level.value]
    assert UrgencyFilter.RISK_CRITICAL == bands["CRITICAL"][0]
    assert BrevityFilter.MAX_WORDS == maaa_config.max_words()


def test_maaa_core_reads_the_same_risk_bands():
    """The two implementations used to hard-code two different band tables."""
    engine = RiskEstimationEngine()
    for name, lo, hi in maaa_config.risk_bands():
        graph = SceneGraph()
        midpoint = (lo + min(hi, 1.0)) / 2
        graph.add_node("x", kind="hazard", risk=midpoint)
        _score, level, _item = engine.score(graph)
        assert level.value == name, (midpoint, level.value, name)


def test_failsafe_thresholds_come_from_config():
    battery_min = maaa_config.failsafe_setting("battery_min_pct", 5.0)
    ok = AutopoieticStatus(True, battery_min + 1, 0.9, 50)
    low = AutopoieticStatus(True, battery_min - 1, 0.9, 50)
    assert not ok.should_failsafe()
    assert low.should_failsafe()


def test_version_is_unified():
    """README said v0.3, main_maaa.py and the REST API said v1.0."""
    import maaa_core
    assert maaa_core.__version__ == maaa_config.VERSION
