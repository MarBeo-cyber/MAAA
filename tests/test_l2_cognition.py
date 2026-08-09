"""L2 regressions: scene coverage, risk aggregation, shared bands."""

from __future__ import annotations

import logging

from layers.l1_perception import L1EmbodiedPerception, SceneCondition
from layers.l2_cognition import (L2SituationalCognition, OBJECT_TEMPLATES,
                                 RISK_THRESHOLDS, RiskLevel, classify_risk)

import maaa_config


def _run(scene, ticks=200, **kw):
    l1, l2 = L1EmbodiedPerception(True), L2SituationalCognition(True)
    l1.inject_scenario(scene, **kw)
    return [l2.process(l1.capture()) for _ in range(ticks)]


def test_every_scene_condition_has_object_templates():
    """DARK, DUSTY and OBSTRUCTED had no entry, so they silently fell back to
    the NORMAL object list — including the /scenario/dark and /scenario/dusty
    endpoints, which therefore modelled a safe room."""
    missing = [s.value for s in SceneCondition if s not in OBJECT_TEMPLATES]
    assert not missing, f"scenes with no object model: {missing}"


def test_degraded_scenes_are_riskier_than_the_normal_scene():
    normal = _run(SceneCondition.NORMAL, stress=0.0, panic=0.0,
                  obstruction=0.0, emergency_sounds=False)
    normal_mean = sum(c.risk_map.global_risk for c in normal) / len(normal)
    for scene in (SceneCondition.DARK, SceneCondition.DUSTY,
                  SceneCondition.OBSTRUCTED):
        frames = _run(scene, stress=0.4, panic=0.0, obstruction=0.3,
                      emergency_sounds=False)
        mean = sum(c.risk_map.global_risk for c in frames) / len(frames)
        assert mean > normal_mean + 0.15, (scene.value, mean, normal_mean)


def test_unmapped_scene_fallback_is_logged(caplog, monkeypatch):
    monkeypatch.delitem(OBJECT_TEMPLATES, SceneCondition.DUSTY)
    with caplog.at_level(logging.WARNING, logger="maaa.l2_cognition"):
        _run(SceneCondition.DUSTY, ticks=3, stress=0.4, panic=0.0,
             obstruction=0.3, emergency_sounds=False)
    assert any("No OBJECT_TEMPLATES entry" in r.getMessage()
               for r in caplog.records), [r.getMessage() for r in caplog.records]


def test_a_close_critical_hazard_drives_the_global_risk():
    """The old aggregate was a flat weighted sum in which the worst object
    contributed at most 0.25, so a CRITICAL hazard 1.2 m away was diluted by
    the average haze and global_risk could not exceed ~0.744."""
    frames = _run(SceneCondition.COLLAPSED, stress=1.0, panic=1.0,
                  obstruction=1.0, emergency_sounds=True)
    peak = max(c.risk_map.global_risk for c in frames)
    assert peak >= 0.80, peak
    assert any(c.risk_map.global_risk_level is RiskLevel.CRITICAL for c in frames)


def test_normal_scene_stays_below_the_medium_band():
    frames = _run(SceneCondition.NORMAL, stress=0.0, panic=0.0,
                  obstruction=0.0, emergency_sounds=False)
    peak = max(c.risk_map.global_risk for c in frames)
    assert peak < RISK_THRESHOLDS[RiskLevel.MEDIUM][0], peak


def test_risk_bands_come_from_the_shared_config():
    """maaa_core and layers used to hard-code two different band tables."""
    for name, lo, hi in maaa_config.risk_bands():
        assert RISK_THRESHOLDS[RiskLevel[name]][0] == lo
        assert RISK_THRESHOLDS[RiskLevel[name]][1] == hi
    assert classify_risk(0.0) is RiskLevel.SAFE
    assert classify_risk(1.0) is RiskLevel.CRITICAL
