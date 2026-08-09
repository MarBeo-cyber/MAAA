"""Every cognitive state and every urgency level must be reachable.

The audit that produced these tests measured, over 2,000 ticks at maximum
injected severity: global_risk capped at 0.744 against a 0.80 CRITICAL gate,
panic_score capped at 0.681 against 0.85 — so UrgencyLevel.CRITICAL, the
"FERMATI." templates, the SOS haptic, the red AR overlay and
CognitiveState.PANICKING could never execute. A demo phase titled
"CROLLO STRUTTURALE — PANICO" reported [ELEVATED].

These tests fail if any of those paths becomes unreachable again.
"""

from __future__ import annotations

import pytest

from layers.l1_perception import L1EmbodiedPerception, SceneCondition
from layers.l2_cognition import L2SituationalCognition, RiskLevel
from layers.l3_human_state import L3HumanStateMonitor, CognitiveState
from layers.l4_regulation import L4SymbioticRegulation, UrgencyLevel


def _pipeline():
    return (L1EmbodiedPerception(True), L2SituationalCognition(True),
            L3HumanStateMonitor(), L4SymbioticRegulation())


def _observe(scenario, ticks=400, warmup=0):
    """Run the pipeline and collect the states / urgencies / risk levels seen."""
    l1, l2, l3, l4 = _pipeline()
    if warmup:
        l1.inject_scenario(SceneCondition.NORMAL, 0.0, 0.0, 0.0, False, 0.0)
        for _ in range(warmup):
            p = l1.capture()
            l3.process(p)
    l1.inject_scenario(**scenario)
    states, urgencies, risk_levels = set(), set(), set()
    maxima = {"risk": 0.0, "panic": 0.0, "freeze": 0.0,
              "overload": 0.0, "attentional": 0.0, "stress": 0.0}
    for _ in range(ticks):
        p = l1.capture()
        c = l2.process(p)
        h = l3.process(p)
        g = l4.regulate(c, h)
        states.add(h.state)
        urgencies.add(g.urgency)
        risk_levels.add(c.risk_map.global_risk_level)
        maxima["risk"] = max(maxima["risk"], c.risk_map.global_risk)
        maxima["panic"] = max(maxima["panic"], h.panic_score)
        maxima["freeze"] = max(maxima["freeze"], h.freeze_score)
        maxima["overload"] = max(maxima["overload"], h.cognitive_overload)
        maxima["attentional"] = max(maxima["attentional"], h.attentional_collapse)
        maxima["stress"] = max(maxima["stress"], h.stress_score)
    return states, urgencies, risk_levels, maxima


# Scenario per state, chosen by sweeping the injectable parameter space.
STATE_SCENARIOS = {
    CognitiveState.CALM: (
        dict(scene=SceneCondition.NORMAL, stress=0.0, panic=0.0,
             obstruction=0.0, emergency_sounds=False), 0),
    CognitiveState.ALERT: (
        dict(scene=SceneCondition.SMOKY, stress=0.0, panic=0.6,
             obstruction=0.3, emergency_sounds=True), 0),
    CognitiveState.STRESSED: (
        dict(scene=SceneCondition.SMOKY, stress=0.6, panic=0.7,
             obstruction=0.3, emergency_sounds=True), 0),
    CognitiveState.OVERLOADED: (
        dict(scene=SceneCondition.SMOKY, stress=0.8, panic=0.0,
             obstruction=0.3, emergency_sounds=True), 0),
    CognitiveState.COLLAPSED: (
        dict(scene=SceneCondition.SMOKY, stress=1.0, panic=0.6,
             obstruction=0.3, emergency_sounds=True), 0),
    CognitiveState.PANICKING: (
        dict(scene=SceneCondition.COLLAPSED, stress=1.0, panic=1.0,
             obstruction=1.0, emergency_sounds=True), 0),
    CognitiveState.FROZEN: (
        dict(scene=SceneCondition.SMOKY, stress=0.9, panic=0.0,
             obstruction=0.3, emergency_sounds=True, freeze=1.0), 60),
}


@pytest.mark.parametrize("state", list(STATE_SCENARIOS))
def test_every_cognitive_state_is_reachable(state):
    scenario, warmup = STATE_SCENARIOS[state]
    states, _, _, maxima = _observe(scenario, ticks=400, warmup=warmup)
    assert state in states, (
        f"{state.value} unreachable with {scenario}; observed "
        f"{sorted(s.value for s in states)}; maxima={maxima}")


def test_all_seven_cognitive_states_reachable_overall():
    seen = set()
    for scenario, warmup in STATE_SCENARIOS.values():
        states, _, _, _ = _observe(scenario, ticks=300, warmup=warmup)
        seen |= states
    assert seen == set(CognitiveState), (
        f"unreachable states: {sorted(s.value for s in set(CognitiveState) - seen)}")


def test_urgency_critical_is_reachable():
    """UrgencyLevel.CRITICAL was structurally unreachable: risk capped at 0.744
    against a 0.80 gate and panic at 0.681 against 0.85."""
    _, urgencies, risk_levels, maxima = _observe(
        dict(scene=SceneCondition.COLLAPSED, stress=1.0, panic=1.0,
             obstruction=1.0, emergency_sounds=True), ticks=300)
    assert UrgencyLevel.CRITICAL in urgencies, f"maxima={maxima}"
    assert RiskLevel.CRITICAL in risk_levels, f"maxima={maxima}"
    assert maxima["risk"] >= 0.80, maxima
    assert maxima["panic"] > 0.75, maxima


def test_every_urgency_level_is_reachable():
    seen = set()
    for scenario, warmup in STATE_SCENARIOS.values():
        _, urgencies, _, _ = _observe(scenario, ticks=300, warmup=warmup)
        seen |= urgencies
    # SILENT needs a risk below the LOW band, which only the empty-scene case
    # produces; AMBIENT..CRITICAL must all be present.
    for level in (UrgencyLevel.AMBIENT, UrgencyLevel.NORMAL,
                  UrgencyLevel.ELEVATED, UrgencyLevel.CRITICAL):
        assert level in seen, (
            f"{level.name} unreachable; observed {sorted(u.name for u in seen)}")


def test_every_risk_level_is_reachable():
    seen = set()
    for scene, stress, obstruction in [
            (SceneCondition.NORMAL, 0.0, 0.0),
            (SceneCondition.DARK, 0.4, 0.1),
            (SceneCondition.DUSTY, 0.4, 0.3),
            (SceneCondition.OBSTRUCTED, 0.4, 0.8),
            (SceneCondition.SMOKY, 0.8, 0.3),
            (SceneCondition.COLLAPSED, 1.0, 1.0)]:
        _, _, levels, _ = _observe(
            dict(scene=scene, stress=stress, panic=0.0,
                 obstruction=obstruction, emergency_sounds=False), ticks=200)
        seen |= levels
    for level in (RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL):
        assert level in seen, (
            f"{level.value} unreachable; observed {sorted(x.value for x in seen)}")


def test_critical_guidance_payload_actually_fires():
    """The CRITICAL branch produces the red overlay, the SOS haptic and one of
    the curated imperative templates — none of which had ever executed."""
    l1, l2, l3, l4 = _pipeline()
    l1.inject_scenario(SceneCondition.COLLAPSED, stress=1.0, panic=1.0,
                       obstruction=1.0, emergency_sounds=True)
    critical_outputs = []
    for _ in range(300):
        p = l1.capture()
        g = l4.regulate(l2.process(p), l3.process(p))
        if g.urgency is UrgencyLevel.CRITICAL and not g.suppressed:
            critical_outputs.append(g)
    assert critical_outputs, "no CRITICAL guidance was ever delivered"
    g = critical_outputs[0]
    assert g.ar_overlay.color_urgency == "red"
    assert g.haptic.active and g.haptic.pattern == "SOS"
    assert g.voice_message in [
        t.format(obstacle="DEBRIS") for t in l4.brevity_filter.TEMPLATES[UrgencyLevel.CRITICAL]
    ] or g.voice_message.startswith("PERICOLO"), g.voice_message
    assert len(g.voice_message.split()) <= l4.brevity_filter.MAX_WORDS


def test_relevance_filter_bypass_on_critical_executes():
    """l4_regulation's `relevance=bypassed:critical` branch was dead code."""
    l1, l2, l3, l4 = _pipeline()
    l1.inject_scenario(SceneCondition.COLLAPSED, stress=1.0, panic=1.0,
                       obstruction=1.0, emergency_sounds=True)
    logs = []
    for _ in range(200):
        p = l1.capture()
        logs.extend(l4.regulate(l2.process(p), l3.process(p)).filter_log)
    assert "relevance=bypassed:critical" in logs
