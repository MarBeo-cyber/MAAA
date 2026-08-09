"""L4 regressions: relevance reason, curated templates, human override."""

from __future__ import annotations

import pytest

from layers.l1_perception import L1EmbodiedPerception, SceneCondition
from layers.l2_cognition import L2SituationalCognition
from layers.l3_human_state import L3HumanStateMonitor
from layers.l4_regulation import (BrevityFilter, L4SymbioticRegulation,
                                  RelevanceFilter, UrgencyLevel)


def _pipeline():
    return (L1EmbodiedPerception(True), L2SituationalCognition(True),
            L3HumanStateMonitor(), L4SymbioticRegulation())


def _cognition(scene=SceneCondition.SMOKY, **kw):
    l1, l2 = L1EmbodiedPerception(True), L2SituationalCognition(True)
    l1.inject_scenario(scene, **{"stress": 0.6, "panic": 0.3,
                                 "obstruction": 0.2, "emergency_sounds": True, **kw})
    return l2.process(l1.capture())


def test_relevance_reason_reports_the_real_delta():
    """`self._last_risk = risk` was executed before the reason string
    interpolated abs(risk - self._last_risk), so it always read 0.00."""
    f = RelevanceFilter(change_threshold=0.15)
    c = _cognition()
    c.risk_map.global_risk = 0.70
    relevant, reason = f.is_relevant(c, "msg")
    assert relevant
    assert reason == "risk_delta=0.70", reason

    c2 = _cognition()
    c2.risk_map.global_risk = 0.20
    relevant, reason = f.is_relevant(c2, "other")
    assert relevant
    assert reason == "risk_delta=0.50", reason


def test_relevance_reason_is_never_a_constant_zero_in_a_live_run():
    l1, l2, l3, l4 = _pipeline()
    reasons = set()
    for scene, stress in [(SceneCondition.NORMAL, 0.0),
                          (SceneCondition.SMOKY, 0.6),
                          (SceneCondition.COLLAPSED, 0.9),
                          (SceneCondition.NORMAL, 0.1)]:
        l1.inject_scenario(scene, stress, 0.2, 0.3, True)
        for _ in range(40):
            p = l1.capture()
            for entry in l4.regulate(l2.process(p), l3.process(p)).filter_log:
                if entry.startswith("relevance=pass:risk_delta="):
                    reasons.add(entry)
    assert reasons, "relevance filter never passed"
    assert reasons != {"relevance=pass:risk_delta=0.00"}, reasons


def test_curated_templates_are_the_messages_actually_spoken():
    """BrevityFilter.TEMPLATES held 18 curated messages that nothing
    referenced; generate() built its own strings instead."""
    bf = BrevityFilter()
    seen = set()
    for scene, stress, panic, obstruction in [
            (SceneCondition.NORMAL, 0.0, 0.0, 0.0),
            (SceneCondition.SMOKY, 0.6, 0.3, 0.2),
            (SceneCondition.DARK, 0.4, 0.2, 0.1),
            (SceneCondition.OBSTRUCTED, 0.4, 0.1, 0.9),
            (SceneCondition.COLLAPSED, 1.0, 1.0, 1.0)]:
        for _ in range(30):
            c = _cognition(scene, stress=stress, panic=panic,
                           obstruction=obstruction)
            for urgency in UrgencyLevel:
                seen.add(bf.generate(urgency, c))
    assert seen, "no messages generated"
    for message in seen:
        assert len(message.split()) <= bf.MAX_WORDS, message
    # Every generated message must trace back to a template.
    patterns = {t.split("{")[0].strip()
                for group in bf.TEMPLATES.values() for t in group}
    for message in seen:
        assert any(message.startswith(p) for p in patterns if p), message


def test_shorten_uses_both_of_its_parameters():
    """shorten(message, urgency, cognition) ignored urgency and cognition and
    truncated the word list mid-sentence."""
    bf = BrevityFilter()
    c = _cognition(SceneCondition.COLLAPSED, stress=1.0, panic=1.0, obstruction=1.0)
    long_message = " ".join(["parola"] * 30)
    out = bf.shorten(long_message, UrgencyLevel.CRITICAL, c)
    assert out != " ".join(["parola"] * bf.MAX_WORDS) + "."
    assert out == bf.select_template(UrgencyLevel.CRITICAL, c)
    assert len(out.split()) <= bf.MAX_WORDS


def test_max_words_comes_from_config():
    import maaa_config
    assert BrevityFilter.MAX_WORDS == maaa_config.max_words()


def test_human_override_mute_silences_non_critical_guidance():
    l1, l2, l3, l4 = _pipeline()
    l1.inject_scenario(SceneCondition.SMOKY, 0.6, 0.3, 0.2, True)
    l4.set_override("mute")
    for _ in range(60):
        p = l1.capture()
        g = l4.regulate(l2.process(p), l3.process(p))
        if g.urgency.value < UrgencyLevel.CRITICAL.value:
            assert g.suppressed
            assert g.suppression_reason == "user_override_mute"


def test_human_override_cannot_silence_critical_guidance():
    """docs/SAFETY.md: the user can switch off advice, not the danger warning."""
    l1, l2, l3, l4 = _pipeline()
    l1.inject_scenario(SceneCondition.COLLAPSED, 1.0, 1.0, 1.0, True)
    l4.set_override("mute")
    delivered = [g for g in (
        l4.regulate(l2.process(p), l3.process(p))
        for p in (l1.capture() for _ in range(200)))
        if not g.suppressed]
    assert delivered, "mute silenced CRITICAL guidance"
    assert all(g.urgency is UrgencyLevel.CRITICAL for g in delivered)


def test_human_override_resume_restores_guidance():
    l1, l2, l3, l4 = _pipeline()
    l1.inject_scenario(SceneCondition.SMOKY, 0.6, 0.3, 0.2, True)
    l4.set_override("mute")
    assert l4.override == "mute"
    l4.set_override("resume")
    assert l4.override is None
    delivered = 0
    for _ in range(60):
        p = l1.capture()
        if not l4.regulate(l2.process(p), l3.process(p)).suppressed:
            delivered += 1
    assert delivered > 0


def test_unknown_override_command_is_rejected():
    l4 = L4SymbioticRegulation()
    with pytest.raises(ValueError):
        l4.set_override("self_destruct")
