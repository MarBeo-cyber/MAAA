from __future__ import annotations

from typing import Optional

import maaa_config
from .models import SceneGraph, HumanState, GuidancePlan, GuidanceMode, OutputChannel, RiskLevel
from .risk import RiskEstimationEngine

#: Human override commands accepted in SensorFrame.user_command (NFR-05).
#:   mute   — latch guidance off; CRITICAL priority still gets through
#:   resume — clear the latch
#:   stop   — one-shot: this frame produces no guidance at all
USER_COMMANDS = ("mute", "resume", "stop")


class RegulatoryEngine:
    """L4 - Symbiotic Regulation.

    Applies four filters before producing output:
    relevance, timing, brevity and urgency.
    """

    def __init__(self, max_words: Optional[int] = None) -> None:
        self.max_words = max_words if max_words is not None else maaa_config.max_words()
        self.risk_engine = RiskEstimationEngine()

    def plan(self, graph: SceneGraph, human: HumanState) -> GuidancePlan:
        risk_score, risk_level, risk_item = self.risk_engine.score(graph)
        safe_path = self.risk_engine.best_safe_path(graph)

        if risk_level == RiskLevel.CRITICAL:
            if safe_path:
                msg = f"STOP. Evita {risk_item}. Vai verso {safe_path}."
                mode = GuidanceMode.STOP
                channels = [OutputChannel.VOICE, OutputChannel.AR, OutputChannel.HAPTIC]
            else:
                msg = f"STOP. Allontanati da {risk_item}."
                mode = GuidanceMode.STOP
                channels = [OutputChannel.VOICE, OutputChannel.HAPTIC]
            return self._shorten(GuidancePlan(mode=mode, message=msg, channels=channels, target_path=safe_path, avoid=risk_item, priority=risk_level, reason="critical_risk"))

        if risk_level == RiskLevel.HIGH:
            if safe_path:
                msg = f"Evita {risk_item}. Procedi verso {safe_path}."
                mode = GuidanceMode.GUIDE
                channels = [OutputChannel.VOICE, OutputChannel.AR]
            else:
                msg = f"Attenzione: {risk_item}. Rallenta."
                mode = GuidanceMode.WARN
                channels = [OutputChannel.VOICE, OutputChannel.HAPTIC]
            return self._shorten(GuidancePlan(mode=mode, message=msg, channels=channels, target_path=safe_path, avoid=risk_item, priority=risk_level, reason="high_risk"))

        if human.cognitive_entropy >= 0.68:
            return self._shorten(GuidancePlan(
                mode=GuidanceMode.GUIDE,
                message="Respira. Guarda avanti. Segui la guida.",
                channels=[OutputChannel.VOICE, OutputChannel.AR],
                target_path=safe_path,
                priority=RiskLevel.MEDIUM,
                reason="cognitive_entropy_high",
            ))

        return GuidancePlan(
            mode=GuidanceMode.OBSERVE,
            message="Monitoraggio attivo.",
            channels=[OutputChannel.LOG],
            target_path=safe_path,
            priority=risk_level,
            reason="below_intervention_threshold",
        )

    def _shorten(self, plan: GuidancePlan) -> GuidancePlan:
        words = plan.message.split()
        if len(words) > self.max_words:
            plan.message = " ".join(words[: self.max_words])
        plan.max_words = self.max_words
        return plan
