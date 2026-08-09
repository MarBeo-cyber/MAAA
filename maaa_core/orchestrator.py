from __future__ import annotations

import logging
from typing import Optional

from .models import (SensorFrame, AutopoieticStatus, GuidancePlan, GuidanceMode,
                     OutputChannel, RiskLevel)
from .perception import EmbodiedPerceptionAdapter, SceneGraphBuilder
from .human_state import HumanStateEstimator
from .regulatory import RegulatoryEngine, USER_COMMANDS
from .continuity import AutopoieticContinuityEngine, EpisodicMemory

logger = logging.getLogger("maaa.core.orchestrator")


class MAAAOrchestrator:
    """Integrated real-time loop prototype.

    L1 acquire -> L2 scene graph/risk -> L3 human state -> L4 regulation
    -> L5 continuity and memory.
    """

    def __init__(self, max_words: Optional[int] = None) -> None:
        self.perception = EmbodiedPerceptionAdapter()
        self.scene_builder = SceneGraphBuilder()
        self.human_estimator = HumanStateEstimator()
        self.regulatory = RegulatoryEngine(max_words=max_words)
        self.continuity = AutopoieticContinuityEngine()
        self.memory = EpisodicMemory()
        self.muted = False

    def process(self, frame: SensorFrame, status: AutopoieticStatus | None = None) -> GuidancePlan:
        acquired = self.perception.acquire(frame)
        graph = self.scene_builder.build(acquired)
        human = self.human_estimator.estimate(acquired)

        override_plan = self._apply_user_command(acquired)

        if status is not None:
            failsafe = self.continuity.check(status)
            if failsafe is not None:
                # Failsafe outranks a mute: the user can silence advice,
                # not the failsafe notice (docs/SAFETY.md).
                self._remember(failsafe, human.cognitive_entropy)
                return failsafe

        plan = self.regulatory.plan(graph, human)

        if override_plan is not None and plan.priority != RiskLevel.CRITICAL:
            self._remember(override_plan, human.cognitive_entropy)
            return override_plan

        self._remember(plan, human.cognitive_entropy)
        return plan

    def _apply_user_command(self, frame: SensorFrame) -> Optional[GuidancePlan]:
        """Handle SensorFrame.user_command; return a plan when output is overridden.

        NFR-05 requires human override to always be available. ``mute`` latches
        until ``resume``; ``stop`` suppresses this frame only. Neither can
        silence CRITICAL priority — see docs/SAFETY.md.
        """
        command = frame.user_command
        one_shot = False
        if command is not None:
            if command not in USER_COMMANDS:
                logger.warning("[core] Ignoring unknown user_command %r (expected %s)",
                               command, USER_COMMANDS)
            elif command == "mute":
                self.muted = True
            elif command == "resume":
                self.muted = False
            elif command == "stop":
                one_shot = True

        if not (self.muted or one_shot):
            return None

        return GuidancePlan(
            mode=GuidanceMode.OBSERVE,
            message="Guida silenziata su richiesta.",
            channels=[OutputChannel.LOG],
            priority=RiskLevel.SAFE,
            reason="user_override_stop" if one_shot else "user_override_mute",
        )

    def _remember(self, plan: GuidancePlan, entropy: float) -> None:
        self.memory.append({
            "mode": plan.mode.value,
            "message": plan.message,
            "channels": [c.value for c in plan.channels],
            "priority": plan.priority.value,
            "reason": plan.reason,
            "cognitive_entropy": entropy,
        })
