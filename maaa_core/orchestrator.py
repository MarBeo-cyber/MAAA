from __future__ import annotations

from .models import SensorFrame, AutopoieticStatus, GuidancePlan
from .perception import EmbodiedPerceptionAdapter, SceneGraphBuilder
from .human_state import HumanStateEstimator
from .regulatory import RegulatoryEngine
from .continuity import AutopoieticContinuityEngine, EpisodicMemory


class MAAAOrchestrator:
    """Integrated real-time loop prototype.

    L1 acquire -> L2 scene graph/risk -> L3 human state -> L4 regulation
    -> L5 continuity and memory.
    """

    def __init__(self, max_words: int = 9) -> None:
        self.perception = EmbodiedPerceptionAdapter()
        self.scene_builder = SceneGraphBuilder()
        self.human_estimator = HumanStateEstimator()
        self.regulatory = RegulatoryEngine(max_words=max_words)
        self.continuity = AutopoieticContinuityEngine()
        self.memory = EpisodicMemory()

    def process(self, frame: SensorFrame, status: AutopoieticStatus | None = None) -> GuidancePlan:
        acquired = self.perception.acquire(frame)
        graph = self.scene_builder.build(acquired)
        human = self.human_estimator.estimate(acquired)

        if status is not None:
            failsafe = self.continuity.check(status)
            if failsafe is not None:
                self._remember(failsafe, human.cognitive_entropy)
                return failsafe

        plan = self.regulatory.plan(graph, human)
        self._remember(plan, human.cognitive_entropy)
        return plan

    def _remember(self, plan: GuidancePlan, entropy: float) -> None:
        self.memory.append({
            "mode": plan.mode.value,
            "message": plan.message,
            "channels": [c.value for c in plan.channels],
            "priority": plan.priority.value,
            "reason": plan.reason,
            "cognitive_entropy": entropy,
        })
