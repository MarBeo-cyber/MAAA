from __future__ import annotations

from .models import SceneGraph, RiskLevel


class RiskEstimationEngine:
    """L2 - estimates risk level from the scene graph."""

    def score(self, graph: SceneGraph) -> tuple[float, RiskLevel, str | None]:
        item, risk = graph.highest_risk()
        if risk >= 0.85:
            return risk, RiskLevel.CRITICAL, item
        if risk >= 0.65:
            return risk, RiskLevel.HIGH, item
        if risk >= 0.35:
            return risk, RiskLevel.MEDIUM, item
        return risk, RiskLevel.LOW, item

    def best_safe_path(self, graph: SceneGraph) -> str | None:
        candidates = [
            (name, data)
            for name, data in graph.nodes.items()
            if data.get("kind") == "path" and data.get("metadata", {}).get("status") == "safe"
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda x: x[1].get("risk", 0.0))[0]
