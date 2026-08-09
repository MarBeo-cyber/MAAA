from __future__ import annotations

import maaa_config
from .models import SceneGraph, RiskLevel


class RiskEstimationEngine:
    """L2 — threshold classifier over caller-supplied hazard scores.

    This does not estimate anything. ``SceneGraphBuilder`` copies the floats in
    ``SensorFrame.hazards`` into the graph unchanged, and ``score()`` returns
    the largest of them and puts it in a band. Given
    ``hazards={"a": 0.42, "b": 0.77}`` it returns ``(0.77, HIGH, "b")``.
    The hazard numbers are invented by whoever builds the SensorFrame — there
    is no perception, no model and no inference between input and output.

    The bands are shared with ``layers.l2_cognition`` via
    ``config/default.yaml`` so the two implementations classify identically.
    """

    def __init__(self) -> None:
        # Descending order so the first match wins.
        self._bands = sorted(maaa_config.risk_bands(), key=lambda b: -b[1])

    def score(self, graph: SceneGraph) -> tuple[float, RiskLevel, str | None]:
        item, risk = graph.highest_risk()
        for name, lo, _hi in self._bands:
            if risk >= lo:
                return risk, RiskLevel[name], item
        return risk, RiskLevel.SAFE, item

    def best_safe_path(self, graph: SceneGraph) -> str | None:
        candidates = [
            (name, data)
            for name, data in graph.nodes.items()
            if data.get("kind") == "path" and data.get("metadata", {}).get("status") == "safe"
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda x: x[1].get("risk", 0.0))[0]
