from __future__ import annotations

from .models import SensorFrame, SceneGraph


class EmbodiedPerceptionAdapter:
    """L1 - Embodied Perception.

    This prototype accepts already simulated sensor frames. Production adapters
    would wrap AR glasses video/audio, IMU, depth estimation, microphone and
    optional eye tracking.
    """

    def acquire(self, frame: SensorFrame) -> SensorFrame:
        return frame


class SceneGraphBuilder:
    """L2 - Situational Cognition: builds a compact scene graph."""

    def build(self, frame: SensorFrame) -> SceneGraph:
        graph = SceneGraph()
        for obj in frame.detected_objects:
            graph.add_node(obj, kind="object", risk=frame.hazards.get(obj, 0.0))

        for hazard, risk in frame.hazards.items():
            if hazard not in graph.nodes:
                graph.add_node(hazard, kind="hazard", risk=risk)

        for path in frame.safe_paths:
            graph.add_node(path, kind="path", risk=frame.hazards.get(path, 0.0), metadata={"status": "safe"})

        for path in frame.blocked_paths:
            graph.add_node(path, kind="path", risk=max(frame.hazards.get(path, 0.75), 0.75), metadata={"status": "blocked"})

        return graph
