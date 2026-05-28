"""
MAAA — Layer 2: Situational Cognition (Cognizione Situazionale)

Costruisce e aggiorna il Scene Graph semantico dell'ambiente in tempo reale:
  - Object Detection e classificazione semantica (YOLOv9 in produzione)
  - Depth Estimation → distanze da ostacoli
  - SLAM → mappa 3D + localizzazione
  - Risk Estimation Engine → probabilità rischio per ogni elemento del grafo
  - Causal Inference → relazioni causa-effetto tra oggetti e rischi
  - Event Prediction → anticipazione eventi pericolosi

Output: SceneGraph aggiornato con RiskMap per ogni elemento
"""

from __future__ import annotations

import time
import math
import random
import logging
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from layers.l1_perception import PerceptionFrame, SceneCondition

logger = logging.getLogger("maaa.l2_cognition")


# ── Risk Model ────────────────────────────────────────────────────────────────

class RiskLevel(Enum):
    SAFE     = "SAFE"
    LOW      = "LOW"
    MEDIUM   = "MEDIUM"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"

    @property
    def numeric(self) -> float:
        return {"SAFE": 0.0, "LOW": 0.25, "MEDIUM": 0.5, "HIGH": 0.75, "CRITICAL": 1.0}[self.value]


RISK_THRESHOLDS = {
    RiskLevel.SAFE:     (0.00, 0.10),
    RiskLevel.LOW:      (0.10, 0.35),
    RiskLevel.MEDIUM:   (0.35, 0.60),
    RiskLevel.HIGH:     (0.60, 0.80),
    RiskLevel.CRITICAL: (0.80, 1.00),
}

def classify_risk(probability: float) -> RiskLevel:
    for level, (lo, hi) in RISK_THRESHOLDS.items():
        if lo <= probability < hi:
            return level
    return RiskLevel.CRITICAL


# ── Scene Graph Data Structures ───────────────────────────────────────────────

@dataclass
class SceneObject:
    """A detected object in the environment."""
    object_id: str
    category: str           # "staircase", "door", "wall", "person", "debris"...
    confidence: float       # Detection confidence 0–1
    distance_m: float       # Distance from user
    bearing_deg: float      # Compass direction from user (0=ahead)
    is_obstacle: bool
    is_exit: bool
    risk_probability: float = 0.0
    risk_level: RiskLevel = RiskLevel.SAFE
    recommended_action: str = ""
    causal_links: list[str] = field(default_factory=list)  # IDs of causally related objects


@dataclass
class RiskMap:
    """Spatial risk distribution around the user."""
    timestamp: float
    objects: list[SceneObject]
    global_risk: float              # 0–1 aggregate environment risk
    global_risk_level: RiskLevel
    passable_directions: list[float]  # Bearings (degrees) that are passable
    recommended_path_bearing: Optional[float]  # Best direction to move
    structural_integrity: float     # 0=collapse imminent, 1=stable
    time_to_act_seconds: Optional[float]  # Estimated time before situation worsens

    def get_critical_objects(self) -> list[SceneObject]:
        return [o for o in self.objects if o.risk_level == RiskLevel.CRITICAL]

    def get_exits(self) -> list[SceneObject]:
        return sorted([o for o in self.objects if o.is_exit],
                      key=lambda o: o.risk_probability)


@dataclass
class SLAMState:
    """SLAM localization and map state."""
    timestamp: float
    position_x: float      # meters from origin
    position_y: float
    position_z: float
    heading_deg: float     # 0=north
    map_coverage: float    # 0–1 fraction of area mapped
    localization_confidence: float  # 0–1
    loop_closure: bool     # True if SLAM confirmed position via known landmark


@dataclass
class CognitionFrame:
    """Full Layer 2 output."""
    timestamp: float
    scene_objects: list[SceneObject]
    risk_map: RiskMap
    slam: SLAMState
    event_predictions: list[str]    # ["structural_collapse_likely", "smoke_increasing"]
    causal_summary: str             # Human-readable causal explanation


# ── Object Detection (simulated / production) ─────────────────────────────────

OBJECT_TEMPLATES = {
    SceneCondition.NORMAL: [
        ("door_east",     "door",      True,  False, 3.5, 45.0),
        ("corridor_n",    "corridor",  False, False, 2.0, 0.0),
        ("staircase",     "staircase", False, True,  8.0, 270.0),
        ("window_w",      "window",    False, True,  4.0, 180.0),
    ],
    SceneCondition.COLLAPSED: [
        ("debris_l",      "debris",    True,  False, 1.2, 30.0),
        ("debris_r",      "debris",    True,  False, 1.5, 330.0),
        ("staircase",     "staircase", True,  False, 5.0, 90.0),   # dangerous
        ("exit_n",        "door",      False, True,  6.0, 0.0),
        ("crack_wall",    "wall",      False, False, 2.0, 180.0),
    ],
    SceneCondition.SMOKY: [
        ("exit_n",        "door",      False, True,  7.0, 15.0),
        ("person",        "person",    False, False, 3.0, 60.0),
        ("window_e",      "window",    False, True,  5.0, 90.0),
    ],
}


def _compute_risk(category: str, distance_m: float,
                  scene: SceneCondition, env_quality: float) -> float:
    """Heuristic risk model — replaced by neural risk estimator in production."""
    base_risk = {
        "debris":    0.85,
        "staircase": 0.70 if scene == SceneCondition.COLLAPSED else 0.15,
        "wall":      0.55 if scene == SceneCondition.COLLAPSED else 0.05,
        "door":      0.10,
        "corridor":  0.20 if scene != SceneCondition.NORMAL else 0.05,
        "window":    0.10,
        "person":    0.10,
    }.get(category, 0.3)

    # Proximity amplifier: closer = riskier
    proximity_factor = max(0, 1.0 - distance_m / 10.0) * 0.2

    # Environmental degradation amplifier
    env_factor = (1.0 - env_quality) * 0.15

    return min(1.0, base_risk + proximity_factor + env_factor + random.gauss(0, 0.02))


def _recommend_action(obj: SceneObject) -> str:
    if obj.risk_level == RiskLevel.CRITICAL:
        return f"Evitare — rischio crollo ({obj.category})"
    if obj.risk_level == RiskLevel.HIGH:
        return f"Allontanarsi da {obj.category}"
    if obj.is_exit and obj.risk_level in (RiskLevel.SAFE, RiskLevel.LOW):
        return f"Percorso prioritario — uscita a {obj.distance_m:.0f}m"
    if obj.risk_level == RiskLevel.MEDIUM:
        return f"Procedere con cautela verso {obj.category}"
    return ""


class L2SituationalCognition:
    """
    Layer 2 — Situational Cognition.

    In production integrates:
      - YOLOv9 object detection on AR video stream
      - Depth Anything v2 for monocular depth estimation
      - ORB-SLAM3 for localization and mapping
      - Neo4j / GNN for scene graph storage and causal inference

    In simulation mode synthesises all outputs from PerceptionFrame.
    """

    def __init__(self, simulation_mode: bool = True):
        self.simulation_mode = simulation_mode
        self._tick = 0
        self._slam_x = 0.0
        self._slam_y = 0.0
        self._slam_heading = 0.0
        logger.info("[L2] Situational Cognition initialized")

    def process(self, perception: PerceptionFrame) -> CognitionFrame:
        self._tick += 1
        ts = time.time()

        # 1. Detect objects
        objects = self._detect_objects(perception)

        # 2. Compute risk per object
        for obj in objects:
            obj.risk_probability = _compute_risk(
                obj.category, obj.distance_m,
                perception.video.scene_condition,
                perception.environment_quality
            )
            obj.risk_level = classify_risk(obj.risk_probability)
            obj.recommended_action = _recommend_action(obj)

        # 3. Build risk map
        risk_map = self._build_risk_map(objects, perception, ts)

        # 4. Update SLAM
        slam = self._update_slam(perception, ts)

        # 5. Predict future events
        predictions = self._predict_events(perception, risk_map)

        # 6. Causal summary
        causal = self._causal_summary(risk_map, predictions)

        return CognitionFrame(
            timestamp=ts,
            scene_objects=objects,
            risk_map=risk_map,
            slam=slam,
            event_predictions=predictions,
            causal_summary=causal,
        )

    def _detect_objects(self, perception: PerceptionFrame) -> list[SceneObject]:
        scene = perception.video.scene_condition
        templates = OBJECT_TEMPLATES.get(scene, OBJECT_TEMPLATES[SceneCondition.NORMAL])
        objects = []
        for obj_id, category, is_obstacle, is_exit, dist, bearing in templates:
            # Slight noise per tick
            noise_dist = random.gauss(0, 0.1)
            objects.append(SceneObject(
                object_id=obj_id,
                category=category,
                confidence=max(0.5, random.gauss(0.85, 0.05)),
                distance_m=max(0.5, dist + noise_dist),
                bearing_deg=bearing + random.gauss(0, 2),
                is_obstacle=is_obstacle,
                is_exit=is_exit,
            ))
        # Add occluded obstacles when smoke/dust present
        if perception.video.smoke_probability > 0.5:
            objects.append(SceneObject(
                object_id="occluded_obstacle",
                category="debris",
                confidence=0.45,
                distance_m=max(0.8, random.gauss(2.0, 0.5)),
                bearing_deg=random.uniform(0, 360),
                is_obstacle=True,
                is_exit=False,
            ))
        return objects

    def _build_risk_map(self, objects: list[SceneObject],
                        perception: PerceptionFrame, ts: float) -> RiskMap:
        global_risk = (
            perception.video.smoke_probability * 0.35 +
            perception.video.dust_level * 0.15 +
            (1.0 - perception.environment_quality) * 0.25 +
            (max((o.risk_probability for o in objects), default=0.0) * 0.25)
        )
        global_risk = min(1.0, global_risk + random.gauss(0, 0.01))

        # Passable directions: bearings without critical obstacles nearby
        blocked = {o.bearing_deg for o in objects
                   if o.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
                   and o.distance_m < 3.0}
        passable = []
        for bearing in range(0, 360, 45):
            if not any(abs(bearing - b) < 30 for b in blocked):
                passable.append(float(bearing))

        # Best exit direction
        exits = [o for o in objects if o.is_exit and
                 o.risk_level in (RiskLevel.SAFE, RiskLevel.LOW)]
        best_exit_bearing = exits[0].bearing_deg if exits else None

        structural = max(0.0, 1.0 - global_risk * 0.8 -
                         perception.video.smoke_probability * 0.2)

        time_to_act = None
        if global_risk > 0.7:
            time_to_act = max(10.0, (1.0 - global_risk) * 120.0)

        return RiskMap(
            timestamp=ts,
            objects=objects,
            global_risk=global_risk,
            global_risk_level=classify_risk(global_risk),
            passable_directions=passable,
            recommended_path_bearing=best_exit_bearing,
            structural_integrity=structural,
            time_to_act_seconds=time_to_act,
        )

    def _update_slam(self, perception: PerceptionFrame, ts: float) -> SLAMState:
        # Simulate slow movement
        self._slam_x += random.gauss(0, 0.05)
        self._slam_y += random.gauss(0, 0.05)
        self._slam_heading += random.gauss(0, 1.0)
        coverage = min(1.0, self._tick * 0.01)
        return SLAMState(
            timestamp=ts,
            position_x=self._slam_x,
            position_y=self._slam_y,
            position_z=0.0,
            heading_deg=self._slam_heading % 360,
            map_coverage=coverage,
            localization_confidence=min(1.0, coverage * 1.2),
            loop_closure=self._tick % 50 == 0,
        )

    def _predict_events(self, perception: PerceptionFrame,
                        risk_map: RiskMap) -> list[str]:
        preds = []
        if perception.video.smoke_probability > 0.4:
            preds.append("smoke_increasing")
        if risk_map.structural_integrity < 0.4:
            preds.append("structural_collapse_likely")
        if risk_map.global_risk > 0.7 and risk_map.time_to_act_seconds:
            preds.append(f"critical_threshold_in_{int(risk_map.time_to_act_seconds)}s")
        if perception.audio.smoke_alarm:
            preds.append("fire_alarm_active")
        return preds

    def _causal_summary(self, risk_map: RiskMap,
                        predictions: list[str]) -> str:
        criticals = risk_map.get_critical_objects()
        if not criticals and risk_map.global_risk < 0.3:
            return "Ambiente stabile. Nessuna minaccia immediata rilevata."
        parts = []
        if criticals:
            names = ", ".join(o.category for o in criticals[:2])
            parts.append(f"Rischio critico rilevato: {names}")
        if "structural_collapse_likely" in predictions:
            parts.append("Rischio crollo strutturale imminente")
        if "smoke_increasing" in predictions:
            parts.append("Fumo in aumento — riduzione visibilità")
        return ". ".join(parts) + "."
