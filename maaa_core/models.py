from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
from uuid import uuid4
from datetime import datetime


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class OutputChannel(str, Enum):
    VOICE = "voice"
    AR = "ar"
    HAPTIC = "haptic"
    LOG = "log"


class GuidanceMode(str, Enum):
    OBSERVE = "observe"
    WARN = "warn"
    GUIDE = "guide"
    STOP = "stop"
    FAILSAFE = "failsafe"


@dataclass
class SensorFrame:
    """Minimal simulated input frame.

    In production this would be populated by AR glasses, IMU, microphone,
    depth estimation, visual detection and human-state sensors.
    """

    frame_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    detected_objects: List[str] = field(default_factory=list)
    blocked_paths: List[str] = field(default_factory=list)
    safe_paths: List[str] = field(default_factory=list)
    hazards: Dict[str, float] = field(default_factory=dict)
    imu_instability: float = 0.0
    audio_stress: float = 0.0
    gaze_fixation_risk: float = 0.0
    user_command: Optional[str] = None


@dataclass
class SceneGraph:
    nodes: Dict[str, Dict] = field(default_factory=dict)
    edges: List[Dict] = field(default_factory=list)

    def add_node(self, name: str, kind: str, risk: float = 0.0, metadata: Dict | None = None) -> None:
        self.nodes[name] = {
            "kind": kind,
            "risk": max(0.0, min(1.0, risk)),
            "metadata": metadata or {},
        }

    def highest_risk(self) -> tuple[str | None, float]:
        if not self.nodes:
            return None, 0.0
        item = max(self.nodes.items(), key=lambda x: x[1].get("risk", 0.0))
        return item[0], item[1].get("risk", 0.0)


@dataclass
class HumanState:
    stress: float
    overload: float
    panic: float
    attention_collapse: float

    @property
    def cognitive_entropy(self) -> float:
        return round(
            0.35 * self.stress
            + 0.30 * self.overload
            + 0.20 * self.panic
            + 0.15 * self.attention_collapse,
            4,
        )


@dataclass
class GuidancePlan:
    mode: GuidanceMode
    message: str
    channels: List[OutputChannel]
    target_path: Optional[str] = None
    avoid: Optional[str] = None
    priority: RiskLevel = RiskLevel.LOW
    max_words: int = 9
    reason: str = ""


@dataclass
class AutopoieticStatus:
    system_ok: bool
    battery_pct: float
    sensor_integrity: float
    latency_ms: float
    fallback_active: bool = False

    def should_failsafe(self, max_latency_ms: float = 200.0) -> bool:
        return (
            not self.system_ok
            or self.battery_pct < 5
            or self.sensor_integrity < 0.45
            or self.latency_ms > max_latency_ms
        )
