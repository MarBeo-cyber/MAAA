from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from .models import AutopoieticStatus, GuidancePlan, GuidanceMode, OutputChannel, RiskLevel


@dataclass
class EpisodicMemory:
    events: list[dict] = field(default_factory=list)
    max_events: int = 120

    def append(self, event: dict) -> None:
        event = dict(event)
        event.setdefault("timestamp", datetime.utcnow().isoformat())
        self.events.append(event)
        if len(self.events) > self.max_events:
            # WAAA-inspired conservative pruning: keep most recent and critical events.
            critical = [e for e in self.events if e.get("priority") in {"HIGH", "CRITICAL"}]
            recent = self.events[-self.max_events // 2 :]
            merged = []
            seen = set()
            for e in critical + recent:
                key = (e.get("timestamp"), e.get("message"))
                if key not in seen:
                    merged.append(e)
                    seen.add(key)
            self.events = merged[-self.max_events :]


class AutopoieticContinuityEngine:
    """L5 - continuity and failsafe."""

    def check(self, status: AutopoieticStatus) -> GuidancePlan | None:
        if status.should_failsafe():
            return GuidancePlan(
                mode=GuidanceMode.FAILSAFE,
                message="Modalita sicura. Segui uscita visibile.",
                channels=[OutputChannel.VOICE, OutputChannel.AR],
                priority=RiskLevel.HIGH,
                reason="autopoietic_failsafe",
            )
        return None
