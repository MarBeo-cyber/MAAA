"""
MAAA — Layer 5: Autopoietic Continuity (Continuità Autopoietica)

Garantisce la continuità operativa del sistema e della relazione uomo-sistema:
  - Self-monitoring: verifica integrità propria (latenze, modelli, sensori)
  - Integrità operativa umana: monitora se l'utente è ancora operativo
  - Preservazione della relazione uomo-sistema: evita rotture del canale cognitivo
  - Memoria autobiografica episodica a 3 livelli (Working / Episodica / Autobiografica)
  - Recovery e Failsafe: degrada gracefully, riavvia moduli in errore

Memoria a 3 livelli (Sez. 6 del documento):
  ┌─────────────────────────────────────────────────────┐
  │ Working Memory   │ 60s rolling window, volatile RAM  │
  │ Episodica        │ Sessione corrente, persistita     │
  │ Autobiografica   │ Profilo utente, VectorDB          │
  └─────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import time
import math
import json
import uuid
import sqlite3
import logging
import threading
from dataclasses import dataclass, field, asdict
from typing import Optional
from collections import deque
from pathlib import Path

import numpy as np

from layers.l2_cognition import CognitionFrame, RiskLevel
from layers.l3_human_state import HumanStateFrame, CognitiveState
from layers.l4_regulation import GuidanceOutput, UrgencyLevel

logger = logging.getLogger("maaa.l5_continuity")


# ── Memory Data Structures ────────────────────────────────────────────────────

@dataclass
class MemoryEvent:
    """Atomic event stored in episodic memory."""
    event_id: str
    timestamp: float
    session_id: str
    event_type: str          # "guidance_delivered", "risk_detected", "panic_spike", ...
    content: dict            # Flexible payload
    gps_lat: float = 0.0
    gps_lon: float = 0.0
    human_state_summary: str = ""
    risk_level: str = "SAFE"
    embedding: Optional[list[float]] = None   # Semantic vector (production: LLM embedding)

    def to_text(self) -> str:
        """Serialize event to text for embedding."""
        return (f"{self.event_type} at t={self.timestamp:.0f}: "
                f"{json.dumps(self.content, ensure_ascii=False)} "
                f"risk={self.risk_level} state={self.human_state_summary}")


# ── Working Memory (volatile, 60s rolling window) ─────────────────────────────

class WorkingMemory:
    """
    Volatile RAM buffer for the last 60 seconds of context.
    Used for: real-time decision coherence, avoiding contradictory instructions.
    """

    WINDOW_SECONDS = 60.0

    def __init__(self):
        self._events: deque[MemoryEvent] = deque(maxlen=500)
        self._lock = threading.Lock()

    def add(self, event: MemoryEvent):
        with self._lock:
            self._events.append(event)
            self._evict_old()

    def _evict_old(self):
        cutoff = time.time() - self.WINDOW_SECONDS
        while self._events and self._events[0].timestamp < cutoff:
            self._events.popleft()

    def recent(self, n: int = 10) -> list[MemoryEvent]:
        with self._lock:
            return list(self._events)[-n:]

    def last_guidance(self) -> Optional[str]:
        """Most recent guidance message delivered."""
        with self._lock:
            for ev in reversed(self._events):
                if ev.event_type == "guidance_delivered":
                    return ev.content.get("message", "")
        return None

    def has_recent_event(self, event_type: str, within_seconds: float = 10.0) -> bool:
        cutoff = time.time() - within_seconds
        with self._lock:
            return any(e.event_type == event_type and e.timestamp >= cutoff
                       for e in self._events)

    @property
    def size(self) -> int:
        return len(self._events)


# ── Episodic Memory (session-persistent, SQLite) ──────────────────────────────

class EpisodicMemory:
    """
    Session-level episodic memory: persists all events for the current mission.
    Storage: SQLite (on-device, encrypted in production via SQLCipher).
    Enables: mission debriefing, pattern detection, coherence across interruptions.
    """

    def __init__(self, db_path: str = "/tmp/maaa_episodes.db"):
        self.db_path = db_path
        self.session_id = str(uuid.uuid4())[:8]
        self._conn = self._init_db()
        self._lock = threading.Lock()
        logger.info("[L5/EpisodicMem] Session %s started, db=%s",
                    self.session_id, db_path)

    def _init_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                event_id         TEXT PRIMARY KEY,
                timestamp        REAL,
                session_id       TEXT,
                event_type       TEXT,
                content_json     TEXT,
                gps_lat          REAL,
                gps_lon          REAL,
                human_state      TEXT,
                risk_level       TEXT,
                embedding_json   TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_session ON episodes(session_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_type ON episodes(event_type)")
        conn.commit()
        return conn

    def add(self, event: MemoryEvent):
        with self._lock:
            self._conn.execute("""
                INSERT OR REPLACE INTO episodes VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (
                event.event_id,
                event.timestamp,
                event.session_id,
                event.event_type,
                json.dumps(event.content, ensure_ascii=False),
                event.gps_lat,
                event.gps_lon,
                event.human_state_summary,
                event.risk_level,
                json.dumps(event.embedding) if event.embedding else None,
            ))
            self._conn.commit()

    def get_session_events(self, limit: int = 200) -> list[MemoryEvent]:
        with self._lock:
            rows = self._conn.execute("""
                SELECT * FROM episodes WHERE session_id=? ORDER BY timestamp DESC LIMIT ?
            """, (self.session_id, limit)).fetchall()
        return [self._row_to_event(r) for r in rows]

    def count_by_type(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute("""
                SELECT event_type, COUNT(*) FROM episodes
                WHERE session_id=? GROUP BY event_type
            """, (self.session_id,)).fetchall()
        return dict(rows)

    def _row_to_event(self, row) -> MemoryEvent:
        emb = json.loads(row[9]) if row[9] else None
        return MemoryEvent(
            event_id=row[0], timestamp=row[1], session_id=row[2],
            event_type=row[3], content=json.loads(row[4]),
            gps_lat=row[5], gps_lon=row[6],
            human_state_summary=row[7], risk_level=row[8],
            embedding=emb,
        )

    def close(self):
        with self._lock:
            self._conn.close()


# ── Autobiographical Memory (persistent, vector similarity) ───────────────────

class AutobiographicalMemory:
    """
    Long-term cross-session memory: user profile, past mission patterns,
    stress responses, effective guidance strategies.

    Production: Weaviate or Qdrant VectorDB with text-embedding-3-large.
    Here: numpy cosine similarity on compact embeddings.
    """

    DIM = 16   # Compact feature vector for simulation (production: 1536-dim)

    def __init__(self, storage_path: str = "/tmp/maaa_autobio.json"):
        self.storage_path = storage_path
        self._memories: list[dict] = []
        self._vectors: list[list[float]] = []
        self._load()
        logger.info("[L5/AutobioMem] Loaded %d long-term memories", len(self._memories))

    def _load(self):
        try:
            with open(self.storage_path) as f:
                data = json.load(f)
                self._memories = data.get("memories", [])
                self._vectors  = data.get("vectors", [])
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def _save(self):
        try:
            with open(self.storage_path, "w") as f:
                json.dump({"memories": self._memories, "vectors": self._vectors}, f)
        except Exception as e:
            logger.warning("[L5/AutobioMem] Save failed: %s", e)

    def _embed(self, event: MemoryEvent) -> list[float]:
        """Compact feature embedding (production: call LLM embedding API)."""
        risk_num = {"SAFE": 0.0, "LOW": 0.25, "MEDIUM": 0.5,
                    "HIGH": 0.75, "CRITICAL": 1.0}.get(event.risk_level, 0.0)
        state_num = hash(event.human_state_summary) % 100 / 100.0
        ts_norm = (event.timestamp % 86400) / 86400.0   # Time of day
        return [
            risk_num, state_num, ts_norm,
            event.gps_lat % 1.0, event.gps_lon % 1.0,
            len(event.content) / 10.0,
        ] + [0.0] * (self.DIM - 6)

    def add(self, event: MemoryEvent):
        vec = event.embedding or self._embed(event)
        self._memories.append({
            "event_id":  event.event_id,
            "timestamp": event.timestamp,
            "type":      event.event_type,
            "summary":   event.to_text()[:200],
            "risk":      event.risk_level,
            "state":     event.human_state_summary,
        })
        self._vectors.append(vec)
        if len(self._memories) % 50 == 0:
            self._save()

    def search(self, query_event: MemoryEvent,
               top_k: int = 3) -> list[tuple[dict, float]]:
        """Find past events similar to the current situation."""
        if not self._vectors:
            return []
        q = np.array(self._embed(query_event), dtype=float)
        q_norm = q / (np.linalg.norm(q) + 1e-9)
        vecs = np.array(self._vectors, dtype=float)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        sims = (vecs / norms) @ q_norm
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [(self._memories[i], float(sims[i])) for i in top_idx]

    def get_user_profile(self) -> dict:
        """Aggregate user profile from autobiographical history."""
        if not self._memories:
            return {"sessions": 0, "avg_stress_missions": 0}
        critical_missions = sum(1 for m in self._memories if m.get("risk") == "CRITICAL")
        return {
            "total_events":       len(self._memories),
            "critical_incidents": critical_missions,
            "profile_complete":   len(self._memories) > 100,
        }

    def close(self):
        self._save()


# ── System Health Monitor ─────────────────────────────────────────────────────

@dataclass
class SystemHealth:
    timestamp: float
    loop_latency_ms: float
    sensor_ok: bool
    l2_ok: bool
    l3_ok: bool
    l4_ok: bool
    memory_ok: bool
    battery_pct: float
    offline_mode: bool
    failsafe_active: bool
    warnings: list[str] = field(default_factory=list)

    @property
    def is_degraded(self) -> bool:
        return not all([self.sensor_ok, self.l2_ok, self.l3_ok, self.l4_ok])

    @property
    def overall_health(self) -> float:
        components = [self.sensor_ok, self.l2_ok, self.l3_ok, self.l4_ok, self.memory_ok]
        return sum(components) / len(components)


# ── Layer 5 ───────────────────────────────────────────────────────────────────

class L5AutopoieticContinuity:
    """
    Layer 5 — Autopoietic Continuity.

    Monitors and preserves the integrity of:
      1. The system itself (latencies, models, sensors)
      2. The human operator (is the user still operational?)
      3. The human-system relationship (is the cognitive channel intact?)

    Also manages the 3-level memory architecture.
    """

    LATENCY_WARN_MS  = 150.0
    LATENCY_CRIT_MS  = 250.0
    BATTERY_WARN_PCT = 20.0

    def __init__(self, db_path: str = "/tmp/maaa_episodes.db",
                 autobio_path: str = "/tmp/maaa_autobio.json"):
        self.working_memory      = WorkingMemory()
        self.episodic_memory     = EpisodicMemory(db_path)
        self.autobiographical    = AutobiographicalMemory(autobio_path)
        self._tick               = 0
        self._loop_latencies: deque[float] = deque(maxlen=30)
        self._system_start       = time.time()
        self._failsafe_active    = False
        logger.info("[L5] Autopoietic Continuity initialized. Session: %s",
                    self.episodic_memory.session_id)

    def process(self,
                cognition: CognitionFrame,
                human: HumanStateFrame,
                guidance: GuidanceOutput,
                loop_latency_ms: float) -> SystemHealth:
        """
        Called once per pipeline cycle.
        Records events to memory, assesses system health, activates failsafe if needed.
        """
        self._tick += 1
        self._loop_latencies.append(loop_latency_ms)
        ts = time.time()

        # ── Record events to memory ───────────────────────────────────────────
        self._record_guidance(guidance, human, cognition)
        self._record_risk_events(cognition, human)
        self._record_human_state(human, cognition)

        # ── Assess system health ──────────────────────────────────────────────
        health = self._assess_health(loop_latency_ms, ts)

        # ── Activate failsafe if needed ───────────────────────────────────────
        if health.is_degraded:
            self._activate_failsafe(health)

        # ── Periodic autobiographical flush ───────────────────────────────────
        if self._tick % 100 == 0:
            self._flush_to_autobiographical()

        return health

    def _record_guidance(self, guidance: GuidanceOutput,
                         human: HumanStateFrame,
                         cognition: CognitionFrame):
        if guidance.suppressed or not guidance.voice_message:
            return
        event = MemoryEvent(
            event_id=str(uuid.uuid4()),
            timestamp=guidance.timestamp,
            session_id=self.episodic_memory.session_id,
            event_type="guidance_delivered",
            content={
                "message": guidance.voice_message,
                "urgency":  guidance.urgency.name,
                "channels": [c.value for c in guidance.active_channels],
            },
            human_state_summary=human.state.value,
            risk_level=cognition.risk_map.global_risk_level.value,
        )
        self.working_memory.add(event)
        self.episodic_memory.add(event)

    def _record_risk_events(self, cognition: CognitionFrame,
                            human: HumanStateFrame):
        if cognition.risk_map.global_risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            if not self.working_memory.has_recent_event("risk_critical", 5.0):
                event = MemoryEvent(
                    event_id=str(uuid.uuid4()),
                    timestamp=cognition.timestamp,
                    session_id=self.episodic_memory.session_id,
                    event_type="risk_critical",
                    content={
                        "global_risk": cognition.risk_map.global_risk,
                        "objects": [o.category for o in
                                    cognition.risk_map.get_critical_objects()],
                        "predictions": cognition.event_predictions,
                    },
                    human_state_summary=human.state.value,
                    risk_level=cognition.risk_map.global_risk_level.value,
                    gps_lat=cognition.slam.position_x,
                    gps_lon=cognition.slam.position_y,
                )
                self.working_memory.add(event)
                self.episodic_memory.add(event)

    def _record_human_state(self, human: HumanStateFrame,
                            cognition: CognitionFrame):
        if human.state in (CognitiveState.PANICKING, CognitiveState.COLLAPSED,
                           CognitiveState.FROZEN):
            if not self.working_memory.has_recent_event("human_crisis", 10.0):
                event = MemoryEvent(
                    event_id=str(uuid.uuid4()),
                    timestamp=human.timestamp,
                    session_id=self.episodic_memory.session_id,
                    event_type="human_crisis",
                    content={
                        "state":    human.state.value,
                        "panic":    human.panic_score,
                        "overload": human.cognitive_overload,
                        "freeze":   human.freeze_score,
                    },
                    human_state_summary=human.state.value,
                    risk_level=cognition.risk_map.global_risk_level.value,
                )
                self.working_memory.add(event)
                self.episodic_memory.add(event)

    def _assess_health(self, loop_latency_ms: float, ts: float) -> SystemHealth:
        warnings = []
        avg_latency = (sum(self._loop_latencies) / len(self._loop_latencies)
                       if self._loop_latencies else 0.0)

        if avg_latency > self.LATENCY_CRIT_MS:
            warnings.append(f"latency_critical:{avg_latency:.0f}ms")
        elif avg_latency > self.LATENCY_WARN_MS:
            warnings.append(f"latency_warning:{avg_latency:.0f}ms")

        # Simulate battery drain
        elapsed_h = (ts - self._system_start) / 3600.0
        battery_pct = max(0.0, 100.0 - elapsed_h * 25.0)  # ~4h endurance
        if battery_pct < self.BATTERY_WARN_PCT:
            warnings.append(f"battery_low:{battery_pct:.0f}%")

        return SystemHealth(
            timestamp=ts,
            loop_latency_ms=loop_latency_ms,
            sensor_ok=True,   # Production: check each adapter's heartbeat
            l2_ok=True,
            l3_ok=True,
            l4_ok=True,
            memory_ok=True,
            battery_pct=battery_pct,
            offline_mode=False,
            failsafe_active=self._failsafe_active,
            warnings=warnings,
        )

    def _activate_failsafe(self, health: SystemHealth):
        if not self._failsafe_active:
            logger.warning("[L5] FAILSAFE ACTIVATED — health=%.2f warnings=%s",
                           health.overall_health, health.warnings)
            self._failsafe_active = True

    def _flush_to_autobiographical(self):
        """Move significant episodic events to autobiographical long-term memory."""
        events = self.episodic_memory.get_session_events(limit=20)
        for ev in events:
            if ev.event_type in ("risk_critical", "human_crisis"):
                self.autobiographical.add(ev)

    def recall_similar(self, cognition: CognitionFrame,
                       human: HumanStateFrame) -> list[tuple[dict, float]]:
        """Retrieve past situations similar to current state from autobiographical memory."""
        query = MemoryEvent(
            event_id="query",
            timestamp=time.time(),
            session_id=self.episodic_memory.session_id,
            event_type="query",
            content={},
            human_state_summary=human.state.value,
            risk_level=cognition.risk_map.global_risk_level.value,
        )
        return self.autobiographical.search(query, top_k=3)

    def session_summary(self) -> dict:
        return {
            "session_id":        self.episodic_memory.session_id,
            "tick":              self._tick,
            "uptime_s":          time.time() - self._system_start,
            "working_mem_size":  self.working_memory.size,
            "episodic_events":   self.episodic_memory.count_by_type(),
            "autobio_memories":  len(self.autobiographical._memories),
            "failsafe_active":   self._failsafe_active,
            "avg_latency_ms":    (sum(self._loop_latencies) /
                                  max(1, len(self._loop_latencies))),
        }

    def close(self):
        self.episodic_memory.close()
        self.autobiographical.close()
        logger.info("[L5] Session closed. Summary: %s", self.session_summary())
