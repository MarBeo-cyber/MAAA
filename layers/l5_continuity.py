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

import os
import tempfile
import time
import json
import uuid
import shutil
import hashlib
import sqlite3

# I percorsi di default vivono nella directory temporanea DEL SISTEMA.
# Scrivere "/tmp/..." a mano funziona su Linux e su Windows finisce in
# C:\tmp\, che spesso non esiste: il file non si apre e l'errore arriva
# lontano dalla causa. tempfile.gettempdir() risolve la cosa ovunque.
DEFAULT_DB_PATH      = os.path.join(tempfile.gettempdir(), "maaa_episodes.db")
DEFAULT_AUTOBIO_PATH = os.path.join(tempfile.gettempdir(), "maaa_autobio.json")
import logging
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional
from collections import deque

import numpy as np

import maaa_config
from layers.l2_cognition import CognitionFrame, RiskLevel
from layers.l3_human_state import HumanStateFrame, CognitiveState
from layers.l4_regulation import GuidanceOutput

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

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self.session_id = str(uuid.uuid4())[:8]
        self.last_error: Optional[str] = None
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

    def healthy(self) -> bool:
        """Probe the store. Feeds SystemHealth.memory_ok (NFR-08 auditability)."""
        try:
            with self._lock:
                self._conn.execute("SELECT 1 FROM episodes LIMIT 1").fetchone()
            self.last_error = None
            return True
        except (sqlite3.Error, AttributeError) as exc:
            self.last_error = str(exc)
            return False

    def get_session_events(self, limit: int = 200) -> list[MemoryEvent]:
        with self._lock:
            rows = self._conn.execute("""
                SELECT * FROM episodes WHERE session_id=? ORDER BY timestamp DESC LIMIT ?
            """, (self.session_id, limit)).fetchall()
        return [self._row_to_event(r) for r in rows]

    def count_by_type(self) -> dict[str, int]:
        try:
            with self._lock:
                rows = self._conn.execute("""
                    SELECT event_type, COUNT(*) FROM episodes
                    WHERE session_id=? GROUP BY event_type
                """, (self.session_id,)).fetchall()
        except sqlite3.Error as exc:
            # A degraded store must not take /status and /session down with it.
            self.last_error = str(exc)
            return {}
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
            try:
                self._conn.close()
            except sqlite3.Error as exc:      # already closed / never opened
                self.last_error = str(exc)


# ── Autobiographical Memory (persistent, vector similarity) ───────────────────

class AutobiographicalMemory:
    """
    Long-term cross-session memory: user profile, past mission patterns,
    stress responses, effective guidance strategies.

    Production: Weaviate or Qdrant VectorDB with text-embedding-3-large.
    Here: numpy cosine similarity on compact embeddings.
    """

    DIM = 16   # Compact feature vector for simulation (production: 1536-dim)

    def __init__(self, storage_path: str = DEFAULT_AUTOBIO_PATH):
        self.storage_path = storage_path
        self._memories: list[dict] = []
        self._vectors: list[list[float]] = []
        self.load_error: Optional[str] = None
        self._load()
        logger.info("[L5/AutobioMem] Loaded %d long-term memories", len(self._memories))

    def _load(self):
        try:
            with open(self.storage_path) as f:
                data = json.load(f)
                self._memories = data.get("memories", [])
                self._vectors  = data.get("vectors", [])
        except FileNotFoundError:
            logger.info("[L5/AutobioMem] No store at %s — starting empty",
                        self.storage_path)
        except (json.JSONDecodeError, OSError) as exc:
            # A corrupt store used to be swallowed silently, so the agent came
            # up with an empty long-term memory and no trace of why.
            backup = f"{self.storage_path}.corrupt-{int(time.time())}"
            try:
                shutil.copy(self.storage_path, backup)
            except OSError:
                backup = "<backup failed>"
            logger.error("[L5/AutobioMem] Store %s unreadable (%s) — "
                         "starting empty, original kept at %s",
                         self.storage_path, exc, backup)
            self.load_error = str(exc)

    def _save(self):
        try:
            with open(self.storage_path, "w") as f:
                json.dump({"memories": self._memories, "vectors": self._vectors}, f)
        except Exception as e:
            logger.warning("[L5/AutobioMem] Save failed: %s", e)

    @staticmethod
    def stable_hash_unit(text: str) -> float:
        """Deterministic [0,1) hash of a string, stable across processes.

        Python's built-in hash() is salted per interpreter (PYTHONHASHSEED),
        so the same state summary embedded in three different runs produced
        0.8 / 0.7 / 0.32 and vectors persisted to disk were not comparable with
        vectors computed after a restart.
        """
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
        return (int.from_bytes(digest, "big") % 1000) / 1000.0

    def _embed(self, event: MemoryEvent) -> list[float]:
        """Compact feature embedding (production: call LLM embedding API)."""
        risk_num = {"SAFE": 0.0, "LOW": 0.25, "MEDIUM": 0.5,
                    "HIGH": 0.75, "CRITICAL": 1.0}.get(event.risk_level, 0.0)
        state_num = self.stable_hash_unit(event.human_state_summary)
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
class Heartbeat:
    """Last report from one pipeline stage."""
    timestamp: float
    ok: bool
    latency_ms: float
    detail: str = ""


@dataclass
class SystemHealth:
    timestamp: float
    loop_latency_ms: float
    sensor_ok: bool
    l2_ok: bool
    l3_ok: bool
    l4_ok: bool
    memory_ok: bool
    #: None when no battery source is attached — this build has no BMS driver.
    battery_pct: Optional[float]
    offline_mode: bool
    failsafe_active: bool
    warnings: list[str] = field(default_factory=list)
    #: Why the system counts as degraded. Empty means healthy.
    degraded_reasons: list[str] = field(default_factory=list)
    #: "unavailable" | "simulated" | "hardware"
    battery_source: str = "unavailable"

    @property
    def is_degraded(self) -> bool:
        return (not all([self.sensor_ok, self.l2_ok, self.l3_ok,
                         self.l4_ok, self.memory_ok])
                or bool(self.degraded_reasons))

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

    #: Stages that must report a heartbeat every cycle for the system to be healthy.
    COMPONENTS = ("sensor", "l2", "l3", "l4")

    def __init__(self, db_path: str = DEFAULT_DB_PATH,
                 autobio_path: str = DEFAULT_AUTOBIO_PATH):
        self.working_memory      = WorkingMemory()
        self.episodic_memory     = EpisodicMemory(db_path)
        self.autobiographical    = AutobiographicalMemory(autobio_path)
        self._tick               = 0
        self._loop_latencies: deque[float] = deque(maxlen=30)
        self._system_start       = time.time()
        self._failsafe_active    = False
        self._heartbeats: dict[str, Heartbeat] = {}
        self._memory_error: Optional[str] = None
        self._healthy_streak     = 0
        self._battery_source_fn: Optional[Callable[[], float]] = None
        self._battery_source_label = "unavailable"

        self.BATTERY_WARN_PCT = maaa_config.failsafe_setting("battery_warn_pct", 20.0)
        self.BATTERY_MIN_PCT  = maaa_config.failsafe_setting("battery_min_pct", 5.0)
        self.HEARTBEAT_TIMEOUT_S = maaa_config.failsafe_setting("heartbeat_timeout_s", 2.0)
        self.STAGE_LATENCY_MAX_MS = maaa_config.failsafe_setting("stage_latency_max_ms", 150.0)
        self.RECOVERY_CYCLES = int(maaa_config.failsafe_setting("recovery_cycles", 10))

        logger.info("[L5] Autopoietic Continuity initialized. Session: %s",
                    self.episodic_memory.session_id)

    # ── Heartbeats ────────────────────────────────────────────────────────────

    def report_heartbeat(self, component: str, ok: bool = True,
                         latency_ms: float = 0.0, detail: str = ""):
        """Called by the orchestrator after each pipeline stage runs.

        Health used to be hard-coded True for every component, so is_degraded
        was always False and _activate_failsafe was dead code. Now a stage that
        raises, stalls or blows its latency budget stops being 'ok' here.
        """
        self._heartbeats[component] = Heartbeat(time.time(), ok, latency_ms, detail)

    def set_battery_source(self, source: Optional[Callable[[], float]],
                           label: str = "hardware"):
        """Attach a battery-percentage source.

        There is no battery-management driver in this repository. Until a
        source is attached, battery_pct is None and battery_source is
        "unavailable" — the previous build reported ``100 - uptime_h * 25`` as
        if it were a real reading.
        """
        self._battery_source_fn = source
        self._battery_source_label = label if source is not None else "unavailable"

    def simulated_battery_source(self, endurance_hours: float = 4.0) -> Callable[[], float]:
        """A clearly-labelled fake battery for demos. Never call this 'hardware'."""
        start = self._system_start

        def _read() -> float:
            elapsed_h = (time.time() - start) / 3600.0
            return max(0.0, 100.0 - elapsed_h * (100.0 / endurance_hours))
        return _read

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
        else:
            self._clear_failsafe(health)
        # Report the state as of the end of this cycle, not the start: the
        # snapshot used to say failsafe_active=False on the very tick that
        # activated it.
        health.failsafe_active = self._failsafe_active

        # ── Periodic autobiographical flush ───────────────────────────────────
        if self._tick % 100 == 0:
            self._flush_to_autobiographical()

        return health

    def _safe_episodic_add(self, event: MemoryEvent):
        """Write to SQLite, recording the failure instead of hiding it."""
        try:
            self.episodic_memory.add(event)
            self._memory_error = None
        except sqlite3.Error as exc:
            self._memory_error = str(exc)
            logger.error("[L5] Episodic write failed: %s", exc)

    def _record_guidance(self, guidance: GuidanceOutput,
                         human: HumanStateFrame,
                         cognition: CognitionFrame):
        """Log the guidance decision — delivered OR suppressed.

        NFR-08 requires every guidance event to be logged. This used to return
        early on suppressed output, so the ~98% of cycles the relevance and
        timing filters silence left no trace at all: the audit trail recorded
        only what the user heard, never what the system chose not to say.
        """
        delivered = not guidance.suppressed and bool(guidance.voice_message)
        content: dict = {
            "urgency":  guidance.urgency.name,
            "filters":  guidance.filter_log,
        }
        if delivered:
            content["message"] = guidance.voice_message
            content["channels"] = [c.value for c in guidance.active_channels]
        else:
            content["candidate_message"] = guidance.voice_message_full
            content["suppression_reason"] = guidance.suppression_reason

        event = MemoryEvent(
            event_id=str(uuid.uuid4()),
            timestamp=guidance.timestamp,
            session_id=self.episodic_memory.session_id,
            event_type="guidance_delivered" if delivered else "guidance_suppressed",
            content=content,
            human_state_summary=human.state.value,
            risk_level=cognition.risk_map.global_risk_level.value,
        )
        self.working_memory.add(event)
        self._safe_episodic_add(event)

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
                self._safe_episodic_add(event)

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
                self._safe_episodic_add(event)

    def _component_ok(self, component: str, now: float,
                      reasons: list[str]) -> bool:
        """Derive one component flag from its most recent heartbeat."""
        hb = self._heartbeats.get(component)
        if hb is None:
            reasons.append(f"{component}_no_heartbeat")
            return False
        age = now - hb.timestamp
        if not hb.ok:
            reasons.append(f"{component}_error:{hb.detail or 'unknown'}")
            return False
        if age > self.HEARTBEAT_TIMEOUT_S:
            reasons.append(f"{component}_heartbeat_stale:{age:.1f}s")
            return False
        if hb.latency_ms > self.STAGE_LATENCY_MAX_MS:
            reasons.append(f"{component}_latency:{hb.latency_ms:.0f}ms")
            return False
        return True

    def _assess_health(self, loop_latency_ms: float, ts: float) -> SystemHealth:
        warnings: list[str] = []
        reasons: list[str] = []
        avg_latency = (sum(self._loop_latencies) / len(self._loop_latencies)
                       if self._loop_latencies else 0.0)

        if avg_latency > self.LATENCY_CRIT_MS:
            warnings.append(f"latency_critical:{avg_latency:.0f}ms")
            reasons.append(f"loop_latency_critical:{avg_latency:.0f}ms")
        elif avg_latency > self.LATENCY_WARN_MS:
            warnings.append(f"latency_warning:{avg_latency:.0f}ms")

        flags = {c: self._component_ok(c, ts, reasons) for c in self.COMPONENTS}

        memory_ok = self._memory_error is None and self.episodic_memory.healthy()
        if not memory_ok:
            detail = self._memory_error or self.episodic_memory.last_error or "probe_failed"
            reasons.append(f"memory_error:{detail}")
            warnings.append("memory_unavailable")

        battery_pct: Optional[float] = None
        if self._battery_source_fn is not None:
            try:
                battery_pct = float(self._battery_source_fn())
            except Exception as exc:                      # noqa: BLE001 — source is external
                reasons.append(f"battery_source_error:{exc}")
                warnings.append("battery_source_error")
        if battery_pct is not None:
            if battery_pct < self.BATTERY_MIN_PCT:
                warnings.append(f"battery_critical:{battery_pct:.0f}%")
                reasons.append(f"battery_critical:{battery_pct:.0f}%")
            elif battery_pct < self.BATTERY_WARN_PCT:
                warnings.append(f"battery_low:{battery_pct:.0f}%")

        return SystemHealth(
            timestamp=ts,
            loop_latency_ms=loop_latency_ms,
            sensor_ok=flags["sensor"],
            l2_ok=flags["l2"],
            l3_ok=flags["l3"],
            l4_ok=flags["l4"],
            memory_ok=memory_ok,
            battery_pct=battery_pct,
            offline_mode=bool(maaa_config.get("maaa.safety.offline_first", True)),
            failsafe_active=self._failsafe_active,
            warnings=warnings,
            degraded_reasons=reasons,
            battery_source=self._battery_source_label,
        )

    def _activate_failsafe(self, health: SystemHealth):
        self._healthy_streak = 0
        if not self._failsafe_active:
            logger.warning("[L5] FAILSAFE ACTIVATED — health=%.2f reasons=%s",
                           health.overall_health, health.degraded_reasons)
            self._failsafe_active = True
            self._record_failsafe_event(health, "failsafe_activated")

    def _clear_failsafe(self, health: SystemHealth):
        """Autopoietic recovery: leave failsafe after a run of healthy cycles."""
        self._healthy_streak += 1
        if self._failsafe_active and self._healthy_streak >= self.RECOVERY_CYCLES:
            logger.info("[L5] Failsafe cleared after %d healthy cycles",
                        self._healthy_streak)
            self._failsafe_active = False
            self._record_failsafe_event(health, "failsafe_cleared")

    def _record_failsafe_event(self, health: SystemHealth, event_type: str):
        event = MemoryEvent(
            event_id=str(uuid.uuid4()),
            timestamp=health.timestamp,
            session_id=self.episodic_memory.session_id,
            event_type=event_type,
            content={
                "reasons": health.degraded_reasons,
                "warnings": health.warnings,
                "overall_health": health.overall_health,
                "battery_pct": health.battery_pct,
                "battery_source": health.battery_source,
            },
            risk_level="CRITICAL" if event_type == "failsafe_activated" else "SAFE",
        )
        self.working_memory.add(event)
        self._safe_episodic_add(event)

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
        # Build the summary BEFORE closing the store: this used to query a
        # closed sqlite connection, so every close() raised ProgrammingError.
        summary = self.session_summary()
        self.episodic_memory.close()
        self.autobiographical.close()
        logger.info("[L5] Session closed. Summary: %s", summary)
