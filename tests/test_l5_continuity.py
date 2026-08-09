"""L5 regressions: stable hashing, health flags, failsafe, NFR-08 audit trail."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import pytest

from layers.l1_perception import SceneCondition
from layers.l5_continuity import (AutobiographicalMemory, L5AutopoieticContinuity,
                                  MemoryEvent)
from core.maaa_agent import MAAAAgent

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture()
def agent(tmp_path):
    a = MAAAAgent(simulation_mode=True, verbose=False,
                  db_path=str(tmp_path / "episodes.db"),
                  autobio_path=str(tmp_path / "autobio.json"))
    yield a
    a.l5.close()


# ── Stable hashing ────────────────────────────────────────────────────────────

EMBED_IN_SUBPROCESS = """
import sys, json
sys.path.insert(0, {root!r})
from layers.l5_continuity import AutobiographicalMemory, MemoryEvent
event = MemoryEvent(event_id="e1", timestamp=1000.0, session_id="s",
                    event_type="human_crisis", content={{}},
                    human_state_summary="panicking", risk_level="CRITICAL")
mem = AutobiographicalMemory({store!r})
print(json.dumps(mem._embed(event)))
"""


def _embed_with_hash_seed(seed: str, store: str) -> str:
    """Compute the real _embed() vector in a fresh interpreter."""
    env = dict(os.environ, PYTHONHASHSEED=seed)
    out = subprocess.run(
        [sys.executable, "-c",
         EMBED_IN_SUBPROCESS.format(root=REPO_ROOT, store=store)],
        env=env, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def test_embedding_is_stable_across_processes(tmp_path):
    """hash() is salted per interpreter, so the same human_state_summary
    embedded in three runs produced 0.8 / 0.7 / 0.32 and vectors persisted to
    /tmp/maaa_autobio.json were not comparable after a restart."""
    store = str(tmp_path / "autobio.json")
    values = {_embed_with_hash_seed(seed, store) for seed in ("0", "1", "12345")}
    assert len(values) == 1, f"_embed() varies with PYTHONHASHSEED: {values}"


def test_persisted_vectors_survive_a_restart(tmp_path):
    """The vector written to disk must still match one computed later, in a
    different interpreter, for the same event."""
    path = str(tmp_path / "autobio.json")
    event = MemoryEvent(event_id="e1", timestamp=1000.0, session_id="s",
                        event_type="human_crisis", content={},
                        human_state_summary="panicking", risk_level="CRITICAL")
    first = AutobiographicalMemory(path)
    first.add(event)
    first.close()

    persisted = json.loads(open(path).read())["vectors"][0]
    recomputed = json.loads(_embed_with_hash_seed("424242", path))
    assert recomputed == persisted, (recomputed, persisted)

    second = AutobiographicalMemory(path)
    hits = second.search(event, top_k=1)
    assert hits and hits[0][1] > 0.99, hits


def test_corrupt_store_is_reported_and_backed_up(tmp_path, caplog):
    path = tmp_path / "autobio.json"
    path.write_text("{not json")
    with caplog.at_level("ERROR", logger="maaa.l5_continuity"):
        mem = AutobiographicalMemory(str(path))
    assert mem.load_error is not None
    assert any("unreadable" in r.getMessage() for r in caplog.records)
    assert list(tmp_path.glob("autobio.json.corrupt-*")), "no backup kept"


# ── Health flags and failsafe ─────────────────────────────────────────────────

def test_health_flags_are_derived_from_heartbeats(agent):
    agent.inject_scenario(SceneCondition.NORMAL, 0.1, 0.0, 0.0, False)
    snap = agent.tick()
    assert snap.health.sensor_ok and snap.health.l2_ok
    assert snap.health.l3_ok and snap.health.l4_ok
    assert not snap.health.is_degraded


def test_failsafe_activates_when_a_layer_stops_reporting(agent):
    agent.inject_scenario(SceneCondition.NORMAL, 0.1, 0.0, 0.0, False)
    snap = agent.tick()
    assert not snap.health.is_degraded

    def boom(_perception):
        raise RuntimeError("model not loaded")

    agent.l2.process = boom
    with pytest.raises(RuntimeError):
        agent.tick()
    health = agent.l5.process(snap.cognition, snap.human, snap.guidance, 10.0)
    assert health.is_degraded
    assert not health.l2_ok
    assert health.failsafe_active
    assert any("l2_error" in r for r in health.degraded_reasons), health.degraded_reasons


def test_failsafe_activates_on_battery_collapse(agent):
    agent.inject_scenario(SceneCondition.COLLAPSED, 0.9, 0.9, 0.7, True)
    agent.tick()
    agent.l5.set_battery_source(lambda: 0.0, label="simulated")
    snap = agent.tick()
    assert snap.health.battery_pct == 0.0
    assert snap.health.is_degraded
    assert snap.health.failsafe_active
    assert any("battery_critical" in r for r in snap.health.degraded_reasons)


def test_failsafe_activation_is_recorded_in_episodic_memory(agent):
    agent.inject_scenario(SceneCondition.NORMAL, 0.1, 0.0, 0.0, False)
    agent.tick()
    agent.l5.set_battery_source(lambda: 0.0, label="simulated")
    agent.tick()
    assert agent.l5.episodic_memory.count_by_type().get("failsafe_activated", 0) >= 1


def test_failsafe_clears_after_healthy_cycles(agent):
    agent.inject_scenario(SceneCondition.NORMAL, 0.1, 0.0, 0.0, False)
    agent.l5.set_battery_source(lambda: 0.0, label="simulated")
    assert agent.tick().health.failsafe_active
    agent.l5.set_battery_source(lambda: 90.0, label="simulated")
    for _ in range(agent.l5.RECOVERY_CYCLES + 1):
        snap = agent.tick()
    assert not snap.health.failsafe_active


def test_battery_is_not_fabricated(agent):
    """battery_pct used to be 100 - uptime_h*25, published via /status as if it
    were a reading. With no source attached it must be explicitly unknown."""
    agent.inject_scenario(SceneCondition.NORMAL, 0.1, 0.0, 0.0, False)
    snap = agent.tick()
    assert snap.health.battery_pct is None
    assert snap.health.battery_source == "unavailable"

    agent.l5.set_battery_source(agent.l5.simulated_battery_source(), label="simulated")
    snap = agent.tick()
    assert snap.health.battery_pct is not None
    assert snap.health.battery_source == "simulated"


def test_stale_heartbeat_degrades_health(tmp_path):
    l5 = L5AutopoieticContinuity(str(tmp_path / "e.db"), str(tmp_path / "a.json"))
    for component in l5.COMPONENTS:
        l5.report_heartbeat(component, ok=True, latency_ms=1.0)
    l5._heartbeats["l3"].timestamp = time.time() - (l5.HEARTBEAT_TIMEOUT_S + 1.0)
    health = l5._assess_health(10.0, time.time())
    assert not health.l3_ok
    assert any("l3_heartbeat_stale" in r for r in health.degraded_reasons)
    l5.close()


# ── NFR-08: every guidance event logged ───────────────────────────────────────

def test_suppressed_guidance_is_logged(agent):
    """_record_guidance returned early on suppressed output, so the ~98% of
    cycles silenced by the relevance and timing filters left no audit trail."""
    agent.inject_scenario(SceneCondition.SMOKY, 0.6, 0.35, 0.2, True)
    for _ in range(60):
        agent.tick()
    counts = agent.l5.episodic_memory.count_by_type()
    stats = agent.l4.output_stats
    assert stats["outputs_suppressed"] > 0, "no output was suppressed in this run"
    assert counts.get("guidance_suppressed", 0) == stats["outputs_suppressed"]
    logged = (counts.get("guidance_suppressed", 0)
              + counts.get("guidance_delivered", 0))
    assert logged == stats["total_cycles"], (counts, stats)


def test_suppressed_guidance_records_why(agent):
    agent.inject_scenario(SceneCondition.SMOKY, 0.6, 0.35, 0.2, True)
    for _ in range(60):
        agent.tick()
    events = [e for e in agent.l5.episodic_memory.get_session_events(200)
              if e.event_type == "guidance_suppressed"]
    assert events
    for e in events:
        assert e.content["suppression_reason"]
        assert e.content["filters"]
        assert "candidate_message" in e.content


def test_memory_write_failure_degrades_health(agent):
    agent.inject_scenario(SceneCondition.NORMAL, 0.1, 0.0, 0.0, False)
    agent.tick()
    agent.l5.episodic_memory._conn.close()
    snap = agent.tick()
    assert not snap.health.memory_ok
    assert snap.health.is_degraded


def test_session_summary_is_json_serialisable(agent):
    agent.inject_scenario(SceneCondition.NORMAL, 0.1, 0.0, 0.0, False)
    agent.tick()
    json.dumps(agent.l5.session_summary())
