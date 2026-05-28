"""
MAAA — REST API  (Flask)

Endpoints:
  GET  /status              → pipeline health + last snapshot summary
  GET  /snapshot            → full last pipeline snapshot
  GET  /human               → current human state
  GET  /risk                → current risk map
  GET  /guidance            → last guidance output
  GET  /memory/working      → recent working memory events
  GET  /memory/episodic     → session episodic events
  GET  /memory/recall       → semantic recall from autobiographical memory
  POST /scenario/<name>     → inject a simulated scenario
  POST /tick/n/<n>          → run n ticks manually
  GET  /session             → session summary from L5
"""

from __future__ import annotations

import threading
from flask import Flask, jsonify, request

from core.maaa_agent import MAAAAgent
from layers.l1_perception import SceneCondition
from layers.l4_regulation import UrgencyLevel


def create_maaa_app(agent: MAAAAgent) -> Flask:
    app = Flask(__name__)
    _lock = threading.Lock()

    def _snap_dict():
        s = agent.last_snapshot
        if not s:
            return {"error": "no_data_yet"}
        c = s.cognition
        h = s.human
        g = s.guidance
        return {
            "tick":       s.tick,
            "latency_ms": round(s.latency_ms, 2),
            "risk": {
                "global":       round(c.risk_map.global_risk, 3),
                "level":        c.risk_map.global_risk_level.value,
                "structural":   round(c.risk_map.structural_integrity, 3),
                "time_to_act":  c.risk_map.time_to_act_seconds,
                "critical_objects": [
                    {"id": o.object_id, "cat": o.category,
                     "dist_m": round(o.distance_m, 1), "action": o.recommended_action}
                    for o in c.risk_map.get_critical_objects()
                ],
                "exits": [
                    {"id": o.object_id, "dist_m": round(o.distance_m, 1),
                     "bearing": round(o.bearing_deg, 0)}
                    for o in c.risk_map.get_exits()
                ],
            },
            "human": {
                "state":            h.state.value,
                "stress":           h.stress_score,
                "panic":            h.panic_score,
                "freeze":           h.freeze_score,
                "overload":         h.cognitive_overload,
                "receptivity":      h.receptivity,
                "decision_capacity": h.decision_capacity,
                "is_critical":      h.is_critical,
                "trend":            agent.l3.get_trend(),
            },
            "guidance": {
                "urgency":   g.urgency.name,
                "message":   g.voice_message,
                "suppressed": g.suppressed,
                "reason":    g.suppression_reason,
                "filters":   g.filter_log,
                "ar": {
                    "active":  g.ar_overlay.active,
                    "text":    g.ar_overlay.text_overlay,
                    "color":   g.ar_overlay.color_urgency,
                    "path":    g.ar_overlay.path_arrow_bearing,
                    "dangers": g.ar_overlay.danger_zones,
                },
                "haptic": {
                    "active":   g.haptic.active,
                    "pattern":  g.haptic.pattern,
                    "intensity": g.haptic.intensity,
                },
            },
            "predictions":    c.event_predictions,
            "causal_summary": c.causal_summary,
            "health": {
                "ok":           not s.health.is_degraded,
                "battery_pct":  round(s.health.battery_pct, 1),
                "failsafe":     s.health.failsafe_active,
                "warnings":     s.health.warnings,
            },
            "l4_stats":   agent.l4.output_stats,
            "output_stats": agent.output.stats,
        }

    @app.get("/status")
    def status():
        return jsonify({
            "agent": "MAAA v1.0",
            "tick":  agent.tick_count,
            "running": agent._running,
            "session": agent.l5.session_summary(),
            "last": _snap_dict(),
        })

    @app.get("/snapshot")
    def snapshot():
        return jsonify(_snap_dict())

    @app.get("/human")
    def human_state():
        s = agent.last_snapshot
        if not s:
            return jsonify({"error": "no_data"}), 503
        h = s.human
        return jsonify({
            "state":             h.state.value,
            "stress_score":      h.stress_score,
            "cognitive_overload": h.cognitive_overload,
            "panic_score":       h.panic_score,
            "freeze_score":      h.freeze_score,
            "attentional_collapse": h.attentional_collapse,
            "arousal":           h.arousal,
            "decision_capacity": h.decision_capacity,
            "receptivity":       h.receptivity,
            "voice_stress":      h.voice_stress_index,
            "gaze_stability":    h.gaze_stability,
            "motor_agitation":   h.motor_agitation,
            "stress_delta":      h.stress_delta,
            "panic_delta":       h.panic_delta,
            "is_critical":       h.is_critical,
            "needs_override":    h.needs_immediate_override,
            "trend":             agent.l3.get_trend(),
        })

    @app.get("/risk")
    def risk():
        s = agent.last_snapshot
        if not s:
            return jsonify({"error": "no_data"}), 503
        rm = s.cognition.risk_map
        return jsonify({
            "global_risk":        rm.global_risk,
            "global_level":       rm.global_risk_level.value,
            "structural":         rm.structural_integrity,
            "time_to_act_s":      rm.time_to_act_seconds,
            "passable_bearings":  rm.passable_directions,
            "recommended_path":   rm.recommended_path_bearing,
            "objects": [
                {
                    "id":       o.object_id,
                    "category": o.category,
                    "dist_m":   o.distance_m,
                    "bearing":  o.bearing_deg,
                    "risk":     o.risk_probability,
                    "level":    o.risk_level.value,
                    "action":   o.recommended_action,
                    "is_exit":  o.is_exit,
                }
                for o in rm.objects
            ],
        })

    @app.get("/guidance")
    def guidance():
        s = agent.last_snapshot
        if not s:
            return jsonify({"error": "no_data"}), 503
        g = s.guidance
        return jsonify({
            "urgency":        g.urgency.name,
            "voice_message":  g.voice_message,
            "full_message":   g.voice_message_full,
            "suppressed":     g.suppressed,
            "reason":         g.suppression_reason,
            "filters":        g.filter_log,
            "delay_ms":       g.delivery_delay_ms,
            "ar_overlay":     {
                "active":   g.ar_overlay.active,
                "text":     g.ar_overlay.text_overlay,
                "color":    g.ar_overlay.color_urgency,
                "path_deg": g.ar_overlay.path_arrow_bearing,
                "dangers":  g.ar_overlay.danger_zones,
            },
            "haptic": {
                "active":    g.haptic.active,
                "pattern":   g.haptic.pattern,
                "intensity": g.haptic.intensity,
            },
            "stats": agent.l4.output_stats,
        })

    @app.get("/memory/working")
    def memory_working():
        events = agent.l5.working_memory.recent(20)
        return jsonify({
            "size": agent.l5.working_memory.size,
            "events": [
                {"type": e.event_type, "ts": e.timestamp,
                 "state": e.human_state_summary, "risk": e.risk_level,
                 "content": e.content}
                for e in events
            ]
        })

    @app.get("/memory/episodic")
    def memory_episodic():
        events = agent.l5.episodic_memory.get_session_events(50)
        counts = agent.l5.episodic_memory.count_by_type()
        return jsonify({
            "session_id":  agent.l5.episodic_memory.session_id,
            "event_counts": counts,
            "recent_events": [
                {"id": e.event_id, "type": e.event_type,
                 "ts": e.timestamp, "risk": e.risk_level,
                 "state": e.human_state_summary}
                for e in events[:20]
            ],
        })

    @app.get("/memory/recall")
    def memory_recall():
        s = agent.last_snapshot
        if not s:
            return jsonify({"error": "no_data"}), 503
        results = agent.l5.recall_similar(s.cognition, s.human)
        return jsonify({
            "query_state": s.human.state.value,
            "query_risk":  s.cognition.risk_map.global_risk_level.value,
            "similar_past": [
                {"similarity": round(sim, 3), "memory": mem}
                for mem, sim in results
            ],
        })

    @app.post("/scenario/<name>")
    def inject_scenario(name: str):
        mapping = {
            "normal":     (SceneCondition.NORMAL,     0.1, 0.0, 0.0, False),
            "smoky":      (SceneCondition.SMOKY,      0.5, 0.3, 0.2, True),
            "dark":       (SceneCondition.DARK,       0.4, 0.2, 0.1, False),
            "collapsed":  (SceneCondition.COLLAPSED,  0.8, 0.7, 0.6, True),
            "panic":      (SceneCondition.COLLAPSED,  0.9, 0.95, 0.7, True),
            "dusty":      (SceneCondition.DUSTY,      0.4, 0.2, 0.3, False),
        }
        if name not in mapping:
            return jsonify({"error": f"unknown scenario '{name}'",
                            "available": list(mapping.keys())}), 400
        scene, stress, panic, obs, sounds = mapping[name]
        agent.inject_scenario(scene, stress, panic, obs, sounds)
        return jsonify({"scenario": name, "injected": True})

    @app.post("/tick/n/<int:n>")
    def run_n_ticks(n: int):
        n = min(n, 200)
        with _lock:
            for _ in range(n):
                agent.tick()
        return jsonify({"ticks_run": n, "total_ticks": agent.tick_count})

    @app.get("/session")
    def session():
        return jsonify(agent.l5.session_summary())

    return app
