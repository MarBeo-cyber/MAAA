"""
MAAA — Main Entry Point
Metacognitive Autopoietic Adaptive Agent v1.0

Run modes:
  python main_maaa.py demo      — four-phase earthquake scenario demo
  python main_maaa.py server    — REST API on :5002
  python main_maaa.py both      — server + demo simultaneously
  python main_maaa.py tick <n>  — run exactly n pipeline ticks

Demo phases:
  Phase 1 (NORMAL, 10s):    Ambiente stabile. Il sistema si calibra.
  Phase 2 (SMOKY, 15s):     Fumo rilevato. Stress sale. Guidance attiva.
  Phase 3 (COLLAPSED, 20s): Crollo strutturale. Panico. Override cognitivo.
  Phase 4 (NORMAL, 10s):    Recovery. De-escalation guidance.
"""

import sys
import time
import logging
import threading

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("maaa.main")
logging.getLogger("werkzeug").setLevel(logging.ERROR)

from core.maaa_agent  import MAAAAgent
from api.maaa_api     import create_maaa_app
from layers.l1_perception import SceneCondition

DIVIDER = "─" * 70


def run_demo(agent: MAAAAgent):
    """Four-phase earthquake emergency scenario."""

    phases = [
        ("FASE 1 — AMBIENTE NORMALE",
         SceneCondition.NORMAL, 0.1, 0.0, 0.0, False, 10),
        ("FASE 2 — FUMO E STRESS",
         SceneCondition.SMOKY, 0.65, 0.35, 0.2, True, 15),
        ("FASE 3 — CROLLO STRUTTURALE — PANICO",
         SceneCondition.COLLAPSED, 0.9, 0.90, 0.7, True, 20),
        ("FASE 4 — RECOVERY E DE-ESCALATION",
         SceneCondition.NORMAL, 0.2, 0.0, 0.0, False, 10),
    ]

    print(f"\n{DIVIDER}")
    print("  MAAA — Demo Scenario: Emergenza Terremoto")
    print(f"{DIVIDER}\n")
    print("  Il MAAA monitora ambiente e stato cognitivo dell'utente.")
    print("  Applica 4 filtri (rilevanza, timing, brevità, urgenza)")
    print("  per minimizzare l'entropia cognitiva in condizioni critiche.\n")

    for phase_name, scene, stress, panic, obs, sounds, duration_s in phases:
        print(f"\n{'═'*70}")
        print(f"  {phase_name}")
        print(f"{'═'*70}")
        agent.inject_scenario(scene, stress, panic, obs, sounds)

        start = time.time()
        tick_interval = 0.3   # 3.3 Hz for readable demo output

        while time.time() - start < duration_s:
            snap = agent.tick()
            elapsed = time.time() - start

            # Print status every ~1.5 seconds
            if agent.tick_count % 5 == 0:
                c = snap.cognition
                h = snap.human
                g = snap.guidance
                print(f"\n  t={elapsed:04.1f}s | tick={snap.tick} | {snap.latency_ms:.1f}ms")
                print(f"  Rischio: {c.risk_map.global_risk_level.value} "
                      f"({c.risk_map.global_risk:.2f}) | "
                      f"Struttura: {c.risk_map.structural_integrity:.2f}")
                print(f"  Umano:   {h.state.value} | "
                      f"stress={h.stress_score:.2f} panic={h.panic_score:.2f} "
                      f"receptivity={h.receptivity:.2f}")
                if not g.suppressed and g.voice_message:
                    urgency_mark = {
                        "CRITICAL": "🔴", "ELEVATED": "⚡",
                        "NORMAL": "▶", "AMBIENT": "●", "SILENT": "○"
                    }.get(g.urgency.name, "▶")
                    print(f"  Output:  {urgency_mark} [{g.urgency.name}] "
                          f"\"{g.voice_message}\"")
                else:
                    print(f"  Output:  ○ [SUPPRESSO] {g.suppression_reason or ''}")
                if c.event_predictions:
                    print(f"  Predict: {', '.join(c.event_predictions)}")

            time.sleep(tick_interval)

        # Phase summary
        print(f"\n  ── Riepilogo fase:")
        print(f"     Tick eseguiti:   {agent.tick_count}")
        stats = agent.l4.output_stats
        print(f"     Output:          {stats['outputs_delivered']} erogati, "
              f"{stats['outputs_suppressed']} soppressi "
              f"({stats['delivery_rate']:.1%} delivery rate)")
        mem = agent.l5.session_summary()
        print(f"     Memoria working: {mem['working_mem_size']} eventi")
        print(f"     Episodica:       {mem['episodic_events']}")

    print(f"\n{DIVIDER}")
    print("  Demo completata.")
    print(f"  Tick totali: {agent.tick_count}")
    summary = agent.l5.session_summary()
    print(f"  Sessione:    {summary['session_id']}")
    print(f"  Uptime:      {summary['uptime_s']:.1f}s")
    print(f"  Memoria autobiografica: {summary['autobio_memories']} eventi")
    print(f"{DIVIDER}\n")


def run_server(agent: MAAAAgent, host: str = "0.0.0.0", port: int = 5002):
    app = create_maaa_app(agent)
    logger.info("MAAA REST API → http://%s:%d", host, port)
    print(f"\n  REST API attiva su http://{host}:{port}")
    print("  Endpoints principali:")
    print("    GET  /status         → riepilogo completo")
    print("    GET  /human          → stato cognitivo utente")
    print("    GET  /risk           → mappa del rischio")
    print("    GET  /guidance       → ultimo output guidance")
    print("    GET  /memory/working → memoria working (60s)")
    print("    GET  /memory/recall  → recall autobiografica")
    print("    POST /scenario/<n>   → inietta scenario (normal/smoky/collapsed/panic)")
    print("    POST /tick/n/<n>     → esegui N tick\n")
    app.run(host=host, port=port, threaded=True)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "demo"
    agent = MAAAAgent(simulation_mode=True, verbose=True)

    if mode == "demo":
        run_demo(agent)

    elif mode == "server":
        # Start background tick loop
        def _bg_loop():
            while True:
                agent.tick()
                time.sleep(0.1)   # 10 Hz background
        t = threading.Thread(target=_bg_loop, daemon=True)
        t.start()
        run_server(agent)

    elif mode == "both":
        def _demo():
            run_demo(agent)
        t = threading.Thread(target=_demo, daemon=True)
        t.start()
        run_server(agent)

    elif mode == "tick":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        print(f"Running {n} ticks...")
        for _ in range(n):
            snap = agent.tick()
        print(f"Done. Last: {snap.cognition.risk_map.global_risk_level.value} | "
              f"{snap.human.state.value}")

    else:
        print("Usage: python main_maaa.py [demo|server|both|tick <n>]")
        sys.exit(1)


if __name__ == "__main__":
    main()
