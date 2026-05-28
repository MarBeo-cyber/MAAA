"""
MAAA — Core Agent (Orchestratore Pipeline)

Implementa il ciclo principale a 8 step (Sez. 4.1 del documento):

  Step 1: Acquisizione Sensori        → L1 EmbodiedPerception.capture()
  Step 2: Pre-processing Edge         → normalizzazione, feature extraction
  Step 3: Percezione Semantica        → L2 SituationalCognition.process()
  Step 4: Scene Graph Update          → già incluso in L2
  Step 5: Human State Estimation      → L3 HumanStateMonitor.process()
  Step 6: Regulatory Engine           → L4 SymbioticRegulation.regulate()
  Step 7: Output Multimodale          → OutputDispatcher
  Step 8: Autopoietic Check           → L5 AutopoieticContinuity.process()

Target: latenza end-to-end < 200ms, tier critico < 50ms
"""

from __future__ import annotations

import sys
import time
import logging
import threading
from dataclasses import dataclass

from layers.l1_perception  import L1EmbodiedPerception,  SceneCondition, PerceptionFrame
from layers.l2_cognition   import L2SituationalCognition, CognitionFrame
from layers.l3_human_state import L3HumanStateMonitor,   HumanStateFrame
from layers.l4_regulation  import L4SymbioticRegulation,  GuidanceOutput, UrgencyLevel
from layers.l5_continuity  import L5AutopoieticContinuity, SystemHealth

logger = logging.getLogger("maaa.agent")


@dataclass
class PipelineSnapshot:
    """Complete state of one pipeline cycle — for REST API / logging."""
    tick:       int
    timestamp:  float
    latency_ms: float
    perception: PerceptionFrame
    cognition:  CognitionFrame
    human:      HumanStateFrame
    guidance:   GuidanceOutput
    health:     SystemHealth


class OutputDispatcher:
    """
    Step 7: Dispatches GuidanceOutput to physical actuators.
    Production: calls TTS engine, AR renderer, haptic motor driver.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._delivered = 0
        self._suppressed = 0

    def dispatch(self, guidance: GuidanceOutput, human: HumanStateFrame):
        if guidance.suppressed or not guidance.voice_message:
            self._suppressed += 1
            return

        self._delivered += 1

        if self.verbose:
            urgency_icons = {
                UrgencyLevel.SILENT:   "○",
                UrgencyLevel.AMBIENT:  "●",
                UrgencyLevel.NORMAL:   "▶",
                UrgencyLevel.ELEVATED: "⚡",
                UrgencyLevel.CRITICAL: "🔴",
            }
            icon = urgency_icons.get(guidance.urgency, "▶")
            delay = ""
            if guidance.delivery_delay_ms > 0:
                delay = f" [delay {guidance.delivery_delay_ms:.0f}ms]"
            print(f"  {icon} [{guidance.urgency.name}]{delay} "
                  f"\"{guidance.voice_message}\"")

            if guidance.ar_overlay.active:
                print(f"     AR: {guidance.ar_overlay.text_overlay} "
                      f"({guidance.ar_overlay.color_urgency})"
                      + (f" → {guidance.ar_overlay.path_arrow_bearing:.0f}°"
                         if guidance.ar_overlay.path_arrow_bearing else ""))

            if guidance.haptic.active:
                print(f"     ⚡ Haptic: {guidance.haptic.pattern} "
                      f"intensity={guidance.haptic.intensity:.1f}")

        # Production hooks:
        # self._tts_engine.speak(guidance.voice_message, priority=guidance.urgency)
        # self._ar_renderer.update(guidance.ar_overlay)
        # self._haptic_driver.pulse(guidance.haptic)

    @property
    def stats(self) -> dict:
        total = self._delivered + self._suppressed
        return {
            "delivered":  self._delivered,
            "suppressed": self._suppressed,
            "rate":       self._delivered / max(1, total),
        }


class MAAAAgent:
    """
    MAAA — Metacognitive Autopoietic Adaptive Agent.
    Orchestrates the 5-layer pipeline in a real-time loop.
    """

    def __init__(self, simulation_mode: bool = True, verbose: bool = True,
                 db_path: str = "/tmp/maaa_episodes.db",
                 autobio_path: str = "/tmp/maaa_autobio.json"):

        logger.info("╔══════════════════════════════════════════════╗")
        logger.info("║  MAAA — Metacognitive Autopoietic Adaptive   ║")
        logger.info("║         Agent  v1.0  — Initializing          ║")
        logger.info("╚══════════════════════════════════════════════╝")

        self.simulation_mode = simulation_mode
        self.verbose = verbose

        # ── Instantiate all 5 layers ──────────────────────────────────────────
        self.l1 = L1EmbodiedPerception(simulation_mode)
        self.l2 = L2SituationalCognition(simulation_mode)
        self.l3 = L3HumanStateMonitor()
        self.l4 = L4SymbioticRegulation()
        self.l5 = L5AutopoieticContinuity(db_path, autobio_path)

        self.output = OutputDispatcher(verbose)

        self._tick = 0
        self._running = False
        self._last_snapshot: PipelineSnapshot | None = None
        self._snapshots: list[PipelineSnapshot] = []   # Rolling buffer for API
        self._MAX_SNAPSHOTS = 60

        logger.info("[MAAA] All 5 layers initialized.")

    # ── Single pipeline cycle ─────────────────────────────────────────────────

    def tick(self) -> PipelineSnapshot:
        """Execute one complete 8-step pipeline cycle."""
        cycle_start = time.time()
        self._tick += 1

        # Step 1-2: Sensor acquisition + pre-processing (L1)
        perception = self.l1.capture()

        # Step 3-4: Semantic perception + scene graph (L2)
        cognition = self.l2.process(perception)

        # Step 5: Human state estimation (L3)
        human = self.l3.process(perception)

        # Step 6: Regulatory engine (L4) — applies 4 filters
        guidance = self.l4.regulate(cognition, human)

        # Step 7: Output dispatch
        self.output.dispatch(guidance, human)

        # Step 8: Autopoietic continuity check (L5)
        latency_ms = (time.time() - cycle_start) * 1000
        health = self.l5.process(cognition, human, guidance, latency_ms)

        snapshot = PipelineSnapshot(
            tick=self._tick,
            timestamp=time.time(),
            latency_ms=latency_ms,
            perception=perception,
            cognition=cognition,
            human=human,
            guidance=guidance,
            health=health,
        )
        self._last_snapshot = snapshot
        self._snapshots.append(snapshot)
        if len(self._snapshots) > self._MAX_SNAPSHOTS:
            self._snapshots.pop(0)

        return snapshot

    # ── Scenario injection ────────────────────────────────────────────────────

    def inject_scenario(self, scene: SceneCondition, stress: float = 0.0,
                        panic: float = 0.0, obstruction: float = 0.0,
                        emergency_sounds: bool = False):
        """Inject a simulated emergency scenario."""
        self.l1.inject_scenario(scene, stress, panic, obstruction, emergency_sounds)

    # ── Continuous run loop ───────────────────────────────────────────────────

    def run_continuous(self, hz: float = 10.0, max_ticks: int | None = None):
        """Run the pipeline at the specified rate (default 10 Hz for demo)."""
        self._running = True
        interval = 1.0 / hz
        logger.info("[MAAA] Starting continuous loop at %.0f Hz", hz)
        try:
            while self._running:
                loop_start = time.time()
                snap = self.tick()

                if self.verbose and self._tick % 5 == 0:
                    self._print_status(snap)

                if max_ticks and self._tick >= max_ticks:
                    break

                # Sleep to maintain target rate
                elapsed = time.time() - loop_start
                sleep = max(0.0, interval - elapsed)
                time.sleep(sleep)
        except KeyboardInterrupt:
            logger.info("[MAAA] Interrupted by user.")
        finally:
            self._running = False
            self.l5.close()
            logger.info("[MAAA] Session summary: %s", self.l5.session_summary())

    def stop(self):
        self._running = False

    def _print_status(self, snap: PipelineSnapshot):
        c = snap.cognition
        h = snap.human
        g = snap.guidance
        health = snap.health
        print(f"\n── Tick {self._tick:04d} | {snap.latency_ms:.1f}ms "
              f"{'⚠ SLOW' if snap.latency_ms > 200 else ''}")
        print(f"  L2 Risk:    {c.risk_map.global_risk_level.value} "
              f"({c.risk_map.global_risk:.2f}) | "
              f"env_quality={snap.perception.environment_quality:.2f}")
        print(f"  L3 Human:   {h.state.value} | "
              f"stress={h.stress_score:.2f} panic={h.panic_score:.2f} "
              f"receptivity={h.receptivity:.2f}")
        print(f"  L4 Output:  {'SUPPRESSED' if g.suppressed else g.urgency.name} "
              f"| delivery_rate={self.l4.output_stats['delivery_rate']:.1%}")
        if health.warnings:
            print(f"  L5 Warnings: {', '.join(health.warnings)}")
        if c.event_predictions:
            print(f"  Predictions: {', '.join(c.event_predictions)}")

    @property
    def last_snapshot(self) -> PipelineSnapshot | None:
        return self._last_snapshot

    @property
    def tick_count(self) -> int:
        return self._tick
