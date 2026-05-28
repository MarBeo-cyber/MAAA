"""
MAAA — Layer 4: Symbiotic Regulation (Regolazione Simbiotica)

Il cuore del MAAA: decide COSA comunicare, QUANDO, COME e a quale URGENZA.
Obiettivo: minimizzare l'entropia cognitiva, non massimizzare l'informazione.

I quattro filtri del Regulatory Engine (Sec. 5.2 del documento):
  1. Filtro di rilevanza   → solo informazioni che cambiano il piano d'azione
  2. Filtro di timing      → output solo quando l'utente è in grado di recepirlo
  3. Filtro di brevità     → massimo 7–9 parole, sintassi imperativa
  4. Filtro di urgenza     → escalation del tono solo se il rischio aumenta

Output: GuidanceOutput con messaggi voce, overlay AR, alert aptico
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from layers.l2_cognition import CognitionFrame, RiskLevel, SceneCondition
from layers.l3_human_state import HumanStateFrame, CognitiveState

logger = logging.getLogger("maaa.l4_regulation")


# ── Output Models ─────────────────────────────────────────────────────────────

class UrgencyLevel(Enum):
    SILENT   = 0   # No output — user not receptive or no new info
    AMBIENT  = 1   # Soft background indicator
    NORMAL   = 2   # Standard guidance tone
    ELEVATED = 3   # Increased urgency, shorter sentences
    CRITICAL = 4   # Maximum urgency, single imperative verb


class OutputChannel(Enum):
    VOICE   = "voice"
    AR_OVERLAY = "ar_overlay"
    HAPTIC  = "haptic"


@dataclass
class AROverlay:
    """AR visual guidance to render on user's glasses."""
    active: bool
    highlight_bearing: Optional[float]     # Direction to highlight (degrees)
    danger_zones: list[float]              # Bearings to mark red
    path_arrow_bearing: Optional[float]   # Green arrow direction
    text_overlay: str                     # Short text on HUD (≤5 words)
    color_urgency: str                    # "green" / "yellow" / "red"


@dataclass
class HapticAlert:
    """Haptic feedback pattern."""
    active: bool
    pattern: str      # "single", "double", "continuous", "SOS"
    intensity: float  # 0.0 – 1.0


@dataclass
class GuidanceOutput:
    """
    Complete guidance package produced by Layer 4.
    Consumed by the output layer for voice synthesis, AR rendering, haptics.
    """
    timestamp: float
    urgency: UrgencyLevel

    # Voice message (post-filter: max 9 words, imperative)
    voice_message: str
    voice_message_full: str          # Unfiltered version for logging / debug

    # AR overlay
    ar_overlay: AROverlay

    # Haptic
    haptic: HapticAlert

    # Metadata
    active_channels: list[OutputChannel]
    filter_log: list[str]            # Which filters fired and why
    suppressed: bool                 # True if output was suppressed by timing filter
    suppression_reason: str = ""

    # Cognitive load budget
    estimated_words: int = 0
    delivery_delay_ms: float = 0.0   # How long to wait before delivering


# ── Filter Implementations ────────────────────────────────────────────────────

class RelevanceFilter:
    """
    Filter 1: Rilevanza
    Passes only information that changes the user's optimal action.
    Compares proposed guidance to last delivered guidance.
    """

    def __init__(self, change_threshold: float = 0.15):
        self._last_risk = 0.0
        self._last_bearing: Optional[float] = None
        self._last_message = ""
        self.change_threshold = change_threshold

    def is_relevant(self, cognition: CognitionFrame,
                    proposed_message: str) -> tuple[bool, str]:
        """Returns (is_relevant, reason)."""
        risk = cognition.risk_map.global_risk
        bearing = cognition.risk_map.recommended_path_bearing

        risk_changed = abs(risk - self._last_risk) > self.change_threshold
        bearing_changed = (bearing != self._last_bearing and
                           bearing is not None and
                           (self._last_bearing is None or
                            abs(bearing - self._last_bearing) > 20))
        new_critical = cognition.risk_map.get_critical_objects() and self._last_risk < 0.6

        if risk_changed or bearing_changed or new_critical:
            self._last_risk = risk
            self._last_bearing = bearing
            self._last_message = proposed_message
            return True, f"risk_delta={abs(risk - self._last_risk):.2f}"

        if proposed_message == self._last_message:
            return False, "identical_to_last_message"

        return False, "no_action_change"

    def update(self, risk: float, bearing: Optional[float], message: str):
        self._last_risk = risk
        self._last_bearing = bearing
        self._last_message = message


class TimingFilter:
    """
    Filter 2: Timing
    Suppresses output when user cannot receive it (panic peak, freeze),
    or enforces minimum intervals to avoid message flooding.
    """

    MIN_INTERVAL_NORMAL_S   = 3.0
    MIN_INTERVAL_ELEVATED_S = 1.5
    MIN_INTERVAL_CRITICAL_S = 0.5

    def __init__(self):
        self._last_output_time = 0.0
        self._consecutive_suppressed = 0

    def should_deliver(self, human: HumanStateFrame,
                       urgency: UrgencyLevel) -> tuple[bool, str]:
        now = time.time()
        elapsed = now - self._last_output_time

        # ── Minimum interval ──────────────────────────────────────────────────
        min_interval = {
            UrgencyLevel.CRITICAL: self.MIN_INTERVAL_CRITICAL_S,
            UrgencyLevel.ELEVATED: self.MIN_INTERVAL_ELEVATED_S,
        }.get(urgency, self.MIN_INTERVAL_NORMAL_S)

        if elapsed < min_interval and urgency != UrgencyLevel.CRITICAL:
            return False, f"too_soon ({elapsed:.1f}s < {min_interval}s)"

        # ── Receptivity gate ──────────────────────────────────────────────────
        # In panic peak, user cannot process speech — wait for brief calm
        if human.panic_score > 0.90 and urgency.value < UrgencyLevel.CRITICAL.value:
            self._consecutive_suppressed += 1
            return False, "panic_peak_receptivity_zero"

        # Frozen user: output IS useful (breaks freeze) — allow delivery
        # Collapsed attentional state: allow only if critical
        if human.attentional_collapse > 0.85 and urgency.value < UrgencyLevel.ELEVATED.value:
            return False, "attentional_collapse"

        self._last_output_time = now
        self._consecutive_suppressed = 0
        return True, "ok"

    @property
    def suppressed_count(self) -> int:
        return self._consecutive_suppressed


class BrevityFilter:
    """
    Filter 3: Brevità
    Enforces the 7–9 word maximum, imperative syntax, simple vocabulary.
    In production: backed by LLM with system prompt for ultra-brief instructions.
    """

    MAX_WORDS = 9

    # Urgency-level message templates
    TEMPLATES = {
        UrgencyLevel.CRITICAL: [
            "FERMATI.",
            "Non muoverti.",
            "Scendi subito.",
            "Esci ora.",
            "Dietro! Pericolo.",
        ],
        UrgencyLevel.ELEVATED: [
            "Gira a destra. Uscita a {dist:.0f} metri.",
            "Evita {obstacle}. Passa a sinistra.",
            "Accelera. Uscita vicina.",
            "Abbassati. Fumo in arrivo.",
        ],
        UrgencyLevel.NORMAL: [
            "Procedi avanti per {dist:.0f} metri.",
            "Uscita a destra. Rischio basso.",
            "Situazione stabile. Continua.",
            "Gira a {bearing:.0f} gradi. Percorso libero.",
        ],
        UrgencyLevel.AMBIENT: [
            "Ambiente monitorato.",
            "Nessuna minaccia immediata.",
        ],
    }

    def shorten(self, message: str, urgency: UrgencyLevel,
                cognition: CognitionFrame) -> str:
        """Ensure message fits in 7–9 words with imperative syntax."""
        words = message.split()
        if len(words) <= self.MAX_WORDS:
            return message

        # Truncate and ensure imperative
        shortened = " ".join(words[:self.MAX_WORDS])
        if not shortened.endswith("."):
            shortened += "."
        return shortened

    def generate(self, urgency: UrgencyLevel,
                 cognition: CognitionFrame) -> str:
        """Generate a contextual brief message."""
        risk_map = cognition.risk_map
        exits = risk_map.get_exits()

        if urgency == UrgencyLevel.CRITICAL:
            criticals = risk_map.get_critical_objects()
            if criticals:
                return f"PERICOLO {criticals[0].category.upper()}. Non avvicinarti."
            return "PERICOLO IMMINENTE. Fermati."

        if urgency == UrgencyLevel.ELEVATED:
            if exits:
                exit_obj = exits[0]
                return (f"Uscita a {exit_obj.bearing_deg:.0f}°, "
                        f"{exit_obj.distance_m:.0f} metri. Muoviti.")
            return "Allontanati dalla zona pericolosa."

        if urgency == UrgencyLevel.NORMAL:
            if exits:
                exit_obj = exits[0]
                return (f"Procedi verso uscita. "
                        f"{exit_obj.distance_m:.0f} metri a destra.")
            if risk_map.recommended_path_bearing is not None:
                return f"Percorso consigliato: {risk_map.recommended_path_bearing:.0f} gradi."
            return "Ambiente stabile. Continua a monitorare."

        return "Sistema attivo."


class UrgencyFilter:
    """
    Filter 4: Urgenza
    Computes urgency level from risk map + human state.
    Escalates ONLY when risk increases — prevents alarm fatigue.
    Never de-escalates faster than the situation warrants.
    """

    def __init__(self):
        self._prev_urgency = UrgencyLevel.SILENT
        self._peak_risk = 0.0

    def compute(self, cognition: CognitionFrame,
                human: HumanStateFrame) -> UrgencyLevel:
        risk = cognition.risk_map.global_risk
        self._peak_risk = max(self._peak_risk, risk)

        # Absolute thresholds
        if risk > 0.80 or human.panic_score > 0.85:
            urgency = UrgencyLevel.CRITICAL
        elif risk > 0.60 or human.cognitive_overload > 0.70:
            urgency = UrgencyLevel.ELEVATED
        elif risk > 0.35 or human.stress_score > 0.55:
            urgency = UrgencyLevel.NORMAL
        elif risk > 0.10:
            urgency = UrgencyLevel.AMBIENT
        else:
            urgency = UrgencyLevel.SILENT

        # Anti-alarm-fatigue: don't oscillate — hold elevated states
        if urgency.value < self._prev_urgency.value:
            # Only de-escalate if risk has genuinely dropped significantly
            if risk < self._peak_risk * 0.7:
                self._peak_risk = risk
                self._prev_urgency = urgency
            else:
                urgency = self._prev_urgency  # Hold
        else:
            self._prev_urgency = urgency

        return urgency


# ── Layer 4 Regulatory Engine ─────────────────────────────────────────────────

class L4SymbioticRegulation:
    """
    Layer 4 — Symbiotic Regulatory Engine.

    Applies the four filters sequentially to produce minimal, calibrated guidance
    that minimizes cognitive entropy rather than maximizing information transfer.
    """

    def __init__(self):
        self.relevance_filter = RelevanceFilter()
        self.timing_filter    = TimingFilter()
        self.brevity_filter   = BrevityFilter()
        self.urgency_filter   = UrgencyFilter()
        self._output_count    = 0
        self._suppressed_count = 0
        logger.info("[L4] Symbiotic Regulatory Engine initialized")

    def regulate(self, cognition: CognitionFrame,
                 human: HumanStateFrame) -> GuidanceOutput:
        ts = time.time()
        filter_log = []

        # ── Filter 4: Urgency (determines tone BEFORE content) ────────────────
        urgency = self.urgency_filter.compute(cognition, human)
        filter_log.append(f"urgency={urgency.name}")

        # ── Filter 3: Brevity (generate the message) ──────────────────────────
        full_message = self.brevity_filter.generate(urgency, cognition)
        brief_message = self.brevity_filter.shorten(full_message, urgency, cognition)
        filter_log.append(f"words={len(brief_message.split())}")

        # ── Filter 1: Relevance ───────────────────────────────────────────────
        if urgency.value < UrgencyLevel.CRITICAL.value:
            relevant, rel_reason = self.relevance_filter.is_relevant(cognition, brief_message)
            filter_log.append(f"relevance={'pass' if relevant else 'block'}:{rel_reason}")
            if not relevant:
                return self._suppressed_output(ts, urgency, brief_message,
                                               filter_log, "relevance_filter")
        else:
            self.relevance_filter.update(
                cognition.risk_map.global_risk,
                cognition.risk_map.recommended_path_bearing,
                brief_message
            )
            filter_log.append("relevance=bypassed:critical")

        # ── Filter 2: Timing ──────────────────────────────────────────────────
        should_deliver, timing_reason = self.timing_filter.should_deliver(human, urgency)
        filter_log.append(f"timing={'pass' if should_deliver else 'block'}:{timing_reason}")
        if not should_deliver:
            return self._suppressed_output(ts, urgency, brief_message,
                                           filter_log, f"timing:{timing_reason}")

        # ── Build AR Overlay ──────────────────────────────────────────────────
        ar = self._build_ar_overlay(cognition, urgency)

        # ── Build Haptic ──────────────────────────────────────────────────────
        haptic = self._build_haptic(urgency, human)

        # ── Determine delivery delay (give user micro-pause to process) ───────
        delay_ms = 0.0
        if human.freeze_score > 0.5:
            delay_ms = 200.0  # Slight delay helps frozen users orient

        # ── Active channels ───────────────────────────────────────────────────
        channels = [OutputChannel.VOICE, OutputChannel.AR_OVERLAY]
        if urgency.value >= UrgencyLevel.ELEVATED.value:
            channels.append(OutputChannel.HAPTIC)

        self._output_count += 1

        return GuidanceOutput(
            timestamp=ts,
            urgency=urgency,
            voice_message=brief_message,
            voice_message_full=full_message,
            ar_overlay=ar,
            haptic=haptic,
            active_channels=channels,
            filter_log=filter_log,
            suppressed=False,
            estimated_words=len(brief_message.split()),
            delivery_delay_ms=delay_ms,
        )

    def _suppressed_output(self, ts: float, urgency: UrgencyLevel,
                           message: str, filter_log: list[str],
                           reason: str) -> GuidanceOutput:
        self._suppressed_count += 1
        return GuidanceOutput(
            timestamp=ts,
            urgency=urgency,
            voice_message="",
            voice_message_full=message,
            ar_overlay=AROverlay(False, None, [], None, "", "green"),
            haptic=HapticAlert(False, "none", 0.0),
            active_channels=[],
            filter_log=filter_log,
            suppressed=True,
            suppression_reason=reason,
        )

    def _build_ar_overlay(self, cognition: CognitionFrame,
                          urgency: UrgencyLevel) -> AROverlay:
        risk_map = cognition.risk_map
        danger_bearings = [o.bearing_deg for o in risk_map.objects
                           if o.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)]
        exits = risk_map.get_exits()
        path_bearing = exits[0].bearing_deg if exits else risk_map.recommended_path_bearing

        color = {"SILENT": "green", "AMBIENT": "green",
                 "NORMAL": "yellow", "ELEVATED": "orange",
                 "CRITICAL": "red"}[urgency.name]

        hud_text = {
            UrgencyLevel.CRITICAL: "⚠ PERICOLO",
            UrgencyLevel.ELEVATED: "Uscita →",
            UrgencyLevel.NORMAL:   "Percorso OK",
            UrgencyLevel.AMBIENT:  "Monitoraggio",
            UrgencyLevel.SILENT:   "",
        }[urgency]

        return AROverlay(
            active=urgency != UrgencyLevel.SILENT,
            highlight_bearing=path_bearing,
            danger_zones=danger_bearings,
            path_arrow_bearing=path_bearing,
            text_overlay=hud_text,
            color_urgency=color,
        )

    def _build_haptic(self, urgency: UrgencyLevel,
                      human: HumanStateFrame) -> HapticAlert:
        patterns = {
            UrgencyLevel.CRITICAL: ("SOS",        1.0),
            UrgencyLevel.ELEVATED: ("double",     0.7),
            UrgencyLevel.NORMAL:   ("single",     0.4),
            UrgencyLevel.AMBIENT:  ("single",     0.2),
            UrgencyLevel.SILENT:   ("none",       0.0),
        }
        pattern, intensity = patterns[urgency]
        return HapticAlert(
            active=urgency.value >= UrgencyLevel.ELEVATED.value,
            pattern=pattern,
            intensity=intensity,
        )

    @property
    def output_stats(self) -> dict:
        total = self._output_count + self._suppressed_count
        return {
            "total_cycles":      total,
            "outputs_delivered": self._output_count,
            "outputs_suppressed": self._suppressed_count,
            "delivery_rate":     self._output_count / max(1, total),
        }
