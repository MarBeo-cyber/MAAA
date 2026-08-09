"""
MAAA — single source for configuration and version.

The repository ships two implementations of the same five-layer idea (see
README.md): ``layers/`` (the reference runtime) and ``maaa_core/`` (a minimal
reference core).  They used to hard-code their own copies of the risk bands,
the brevity limit and the failsafe thresholds, and ``config/default.yaml`` was
never read by anything.  This module loads that YAML file once and hands the
same numbers to both, so a threshold is defined in exactly one place.

Resolution order for the configuration file:
  1. ``$MAAA_CONFIG``
  2. ``<repo root>/config/default.yaml``
  3. ``./config/default.yaml``
If none exists the embedded ``FALLBACK`` dict is used and a warning is logged.
``tests/test_config.py`` asserts that the YAML file and ``FALLBACK`` agree, so
the two cannot silently drift apart.
"""

from __future__ import annotations

import os
import copy
import logging
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger("maaa.config")

# Single version string for the whole repository.  pyproject.toml reads it from
# here (``[tool.setuptools.dynamic] version = {attr = "maaa_config.VERSION"}``),
# and maaa_core, main_maaa.py and the REST API all report this value.
VERSION = "0.4.0"
__version__ = VERSION

CONFIG_ENV_VAR = "MAAA_CONFIG"
_REPO_ROOT = Path(__file__).resolve().parent
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "default.yaml"

#: Used only when no configuration file can be found.  Kept in sync with
#: config/default.yaml by tests/test_config.py.
FALLBACK: dict[str, Any] = {
    "maaa": {
        "max_words": 9,
        "latency": {
            "tier0_guidance_ms": 50,
            "tier1_cognitive_loop_ms": 200,
            "tier2_memory_ms": 1000,
        },
        "safety": {
            "offline_first": True,
            "human_override": True,
            "raw_video_cloud_upload": False,
        },
        "risk_thresholds": {
            "low": 0.10,
            "medium": 0.35,
            "high": 0.60,
            "critical": 0.80,
        },
        "human_state_thresholds": {
            "alert": 0.30,
            "stressed": 0.60,
            "overloaded": 0.70,
            "frozen": 0.65,
            "collapsed": 0.75,
            "panicking": 0.75,
            "is_critical_panic": 0.75,
            "is_critical_freeze": 0.65,
            "is_critical_attentional": 0.75,
            "immediate_override_panic": 0.78,
            "immediate_override_overload": 0.80,
        },
        "failsafe": {
            "battery_min_pct": 5.0,
            "battery_warn_pct": 20.0,
            "sensor_integrity_min": 0.45,
            "max_latency_ms": 200.0,
            "stage_latency_max_ms": 150.0,
            "heartbeat_timeout_s": 2.0,
            "recovery_cycles": 10,
        },
        "resource_governor": {
            "llm_in_primary_emergency_loop": False,
            "local_models_only_default": True,
            "max_context_tokens_daily": 2048,
        },
    }
}

_cache: Optional[dict[str, Any]] = None
_cache_path: Optional[Path] = None


def config_path() -> Optional[Path]:
    """Return the configuration file that will be used, or None if there is none."""
    env = os.environ.get(CONFIG_ENV_VAR)
    if env:
        p = Path(env).expanduser()
        return p if p.is_file() else None
    for candidate in (_DEFAULT_CONFIG_PATH, Path.cwd() / "config" / "default.yaml"):
        if candidate.is_file():
            return candidate
    return None


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: Optional[str | Path] = None, force: bool = False) -> dict[str, Any]:
    """Load (and cache) the YAML configuration merged over ``FALLBACK``."""
    global _cache, _cache_path
    if _cache is not None and not force and path is None:
        return _cache

    chosen = Path(path) if path is not None else config_path()
    data: dict[str, Any] = {}
    if chosen is not None and chosen.is_file():
        with open(chosen, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        logger.debug("[config] loaded %s", chosen)
    else:
        logger.warning("[config] no config file found — using embedded FALLBACK")

    merged = _deep_merge(FALLBACK, data)
    if path is None:
        _cache, _cache_path = merged, chosen
    return merged


def get(dotted: str, default: Any = None) -> Any:
    """Fetch a value by dotted path, e.g. ``get("maaa.risk_thresholds.high")``."""
    node: Any = load_config()
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def risk_bands() -> list[tuple[str, float, float]]:
    """Ordered (name, lower_inclusive, upper_exclusive) risk bands from config.

    Shared by ``layers.l2_cognition.RISK_THRESHOLDS`` and
    ``maaa_core.risk.RiskEstimationEngine`` so both classify identically.
    """
    t = get("maaa.risk_thresholds", {}) or {}
    low = float(t.get("low", 0.10))
    medium = float(t.get("medium", 0.35))
    high = float(t.get("high", 0.60))
    critical = float(t.get("critical", 0.80))
    return [
        ("SAFE", 0.0, low),
        ("LOW", low, medium),
        ("MEDIUM", medium, high),
        ("HIGH", high, critical),
        ("CRITICAL", critical, 1.0 + 1e-9),
    ]


def max_words() -> int:
    return int(get("maaa.max_words", 9))


def human_state_threshold(name: str, default: float) -> float:
    return float(get(f"maaa.human_state_thresholds.{name}", default))


def failsafe_setting(name: str, default: float) -> float:
    return float(get(f"maaa.failsafe.{name}", default))
