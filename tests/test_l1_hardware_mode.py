"""With simulation_mode=False nothing may return synthetic data.

Before the fix, L1EmbodiedPerception(simulation_mode=False) logged
"[ARGlasses] Hardware connected", silently downgraded itself to simulation and
then emitted synthetic frames — including GPS at hardcoded Milan coordinates
plus gaussian noise — that were indistinguishable from real sensor readings.
"""

from __future__ import annotations

import logging

import pytest

from layers.l1_perception import (ARGlassesAdapter, DepthCameraAdapter,
                                  EyeTrackerAdapter, GPSAdapter,
                                  HardwareUnavailableError, IMUAdapter,
                                  L1EmbodiedPerception, MicrophoneAdapter)


def test_ar_glasses_refuse_to_construct_in_hardware_mode():
    with pytest.raises(HardwareUnavailableError):
        ARGlassesAdapter(simulation_mode=False)


def test_layer_one_refuses_to_construct_in_hardware_mode():
    with pytest.raises(HardwareUnavailableError):
        L1EmbodiedPerception(simulation_mode=False)


@pytest.mark.parametrize("adapter_cls", [
    IMUAdapter, DepthCameraAdapter, EyeTrackerAdapter,
    MicrophoneAdapter, GPSAdapter,
])
def test_every_adapter_raises_instead_of_synthesising(adapter_cls):
    """These five never even branched on simulation_mode."""
    adapter = adapter_cls(simulation_mode=False)
    with pytest.raises(HardwareUnavailableError):
        adapter.read()


def test_capture_refuses_when_flipped_to_hardware_mode_after_construction():
    l1 = L1EmbodiedPerception(simulation_mode=True)
    l1.simulation_mode = False
    with pytest.raises(HardwareUnavailableError):
        l1.capture()


def test_init_hardware_does_not_log_a_success_it_did_not_achieve(caplog):
    with caplog.at_level(logging.INFO, logger="maaa.l1_perception"):
        with pytest.raises(HardwareUnavailableError):
            ARGlassesAdapter(simulation_mode=False)
    messages = [r.getMessage() for r in caplog.records]
    assert not any("Hardware connected" in m for m in messages), messages
    assert any("hardware mode unavailable" in m.lower() for m in messages), messages


def test_simulation_mode_still_works():
    l1 = L1EmbodiedPerception(simulation_mode=True)
    frame = l1.capture()
    assert frame.video.width == 1920
    assert 0.0 <= frame.video.luminance <= 1.0
