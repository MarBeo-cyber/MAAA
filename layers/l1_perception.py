"""
MAAA — Layer 1: Embodied Perception (Percezione Incarnata)

Astrazione dei sensori fisici indossabili:
  - Occhiali AR (video frame, audio ambientale)
  - IMU (accelerometro + giroscopio → orientamento, rilevamento caduta)
  - GPS / Geolocalizzazione
  - Eye Tracker (focus attenzione, stima carico cognitivo)
  - Microfono direzionale (voce utente, suoni ambiente)
  - Depth Camera (ToF / RealSense)

In produzione ogni SensorAdapter si collega al driver hardware reale.
In modalità simulazione genera dati sintetici per testing e demo.
"""

from __future__ import annotations

import time
import math
import random
import logging
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

logger = logging.getLogger("maaa.l1_perception")


# ── Data Models ───────────────────────────────────────────────────────────────

class SceneCondition(Enum):
    NORMAL     = "normal"
    SMOKY      = "smoky"
    DARK       = "dark"
    DUSTY      = "dusty"
    OBSTRUCTED = "obstructed"
    COLLAPSED  = "collapsed"


@dataclass
class VideoFrame:
    """Single processed frame from AR glasses."""
    timestamp: float
    width: int
    height: int
    luminance: float          # 0.0 (pitch black) – 1.0 (bright)
    contrast: float           # 0.0 – 1.0
    motion_magnitude: float   # optical flow magnitude 0.0 – 1.0
    smoke_probability: float  # 0.0 – 1.0
    dust_level: float         # 0.0 – 1.0
    scene_condition: SceneCondition
    raw_pixels: Optional[object] = None   # np.ndarray in production


@dataclass
class DepthMap:
    """Depth estimation output from ToF / monocular model."""
    timestamp: float
    min_distance_m: float     # closest obstacle
    max_distance_m: float
    mean_distance_m: float
    obstacle_grid: list[list[float]] = field(default_factory=list)  # 8×8 coarse grid (meters)
    passable_path_exists: bool = True


@dataclass
class IMUData:
    """Inertial Measurement Unit — accelerometer + gyroscope."""
    timestamp: float
    accel_x: float            # m/s²
    accel_y: float
    accel_z: float
    gyro_x: float             # rad/s
    gyro_y: float
    gyro_z: float
    orientation_pitch: float  # degrees
    orientation_roll: float
    orientation_yaw: float
    fall_detected: bool = False
    sudden_movement: bool = False

    @property
    def total_acceleration(self) -> float:
        return math.sqrt(self.accel_x**2 + self.accel_y**2 + self.accel_z**2)


@dataclass
class GPSData:
    """GPS / indoor positioning data."""
    timestamp: float
    latitude: float
    longitude: float
    altitude_m: float
    accuracy_m: float
    floor: Optional[int] = None       # indoor floor level
    building_id: Optional[str] = None
    is_indoor: bool = False


@dataclass
class EyeTrackingData:
    """Eye tracker output — gaze, blink, pupil."""
    timestamp: float
    gaze_x: float             # normalized screen coords 0–1
    gaze_y: float
    blink_rate_per_min: float # normal: 15–20; stress: <10 or >30
    pupil_diameter_mm: float  # dilation indicates arousal/stress
    fixation_duration_ms: float  # short fixations → scanning; long → freeze
    saccade_velocity: float   # high velocity → panic; low → freeze
    vergence_angle: float     # degrees; measures binocular convergence


@dataclass
class AudioFrame:
    """Processed audio frame from directional microphone."""
    timestamp: float
    user_voice_detected: bool
    voice_pitch_hz: float     # stress raises pitch
    voice_tremor: float       # 0.0 – 1.0
    speech_rate_wpm: float    # normal ~130; panic >200; freeze <60
    ambient_db: float         # environmental noise level
    impact_sounds: bool       # bangs, crashes (structural collapse indicator)
    smoke_alarm: bool
    human_cries: bool         # other people in distress


@dataclass
class PerceptionFrame:
    """Aggregated perception snapshot — output of Layer 1."""
    timestamp: float
    video: VideoFrame
    depth: DepthMap
    imu: IMUData
    gps: GPSData
    eye: EyeTrackingData
    audio: AudioFrame

    # Derived quick-access fields
    @property
    def environment_quality(self) -> float:
        """0 = completely degraded, 1 = fully operational."""
        return (
            self.video.luminance * 0.3 +
            (1.0 - self.video.smoke_probability) * 0.3 +
            (1.0 - self.video.dust_level) * 0.2 +
            min(self.depth.min_distance_m / 2.0, 1.0) * 0.2
        )

    @property
    def is_physical_emergency(self) -> bool:
        return (
            self.imu.fall_detected or
            self.audio.impact_sounds or
            self.video.scene_condition == SceneCondition.COLLAPSED or
            self.video.smoke_probability > 0.7
        )


# ── Sensor Adapters ───────────────────────────────────────────────────────────

class ARGlassesAdapter:
    """
    Adapter for Ray-Ban Meta Smart Glasses / equivalent AR device.
    Production: connects to device SDK via USB/BT.
    Simulation: generates synthetic frames for specified scenario.
    """

    def __init__(self, simulation_mode: bool = True):
        self.simulation_mode = simulation_mode
        self._scenario = SceneCondition.NORMAL
        self._frame_count = 0

        if not simulation_mode:
            self._init_hardware()

    def _init_hardware(self):
        """Connect to real AR glasses SDK."""
        try:
            # Production: import device-specific SDK
            # import meta_ar_sdk as sdk
            # self._device = sdk.connect()
            logger.info("[ARGlasses] Hardware connected")
        except ImportError:
            logger.warning("[ARGlasses] SDK not found — falling back to simulation")
            self.simulation_mode = True

    def set_scenario(self, scenario: SceneCondition):
        """For simulation only: inject a scene condition."""
        self._scenario = scenario

    def capture_frame(self) -> VideoFrame:
        self._frame_count += 1
        ts = time.time()

        if self.simulation_mode:
            return self._synthesise_frame(ts)

        # Production path — read from device SDK
        # raw = self._device.get_frame()
        # return self._process_raw_frame(raw, ts)
        return self._synthesise_frame(ts)

    def _synthesise_frame(self, ts: float) -> VideoFrame:
        s = self._scenario
        base_noise = random.gauss(0, 0.02)

        params = {
            SceneCondition.NORMAL:     (0.85, 0.75, 0.1,  0.02, 0.01),
            SceneCondition.DARK:       (0.15, 0.3,  0.05, 0.03, 0.02),
            SceneCondition.SMOKY:      (0.4,  0.35, 0.2,  0.75, 0.4),
            SceneCondition.DUSTY:      (0.55, 0.45, 0.15, 0.3,  0.65),
            SceneCondition.OBSTRUCTED: (0.6,  0.5,  0.3,  0.1,  0.1),
            SceneCondition.COLLAPSED:  (0.25, 0.2,  0.6,  0.6,  0.5),
        }[s]

        lum, cont, motion, smoke, dust = params
        return VideoFrame(
            timestamp=ts,
            width=1920, height=1080,
            luminance=max(0, min(1, lum + base_noise)),
            contrast=max(0, min(1, cont + base_noise)),
            motion_magnitude=max(0, motion + abs(base_noise)),
            smoke_probability=max(0, min(1, smoke + base_noise * 0.5)),
            dust_level=max(0, min(1, dust + base_noise * 0.5)),
            scene_condition=s
        )


class IMUAdapter:
    def __init__(self, simulation_mode: bool = True):
        self.simulation_mode = simulation_mode
        self._panic_mode = False

    def set_panic(self, active: bool):
        self._panic_mode = active

    def read(self) -> IMUData:
        ts = time.time()
        if self._panic_mode:
            # Rapid, erratic movement
            ax, ay, az = random.gauss(2.0, 1.5), random.gauss(1.5, 1.0), random.gauss(9.8, 2.0)
            gx, gy, gz = random.gauss(0.8, 0.4), random.gauss(0.6, 0.3), random.gauss(0.5, 0.3)
        else:
            ax, ay, az = random.gauss(0.1, 0.05), random.gauss(0.1, 0.05), random.gauss(9.8, 0.1)
            gx, gy, gz = random.gauss(0, 0.02), random.gauss(0, 0.02), random.gauss(0, 0.02)

        total_a = math.sqrt(ax**2 + ay**2 + az**2)
        return IMUData(
            timestamp=ts,
            accel_x=ax, accel_y=ay, accel_z=az,
            gyro_x=gx, gyro_y=gy, gyro_z=gz,
            orientation_pitch=random.gauss(0, 5 if self._panic_mode else 1),
            orientation_roll=random.gauss(0, 3 if self._panic_mode else 0.5),
            orientation_yaw=random.gauss(0, 10 if self._panic_mode else 2),
            fall_detected=total_a > 25.0,
            sudden_movement=total_a > 15.0,
        )


class DepthCameraAdapter:
    def __init__(self, simulation_mode: bool = True):
        self.simulation_mode = simulation_mode
        self._obstruction_level = 0.0   # 0=clear, 1=fully blocked

    def set_obstruction(self, level: float):
        self._obstruction_level = max(0.0, min(1.0, level))

    def read(self) -> DepthMap:
        ts = time.time()
        min_d = max(0.3, 0.5 + random.gauss(0, 0.1) - self._obstruction_level * 0.4)
        mean_d = min_d + random.uniform(1.0, 3.0) * (1.0 - self._obstruction_level)
        return DepthMap(
            timestamp=ts,
            min_distance_m=min_d,
            max_distance_m=mean_d + random.uniform(1.0, 5.0),
            mean_distance_m=mean_d,
            passable_path_exists=min_d > 0.6 and self._obstruction_level < 0.8,
        )


class EyeTrackerAdapter:
    def __init__(self, simulation_mode: bool = True):
        self.simulation_mode = simulation_mode
        self._stress_level = 0.0

    def set_stress(self, level: float):
        self._stress_level = max(0.0, min(1.0, level))

    def read(self) -> EyeTrackingData:
        ts = time.time()
        s = self._stress_level
        return EyeTrackingData(
            timestamp=ts,
            gaze_x=random.uniform(0.2, 0.8),
            gaze_y=random.uniform(0.2, 0.8),
            blink_rate_per_min=max(2, random.gauss(15 - s * 8, 2)),   # stress reduces blinking
            pupil_diameter_mm=max(2.0, random.gauss(4.0 + s * 3.0, 0.5)),  # stress dilates
            fixation_duration_ms=max(50, random.gauss(300 - s * 200, 30)),
            saccade_velocity=max(0, random.gauss(100 + s * 300, 20)),
            vergence_angle=random.gauss(2.5, 0.3),
        )


class MicrophoneAdapter:
    def __init__(self, simulation_mode: bool = True):
        self.simulation_mode = simulation_mode
        self._panic_level = 0.0
        self._emergency_sounds = False

    def set_panic(self, level: float):
        self._panic_level = max(0.0, min(1.0, level))

    def set_emergency_sounds(self, active: bool):
        self._emergency_sounds = active

    def read(self) -> AudioFrame:
        ts = time.time()
        p = self._panic_level
        return AudioFrame(
            timestamp=ts,
            user_voice_detected=random.random() > 0.3,
            voice_pitch_hz=max(80, random.gauss(150 + p * 100, 20)),   # panic raises pitch
            voice_tremor=max(0, random.gauss(p * 0.7, 0.1)),
            speech_rate_wpm=max(40, random.gauss(130 + p * 80, 15)),
            ambient_db=random.gauss(45 + p * 30, 5),
            impact_sounds=self._emergency_sounds and random.random() < 0.3,
            smoke_alarm=self._emergency_sounds and random.random() < 0.2,
            human_cries=self._emergency_sounds and p > 0.6 and random.random() < 0.2,
        )


class GPSAdapter:
    def __init__(self, simulation_mode: bool = True,
                 base_lat: float = 45.4642,  # Milan
                 base_lon: float = 9.1900):
        self.simulation_mode = simulation_mode
        self.base_lat = base_lat
        self.base_lon = base_lon

    def read(self) -> GPSData:
        return GPSData(
            timestamp=time.time(),
            latitude=self.base_lat + random.gauss(0, 0.00001),
            longitude=self.base_lon + random.gauss(0, 0.00001),
            altitude_m=random.gauss(50.0, 0.5),
            accuracy_m=random.uniform(2.0, 8.0),
            floor=1,
            building_id="building_001",
            is_indoor=True,
        )


# ── Layer 1 Hub ───────────────────────────────────────────────────────────────

class L1EmbodiedPerception:
    """
    Layer 1 — aggregates all sensor adapters and emits PerceptionFrame objects
    at the configured sample rate (default 30 fps).
    """

    def __init__(self, simulation_mode: bool = True):
        self.simulation_mode = simulation_mode
        self.ar_glasses  = ARGlassesAdapter(simulation_mode)
        self.imu         = IMUAdapter(simulation_mode)
        self.depth_cam   = DepthCameraAdapter(simulation_mode)
        self.eye_tracker = EyeTrackerAdapter(simulation_mode)
        self.microphone  = MicrophoneAdapter(simulation_mode)
        self.gps         = GPSAdapter(simulation_mode)

        self._sample_count = 0
        logger.info("[L1] Embodied Perception initialized (sim=%s)", simulation_mode)

    def capture(self) -> PerceptionFrame:
        """Capture one synchronized frame from all sensors."""
        ts = time.time()
        self._sample_count += 1

        frame = PerceptionFrame(
            timestamp=ts,
            video=self.ar_glasses.capture_frame(),
            depth=self.depth_cam.read(),
            imu=self.imu.read(),
            gps=self.gps.read(),
            eye=self.eye_tracker.read(),
            audio=self.microphone.read(),
        )

        return frame

    # ── Scenario injection helpers (simulation only) ──────────────────────────

    def inject_scenario(self, scene: SceneCondition, stress: float = 0.0,
                        panic: float = 0.0, obstruction: float = 0.0,
                        emergency_sounds: bool = False):
        """Inject a simulated emergency scenario for testing."""
        self.ar_glasses.set_scenario(scene)
        self.eye_tracker.set_stress(stress)
        self.imu.set_panic(panic > 0.5)
        self.depth_cam.set_obstruction(obstruction)
        self.microphone.set_panic(panic)
        self.microphone.set_emergency_sounds(emergency_sounds)
        logger.info("[L1] Scenario injected: %s stress=%.2f panic=%.2f", scene.value, stress, panic)

    @property
    def sample_count(self) -> int:
        return self._sample_count
