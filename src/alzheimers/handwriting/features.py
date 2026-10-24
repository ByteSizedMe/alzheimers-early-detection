"""Kinematic features from a pen/mouse trace.

A trace is a list of strokes (pen-down to pen-up); each stroke is an array of
``(x, y, t)`` rows with ``t`` in milliseconds. The same function scores a drawing
captured live in the browser and a synthetic one from ``synthetic.py``, so the
two cannot drift apart.

Every feature is expressed relative to the canvas diagonal, so the numbers do not
depend on screen size or zoom.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

RESAMPLE_HZ = 100.0
PAUSE_SPEED = 0.05  # canvas-diagonals per second (5 Hz low-passed); below this the pen is "stalled"
TREMOR_BAND = (4.0, 12.0)  # Hz

FEATURE_NAMES = (
    "duration_s",
    "mean_speed",
    "speed_cv",
    "pause_ratio",
    "in_air_ratio",
    "speed_peaks",
    "tremor_amp",
    "spiral_irregularity",
    "size_norm",
)


@dataclass(frozen=True)
class FeatureInfo:
    label: str
    unit: str
    meaning: str


FEATURE_INFO: dict[str, FeatureInfo] = {
    "duration_s": FeatureInfo(
        "Drawing time", "s", "Total time from first to last touch. Slowed psychomotor speed lengthens it."
    ),
    "mean_speed": FeatureInfo(
        "Mean pen speed", "diag/s", "Path length per second of pen contact, as a fraction of the canvas diagonal."
    ),
    "speed_cv": FeatureInfo(
        "Speed variability", "", "Spread of pen speed relative to its mean. Uneven, stop-start motion raises it."
    ),
    "pause_ratio": FeatureInfo(
        "Stalled time", "%", "Share of pen-down time spent nearly stationary (hesitation)."
    ),
    "in_air_ratio": FeatureInfo(
        "Pen-up time", "%", "Share of the whole task spent with the pen lifted. Longer in-air time is a reported marker."
    ),
    "speed_peaks": FeatureInfo(
        "Speed peaks", "per diag", "Velocity peaks per canvas-diagonal of path. A fluent stroke has few; hesitant ones have many."
    ),
    "tremor_amp": FeatureInfo(
        "Tremor amplitude", "‰ diag", "RMS of 4-12 Hz oscillation in pen position, in thousandths of the diagonal."
    ),
    "spiral_irregularity": FeatureInfo(
        "Spiral irregularity", "%", "How far the radius wobbles around a smooth spiral, relative to the mean radius."
    ),
    "size_norm": FeatureInfo(
        "Drawing size", "% of canvas", "Extent of the drawing relative to the canvas. Shrinking size is micrographia-like."
    ),
}


def _clean_stroke(stroke) -> np.ndarray | None:
    arr = np.asarray(stroke, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 3 or len(arr) < 2:
        return None
    arr = arr[np.argsort(arr[:, 2], kind="stable")][:, :3]
    # Drop repeated timestamps; interpolation needs strictly increasing t.
    keep = np.concatenate([[True], np.diff(arr[:, 2]) > 0])
    arr = arr[keep]
    return arr if len(arr) >= 2 else None


def _resample(stroke: np.ndarray, hz: float = RESAMPLE_HZ) -> np.ndarray:
    """Uniformly resample one stroke; returns columns x, y, t(s)."""
    t = stroke[:, 2] / 1000.0
    grid = np.arange(t[0], t[-1], 1.0 / hz)
    if len(grid) < 2:
        grid = np.array([t[0], t[-1]])
    return np.column_stack([np.interp(grid, t, stroke[:, 0]), np.interp(grid, t, stroke[:, 1]), grid])


def _speed(resampled: np.ndarray, diag: float, hz: float = RESAMPLE_HZ) -> np.ndarray:
    """Pen speed in canvas-diagonals per second, lightly smoothed.

    Pointer coordinates are integers, so raw differences are staircase-shaped; a
    ~50 ms moving average removes that quantisation without hiding real stalls.
    """
    d = np.hypot(np.diff(resampled[:, 0]), np.diff(resampled[:, 1])) * hz / diag
    if len(d) < 5:
        return d
    kernel = np.ones(5) / 5
    return np.convolve(d, kernel, mode="same")


def _tremor_rms(resampled: np.ndarray, diag: float, hz: float = RESAMPLE_HZ) -> float | None:
    if len(resampled) < 60:  # filtfilt needs headroom; shorter strokes carry no tremor signal
        return None
    sos = signal.butter(4, TREMOR_BAND, btype="bandpass", fs=hz, output="sos")
    xs = signal.sosfiltfilt(sos, resampled[:, 0])
    ys = signal.sosfiltfilt(sos, resampled[:, 1])
    return float(np.sqrt(np.mean(xs**2 + ys**2)) / diag)


def _spiral_irregularity(strokes: list[np.ndarray]) -> float:
    """Radial residual of the whole trace around a smooth spiral.

    The task asks the person to start at the centre, so the first point is the
    origin. Radius is regressed on unwrapped angle with a low-order polynomial;
    what is left over is wobble.
    """
    pts = np.vstack([s[:, :2] for s in strokes])
    if len(pts) < 20:
        return 0.0
    centre = pts[0]
    rel = pts - centre
    r = np.hypot(rel[:, 0], rel[:, 1])
    theta = np.unwrap(np.arctan2(rel[:, 1], rel[:, 0]))
    moving = r > 0.05 * r.max()  # angle is meaningless right at the origin
    r, theta = r[moving], theta[moving]
    if len(r) < 20 or np.ptp(theta) < np.pi:
        return 0.0
    # Low-order fit in angle-normalised units: well conditioned, and wobble shorter
    # than the fit's own curvature stays in the residual.
    u = (theta - theta.min()) / np.ptp(theta)
    coeffs = np.polyfit(u, r, 3)
    resid = r - np.polyval(coeffs, u)
    return float(np.std(resid) / np.mean(r))


def extract_features(strokes, canvas_size: tuple[float, float]) -> dict[str, float]:
    """Compute the feature vector for one drawing.

    Raises ``ValueError`` when there is too little data to measure anything.
    """
    width, height = canvas_size
    diag = float(np.hypot(width, height))
    if diag <= 0:
        raise ValueError("canvas size must be positive")

    cleaned = [s for s in (_clean_stroke(s) for s in strokes) if s is not None]
    if not cleaned:
        raise ValueError("no usable strokes: draw for at least a moment")

    resampled = [_resample(s) for s in cleaned]
    speeds = np.concatenate([_speed(r, diag) for r in resampled])
    if len(speeds) < 30:
        raise ValueError("drawing too short to analyse; draw the whole spiral")

    t_start = min(s[0, 2] for s in cleaned) / 1000.0
    t_end = max(s[-1, 2] for s in cleaned) / 1000.0
    duration = t_end - t_start
    contact = sum((s[-1, 2] - s[0, 2]) / 1000.0 for s in cleaned)

    path_len = sum(
        float(np.sum(np.hypot(np.diff(r[:, 0]), np.diff(r[:, 1])))) for r in resampled
    ) / diag
    mean_speed = path_len / contact if contact > 0 else 0.0

    # Peaks and stalls are both read off a 5 Hz low-passed speed, so neither
    # pointer quantisation nor tremor (4-12 Hz) registers as hesitation.
    peaks = 0
    stalled = 0
    total_samples = 0
    sos = signal.butter(2, 5.0, btype="low", fs=RESAMPLE_HZ, output="sos")
    for r in resampled:
        v = _speed(r, diag)
        if len(v) < 40:
            continue
        smooth = signal.sosfiltfilt(sos, v)
        stalled += int(np.sum(smooth < PAUSE_SPEED))
        total_samples += len(smooth)
        floor = max(np.median(smooth), 1e-6)
        found, _ = signal.find_peaks(smooth, prominence=0.15 * floor, distance=int(0.15 * RESAMPLE_HZ))
        peaks += len(found)

    tremors = [v for v in (_tremor_rms(r, diag) for r in resampled) if v is not None]
    pts = np.vstack([s[:, :2] for s in cleaned])
    extent = float(np.hypot(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])))

    return {
        "duration_s": float(duration),
        "mean_speed": float(mean_speed),
        "speed_cv": float(np.std(speeds) / max(np.mean(speeds), 1e-9)),
        "pause_ratio": float(stalled / total_samples) if total_samples else 0.0,
        "in_air_ratio": float(max(duration - contact, 0.0) / duration) if duration > 0 else 0.0,
        "speed_peaks": float(peaks / path_len) if path_len > 0 else 0.0,
        "tremor_amp": float(np.mean(tremors) * 1000.0) if tremors else 0.0,
        "spiral_irregularity": _spiral_irregularity(cleaned),
        "size_norm": float(extent / diag),
    }


def speed_trace(strokes, canvas_size: tuple[float, float]) -> list[list[dict[str, float]]]:
    """The drawing as coloured-by-speed points, for the front end to render.

    Speed is scaled to 0-1 against the trace's own 95th percentile so the colour
    ramp always spans the drawing; absolute speed is already in the features.
    """
    diag = float(np.hypot(*canvas_size))
    cleaned = [s for s in (_clean_stroke(s) for s in strokes) if s is not None]
    all_speeds = []
    per_stroke = []
    for s in cleaned:
        dt = np.diff(s[:, 2]) / 1000.0
        d = np.hypot(np.diff(s[:, 0]), np.diff(s[:, 1])) / diag
        v = d / np.maximum(dt, 1e-3)
        if len(v) >= 5:
            v = np.convolve(v, np.ones(5) / 5, mode="same")
        v = np.concatenate([v[:1], v])
        per_stroke.append((s, v))
        all_speeds.append(v)
    if not all_speeds:
        return []
    scale = max(float(np.percentile(np.concatenate(all_speeds), 95)), 1e-9)
    return [
        [
            {"x": round(float(x), 1), "y": round(float(y), 1), "s": round(float(min(sp / scale, 1.0)), 3)}
            for (x, y), sp in zip(s[:, :2], v, strict=True)
        ]
        for s, v in per_stroke
    ]
