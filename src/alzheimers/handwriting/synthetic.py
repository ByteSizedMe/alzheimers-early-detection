"""Synthetic spiral-drawing traces.

No handwriting cohort ships with this repository, so the classifier trains on
simulated drawings. The simulator encodes the *direction* of the kinematic
changes reported in the handwriting literature for cognitive decline -- slower
and more variable pen speed, more stalls, longer pen-up time, more tremor, a
wobblier and smaller spiral. It does not encode their size, and the numbers it
produces say nothing about real patients. Replace it with a recorded dataset
(e.g. DARWIN, Cilia et al. 2022) before reading anything into accuracy.

To keep the problem from being trivially separable, every subject gets their own
baseline speed, size and tremor, and the two groups' latent severity overlap.
"""

from __future__ import annotations

import numpy as np

from .features import FEATURE_NAMES, extract_features

CANVAS = (720.0, 520.0)
FS = 200  # internal simulation rate, Hz

# Fixed severities for the replayable demo profiles.
DEMO_SEVERITY = {"typical": 0.05, "borderline": 0.32, "elevated": 0.85}


def _smooth_noise(rng: np.random.Generator, n: int, window_s: float) -> np.ndarray:
    raw = rng.standard_normal(n + int(window_s * FS))
    k = max(int(window_s * FS), 1)
    out = np.convolve(raw, np.ones(k) / k, mode="valid")[:n]
    return (out - out.mean()) / (out.std() + 1e-9)


def generate_drawing(severity: float, rng: np.random.Generator, canvas=CANVAS) -> list[np.ndarray]:
    """One simulated spiral as a list of ``(x, y, t_ms)`` strokes."""
    w, h = canvas
    diag = float(np.hypot(w, h))
    s = float(np.clip(severity, 0.0, 1.0))

    # --- ideal path: an Archimedean spiral from the centre ----------------------
    turns = rng.uniform(2.5, 3.5)
    size = (1.0 - 0.35 * s * rng.uniform(0.5, 1.2)) * rng.uniform(0.85, 1.15)
    radius = 0.40 * min(w, h) * size
    theta = np.linspace(0.0, 2 * np.pi * turns, 4000)
    wobble = (0.012 + 0.05 * s * rng.uniform(0.5, 1.5)) * radius
    wob = _smooth_noise(rng, len(theta), 0.3)  # smooth in index space, ~1 turn long
    r = radius * theta / theta[-1] + wobble * wob * (theta / theta[-1])
    sign = rng.choice([-1.0, 1.0])
    cx = w / 2 + rng.uniform(-30, 30)
    cy = h / 2 + rng.uniform(-30, 30)
    px = cx + r * np.cos(sign * theta)
    py = cy + r * np.sin(sign * theta)
    arclen = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(px), np.diff(py)))])
    total = arclen[-1]

    # --- pen speed over contact time -------------------------------------------
    n_max = 150 * FS
    v0 = 0.35 * np.exp(rng.normal(0.0, 0.25)) * (1.0 - 0.35 * s) * diag
    sigma = 0.10 + 0.50 * s
    speed = v0 * np.exp(sigma * _smooth_noise(rng, n_max, 0.35))
    expected = int(total / max(v0, 1.0) * FS)  # stalls must land inside the drawing
    for _ in range(rng.poisson(0.2 + 4.0 * s)):  # stalls
        start = rng.integers(FS // 2, max(int(0.9 * expected), FS))
        speed[start : start + int(rng.uniform(0.25, 0.8) * FS)] = 0.0
    arc = np.cumsum(speed) / FS
    n_end = int(np.searchsorted(arc, total))
    if n_end >= n_max:  # pathological draw; make it finish rather than loop
        n_end = n_max - 1
        arc = arc * (total / arc[n_end])
    arc = arc[:n_end]
    t_contact = np.arange(n_end) / FS

    x = np.interp(arc, arclen, px)
    y = np.interp(arc, arclen, py)

    # --- tremor -----------------------------------------------------------------
    amp = diag * (0.0004 + 0.0018 * s * rng.uniform(0.3, 1.3))
    freq = rng.uniform(4.5, 8.0)
    mod = 1.0 + 0.2 * _smooth_noise(rng, n_end, 1.0)
    x = x + amp * mod * np.sin(2 * np.pi * freq * t_contact + rng.uniform(0, 2 * np.pi))
    y = y + amp * mod * np.sin(2 * np.pi * freq * t_contact + rng.uniform(0, 2 * np.pi))

    # --- pen lifts: split into strokes, with a time gap at each ----------------
    n_lifts = rng.poisson(0.15 + 3.0 * s * s)
    lift_idx = np.sort(rng.integers(FS, max(n_end - FS, FS + 1), size=n_lifts)) if n_lifts else []
    gaps = rng.uniform(0.3, 1.2, size=n_lifts) * (1.0 + s)
    real_t = t_contact.copy()
    for idx, gap in zip(lift_idx, gaps, strict=True):
        real_t[idx:] += gap
    bounds = [0, *[int(i) for i in lift_idx], n_end]

    # --- sample like a pointer device: irregular ~60-125 Hz ---------------------
    rate = rng.uniform(60, 125)
    t0 = rng.uniform(500, 3000)
    strokes = []
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        if b - a < 3:
            continue
        seg_t = real_t[a:b]
        times = [seg_t[0]]
        while times[-1] < seg_t[-1]:
            times.append(times[-1] + max(rng.normal(1.0 / rate, 0.15 / rate), 0.003))
        times = np.array(times[:-1]) if len(times) > 2 else np.array(times)
        sx = np.round(np.interp(times, seg_t, x[a:b]))
        sy = np.round(np.interp(times, seg_t, y[a:b]))
        strokes.append(np.column_stack([sx, sy, times * 1000.0 + t0]))
    return strokes


def make_cohort(
    n_per_class: int = 300, seed: int = 0, canvas=CANVAS
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Feature matrix, labels (0 = typical, 1 = impaired) and latent severity."""
    rng = np.random.default_rng(seed)
    rows, labels, severities = [], [], []
    for label, (mu, sd) in enumerate([(0.15, 0.12), (0.60, 0.20)]):
        for _ in range(n_per_class):
            severity = float(np.clip(rng.normal(mu, sd), 0.0, 1.0))
            try:
                feats = extract_features(generate_drawing(severity, rng, canvas), canvas)
            except ValueError:
                continue  # a degenerate simulated drawing; skip rather than pad
            rows.append([feats[name] for name in FEATURE_NAMES])
            labels.append(label)
            severities.append(severity)
    return np.array(rows), np.array(labels), np.array(severities)
