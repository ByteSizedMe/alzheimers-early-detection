"""Handwriting pipeline: feature extraction, simulator, model, and API."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from alzheimers.api.main import app
from alzheimers.handwriting.features import FEATURE_NAMES, extract_features
from alzheimers.handwriting.model import HandwritingAnalyzer, train
from alzheimers.handwriting.synthetic import CANVAS, DEMO_SEVERITY, generate_drawing, make_cohort


def _spiral(tremor_px: float = 0.0, speed_mod: float = 0.0, seed: int = 0):
    """A clean spiral at constant speed, optionally with tremor / speed modulation."""
    rng = np.random.default_rng(seed)
    n = 800
    t = np.arange(n) * 12.0  # 12 ms apart
    u = np.arange(n) / n
    theta = u * 6 * np.pi
    r = 180 * u
    x = 360 + r * np.cos(theta) + tremor_px * np.sin(2 * np.pi * 7 * t / 1000)
    y = 260 + r * np.sin(theta) + tremor_px * np.sin(2 * np.pi * 7 * t / 1000 + 1)
    if speed_mod:
        t = t + speed_mod * 400 * np.sin(2 * np.pi * u * 9) + rng.normal(0, 0, n)
        t = np.maximum.accumulate(t + np.arange(n) * 0.01)
    return [np.column_stack([x, y, t])]


def test_feature_vector_is_complete_and_finite():
    feats = extract_features(_spiral(), CANVAS)
    assert set(feats) == set(FEATURE_NAMES)
    assert all(np.isfinite(v) for v in feats.values())


def test_tremor_amplitude_tracks_injected_tremor():
    clean = extract_features(_spiral(0.0), CANVAS)["tremor_amp"]
    shaky = extract_features(_spiral(2.0), CANVAS)["tremor_amp"]
    assert shaky > 3 * clean


def test_uneven_pace_raises_speed_variability():
    steady = extract_features(_spiral(), CANVAS)["speed_cv"]
    uneven = extract_features(_spiral(speed_mod=1.0), CANVAS)["speed_cv"]
    assert uneven > steady


def test_features_do_not_depend_on_canvas_scale():
    base = _spiral()
    big = [s * np.array([2.0, 2.0, 1.0]) for s in base]
    a = extract_features(base, CANVAS)
    b = extract_features(big, (CANVAS[0] * 2, CANVAS[1] * 2))
    for name in ("mean_speed", "speed_cv", "size_norm", "spiral_irregularity"):
        assert b[name] == pytest.approx(a[name], rel=0.05)


def test_pen_lift_time_is_measured():
    stroke = _spiral()[0]
    half = len(stroke) // 2
    first, second = stroke[:half].copy(), stroke[half:].copy()
    second[:, 2] += 2000.0  # a two-second lift in the middle
    lifted = extract_features([first, second], CANVAS)
    assert lifted["in_air_ratio"] > 0.15
    assert extract_features(_spiral(), CANVAS)["in_air_ratio"] == 0.0


@pytest.mark.parametrize("strokes", [[], [[(1, 1, 0)]], [[(1, 1, 0), (2, 2, 10)]]])
def test_too_little_data_is_rejected(strokes):
    with pytest.raises(ValueError):
        extract_features(strokes, CANVAS)


def test_simulator_severity_moves_features_in_the_documented_direction():
    rng = np.random.default_rng(0)
    low = [extract_features(generate_drawing(0.05, rng), CANVAS) for _ in range(15)]
    high = [extract_features(generate_drawing(0.9, rng), CANVAS) for _ in range(15)]
    mean = lambda rows, k: float(np.mean([r[k] for r in rows]))  # noqa: E731
    for name in ("duration_s", "speed_cv", "speed_peaks", "tremor_amp", "spiral_irregularity"):
        assert mean(high, name) > mean(low, name), name
    assert mean(high, "mean_speed") < mean(low, "mean_speed")
    assert mean(high, "size_norm") < mean(low, "size_norm")


def test_cohort_is_not_trivially_separable():
    """Overlapping severities must keep cross-validated accuracy below perfect.

    A perfect score on the simulator would mean the groups are disjoint by
    construction, which would make the reported figure meaningless.
    """
    metrics = train(n_per_class=120, seed=3)["metrics"]
    assert 0.80 < metrics["cv_accuracy"] < 0.99
    assert metrics["cv_auc"] > 0.9


def test_cohort_shape():
    X, y, s = make_cohort(20, seed=1)
    assert X.shape[1] == len(FEATURE_NAMES)
    assert len(X) == len(y) == len(s)
    assert set(y) == {0, 1}


@pytest.fixture(scope="module")
def analyzer(tmp_path_factory):
    # A throwaway path, so the test never reads or writes the real artifact.
    return HandwritingAnalyzer(tmp_path_factory.mktemp("hw") / "model.joblib")


def test_demo_profiles_order_correctly(analyzer):
    def median_p(profile):
        return float(
            np.median(
                [
                    analyzer.analyze(generate_drawing(DEMO_SEVERITY[profile], np.random.default_rng(k)))[
                        "probability"
                    ]
                    for k in range(10)
                ]
            )
        )

    assert median_p("typical") < median_p("borderline") < median_p("elevated")


def test_contributions_explain_the_logit(analyzer):
    """Contributions plus the intercept must reproduce the model's own logit."""
    result = analyzer.analyze(generate_drawing(0.8, np.random.default_rng(5)))
    clf = analyzer.pipeline.named_steps["logisticregression"]
    logit = sum(f["contribution"] for f in result["features"]) + float(clf.intercept_[0])
    p = 1 / (1 + np.exp(-logit))
    assert p == pytest.approx(result["probability"], abs=0.01)


client = TestClient(app)


def _payload(profile: str, seed: int = 0) -> dict:
    d = client.get(f"/handwriting/demo/{profile}", params={"seed": seed}).json()
    return {k: d[k] for k in ("strokes", "canvas_width", "canvas_height")}


def test_api_round_trip():
    low = client.post("/handwriting/analyze", json=_payload("typical")).json()
    high = client.post("/handwriting/analyze", json=_payload("elevated")).json()
    assert low["tier"] == "Low" and high["tier"] == "High"
    assert high["probability"] > low["probability"]
    assert low["trace"] and low["features"]


def test_api_rejects_unusable_drawing():
    resp = client.post(
        "/handwriting/analyze",
        json={"strokes": [[{"x": 1, "y": 1, "t": 0}]], "canvas_width": 100, "canvas_height": 100},
    )
    assert resp.status_code == 422


def test_api_rejects_oversized_trace():
    stroke = [{"x": float(i % 50), "y": 1.0, "t": float(i)} for i in range(31_000)]
    resp = client.post(
        "/handwriting/analyze",
        json={"strokes": [stroke], "canvas_width": 100, "canvas_height": 100},
    )
    assert resp.status_code == 413


def test_api_unknown_demo_profile():
    assert client.get("/handwriting/demo/nope").status_code == 404
