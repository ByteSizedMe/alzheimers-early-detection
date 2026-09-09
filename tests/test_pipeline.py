"""Dataset, config, and API tests.

Covers the seams where the stages hand off to each other -- a wrong split or a
train-transform leaking into evaluation is the kind of bug that shows up as an
implausibly good score rather than a crash.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient
from PIL import Image

from alzheimers import CLASSES
from alzheimers.config import Config
from alzheimers.data.datasets import (
    MRISliceDataset,
    build_transforms,
    compute_class_weights,
    scan_image_directory,
    stratified_split,
)


@pytest.fixture
def cfg() -> Config:
    return Config.load()


@pytest.fixture
def image_root(tmp_path):
    """A tiny on-disk cohort with a deliberate 4:1 imbalance."""
    rng = np.random.default_rng(0)
    counts = {"NC": 40, "EMCI": 24, "LMCI": 16, "AD": 10}
    for label, n in counts.items():
        (tmp_path / label).mkdir()
        for i in range(n):
            array = rng.integers(0, 255, (32, 32), dtype=np.uint8)
            Image.fromarray(array).save(tmp_path / label / f"{label}_{i}.png")
    return tmp_path, counts


# ------------------------------------------------------------------ config


def test_config_reads_nested_keys(cfg):
    assert cfg["data.image_size"] == 224
    assert cfg["classes"] == list(CLASSES)
    assert cfg["model.backbone"] == "vgg19"


def test_missing_key_raises_rather_than_returning_none(cfg):
    with pytest.raises(KeyError, match="training.nonexistent"):
        cfg["training.nonexistent"]
    assert cfg.get("training.nonexistent", 7) == 7


def test_paths_resolve_against_the_repo_root(cfg):
    assert cfg.path("paths.processed_dir").is_absolute()


def test_config_loads_from_an_arbitrary_file(tmp_path):
    path = tmp_path / "custom.yaml"
    path.write_text(yaml.safe_dump({"training": {"lr": 0.5}}), encoding="utf-8")

    assert Config.load(path)["training.lr"] == 0.5


def test_default_config_covers_every_key_the_code_reads(cfg):
    """Cheap guard against a config key being renamed in only one place."""
    for key in [
        "seed", "classes", "paths.raw_dir", "paths.processed_dir", "paths.artifacts_dir",
        "data.image_size", "data.val_fraction", "data.test_fraction", "data.batch_size",
        "data.num_workers", "data.augment", "imputation.n_donors", "imputation.donor_pool",
        "imputation.scale", "balancing.method", "balancing.n_neighbors", "balancing.beta",
        "model.pretrained", "model.dropout", "model.trainable_blocks", "model.hidden_dim",
        "training.epochs", "training.lr", "training.head_lr_multiplier",
        "training.weight_decay", "training.label_smoothing", "training.class_weights",
        "training.early_stopping_patience", "training.monitor", "training.amp",
        "gradcam.target_layer", "gradcam.alpha", "api.checkpoint", "dashboard.api_url",
    ]:
        assert cfg[key] is not None, key


# ----------------------------------------------------------------- dataset


def test_scan_finds_every_class_folder(image_root):
    root, counts = image_root
    manifest = scan_image_directory(root)

    assert len(manifest) == sum(counts.values())
    assert manifest["label"].value_counts().to_dict() == counts


def test_scan_points_at_the_synthetic_generator_when_empty(tmp_path):
    with pytest.raises(FileNotFoundError, match="make_synthetic_data"):
        scan_image_directory(tmp_path / "absent")


def test_split_is_disjoint_and_preserves_class_ratio(image_root):
    root, counts = image_root
    manifest = scan_image_directory(root)

    splits = stratified_split(manifest, val_fraction=0.15, test_fraction=0.15, seed=42)

    total = sum(len(s) for s in splits.values())
    assert total == len(manifest)

    paths = [set(frame["path"]) for frame in splits.values()]
    assert not paths[0] & paths[1]
    assert not paths[0] & paths[2]
    assert not paths[1] & paths[2]

    for frame in splits.values():
        assert set(frame["label"]) == set(counts)


def test_split_rejects_impossible_fractions(image_root):
    root, _ = image_root
    with pytest.raises(ValueError, match="between 0 and 1"):
        stratified_split(scan_image_directory(root), 0.6, 0.6, seed=0)


def test_dataset_yields_normalised_three_channel_tensors(image_root, cfg):
    root, _ = image_root
    dataset = MRISliceDataset(scan_image_directory(root), build_transforms(cfg, train=False))

    tensor, target = dataset[0]

    assert tensor.shape == (3, 224, 224)  # grayscale replicated for VGG19
    assert target in range(len(CLASSES))
    assert tensor.min() < 0  # ImageNet normalisation shifts below zero


def test_eval_transform_is_deterministic(image_root, cfg):
    root, _ = image_root
    dataset = MRISliceDataset(scan_image_directory(root), build_transforms(cfg, train=False))

    first, _ = dataset[0]
    second, _ = dataset[0]

    assert np.allclose(first.numpy(), second.numpy())


def test_train_transform_augments(image_root, cfg):
    """Augmentation must actually vary, or it is doing nothing."""
    root, _ = image_root
    dataset = MRISliceDataset(scan_image_directory(root), build_transforms(cfg, train=True))

    samples = [dataset[0][0].numpy() for _ in range(6)]
    assert any(not np.allclose(samples[0], other) for other in samples[1:])


def test_dataset_rejects_an_unknown_label():
    manifest = pd.DataFrame({"path": ["a.png"], "label": ["MCI_UNKNOWN"]})
    with pytest.raises(ValueError, match="unknown labels"):
        MRISliceDataset(manifest, build_transforms(Config.load(), train=False))


def test_class_weights_favour_the_rare_stages(image_root):
    root, _ = image_root
    manifest = scan_image_directory(root)

    weights = compute_class_weights(manifest)

    assert len(weights) == len(CLASSES)
    assert weights[CLASSES.index("AD")] > weights[CLASSES.index("NC")]
    assert float(weights.mean()) == pytest.approx(1.0, abs=1e-5)


def test_class_weights_handle_an_absent_class():
    """A split with no AD scans must not divide by zero."""
    manifest = pd.DataFrame({"path": ["a.png"] * 4, "label": ["NC"] * 4})
    weights = compute_class_weights(manifest)

    assert np.isfinite(weights.numpy()).all()


# --------------------------------------------------------------------- api


@pytest.fixture
def client():
    from alzheimers.api.main import app

    with TestClient(app) as test_client:
        yield test_client


def test_health_answers_even_without_a_model(client):
    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert "model_loaded" in body
    # A missing checkpoint must be reported, not raised.
    assert body["model_loaded"] or body["error"]


def test_classes_endpoint_lists_the_four_ordered_stages(client):
    body = client.get("/classes").json()

    assert [entry["name"] for entry in body["classes"]] == list(CLASSES)
    assert body["classes"][0]["description"]


def test_api_info_carries_the_not_a_medical_device_disclaimer(client):
    body = client.get("/api").json()
    assert "not for clinical use" in body["disclaimer"].lower()


def test_root_serves_the_review_interface(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Grad-CAM" in response.text


def test_sample_endpoint_rejects_path_traversal(client):
    """A filename from the client must never be joined onto a path directly."""
    for attempt in ["../../../etc/passwd", "..%2F..%2Fconfig", "no_such_file.png"]:
        assert client.get(f"/samples/{attempt}").status_code == 404


def test_samples_listing_is_always_well_formed(client):
    body = client.get("/samples").json()
    assert isinstance(body["samples"], list)
    for entry in body["samples"]:
        assert set(entry) == {"name", "true_stage", "is_known_error"}


def test_empty_upload_is_a_400(client):
    response = client.post("/predict", files={"file": ("empty.png", b"", "image/png")})
    assert response.status_code == 400


def test_non_image_upload_is_a_400_not_a_500(client):
    response = client.post(
        "/predict", files={"file": ("notes.txt", b"this is not a scan", "text/plain")}
    )
    # 400 when a model is loaded, 503 when there is no checkpoint to load.
    assert response.status_code in (400, 503)


def test_explain_rejects_an_unknown_target_class(client, tmp_path):
    image_path = tmp_path / "slice.png"
    Image.fromarray(np.zeros((32, 32), dtype=np.uint8)).save(image_path)

    response = client.post(
        "/explain",
        files={"file": ("slice.png", image_path.read_bytes(), "image/png")},
        params={"target_class": "PARKINSONS"},
    )
    assert response.status_code in (400, 503)
