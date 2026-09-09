"""Model, Grad-CAM, and metric tests.

Everything here builds VGG19 with ``pretrained=False`` -- the architecture is
identical and skipping the 548 MB ImageNet download keeps the suite fast and
runnable offline.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from alzheimers import CLASSES
from alzheimers.models.gradcam import GradCAM, cam_focus_summary, denormalize, overlay_cam
from alzheimers.models.vgg19 import (
    build_model,
    count_parameters,
    load_checkpoint,
    parameter_groups,
    save_checkpoint,
)
from alzheimers.training.metrics import (
    compute_metrics,
    early_detection_recall,
    format_confusion_matrix,
)


@pytest.fixture(scope="module")
def model():
    return build_model(pretrained=False, trainable_blocks=1)


@pytest.fixture
def batch():
    torch.manual_seed(0)
    return torch.randn(2, 3, 224, 224)


# ------------------------------------------------------------------- model


def test_output_shape_matches_class_count(model, batch):
    with torch.no_grad():
        logits = model(batch)
    assert logits.shape == (2, len(CLASSES))


def test_only_the_last_block_and_head_are_trainable(model):
    trainable, total = count_parameters(model)

    assert 0 < trainable < total
    # The frozen early blocks are the bulk of VGG19's conv weights.
    assert trainable / total < 0.75

    frozen_early = [
        p.requires_grad for i, layer in enumerate(model.features) if i < 20
        for p in layer.parameters()
    ]
    assert not any(frozen_early)


def test_trainable_blocks_zero_freezes_the_whole_backbone():
    frozen = build_model(pretrained=False, trainable_blocks=0)
    assert not any(p.requires_grad for p in frozen.features.parameters())
    assert all(p.requires_grad for p in frozen.classifier.parameters())


def test_rejects_out_of_range_trainable_blocks():
    with pytest.raises(ValueError, match="trainable_blocks"):
        build_model(pretrained=False, trainable_blocks=9)


def test_head_gets_a_higher_learning_rate(model):
    groups = parameter_groups(model, lr=1e-4, head_lr_multiplier=10.0)
    assert groups[0]["lr"] == pytest.approx(1e-3)
    assert groups[1]["lr"] == pytest.approx(1e-4)


def test_checkpoint_round_trip(tmp_path, model, batch):
    path = tmp_path / "model.pt"
    metadata = {"classes": list(CLASSES), "trainable_blocks": 1, "hidden_dim": 512, "epoch": 3}
    save_checkpoint(path, model, metadata)

    restored, restored_meta = load_checkpoint(path)

    assert restored_meta["epoch"] == 3
    model.eval()
    with torch.no_grad():
        np.testing.assert_allclose(
            model(batch).numpy(), restored(batch).numpy(), rtol=1e-5, atol=1e-5
        )


def test_missing_checkpoint_names_the_fix(tmp_path):
    with pytest.raises(FileNotFoundError, match="alzheimers.training.train"):
        load_checkpoint(tmp_path / "nope.pt")


# ---------------------------------------------------------------- grad-cam


def test_gradcam_returns_a_normalised_map_at_input_size(model, batch):
    with GradCAM(model) as explainer:
        cam, class_index, probabilities = explainer(batch[:1])

    assert cam.shape == (224, 224)
    assert cam.min() >= 0.0 and cam.max() <= 1.0
    assert 0 <= class_index < len(CLASSES)
    assert probabilities.shape == (len(CLASSES),)
    assert probabilities.sum() == pytest.approx(1.0, abs=1e-5)


def test_gradcam_explains_the_class_you_ask_for(model, batch):
    with GradCAM(model) as explainer:
        _, class_index, _ = explainer(batch[:1], class_index=3)
    assert class_index == 3


def test_gradcam_accepts_an_unbatched_tensor(model, batch):
    with GradCAM(model) as explainer:
        cam, _, _ = explainer(batch[0])
    assert cam.shape == (224, 224)


def test_gradcam_rejects_a_real_batch(model, batch):
    with GradCAM(model) as explainer:
        with pytest.raises(ValueError, match="one image at a time"):
            explainer(batch)


def test_gradcam_rejects_an_unknown_layer(model):
    with pytest.raises((AttributeError, IndexError)):
        GradCAM(model, "features.999")


def test_gradcam_leaves_the_model_in_its_original_mode(model, batch):
    model.train()
    with GradCAM(model) as explainer:
        explainer(batch[:1])
    assert model.training
    model.eval()


def test_release_removes_the_hooks(model, batch):
    explainer = GradCAM(model)
    explainer(batch[:1])
    explainer.release()

    before = explainer.activations
    with torch.no_grad():
        model(batch)
    assert explainer.activations is before  # hook no longer firing


# ------------------------------------------------------------- cam helpers


def test_denormalize_returns_uint8_rgb(batch):
    image = denormalize(batch[:1])
    assert image.shape == (224, 224, 3)
    assert image.dtype == np.uint8


def test_overlay_resizes_a_mismatched_cam():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    cam = np.linspace(0, 1, 32 * 32).reshape(32, 32).astype(np.float32)

    blended = overlay_cam(image, cam)

    assert blended.shape == (64, 64, 3)
    assert blended.dtype == np.uint8


def test_focus_summary_locates_a_corner_activation():
    cam = np.zeros((100, 100), dtype=np.float32)
    cam[:20, :20] = 1.0

    summary = cam_focus_summary(cam)

    assert summary["dominant_quadrant"] == "upper_left"
    assert summary["central_share"] < 0.1
    assert summary["quadrant_share"]["upper_left"] == pytest.approx(1.0)


def test_focus_summary_handles_a_dead_map():
    summary = cam_focus_summary(np.zeros((32, 32), dtype=np.float32))
    assert summary["dominant_quadrant"] is None
    assert summary["central_share"] == 0.0


# ----------------------------------------------------------------- metrics


def test_metrics_on_a_perfect_prediction():
    y = np.array([0, 1, 2, 3, 0, 1])
    probabilities = np.eye(4)[y]

    metrics = compute_metrics(y, y, probabilities, CLASSES)

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["macro_auc"] == 1.0


def test_macro_f1_punishes_the_majority_only_predictor():
    """Accuracy rewards always guessing NC; macro F1 must not."""
    y_true = np.array([0] * 90 + [1, 2, 3] * 3 + [0])
    y_pred = np.zeros_like(y_true)

    metrics = compute_metrics(y_true, y_pred, None, CLASSES)

    assert metrics["accuracy"] > 0.9
    assert metrics["macro_f1"] < 0.3
    assert metrics["balanced_accuracy"] < 0.3


def test_early_detection_recall_counts_missed_impairment():
    classes = ("NC", "EMCI", "LMCI", "AD")
    y_true = np.array([1, 2, 3, 1])       # all impaired
    y_pred = np.array([0, 2, 3, 0])       # two sent home as normal

    assert early_detection_recall(y_true, y_pred, classes) == pytest.approx(0.5)


def test_early_detection_recall_is_nan_without_impaired_cases():
    result = early_detection_recall(np.array([0, 0]), np.array([0, 0]), CLASSES)
    assert np.isnan(result)


def test_confusion_matrix_formats_all_classes():
    y = np.array([0, 1, 2, 3])
    text = format_confusion_matrix(compute_metrics(y, y, None, CLASSES)["confusion_matrix"], CLASSES)

    for name in CLASSES:
        assert name in text
    assert "predicted" in text

def test_configured_target_layer_is_post_activation(model, batch):
    """The default target must be a ReLU output, not a raw conv output.

    Grad-CAM weights post-activation maps. Pointing it at the conv itself feeds
    in large negative values, the channel-weighted sum goes negative across the
    whole map, and the final ReLU flattens every CAM to zero -- a silent failure
    that produces blank heatmaps rather than an error.
    """
    from alzheimers.config import Config

    target = str(Config.load()["gradcam.target_layer"])
    with GradCAM(model, target) as explainer:
        explainer(batch[:1])
        assert explainer.activations.min() >= 0.0, f"{target} is not post-activation"


def test_gradcam_on_the_raw_conv_layer_is_the_degenerate_case(model, batch):
    """Pins the failure the default guards against, so a future retarget of the
    config back to features.34 is caught by the test above rather than in QA."""
    with GradCAM(model, "features.34") as explainer:
        explainer(batch[:1])
        assert explainer.activations.min() < 0.0
