"""Phase 2 entry point: fine-tune VGG19 on MRI slices.

    python -m alzheimers.training.train --config configs/default.yaml

Reads the manifests written by ``alzheimers.data.prepare``, trains with
class-weighted cross-entropy, and keeps the checkpoint that scores best on the
validation split (not the last epoch -- these models overfit within a few
epochs on small cohorts).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .. import CLASSES
from ..config import Config, set_seed
from ..data.datasets import build_dataloaders, compute_class_weights
from ..models.vgg19 import build_model, count_parameters, parameter_groups, save_checkpoint
from .metrics import compute_metrics, early_detection_recall, format_confusion_matrix


def resolve_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_splits(processed_dir: Path) -> dict[str, pd.DataFrame]:
    splits = {}
    for name in ("train", "val", "test"):
        path = processed_dir / f"{name}_manifest.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing. Run: python -m alzheimers.data.prepare"
            )
        splits[name] = pd.read_csv(path)
    return splits


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler=None,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """One pass. Trains when ``optimizer`` is given, evaluates otherwise."""
    training = optimizer is not None
    model.train(training)

    total_loss, n_seen = 0.0, 0
    all_targets, all_preds, all_probs = [], [], []

    with torch.set_grad_enabled(training):
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            use_amp = scaler is not None and device.type == "cuda"
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)

            if training:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * images.size(0)
            n_seen += images.size(0)
            all_targets.append(targets.cpu().numpy())
            all_probs.append(torch.softmax(logits.float(), dim=1).detach().cpu().numpy())
            all_preds.append(logits.argmax(dim=1).cpu().numpy())

    return (
        total_loss / max(n_seen, 1),
        np.concatenate(all_targets),
        np.concatenate(all_preds),
        np.concatenate(all_probs),
    )


def train(cfg: Config, device_str: str | None = None, epochs: int | None = None) -> dict:
    set_seed(int(cfg["seed"]))
    device = resolve_device(device_str)
    artifacts = cfg.path("paths.artifacts_dir")
    artifacts.mkdir(parents=True, exist_ok=True)

    splits = load_splits(cfg.path("paths.processed_dir"))
    loaders = build_dataloaders(cfg, splits)
    print(f"[data] train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

    model = build_model(
        num_classes=len(CLASSES),
        pretrained=bool(cfg["model.pretrained"]),
        dropout=float(cfg["model.dropout"]),
        trainable_blocks=int(cfg["model.trainable_blocks"]),
        hidden_dim=int(cfg["model.hidden_dim"]),
    ).to(device)

    trainable, total = count_parameters(model)
    print(f"[model] VGG19 on {device} -- {trainable:,} trainable / {total:,} total")

    weights = None
    if cfg["training.class_weights"] == "balanced":
        weights = compute_class_weights(splits["train"]).to(device)
        print(f"[model] class weights {dict(zip(CLASSES, weights.cpu().numpy().round(3)))}")

    criterion = nn.CrossEntropyLoss(
        weight=weights, label_smoothing=float(cfg["training.label_smoothing"])
    )
    optimizer = torch.optim.AdamW(
        parameter_groups(
            model, float(cfg["training.lr"]), float(cfg["training.head_lr_multiplier"])
        ),
        weight_decay=float(cfg["training.weight_decay"]),
    )

    n_epochs = epochs if epochs is not None else int(cfg["training.epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(n_epochs, 1))
    scaler = (
        torch.amp.GradScaler(device.type)
        if bool(cfg["training.amp"]) and device.type == "cuda"
        else None
    )

    monitor = str(cfg["training.monitor"]).replace("val_", "")
    patience = int(cfg["training.early_stopping_patience"])
    best_score, best_epoch, history = -float("inf"), -1, []
    checkpoint_path = artifacts / "best_model.pt"

    for epoch in range(1, n_epochs + 1):
        started = time.time()
        train_loss, y_true, y_pred, _ = run_epoch(
            model, loaders["train"], criterion, device, optimizer, scaler
        )
        train_metrics = compute_metrics(y_true, y_pred, None, CLASSES)

        val_loss, v_true, v_pred, v_prob = run_epoch(model, loaders["val"], criterion, device)
        val_metrics = compute_metrics(v_true, v_pred, v_prob, CLASSES)
        scheduler.step()

        score = val_metrics.get(monitor, val_metrics["macro_f1"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_metrics["accuracy"], 4),
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_metrics["accuracy"], 4),
                "val_macro_f1": round(val_metrics["macro_f1"], 4),
                "lr": round(optimizer.param_groups[0]["lr"], 8),
            }
        )

        marker = ""
        if score > best_score:
            best_score, best_epoch = score, epoch
            save_checkpoint(
                checkpoint_path,
                model,
                {
                    "classes": list(CLASSES),
                    "dropout": float(cfg["model.dropout"]),
                    "trainable_blocks": int(cfg["model.trainable_blocks"]),
                    "hidden_dim": int(cfg["model.hidden_dim"]),
                    "image_size": int(cfg["data.image_size"]),
                    "epoch": epoch,
                    "val_score": round(score, 4),
                    "monitor": monitor,
                    "config": str(cfg.source),
                },
            )
            marker = "  <- best"

        print(
            f"epoch {epoch:>3}/{n_epochs}  "
            f"train_loss {train_loss:.4f}  val_loss {val_loss:.4f}  "
            f"val_acc {val_metrics['accuracy']:.4f}  val_f1 {val_metrics['macro_f1']:.4f}  "
            f"({time.time() - started:.1f}s){marker}"
        )

        if epoch - best_epoch >= patience:
            print(f"[train] no {monitor} gain for {patience} epochs -- stopping early")
            break

    history_path = artifacts / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"\n[train] best epoch {best_epoch} ({monitor}={best_score:.4f}) -> {checkpoint_path}")

    # Final read of the untouched test split, from the best checkpoint.
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    _, t_true, t_pred, t_prob = run_epoch(model, loaders["test"], criterion, device)
    test_metrics = compute_metrics(t_true, t_pred, t_prob, CLASSES)
    test_metrics["early_detection_recall"] = early_detection_recall(t_true, t_pred, CLASSES)

    print("\n=== test split ===")
    print(f"accuracy {test_metrics['accuracy']:.4f}   macro F1 {test_metrics['macro_f1']:.4f}")
    print(f"impaired-case recall {test_metrics['early_detection_recall']:.4f}")
    print(format_confusion_matrix(test_metrics["confusion_matrix"], CLASSES))

    (artifacts / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2), encoding="utf-8"
    )
    return test_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the MRI stage classifier")
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default=None, help="cuda, cpu (default: auto)")
    parser.add_argument("--epochs", type=int, default=None, help="override config epochs")
    args = parser.parse_args()

    train(Config.load(args.config), device_str=args.device, epochs=args.epochs)


if __name__ == "__main__":
    main()
