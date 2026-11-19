
import argparse
import json
import random
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.cached_audio_dataset import CachedAudioDataset
from src.audio_validate_checkpoint import AEGISAudioTransformer


ROOT = Path(__file__).resolve().parents[1]

def atomic_torch_save(obj, destination):
    """Save a checkpoint through a temporary file, then replace the target."""
    destination = Path(destination)
    temp_path = destination.with_name(destination.name + ".tmp")

    try:
        with temp_path.open("wb") as f:
            torch.save(obj, f)
            f.flush()
            os.fsync(f.fileno())

        os.replace(temp_path, destination)
    finally:
        if temp_path.exists():
            temp_path.unlink()



def load_config():
    path = ROOT / "configs/audio_transformer_reproducible.json"
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_inputs(config):
    dataset_cfg = config["dataset"]
    index_path = resolve_path(dataset_cfg["index_path"])
    cache_path = resolve_path(dataset_cfg["cache_path"])
    spec_path = resolve_path(dataset_cfg["cache_spec_path"])

    for path in (index_path, cache_path, spec_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required input is missing: {path}")

    index = pd.read_csv(index_path)
    required = {
        "window_id", "sample_id", "label", "split"
    }
    missing = required - set(index.columns)
    if missing:
        raise ValueError(f"Index is missing columns: {sorted(missing)}")

    if index["window_id"].duplicated().any():
        raise ValueError("Window IDs are not unique.")

    if set(index["label"].dropna().unique()) != {"bonafide", "spoof"}:
        raise ValueError("Unexpected labels in the window index.")

    if set(index["split"].dropna().unique()) != {
        dataset_cfg["training_split"],
        dataset_cfg["validation_split"],
    }:
        raise ValueError("Unexpected dataset splits.")

    with spec_path.open("r", encoding="utf-8-sig") as f:
        spec = json.load(f)

    expected_shape = dataset_cfg["feature_shape"]
    if spec.get("shape_per_window") != expected_shape:
        raise ValueError("Cache feature shape does not match configuration.")
    if spec.get("dtype") != dataset_cfg["cache_dtype"]:
        raise ValueError("Cache dtype does not match configuration.")
    if spec.get("num_windows") != len(index):
        raise ValueError("Cache window count does not match index.")
    if spec.get("source_window_index") != index_path.name:
        raise ValueError("Cache source index name does not match.")

    expected_bytes = (
        len(index) * int(np.prod(expected_shape)) * 2
    )
    actual_bytes = cache_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"Cache size mismatch: expected {expected_bytes:,} bytes, "
            f"found {actual_bytes:,} bytes."
        )

    train_index = index[
        index["split"] == dataset_cfg["training_split"]
    ].reset_index(drop=True)
    dev_index = index[
        index["split"] == dataset_cfg["validation_split"]
    ].reset_index(drop=True)

    if train_index.empty or dev_index.empty:
        raise ValueError("Training or validation split is empty.")

    return index, train_index, dev_index, cache_path


def make_model(config, device):
    model_cfg = config["model"]
    model = AEGISAudioTransformer(
        in_channels=model_cfg["in_channels"],
        embed_dim=model_cfg["embed_dim"],
        patch_size=tuple(model_cfg["patch_size"]),
        num_heads=model_cfg["num_heads"],
        num_layers=model_cfg["num_layers"],
        num_classes=model_cfg["num_classes"],
        dropout=model_cfg["dropout"],
    )
    return model.to(device)


def make_loaders(config, full_index, train_index, dev_index, cache_path):
    shape = tuple(config["dataset"]["feature_shape"])

    train_ds = CachedAudioDataset(
        window_index=train_index,
        cache_path=cache_path,
        full_window_index=full_index,
        target_shape=shape,
    )
    dev_ds = CachedAudioDataset(
        window_index=dev_index,
        cache_path=cache_path,
        full_window_index=full_index,
        target_shape=shape,
    )

    batch_size = config["training"]["batch_size"]
    workers = config["training"].get("num_workers", 0)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=False,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=False,
    )
    return train_loader, dev_loader


def run_one_batch_smoke_test(config, train_loader, device):
    model = make_model(config, device)
    model.train()

    training_cfg = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
    )
    weights = torch.tensor(
        training_cfg["class_weights"], dtype=torch.float32, device=device
    )
    criterion = nn.CrossEntropyLoss(weight=weights)

    batch = next(iter(train_loader))
    features = batch["features"].to(device)
    labels = batch["label"].to(device)

    if features.ndim != 4 or not torch.isfinite(features).all():
        raise ValueError("Smoke-test features have an invalid shape or values.")
    if labels.ndim != 1 or labels.numel() != features.shape[0]:
        raise ValueError("Smoke-test labels have an invalid shape.")

    optimizer.zero_grad(set_to_none=True)
    logits = model(features)
    loss = criterion(logits, labels)

    if logits.shape != (labels.shape[0], 2):
        raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")
    if not torch.isfinite(loss):
        raise ValueError("Smoke-test loss is not finite.")

    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        training_cfg["gradient_clip_max_norm"],
    )
    if not torch.isfinite(torch.as_tensor(grad_norm)):
        raise ValueError("Smoke-test gradient norm is not finite.")

    optimizer.step()

    print("Smoke-test forward pass: PASS")
    print("Smoke-test loss:", float(loss.detach()))
    print("Smoke-test backward pass: PASS")
    print("Smoke-test optimizer update: PASS")
    print("Batch size:", labels.shape[0])
    print("Device:", device)
    print("Checkpoint saved: NO")
    print("RESULT: smoke test passed; no experiment outputs were written.")


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    tp = fp = tn = fn = 0

    for batch in loader:
        features = batch["features"].to(device)
        labels = batch["label"].to(device)
        logits = model(features)
        loss = criterion(logits, labels)

        predictions = logits.argmax(dim=1)
        count = labels.numel()
        total_loss += float(loss) * count
        total += count
        correct += int((predictions == labels).sum())

        tp += int(((predictions == 1) & (labels == 1)).sum())
        fp += int(((predictions == 1) & (labels == 0)).sum())
        tn += int(((predictions == 0) & (labels == 0)).sum())
        fn += int(((predictions == 0) & (labels == 1)).sum())

    if total == 0:
        raise ValueError("Validation loader produced no examples.")

    recall_spoof = tp / max(tp + fn, 1)
    recall_bonafide = tn / max(tn + fp, 1)
    precision_spoof = tp / max(tp + fp, 1)
    f1_spoof = (
        2 * precision_spoof * recall_spoof
        / max(precision_spoof + recall_spoof, 1e-12)
    )

    return {
        "val_loss": total_loss / total,
        "val_accuracy": correct / total,
        "val_balanced_accuracy": (
            recall_spoof + recall_bonafide
        ) / 2,
        "val_spoof_precision": precision_spoof,
        "val_spoof_recall": recall_spoof,
        "val_spoof_f1": f1_spoof,
        "val_confusion_matrix": [[tn, fp], [fn, tp]],
        "val_examples": total,
    }


def run_training(config, train_loader, dev_loader, device):
    safety = config["safety"]
    if safety.get("overwrite_existing") is not False:
        raise ValueError("Safe runner requires overwrite_existing=false.")

    model_dir = resolve_path(safety["output_directory"])
    results_dir = resolve_path("results/audio_transformer_reproducible_v1")

    # Refuse to use existing directories: never overwrite prior experiment data.
    if model_dir.exists() or results_dir.exists():
        raise FileExistsError(
            "Training stopped because an output directory already exists. "
            "Inspect it and choose a new experiment directory before retrying."
        )

    training_cfg = config["training"]
    model = make_model(config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
    )

    # Create experiment directories only after model/optimizer initialization.
    model_dir.mkdir(parents=True, exist_ok=False)
    try:
        results_dir.mkdir(parents=True, exist_ok=False)
    except Exception:
        # Roll back only the newly created model directory if it is empty.
        try:
            model_dir.rmdir()
        except OSError:
            pass
        raise

    scheduler_cfg = training_cfg["scheduler"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=scheduler_cfg["mode"],
        factor=scheduler_cfg["factor"],
        patience=scheduler_cfg["patience"],
    )
    weights = torch.tensor(
        training_cfg["class_weights"], dtype=torch.float32, device=device
    )
    criterion = nn.CrossEntropyLoss(weight=weights)

    history = []
    best_val_loss = float("inf")
    model_cfg = dict(config["model"])
    model_cfg["patch_size"] = tuple(model_cfg["patch_size"])

    with (results_dir / "run_config.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(config, f, indent=2)

    for epoch in range(1, training_cfg["epochs"] + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for batch in train_loader:
            features = batch["features"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite training loss at epoch {epoch}."
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                training_cfg["gradient_clip_max_norm"],
            )
            optimizer.step()

            count = labels.numel()
            running_loss += float(loss.detach()) * count
            seen += count

        metrics = evaluate(model, dev_loader, criterion, device)
        scheduler.step(metrics["val_loss"])

        record = {
            "epoch": epoch,
            "train_loss": running_loss / max(seen, 1),
            "learning_rate": optimizer.param_groups[0]["lr"],
            **metrics,
        }
        history.append(record)

        print(
            f"Epoch {epoch}/{training_cfg['epochs']} | "
            f"train_loss={record['train_loss']:.6f} | "
            f"val_loss={metrics['val_loss']:.6f} | "
            f"val_balanced_accuracy="
            f"{metrics['val_balanced_accuracy']:.4f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": metrics["val_loss"],
            "val_balanced_accuracy": metrics["val_balanced_accuracy"],
            "val_spoof_recall": metrics["val_spoof_recall"],
            "val_spoof_f1": metrics["val_spoof_f1"],
            "model_config": model_cfg,
            "experiment_config": config,
            "seed": config["seed"],
        }

        # Each output path belongs exclusively to this new experiment.
        atomic_torch_save(
            checkpoint, model_dir / "last_audio_transformer.pt"
        )
        if metrics["val_loss"] < best_val_loss:
            best_val_loss = metrics["val_loss"]
            atomic_torch_save(
                checkpoint, model_dir / "best_audio_transformer.pt"
            )

        pd.DataFrame(history).to_csv(
            results_dir / "training_history.csv", index=False
        )

    with (results_dir / "training_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(history[-1], f, indent=2)

    print("Training complete.")
    print("New checkpoints:", model_dir)
    print("New results:", results_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Safe, reproducible AEGIS audio-transformer runner."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--smoke-test", action="store_true",
        help="Run one training batch without saving a checkpoint."
    )
    mode.add_argument(
        "--train", action="store_true",
        help="Explicitly start the configured full training run."
    )
    args = parser.parse_args()

    config = load_config()
    set_seed(config["seed"])
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    full_index, train_index, dev_index, cache_path = validate_inputs(config)
    train_loader, dev_loader = make_loaders(
        config, full_index, train_index, dev_index, cache_path
    )

    print("Experiment:", config["experiment_name"])
    print("Device:", device)
    print("Training windows:", len(train_index))
    print("Validation windows:", len(dev_index))
    print("Existing checkpoint will not be modified.")

    if args.smoke_test:
        run_one_batch_smoke_test(config, train_loader, device)
        return

    if not args.train:
        raise RuntimeError("No execution mode selected.")

    if config["safety"].get("run_training_automatically") is not False:
        raise ValueError("Expected automatic training to remain disabled.")

    run_training(config, train_loader, dev_loader, device)


if __name__ == "__main__":
    main()
