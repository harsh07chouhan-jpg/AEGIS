import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from src.cached_audio_dataset import load_cached_audio_splits


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "audio_transformer_colab_v1.json"


class AEGISAudioTransformer(nn.Module):
    """Exact baseline Audio Spectrogram Transformer architecture (875,906 parameters).

    Defined directly here so Colab execution does not depend on importing
    `src/audio_validate_checkpoint.py` (which contains a hardcoded Windows path).
    """

    def __init__(
        self,
        in_channels=2,
        embed_dim=128,
        patch_size=(16, 16),
        num_heads=4,
        num_layers=4,
        num_classes=2,
        dropout=0.1,
    ):
        super().__init__()
        patch_size = tuple(patch_size)
        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        freq_patches = 128 // patch_size[0]
        time_patches = 257 // patch_size[1]
        num_patches = freq_patches * time_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embedding[:, : x.size(1)]
        x = self.transformer(x)
        x = self.norm(x[:, 0])
        return self.classifier(x)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path):
    if not config_path.exists():
        raise FileNotFoundError(f"Missing configuration file: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("safety", {}).get("overwrite_existing") is not False:
        raise ValueError("Safety setting overwrite_existing must remain false.")
    return config


def build_model(model_cfg):
    if model_cfg.get("architecture") != "AEGISAudioTransformer":
        raise ValueError(f"Unsupported architecture: {model_cfg.get('architecture')}")
    return AEGISAudioTransformer(
        in_channels=int(model_cfg["in_channels"]),
        embed_dim=int(model_cfg["embed_dim"]),
        patch_size=tuple(model_cfg["patch_size"]),
        num_heads=int(model_cfg["num_heads"]),
        num_layers=int(model_cfg["num_layers"]),
        num_classes=int(model_cfg["num_classes"]),
        dropout=float(model_cfg["dropout"]),
    )


def verify_split_integrity(config, train_dataset, val_dataset, full_index):
    data_cfg = config["data"]
    if len(full_index) != int(data_cfg["expected_total_windows"]):
        raise ValueError(
            f"Unexpected total windows: {len(full_index)} != "
            f"{data_cfg['expected_total_windows']}"
        )
    if len(train_dataset) != int(data_cfg["expected_train_windows"]):
        raise ValueError(
            f"Unexpected train windows: {len(train_dataset)} != "
            f"{data_cfg['expected_train_windows']}"
        )
    if len(val_dataset) != int(data_cfg["expected_val_windows"]):
        raise ValueError(
            f"Unexpected validation windows: {len(val_dataset)} != "
            f"{data_cfg['expected_val_windows']}"
        )

    train_positions = set(train_dataset.positions.tolist())
    val_positions = set(val_dataset.positions.tolist())
    if train_positions & val_positions:
        raise AssertionError("Train and validation cache positions overlap.")
    if len(train_positions) + len(val_positions) != len(full_index):
        raise AssertionError(
            "Train and validation splits do not cover the full index."
        )

    if int(val_dataset.positions.min()) != int(data_cfg["expected_train_windows"]):
        raise AssertionError("Validation cache positions do not start at 55,630.")
    if int(val_dataset.positions.max()) != int(data_cfg["expected_total_windows"]) - 1:
        raise AssertionError("Validation cache positions do not end at 111,235.")


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    y_true = []
    y_pred = []

    with torch.inference_mode():
        for features, labels, _ in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if not torch.isfinite(features).all():
                raise ValueError("Non-finite validation features encountered.")

            logits = model(features)
            if not torch.isfinite(logits).all():
                raise ValueError("Non-finite validation logits encountered.")

            loss = criterion(logits, labels)
            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_examples += int(batch_size)

            preds = logits.argmax(dim=1)
            y_true.extend(labels.cpu().numpy().astype(int).tolist())
            y_pred.extend(preds.cpu().numpy().astype(int).tolist())

    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_pred_np = np.asarray(y_pred, dtype=np.int64)
    cm = confusion_matrix(y_true_np, y_pred_np, labels=[0, 1])

    return {
        "val_loss": float(total_loss / max(total_examples, 1)),
        "val_accuracy": float(accuracy_score(y_true_np, y_pred_np)),
        "val_balanced_accuracy": float(balanced_accuracy_score(y_true_np, y_pred_np)),
        "val_spoof_precision": float(
            precision_score(y_true_np, y_pred_np, pos_label=1, zero_division=0)
        ),
        "val_spoof_recall": float(
            recall_score(y_true_np, y_pred_np, pos_label=1, zero_division=0)
        ),
        "val_spoof_f1": float(
            f1_score(y_true_np, y_pred_np, pos_label=1, zero_division=0)
        ),
        "val_confusion_matrix": cm.tolist(),
        "val_examples": int(total_examples),
    }


def train_one_epoch(model, loader, optimizer, criterion, device, max_norm, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    total_examples = 0

    for batch_idx, (features, labels, _) in enumerate(loader, start=1):
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if not torch.isfinite(features).all():
            raise ValueError(f"Non-finite training features in batch {batch_idx}.")

        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        if not torch.isfinite(logits).all():
            raise ValueError(f"Non-finite training logits in batch {batch_idx}.")

        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite training loss in batch {batch_idx}.")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_examples += int(batch_size)

        if batch_idx % 500 == 0 or batch_idx == len(loader):
            print(
                f"Epoch {epoch}/{total_epochs} | "
                f"batch {batch_idx}/{len(loader)} | "
                f"running_loss={total_loss / max(total_examples, 1):.6f}"
            )

    return float(total_loss / max(total_examples, 1))


def resolve_paths(config, cache_dir_override=None, index_csv_override=None):
    index_path = (
        Path(index_csv_override).resolve()
        if index_csv_override
        else ROOT / config["data"]["window_index_csv"]
    )
    cache_dir = (
        Path(cache_dir_override).resolve()
        if cache_dir_override
        else ROOT / config["data"]["feature_cache_dir"]
    )
    model_dir = ROOT / config["outputs"]["model_dir"]
    results_dir = ROOT / config["outputs"]["results_dir"]
    return index_path, cache_dir, model_dir, results_dir


def run(config, smoke_test=False, cache_dir_override=None, index_csv_override=None):
    seed = int(config["seed"])
    set_seed(seed)

    index_path, cache_dir, model_dir, results_dir = resolve_paths(
        config,
        cache_dir_override=cache_dir_override,
        index_csv_override=index_csv_override,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_pin_memory = device.type == "cuda"
    print(f"Device: {device} (pin_memory={use_pin_memory})")
    print(f"Window index CSV: {index_path}")
    print(f"Feature cache dir: {cache_dir}")

    train_dataset, val_dataset, full_index, _ = load_cached_audio_splits(
        index_path=index_path,
        cache_dir=cache_dir,
        train_split=config["data"]["train_split"],
        val_split=config["data"]["val_split"],
    )
    verify_split_integrity(config, train_dataset, val_dataset, full_index)

    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["training"]["num_workers"])
    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    model = build_model(config["model"]).to(device)
    class_weights = torch.tensor(
        config["training"]["class_weights"],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler_cfg = config["training"]["scheduler"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=scheduler_cfg["mode"],
        factor=float(scheduler_cfg["factor"]),
        patience=int(scheduler_cfg["patience"]),
    )

    print(f"Train windows: {len(train_dataset):,}")
    print(f"Validation windows: {len(val_dataset):,}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    if smoke_test:
        print("Running read-only smoke test on one train batch and one val batch...")
        model.eval()
        with torch.inference_mode():
            train_x, train_y, train_pos = next(iter(train_loader))
            val_x, val_y, val_pos = next(iter(val_loader))
            train_logits = model(train_x.to(device))
            val_logits = model(val_x.to(device))
            train_loss = criterion(train_logits, train_y.to(device))
            val_loss = criterion(val_logits, val_y.to(device))

        if train_logits.shape != (batch_size, 2) or val_logits.shape != (batch_size, 2):
            raise ValueError("Unexpected logit shape during smoke test.")
        if not (torch.isfinite(train_loss) and torch.isfinite(val_loss)):
            raise ValueError("Non-finite loss during smoke test.")

        print(
            "Smoke test passed:",
            {
                "train_batch_shape": list(train_x.shape),
                "val_batch_shape": list(val_x.shape),
                "train_first_position": int(train_pos[0]),
                "val_first_position": int(val_pos[0]),
                "train_loss": round(float(train_loss.item()), 6),
                "val_loss": round(float(val_loss.item()), 6),
            },
        )
        print("No model checkpoints or result files were written.")
        return

    if model_dir.exists() or results_dir.exists():
        raise FileExistsError(
            f"Output directory already exists ({model_dir} or {results_dir}); "
            "refusing to overwrite existing artifacts."
        )

    model_dir.mkdir(parents=True, exist_ok=False)
    results_dir.mkdir(parents=True, exist_ok=False)

    best_checkpoint_path = model_dir / config["outputs"]["checkpoint_name"]
    last_checkpoint_path = model_dir / config["outputs"]["last_checkpoint_name"]
    history_csv_path = results_dir / "training_history.csv"
    summary_json_path = results_dir / "training_summary.json"
    config_copy_path = results_dir / "run_config.json"

    config_copy_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    best_val_loss = float("inf")
    best_epoch = None
    history = []
    total_epochs = int(config["training"]["epochs"])
    max_norm = float(config["training"]["gradient_clip_max_norm"])
    start_time = time.time()

    for epoch in range(1, total_epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            max_norm,
            epoch,
            total_epochs,
        )
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_metrics["val_loss"])

        current_lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "learning_rate": current_lr,
            **val_metrics,
        }
        history.append(row)

        checkpoint_payload = {
            "epoch": epoch,
            "seed": seed,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_metrics["val_loss"],
            "val_metrics": val_metrics,
            "model_config": {
                key: value
                for key, value in config["model"].items()
                if key != "architecture"
            },
            "run_config": config,
        }
        torch.save(checkpoint_payload, last_checkpoint_path)

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            best_epoch = epoch
            torch.save(checkpoint_payload, best_checkpoint_path)

        print(
            f"Epoch {epoch}/{total_epochs} complete | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_metrics['val_loss']:.6f} | "
            f"val_acc={val_metrics['val_accuracy']:.4%} | "
            f"val_bal_acc={val_metrics['val_balanced_accuracy']:.4%}"
        )

    with history_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = list(history[0].keys())
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            row_copy = dict(row)
            row_copy["val_confusion_matrix"] = json.dumps(
                row_copy["val_confusion_matrix"]
            )
            writer.writerow(row_copy)

    summary = {
        "experiment_name": config["experiment_name"],
        "device": str(device),
        "seed": seed,
        "epochs_completed": total_epochs,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_checkpoint_path": str(best_checkpoint_path),
        "last_checkpoint_path": str(last_checkpoint_path),
        "elapsed_seconds": float(time.time() - start_time),
        "final_epoch_metrics": history[-1],
    }
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Training complete. Summary saved to:", summary_json_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Colab-ready reproducible AEGIS Audio Transformer trainer."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to training JSON configuration (default: configs/audio_transformer_colab_v1.json).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=os.environ.get("AEGIS_CACHE_DIR"),
        help="Optional override path for audio_feature_cache directory (e.g. /content/audio_feature_cache).",
    )
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=os.environ.get("AEGIS_INDEX_CSV"),
        help="Optional override path for asvspoof2019_la_window_index.csv.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--smoke-test",
        action="store_true",
        help="Verify dataset/model pipeline on one batch without training or writing files.",
    )
    mode.add_argument(
        "--train",
        action="store_true",
        help="Run full training and write outputs to the new experiment directory.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        args = parse_args()
        cfg = load_config(args.config)
        if args.train and cfg["safety"].get("run_training_automatically") is not False:
            raise ValueError("Invalid safety configuration.")
        run(
            cfg,
            smoke_test=args.smoke_test,
            cache_dir_override=args.cache_dir,
            index_csv_override=args.index_csv,
        )
    except Exception as exc:
        print(f"TRAINING SCRIPT FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
