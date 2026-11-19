from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(r"D:\Projects\AEGIS")
INDEX_PATH = ROOT / "data/audio_dev/processed/asvspoof2019_la_window_index.csv"
SPEC_PATH = ROOT / "data/audio_dev/processed/audio_feature_cache/cache_spec.json"
CACHE_PATH = ROOT / "data/audio_dev/processed/audio_feature_cache/features.dat"
CHECKPOINT_PATH = ROOT / "models/audio_transformer/best_audio_transformer.pt"
OUTPUT_DIR = ROOT / "results/audio_transformer_validation_corrected"
PREDICTIONS_PATH = OUTPUT_DIR / "dev_window_predictions.csv"
SUMMARY_PATH = OUTPUT_DIR / "validation_summary.json"

BATCH_SIZE = 16
NUM_WORKERS = 0
CLASS_WEIGHTS = torch.tensor([5.0453472, 0.5550013], dtype=torch.float32)


def verify_inputs():
    for path in [INDEX_PATH, SPEC_PATH, CACHE_PATH, CHECKPOINT_PATH]:
        if not path.is_file():
            raise FileNotFoundError(f"Required input missing: {path}")

    if OUTPUT_DIR.exists() or PREDICTIONS_PATH.exists() or SUMMARY_PATH.exists():
        raise FileExistsError(
            "Safety stop: validation output path already exists. "
            f"Nothing was written. Inspect: {OUTPUT_DIR}"
        )


class AEGISAudioTransformer(nn.Module):
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
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
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
            encoder_layer, num_layers=num_layers
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embedding[:, :x.size(1)]
        x = self.transformer(x)
        x = self.norm(x[:, 0])
        return self.classifier(x)


class CachedDevelopmentDataset(Dataset):
    def __init__(self, index_df, full_cache_positions, cache):
        self.index_df = index_df.reset_index(drop=True)
        self.positions = np.asarray(full_cache_positions, dtype=np.int64)
        self.cache = cache

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        full_pos = int(self.positions[idx])
        row = self.index_df.iloc[idx]
        features = np.asarray(self.cache[full_pos], dtype=np.float32)

        label_text = str(row["label"]).strip().lower()
        if label_text == "spoof":
            label = 1
        elif label_text == "bonafide":
            label = 0
        else:
            raise ValueError(
                f"Unexpected label {row['label']!r} at cache position {full_pos}"
            )

        return (
            torch.from_numpy(features),
            torch.tensor(label, dtype=torch.long),
            full_pos,
        )


def main():
    verify_inputs()
    torch.set_num_threads(min(10, torch.get_num_threads()))
    print("Device: CPU")
    print("Reading window index and cache specification...")

    full_index = pd.read_csv(INDEX_PATH)
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))

    expected_shape = [2, 128, 257]
    if spec.get("shape_per_window") != expected_shape:
        raise ValueError(f"Unexpected cache shape: {spec.get('shape_per_window')}")
    if spec.get("dtype") != "float16":
        raise ValueError(f"Unexpected cache dtype: {spec.get('dtype')}")
    if int(spec.get("num_windows", -1)) != len(full_index):
        raise ValueError("Cache spec window count does not match index row count")

    expected_bytes = len(full_index) * int(np.prod(expected_shape)) * 2
    actual_bytes = CACHE_PATH.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"Cache byte-size mismatch: expected {expected_bytes}, got {actual_bytes}"
        )

    dev_mask = full_index["split"].astype(str).str.lower().eq("dev")
    dev_positions = np.flatnonzero(dev_mask.to_numpy())
    dev_index = full_index.iloc[dev_positions].reset_index(drop=True)

    if len(dev_positions) == 0:
        raise ValueError("No development windows found")
    if not np.all(full_index.iloc[dev_positions]["split"].eq("dev")):
        raise AssertionError("Development position verification failed")

    print(f"Full index rows: {len(full_index):,}")
    print(f"Development windows: {len(dev_positions):,}")
    print(
        f"Development cache positions: "
        f"{dev_positions[0]}–{dev_positions[-1]}"
    )
    print("Opening feature cache read-only...")

    cache = np.memmap(
        CACHE_PATH,
        dtype=np.float16,
        mode="r",
        shape=(len(full_index), *expected_shape),
    )
    dataset = CachedDevelopmentDataset(dev_index, dev_positions, cache)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    print("Loading checkpoint on CPU...")
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )
    model_config = dict(checkpoint["model_config"])
    model = AEGISAudioTransformer(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Running corrected development inference...")

    all_positions = []
    all_labels = []
    all_scores = []
    all_predictions = []
    total_weighted_loss = 0.0
    total_target_weight = 0.0
    class_weights = CLASS_WEIGHTS

    with torch.inference_mode():
        for batch_number, (features, labels, positions) in enumerate(loader, start=1):
            if not torch.isfinite(features).all():
                raise ValueError(f"Non-finite feature tensor in batch {batch_number}")

            logits = model(features)
            if logits.ndim != 2 or logits.shape[1] != 2:
                raise ValueError(f"Unexpected model output shape: {tuple(logits.shape)}")

            # Match the original training class-weighted cross-entropy objective.
            batch_loss_sum = nn.functional.cross_entropy(
                logits, labels, weight=class_weights, reduction="sum"
            )
            total_weighted_loss += float(batch_loss_sum.item())
            total_target_weight += float(class_weights[labels].sum().item())

            probabilities = torch.softmax(logits, dim=1)[:, 1]
            predictions = logits.argmax(dim=1)

            all_positions.extend(positions.numpy().astype(int).tolist())
            all_labels.extend(labels.numpy().astype(int).tolist())
            all_scores.extend(probabilities.numpy().astype(float).tolist())
            all_predictions.extend(predictions.numpy().astype(int).tolist())

            if batch_number % 250 == 0 or batch_number == len(loader):
                print(f"Processed {min(batch_number * BATCH_SIZE, len(dataset)):,}"
                      f"/{len(dataset):,} windows")

    y_true = np.asarray(all_labels, dtype=np.int64)
    y_score = np.asarray(all_scores, dtype=np.float64)
    y_pred = np.asarray(all_predictions, dtype=np.int64)
    positions = np.asarray(all_positions, dtype=np.int64)

    if len(y_true) != len(dev_index) or len(np.unique(positions)) != len(dev_index):
        raise AssertionError("Prediction count or cache-position uniqueness check failed")
    if not np.array_equal(positions, dev_positions):
        raise AssertionError("Prediction cache positions do not match development rows")
    if not np.isfinite(y_score).all():
        raise ValueError("Non-finite prediction scores detected")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    summary = {
        "evaluation_split": "dev",
        "checkpoint_path": str(CHECKPOINT_PATH),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_model_config": model_config,
        "device": "cpu",
        "number_of_windows": int(len(y_true)),
        "cache_position_start": int(positions.min()),
        "cache_position_end": int(positions.max()),
        "cache_positions_match_development_index": True,
        "class_mapping": {"bonafide": 0, "spoof": 1},
        "class_counts": {
            "bonafide": int((y_true == 0).sum()),
            "spoof": int((y_true == 1).sum()),
        },
        "weighted_cross_entropy_loss": float(
            total_weighted_loss / total_target_weight
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "spoof_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "spoof_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "spoof_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "roc_auc_spoof_positive": float(roc_auc_score(y_true, y_score)),
        "average_precision_spoof_positive": float(average_precision_score(y_true, y_score)),
        "confusion_matrix_labels_order": ["bonafide_0", "spoof_1"],
        "confusion_matrix": cm.tolist(),
        "score_mean_bonafide": float(y_score[y_true == 0].mean()),
        "score_mean_spoof": float(y_score[y_true == 1].mean()),
        "notes": [
            "This evaluates the existing checkpoint without retraining.",
            "Development features are read using original full-cache row positions.",
            "Metrics are window-level, not utterance-level.",
            "ROC-AUC and average precision use spoof as the positive class.",
        ],
    }

    output = dev_index.copy()
    output["cache_position"] = positions
    output["true_label_binary"] = y_true
    output["spoof_score"] = y_score
    output["predicted_label_binary"] = y_pred
    output["predicted_label"] = np.where(y_pred == 1, "spoof", "bonafide")

    # Final collision check immediately before creating output files.
    if OUTPUT_DIR.exists() or PREDICTIONS_PATH.exists() or SUMMARY_PATH.exists():
        raise FileExistsError("Output path appeared during execution; refusing to write.")

    OUTPUT_DIR.mkdir(parents=False, exist_ok=False)
    output.to_csv(PREDICTIONS_PATH, index=False)
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("\n=== CORRECTED DEVELOPMENT VALIDATION ===")
    print(f"Windows: {summary['number_of_windows']:,}")
    print(f"Weighted cross-entropy loss: {summary['weighted_cross_entropy_loss']:.6f}")
    print(f"Accuracy: {summary['accuracy']:.4%}")
    print(f"Balanced accuracy: {summary['balanced_accuracy']:.4%}")
    print(f"Spoof precision: {summary['spoof_precision']:.4f}")
    print(f"Spoof recall: {summary['spoof_recall']:.4f}")
    print(f"Spoof F1: {summary['spoof_f1']:.4f}")
    print(f"ROC-AUC: {summary['roc_auc_spoof_positive']:.6f}")
    print(f"Average precision: {summary['average_precision_spoof_positive']:.6f}")
    print("Confusion matrix (rows=true, columns=predicted; bonafide, spoof):")
    print(cm)
    print("\nSaved predictions:", PREDICTIONS_PATH)
    print("Saved summary:", SUMMARY_PATH)
    print("Existing checkpoint and evaluation predictions were not modified.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nVALIDATION FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
