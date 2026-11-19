
from pathlib import Path
import hashlib
import json
import time

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import resample
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

# =========================================================
# Configuration
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

METADATA_PATH = (
    ROOT / "data/audio_dev/processed/asvspoof2019_la_eval_metadata.csv"
)

CHECKPOINT_PATH = (
    ROOT / "models/audio_transformer_reproducible_v1/best_audio_transformer.pt"
)

OUTPUT_DIR = ROOT / "results/audio_transformer_reproducible_v1/eval"
PREDICTIONS_PATH = OUTPUT_DIR / "eval_utterance_predictions.csv"
FAILURES_PATH = OUTPUT_DIR / "eval_failures.csv"
MANIFEST_PATH = OUTPUT_DIR / "run_manifest.json"
SUMMARY_PATH = OUTPUT_DIR / "evaluation_summary.json"

EXPECTED_CHECKPOINT_SHA256 = (
    "B937B6BB5C95650D361F75FB9FBE4E8BFFC010070F2853D91E36C64D9D8FD7A4"
)

WINDOW_SEC = 2.56
HOP_SEC = 1.28
TARGET_SR = 16000
SAVE_EVERY = 100


# =========================================================
# Utility functions
# =========================================================

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def preprocess_audio(audio_path, start_sec):
    waveform, sr = librosa.load(
        str(audio_path),
        sr=TARGET_SR,
        mono=True,
        offset=float(start_sec),
        duration=WINDOW_SEC,
    )

    target_samples = int(TARGET_SR * WINDOW_SEC)

    if len(waveform) < target_samples:
        waveform = np.pad(
            waveform,
            (0, target_samples - len(waveform)),
            mode="constant",
        )
    else:
        waveform = waveform[:target_samples]

    waveform = waveform.astype(np.float32)

    if not np.isfinite(waveform).all():
        raise ValueError(f"Non-finite audio values: {audio_path}")

    return waveform, sr


def extract_audio_features(waveform, sr):
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_fft=1024,
        hop_length=160,
        n_mels=128,
        fmin=20,
        fmax=8000,
    )

    log_mel = librosa.power_to_db(mel, ref=np.max)

    cqt = librosa.cqt(
        waveform,
        sr=sr,
        hop_length=160,
        fmin=65.4,
        n_bins=128,
        bins_per_octave=24,
    )

    cqt_aligned = resample(np.abs(cqt), 257, axis=1)

    cqt_db = librosa.amplitude_to_db(
        np.maximum(cqt_aligned, 1e-10),
        ref=np.max,
    )

    features = np.stack(
        [log_mel, cqt_db],
        axis=0,
    ).astype(np.float32)

    if features.shape != (2, 128, 257):
        raise ValueError(f"Unexpected feature shape: {features.shape}")

    if not np.isfinite(features).all():
        raise ValueError("Non-finite audio features.")

    return features


# =========================================================
# Exact checkpoint architecture
# =========================================================

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

        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        num_patches = (
            (128 // patch_size[0])
            * (257 // patch_size[1])
        )

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )

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
        x = x + self.pos_embedding[:, :x.size(1)]

        x = self.transformer(x)
        return self.classifier(self.norm(x[:, 0]))


# =========================================================
# Main evaluation
# =========================================================

def main():
    # Verify checkpoint identity before creating outputs.
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(CHECKPOINT_PATH)

    checkpoint_hash = sha256_file(CHECKPOINT_PATH)

    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            "Checkpoint SHA-256 mismatch. Evaluation cancelled."
        )

    # Load checkpoint BEFORE using checkpoint metadata.
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    if checkpoint.get("epoch") != 5:
        raise RuntimeError("Expected the verified epoch-5 checkpoint.")

    print("Checkpoint SHA-256:", checkpoint_hash)
    print("Checkpoint epoch:", checkpoint["epoch"])

    # Validate metadata.
    metadata = pd.read_csv(METADATA_PATH)

    required = {
        "sample_id",
        "speaker_id",
        "attack_id",
        "label",
        "file_path",
        "duration_sec",
        "num_windows",
    }

    if not required.issubset(metadata.columns):
        raise ValueError(
            f"Missing metadata columns: {required - set(metadata.columns)}"
        )

    if metadata["sample_id"].isna().any():
        raise ValueError("Metadata contains missing sample IDs.")

    if metadata["sample_id"].duplicated().any():
        raise ValueError("Metadata contains duplicate sample IDs.")

    manifest = {
        "checkpoint_path": str(CHECKPOINT_PATH.relative_to(ROOT)),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "metadata_path": str(METADATA_PATH.relative_to(ROOT)),
        "metadata_rows": int(len(metadata)),
        "window_seconds": WINDOW_SEC,
        "hop_seconds": HOP_SEC,
        "sample_rate": TARGET_SR,
        "feature_shape": [2, 128, 257],
        "score_definition": (
            "Mean and maximum window-level softmax probability "
            "for class index 1 (spoof)."
        ),
    }

    # Only use a directory created for this evaluation.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if MANIFEST_PATH.exists():
        old_manifest = json.loads(
            MANIFEST_PATH.read_text(encoding="utf-8")
        )

        for key, value in old_manifest.items():
            if key == "score_definition":
                if value != "mean and max window-level softmax probability for class index 1":
                    raise RuntimeError(
                        "Existing manifest score definition differs."
                    )
            elif key not in manifest or manifest[key] != value:
                raise RuntimeError(
                    f"Existing manifest differs for field: {key}"
                )

        required_existing_fields = {
            "checkpoint_path",
            "checkpoint_sha256",
            "metadata_path",
            "metadata_rows",
            "window_seconds",
            "hop_seconds",
            "sample_rate",
            "feature_shape",
            "score_definition",
        }
        if not required_existing_fields.issubset(old_manifest):
            raise RuntimeError(
                "Existing manifest is missing required fields."
            )
    else:
        if (
            PREDICTIONS_PATH.exists()
            or FAILURES_PATH.exists()
            or SUMMARY_PATH.exists()
        ):
            raise RuntimeError(
                "Evaluation outputs exist without a matching manifest."
            )

        MANIFEST_PATH.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

    if checkpoint.get("epoch") != 5:
        raise RuntimeError("Expected the verified epoch-5 checkpoint.")
    # Load model with strict state-dict validation.
    device = torch.device("cpu")

    model = AEGISAudioTransformer(**checkpoint["model_config"])
    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )
    model.to(device).eval()

    print("Evaluation utterances:", len(metadata))
    print("Output directory:", OUTPUT_DIR)

    # Load existing progress for safe resumption.
    if PREDICTIONS_PATH.exists():
        predictions = pd.read_csv(PREDICTIONS_PATH)
    else:
        predictions = pd.DataFrame()

    if FAILURES_PATH.exists():
        failures = pd.read_csv(FAILURES_PATH)
    else:
        failures = pd.DataFrame(columns=["sample_id", "error"])

    if not predictions.empty:
        if predictions["sample_id"].duplicated().any():
            raise RuntimeError("Saved predictions contain duplicate IDs.")

        if not set(predictions["sample_id"]).issubset(
            set(metadata["sample_id"])
        ):
            raise RuntimeError("Saved predictions contain unknown IDs.")

    if not failures.empty and failures["sample_id"].duplicated().any():
        raise RuntimeError("Saved failures contain duplicate IDs.")

    predicted_ids = (
        set(predictions["sample_id"])
        if not predictions.empty
        else set()
    )

    failed_ids = (
        set(failures["sample_id"])
        if not failures.empty
        else set()
    )

    if predicted_ids & failed_ids:
        raise RuntimeError(
            "Some IDs appear in both prediction and failure files."
        )

    # Previously failed utterances are not automatically retried.
    # This prevents duplicate failure records when resuming.
    resolved_ids = predicted_ids | failed_ids

    pending = metadata[
        ~metadata["sample_id"].isin(resolved_ids)
    ]

    print("Already predicted:", len(predicted_ids))
    print("Previously failed:", len(failed_ids))
    print("Remaining utterances:", len(pending))

    new_rows = []
    new_failures = []
    start_time = time.time()
    resolved_this_run = 0

    for _, row in pending.iterrows():
        sample_id = row["sample_id"]

        try:
            num_windows = int(row["num_windows"])

            if num_windows <= 0:
                raise ValueError(
                    f"Invalid num_windows: {num_windows}"
                )

            scores = []

            for window_idx in range(num_windows):
                start_sec = window_idx * HOP_SEC

                waveform, sr = preprocess_audio(
                    row["file_path"],
                    start_sec,
                )

                features = extract_audio_features(waveform, sr)

                x = torch.from_numpy(features).unsqueeze(0).to(device)

                with torch.inference_mode():
                    logits = model(x)
                    probability = torch.softmax(
                        logits,
                        dim=1,
                    )[0, 1].item()

                if not np.isfinite(probability):
                    raise ValueError("Non-finite model score.")

                scores.append(float(probability))

            new_rows.append({
                "sample_id": sample_id,
                "speaker_id": row["speaker_id"],
                "attack_id": row["attack_id"],
                "label": row["label"],
                "duration_sec": row["duration_sec"],
                "num_windows": len(scores),
                "mean_score": float(np.mean(scores)),
                "max_score": float(np.max(scores)),
            })

        except Exception as exc:
            new_failures.append({
                "sample_id": sample_id,
                "error": f"{type(exc).__name__}: {exc}",
            })

            print(
                f"FAILED {sample_id}: "
                f"{type(exc).__name__}: {exc}"
            )

        resolved_this_run += 1

        # Persist progress periodically.
        if resolved_this_run % SAVE_EVERY == 0:
            if new_rows:
                chunk = pd.DataFrame(new_rows)
                chunk.to_csv(
                    PREDICTIONS_PATH,
                    mode="a" if PREDICTIONS_PATH.exists() else "w",
                    header=not PREDICTIONS_PATH.exists(),
                    index=False,
                )
                predicted_ids.update(chunk["sample_id"])
                new_rows = []

            if new_failures:
                chunk = pd.DataFrame(new_failures)
                chunk.to_csv(
                    FAILURES_PATH,
                    mode="a" if FAILURES_PATH.exists() else "w",
                    header=not FAILURES_PATH.exists(),
                    index=False,
                )
                failed_ids.update(chunk["sample_id"])
                new_failures = []

            print(
                f"Resolved this run: {resolved_this_run}/"
                f"{len(pending)} | predictions: {len(predicted_ids)} "
                f"| failures: {len(failed_ids)} "
                f"| elapsed: {(time.time() - start_time) / 60:.1f} min"
            )

    # Save remaining progress.
    if new_rows:
        chunk = pd.DataFrame(new_rows)
        chunk.to_csv(
            PREDICTIONS_PATH,
            mode="a" if PREDICTIONS_PATH.exists() else "w",
            header=not PREDICTIONS_PATH.exists(),
            index=False,
        )

    if new_failures:
        chunk = pd.DataFrame(new_failures)
        chunk.to_csv(
            FAILURES_PATH,
            mode="a" if FAILURES_PATH.exists() else "w",
            header=not FAILURES_PATH.exists(),
            index=False,
        )

    # Read saved files and validate coverage.
    predictions = (
        pd.read_csv(PREDICTIONS_PATH)
        if PREDICTIONS_PATH.exists()
        else pd.DataFrame()
    )

    failures = (
        pd.read_csv(FAILURES_PATH)
        if FAILURES_PATH.exists()
        else pd.DataFrame(columns=["sample_id", "error"])
    )

    if not predictions.empty and predictions["sample_id"].duplicated().any():
        raise RuntimeError("Duplicate prediction IDs after evaluation.")

    if not failures.empty and failures["sample_id"].duplicated().any():
        raise RuntimeError("Duplicate failure IDs after evaluation.")

    prediction_ids = (
        set(predictions["sample_id"])
        if not predictions.empty
        else set()
    )

    failure_ids = (
        set(failures["sample_id"])
        if not failures.empty
        else set()
    )

    if prediction_ids & failure_ids:
        raise RuntimeError(
            "Some IDs appear in both predictions and failures."
        )

    all_ids = set(metadata["sample_id"])
    unknown_ids = (prediction_ids | failure_ids) - all_ids
    unresolved_ids = all_ids - prediction_ids - failure_ids

    summary = {
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "metadata_utterances": int(len(metadata)),
        "prediction_rows": int(len(predictions)),
        "failure_rows": int(len(failures)),
        "unresolved_rows": int(len(unresolved_ids)),
        "unknown_result_ids": int(len(unknown_ids)),
        "elapsed_minutes_this_run": round(
            (time.time() - start_time) / 60,
            2,
        ),
    }

    # Calculate metrics only on successfully predicted utterances.
    if not predictions.empty:
        labels = predictions["label"].astype(str).str.strip().str.lower()

        valid_labels = labels.isin(["bonafide", "spoof"])

        if not valid_labels.all():
            raise ValueError(
                "Unexpected labels in prediction CSV: "
                f"{sorted(set(labels[~valid_labels]))}"
            )

        y_true = (labels == "spoof").astype(int).to_numpy()
        y_score = predictions["mean_score"].astype(float).to_numpy()
        y_pred = (y_score >= 0.5).astype(int)

        summary.update({
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(
                balanced_accuracy_score(y_true, y_pred)
            ),
            "spoof_precision": float(
                precision_score(y_true, y_pred, zero_division=0)
            ),
            "spoof_recall": float(
                recall_score(y_true, y_pred, zero_division=0)
            ),
            "spoof_f1": float(
                f1_score(y_true, y_pred, zero_division=0)
            ),
            "confusion_matrix_labels_bonafide_spoof": confusion_matrix(
                y_true,
                y_pred,
                labels=[0, 1],
            ).tolist(),
        })

        if len(np.unique(y_true)) == 2:
            summary["roc_auc"] = float(
                roc_auc_score(y_true, y_score)
            )

            fpr, tpr, _ = roc_curve(y_true, y_score)
            fnr = 1.0 - tpr
            idx = int(np.nanargmin(np.abs(fnr - fpr)))

            summary["eer_approx"] = float(
                (fpr[idx] + fnr[idx]) / 2
            )

    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nEvaluation summary:")
    print(json.dumps(summary, indent=2))
    print("Saved summary:", SUMMARY_PATH)


if __name__ == "__main__":
    main()
