
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import librosa
from scipy.signal import resample


# =========================================================
# Configuration
# =========================================================

PROJECT_ROOT = Path(r"D:\Projects\AEGIS")

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "audio_dev"
    / "processed"
    / "asvspoof2019_la_eval_metadata.csv"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "audio_transformer"
    / "best_audio_transformer.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "audio_transformer_eval"
)

OUTPUT_PATH = OUTPUT_DIR / "eval_utterance_predictions.csv"

WINDOW_SEC = 2.56
HOP_SEC = 1.28
TARGET_SR = 16000

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Audio preprocessing
# =========================================================

def preprocess_audio(
    audio_path,
    start_sec=0.0,
    window_seconds=2.56,
    target_sr=16000
):
    waveform, sample_rate = librosa.load(
        audio_path,
        sr=target_sr,
        mono=True,
        offset=float(start_sec),
        duration=float(window_seconds)
    )

    target_samples = int(target_sr * window_seconds)

    if len(waveform) < target_samples:
        waveform = np.pad(
            waveform,
            (0, target_samples - len(waveform)),
            mode="constant"
        )
    elif len(waveform) > target_samples:
        waveform = waveform[:target_samples]

    waveform = waveform.astype(np.float32)

    if not np.isfinite(waveform).all():
        raise ValueError(
            f"Non-finite audio values detected: {audio_path}"
        )

    return waveform, sample_rate


def extract_audio_features(
    waveform,
    sr=16000,
    n_fft=1024,
    hop_length=160,
    n_mels=128,
    fmin=20,
    fmax=8000,
    cqt_fmin=65.4,
    cqt_bins=128,
    cqt_bins_per_octave=24,
    target_time_frames=257
):
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax
    )

    log_mel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    cqt = librosa.cqt(
        waveform,
        sr=sr,
        hop_length=hop_length,
        fmin=cqt_fmin,
        n_bins=cqt_bins,
        bins_per_octave=cqt_bins_per_octave
    )

    cqt_magnitude = np.abs(cqt)

    cqt_aligned = resample(
        cqt_magnitude,
        target_time_frames,
        axis=1
    )

    cqt_aligned = librosa.amplitude_to_db(
        np.maximum(cqt_aligned, 1e-10),
        ref=np.max
    )

    audio_features = np.stack(
        [log_mel, cqt_aligned],
        axis=0
    ).astype(np.float32)

    if not np.isfinite(audio_features).all():
        raise ValueError(
            "Non-finite values detected in audio features."
        )

    return audio_features


# =========================================================
# Exact Transformer architecture
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
        dropout=0.1
    ):
        super().__init__()

        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        freq_patches = 128 // patch_size[0]
        time_patches = 257 // patch_size[1]

        num_patches = (
            freq_patches * time_patches
        )

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )

        self.pos_embedding = nn.Parameter(
            torch.zeros(
                1,
                num_patches + 1,
                embed_dim
            )
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(embed_dim)

        self.classifier = nn.Linear(
            embed_dim,
            num_classes
        )

    def forward(self, x):

        x = self.patch_embed(x)

        x = x.flatten(2).transpose(1, 2)

        batch_size = x.size(0)

        cls_tokens = self.cls_token.expand(
            batch_size,
            -1,
            -1
        )

        x = torch.cat(
            [cls_tokens, x],
            dim=1
        )

        x = x + self.pos_embedding[:, :x.size(1)]

        x = self.transformer(x)

        x = self.norm(x[:, 0])

        return self.classifier(x)


# =========================================================
# Load model
# =========================================================

print("Loading metadata...")

eval_metadata = pd.read_csv(
    METADATA_PATH
)

print(
    f"EVAL utterances: {len(eval_metadata):,}"
)

print("Loading checkpoint...")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu"
)

model_config = checkpoint["model_config"]

model = AEGISAudioTransformer(
    **model_config
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(
    "Checkpoint epoch:",
    checkpoint["epoch"]
)

print(
    "Parameters:",
    sum(p.numel() for p in model.parameters())
)


# =========================================================
# Resume support
# =========================================================

if OUTPUT_PATH.exists():

    existing = pd.read_csv(
        OUTPUT_PATH
    )

    completed_ids = set(
        existing["sample_id"]
    )

    print(
        f"Existing completed utterances: "
        f"{len(completed_ids):,}"
    )

else:

    existing = pd.DataFrame()

    completed_ids = set()

    print(
        "No previous results found. "
        "Starting from zero."
    )


# =========================================================
# Inference
# =========================================================

new_results = []

start_total = time.time()

for index, row in eval_metadata.iterrows():

    sample_id = row["sample_id"]

    if sample_id in completed_ids:
        continue

    window_scores = []

    num_windows = int(row["num_windows"])

    try:

        for window_idx in range(num_windows):

            start_sec = (
                window_idx * HOP_SEC
            )

            waveform, sr = preprocess_audio(
                row["file_path"],
                start_sec=start_sec,
                window_seconds=WINDOW_SEC,
                target_sr=TARGET_SR
            )

            features = extract_audio_features(
                waveform,
                sr=sr
            )

            X = torch.from_numpy(
                features
            ).unsqueeze(0).float()

            with torch.no_grad():

                logits = model(X)

                spoof_probability = (
                    torch.softmax(
                        logits,
                        dim=1
                    )[0, 1].item()
                )

            if not np.isfinite(
                spoof_probability
            ):
                raise ValueError(
                    "Non-finite model output."
                )

            window_scores.append(
                spoof_probability
            )

        mean_score = float(
            np.mean(window_scores)
        )

        max_score = float(
            np.max(window_scores)
        )

        new_results.append({

            "sample_id": sample_id,

            "speaker_id": row["speaker_id"],

            "attack_id": row["attack_id"],

            "label": row["label"],

            "duration_sec": row["duration_sec"],

            "num_windows": num_windows,

            "mean_score": mean_score,

            "max_score": max_score
        })

    except Exception as exc:

        print(
            f"\nERROR: {sample_id}: {exc}"
        )

    # Save every 100 newly processed utterances
    if len(new_results) >= 100:

        chunk = pd.DataFrame(
            new_results
        )

        if OUTPUT_PATH.exists():

            chunk.to_csv(
                OUTPUT_PATH,
                mode="a",
                header=False,
                index=False
            )

        else:

            chunk.to_csv(
                OUTPUT_PATH,
                index=False
            )

        completed_ids.update(
            chunk["sample_id"]
        )

        new_results = []

        elapsed = (
            time.time()
            - start_total
        )

        print(
            f"Completed: "
            f"{len(completed_ids):,}/"
            f"{len(eval_metadata):,} "
            f"| elapsed: "
            f"{elapsed / 60:.1f} min"
        )


# Save remaining results

if new_results:

    chunk = pd.DataFrame(
        new_results
    )

    if OUTPUT_PATH.exists():

        chunk.to_csv(
            OUTPUT_PATH,
            mode="a",
            header=False,
            index=False
        )

    else:

        chunk.to_csv(
            OUTPUT_PATH,
            index=False
        )


elapsed_total = (
    time.time()
    - start_total
)

print("\n================================")
print("EVAL INFERENCE COMPLETE")
print("================================")

print(
    f"Elapsed: "
    f"{elapsed_total / 3600:.2f} hours"
)

print(
    f"Results saved to:\n{OUTPUT_PATH}"
)
