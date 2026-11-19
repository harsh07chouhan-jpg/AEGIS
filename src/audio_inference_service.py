"""AEGIS Stage 1 - standalone audio inference service.

Runs the *evaluated* reproducible-v1 audio checkpoint on a single audio file
and returns a structured, JSON-serializable result. It is a new file: no
existing AEGIS module is imported, edited or executed, and nothing is written
to disk.

Compatibility policy (all items verified against src/audio_eval_reproducible.py):
  * Checkpoint SHA-256 is verified on the exact bytes that are then loaded.
  * AEGISAudioTransformer and extract_audio_features are verbatim copies of the
    evaluation script's definitions (copied, not imported, so this service does
    not depend on sklearn/pandas or on future edits to research code). A parity
    test against the original functions is part of the test plan.
  * 16 kHz mono, 2.56 s windows, 1.28 s hop, zero-padded final/short window.
  * Per-window score = softmax(logits)[0, 1] (class 1 = spoof), batch size 1.
  * Aggregate = mean and max of window scores. The evaluation's label rule is
    mean_score >= 0.5 (verified: reproduces evaluation_summary.json exactly).

Scores are raw model outputs. They are NOT calibrated probabilities.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import scipy
import torch
import torch.nn as nn
from scipy.signal import resample

# =========================================================
# Pinned identity and policy (from the evaluated run)
# =========================================================

SCHEMA_VERSION = "aegis.audio_inference.v1"
MODEL_NAME = "aegis_audio_transformer_reproducible_v1"

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_PATH = (
    ROOT / "models/audio_transformer_reproducible_v1/best_audio_transformer.pt"
)
DEFAULT_EVAL_SUMMARY_PATH = (
    ROOT / "results/audio_transformer_reproducible_v1/eval/evaluation_summary.json"
)

EXPECTED_CHECKPOINT_SHA256 = (
    "B937B6BB5C95650D361F75FB9FBE4E8BFFC010070F2853D91E36C64D9D8FD7A4"
)
EXPECTED_CHECKPOINT_EPOCH = 5
EXPECTED_MODEL_CONFIG = {
    "in_channels": 2,
    "embed_dim": 128,
    "patch_size": (16, 16),
    "num_heads": 4,
    "num_layers": 4,
    "num_classes": 2,
    "dropout": 0.1,
}
CLASS_MAPPING = {"bonafide": 0, "spoof": 1}

TARGET_SR = 16000
WINDOW_SEC = 2.56
HOP_SEC = 1.28
WINDOW_SAMPLES = int(TARGET_SR * WINDOW_SEC)  # 40960
HOP_SAMPLES = int(TARGET_SR * HOP_SEC)  # 20480
FEATURE_SHAPE = (2, 128, 257)

# Evaluation rule: y_pred = (mean_score >= 0.5).
DECISION_THRESHOLD = 0.5

# Observed duration range of the 71,237 evaluated utterances
# (eval_utterance_predictions.csv). Outside it, behaviour is unvalidated.
EVAL_OBSERVED_MIN_DURATION_SEC = 0.4699375
EVAL_OBSERVED_MAX_DURATION_SEC = 13.0255625

LIMITATIONS: Tuple[str, ...] = (
    "Scores are raw softmax outputs of a window-level classifier averaged over "
    "windows. They are not calibrated probabilities and are not a probability "
    "that the audio is fake.",
    "The checkpoint was evaluated only on the ASVspoof 2019 LA evaluation set. "
    "Performance on other datasets, codecs, noise conditions, languages, "
    "recording devices or real-world uploads has not been measured.",
    "At the 0.5 threshold the benchmark evaluation showed a large error rate on "
    "bona fide audio (see reference_performance.bonafide_recall_at_0_5). A "
    "'bonafide' result is therefore NOT evidence that audio is authentic.",
    "Benchmark precision figures depend on the evaluation set's class balance "
    "(about 90% spoof) and will differ in other populations.",
    "Audio outside the evaluated duration range, or not originally 16 kHz "
    "ASVspoof-style speech, is unvalidated. Inputs are decoded to 16 kHz mono "
    "before analysis.",
    "Short inputs and the final window are zero-padded, as in the evaluation. "
    "Padded windows are included in the aggregate.",
    "This is an audio-only model score and is not forensic proof. It must not "
    "be used as the sole basis for any decision.",
)


# =========================================================
# Errors and limits
# =========================================================

class ModelIntegrityError(RuntimeError):
    """Checkpoint missing, hash mismatch, or incompatible with the architecture."""


class AudioInputError(Exception):
    """The input audio cannot be analysed. Carries a stable machine code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class InputLimits:
    """Engineering safety limits. Not derived from evaluation results."""

    max_file_bytes: int = 50 * 1024 * 1024
    min_duration_sec: float = 0.25
    max_duration_sec: float = 60.0
    silence_peak_threshold: float = 1e-4


# =========================================================
# Hashing
# =========================================================

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


# =========================================================
# Exact checkpoint architecture (verbatim from audio_eval_reproducible.py)
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
# Exact feature extraction (verbatim from audio_eval_reproducible.py)
# =========================================================

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
# Audio loading, validation and windowing
# =========================================================

@dataclass(frozen=True)
class LoadedAudio:
    waveform: np.ndarray  # float32, 16 kHz mono
    num_samples: int
    duration_sec: float
    peak_amplitude: float


def load_and_validate_audio(path, limits: InputLimits = InputLimits()) -> LoadedAudio:
    """Decode to 16 kHz mono float32 and reject unusable input.

    Raises AudioInputError with a stable code; never returns bad audio.
    """
    p = Path(path)
    if not p.is_file():
        raise AudioInputError("file_not_found", "Input file does not exist.")

    size = p.stat().st_size
    if size == 0:
        raise AudioInputError("empty_file", "Input file is empty (0 bytes).")
    if size > limits.max_file_bytes:
        raise AudioInputError(
            "file_too_large",
            f"Input file exceeds the {limits.max_file_bytes} byte limit.",
        )

    # Cheap header-based duration check so very long files are not fully decoded.
    # Best effort only: some formats/librosa versions cannot report this.
    try:
        header_duration = float(librosa.get_duration(path=str(p)))
    except Exception:
        header_duration = None
    if header_duration is not None and header_duration > limits.max_duration_sec:
        raise AudioInputError(
            "audio_too_long",
            f"Audio exceeds the {limits.max_duration_sec:g} s limit.",
        )

    try:
        waveform, _ = librosa.load(str(p), sr=TARGET_SR, mono=True)
    except librosa.util.exceptions.ParameterError as exc:
        # librosa validates finiteness itself while down-mixing/resampling.
        if "finite" in str(exc).lower():
            raise AudioInputError(
                "non_finite_audio", "Audio contains NaN or infinite sample values."
            ) from exc
        raise AudioInputError(
            "unreadable_audio", "Audio could not be decoded (ParameterError)."
        ) from exc
    except Exception as exc:
        raise AudioInputError(
            "unreadable_audio",
            f"Audio could not be decoded ({type(exc).__name__}).",
        ) from exc

    waveform = np.asarray(waveform, dtype=np.float32)

    if waveform.size == 0:
        raise AudioInputError("empty_audio", "File decoded to zero audio samples.")
    if not np.isfinite(waveform).all():
        raise AudioInputError(
            "non_finite_audio", "Audio contains NaN or infinite sample values."
        )

    duration = waveform.size / TARGET_SR
    if duration < limits.min_duration_sec:
        raise AudioInputError(
            "audio_too_short",
            f"Audio is {duration:.3f} s; minimum is {limits.min_duration_sec:g} s.",
        )
    if duration > limits.max_duration_sec:
        raise AudioInputError(
            "audio_too_long",
            f"Audio exceeds the {limits.max_duration_sec:g} s limit.",
        )

    peak = float(np.max(np.abs(waveform)))
    if peak < limits.silence_peak_threshold:
        raise AudioInputError(
            "silent_input", "Audio contains no usable signal (effectively silent)."
        )

    return LoadedAudio(
        waveform=waveform,
        num_samples=int(waveform.size),
        duration_sec=float(duration),
        peak_amplitude=peak,
    )


def plan_windows(num_samples: int) -> List[Tuple[int, int]]:
    """(start, end) sample ranges following the evaluation policy.

    Count = max(1, ceil((N - window) / hop) + 1), which matched the
    num_windows of all 71,237 evaluated utterances. The last window may be
    shorter than WINDOW_SAMPLES and is zero-padded by pad_window().
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    shortfall = num_samples - WINDOW_SAMPLES
    count = max(1, -(-shortfall // HOP_SAMPLES) + 1)  # integer ceil division
    return [
        (i * HOP_SAMPLES, min(i * HOP_SAMPLES + WINDOW_SAMPLES, num_samples))
        for i in range(count)
    ]


def pad_window(segment: np.ndarray) -> np.ndarray:
    """Zero-pad (end) to exactly WINDOW_SAMPLES float32, as np.pad did in eval."""
    window = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    n = min(len(segment), WINDOW_SAMPLES)
    window[:n] = segment[:n]
    return window


# =========================================================
# Checkpoint verification and loading
# =========================================================

def _normalise_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    merged = {**EXPECTED_MODEL_CONFIG, **dict(config)}
    merged["patch_size"] = tuple(merged["patch_size"])
    return merged


def verify_and_load_model(
    checkpoint_path: Path, expected_sha256: str
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Hash the checkpoint bytes, then load those same bytes (no TOCTOU gap).

    torch.load(weights_only=False) unpickles arbitrary objects; that is only
    acceptable because the SHA-256 of the exact bytes has just been verified.
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise ModelIntegrityError(f"Checkpoint not found: {path}")

    data = path.read_bytes()
    actual = sha256_bytes(data)
    if actual != expected_sha256.upper():
        raise ModelIntegrityError(
            "Checkpoint SHA-256 mismatch; refusing to load. "
            f"expected={expected_sha256.upper()} actual={actual}"
        )

    checkpoint = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
    del data

    for key in ("model_config", "model_state_dict", "epoch"):
        if key not in checkpoint:
            raise ModelIntegrityError(f"Checkpoint is missing key: {key!r}")
    if checkpoint["epoch"] != EXPECTED_CHECKPOINT_EPOCH:
        raise ModelIntegrityError(
            f"Expected epoch {EXPECTED_CHECKPOINT_EPOCH}, found {checkpoint['epoch']}."
        )

    config = _normalise_model_config(checkpoint["model_config"])
    if config != EXPECTED_MODEL_CONFIG:
        raise ModelIntegrityError(
            f"Checkpoint model_config differs from the evaluated architecture: {config}"
        )

    model = AEGISAudioTransformer(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(torch.device("cpu")).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    identity = {
        "checkpoint_sha256": actual,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
    }
    return model, identity


# =========================================================
# Reference performance (read-only, from the existing evaluation summary)
# =========================================================

def load_reference_performance(
    summary_path: Path, expected_sha256: str
) -> Optional[Dict[str, Any]]:
    """Reference metrics from evaluation_summary.json, or None if unusable.

    Returned only if the summary names the same checkpoint hash.
    Derived rates come from the stored confusion matrix (rows = true class,
    columns = predicted class, order bonafide, spoof).
    """
    try:
        data = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        if str(data.get("checkpoint_sha256", "")).upper() != expected_sha256.upper():
            return None
        (tn, fp), (fn, tp) = data["confusion_matrix_labels_bonafide_spoof"]
        total = tn + fp + fn + tp
        return {
            "source": "evaluation_summary.json (ASVspoof 2019 LA eval, as evaluated)",
            "utterances": int(data["metadata_utterances"]),
            "decision_rule": f"mean_score >= {DECISION_THRESHOLD}",
            "accuracy": float(data["accuracy"]),
            "balanced_accuracy": float(data["balanced_accuracy"]),
            "roc_auc": float(data["roc_auc"]),
            "eer_approx": float(data["eer_approx"]),
            "confusion_matrix_labels_bonafide_spoof": [[tn, fp], [fn, tp]],
            "bonafide_recall_at_0_5": tn / (tn + fp),
            "spoof_recall_at_0_5": tp / (tp + fn),
            "bonafide_prediction_precision_at_0_5": tn / (tn + fn),
            "spoof_prediction_precision_at_0_5": tp / (tp + fp),
            "spoof_prevalence_in_evaluation_set": (fn + tp) / total,
        }
    except Exception:
        return None


# =========================================================
# Service
# =========================================================

class AudioInferenceService:
    """Load once, call analyze_file() many times.

    Intended for FastAPI: construct one instance at application start-up
    (lifespan) and call analyze_file() from a sync endpoint/threadpool.
    analyze_file() never raises for bad input; it returns a result dict whose
    "status" is "ok", "rejected" (unusable input) or "error" (unexpected
    failure). Only construction can raise ModelIntegrityError.
    """

    def __init__(
        self,
        checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
        expected_sha256: str = EXPECTED_CHECKPOINT_SHA256,
        evaluation_summary_path: Path = DEFAULT_EVAL_SUMMARY_PATH,
        limits: InputLimits = InputLimits(),
        num_threads: Optional[int] = None,
    ):
        if num_threads is not None:
            torch.set_num_threads(int(num_threads))
        self.limits = limits
        self.model, self._identity = verify_and_load_model(
            checkpoint_path, expected_sha256
        )
        self.reference_performance = load_reference_performance(
            evaluation_summary_path, expected_sha256
        )

    # ---- identity / metadata -------------------------------------------

    def _model_block(self) -> Dict[str, Any]:
        return {
            "name": MODEL_NAME,
            "checkpoint_sha256": self._identity["checkpoint_sha256"],
            "checkpoint_epoch": self._identity["checkpoint_epoch"],
            "class_mapping": dict(CLASS_MAPPING),
            "score_definition": "softmax(logits)[class 1 = spoof] per window",
        }

    @staticmethod
    def _runtime_block() -> Dict[str, str]:
        return {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "librosa": librosa.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "device": "cpu",
        }

    @staticmethod
    def _policy_block() -> Dict[str, Any]:
        return {
            "sample_rate": TARGET_SR,
            "window_sec": WINDOW_SEC,
            "hop_sec": HOP_SEC,
            "feature_shape": list(FEATURE_SHAPE),
            "final_window": "zero-padded at the end (as in evaluation)",
            "window_scoring": "batch size 1, torch.inference_mode()",
            "aggregation": "mean and max of window spoof scores",
            "decision_rule": f"mean_spoof_score >= {DECISION_THRESHOLD}",
            "decision_threshold": DECISION_THRESHOLD,
            "evaluated_duration_range_sec": [
                EVAL_OBSERVED_MIN_DURATION_SEC,
                EVAL_OBSERVED_MAX_DURATION_SEC,
            ],
        }

    def model_info(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "model": {
                **self._model_block(),
                "parameter_count": self._identity["parameter_count"],
                "architecture": {
                    **{k: v for k, v in EXPECTED_MODEL_CONFIG.items() if k != "patch_size"},
                    "patch_size": list(EXPECTED_MODEL_CONFIG["patch_size"]),
                },
            },
            "policy": self._policy_block(),
            "input_limits": {
                "max_file_bytes": self.limits.max_file_bytes,
                "min_duration_sec": self.limits.min_duration_sec,
                "max_duration_sec": self.limits.max_duration_sec,
                "silence_peak_threshold": self.limits.silence_peak_threshold,
            },
            "runtime": self._runtime_block(),
            "reference_performance": self.reference_performance,
            "limitations": list(LIMITATIONS),
        }

    # ---- inference ------------------------------------------------------

    def _score_window(self, window: np.ndarray) -> float:
        features = extract_audio_features(window, TARGET_SR)
        x = torch.from_numpy(features).unsqueeze(0)
        with torch.inference_mode():
            logits = self.model(x)
            if tuple(logits.shape) != (1, 2):
                raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")
            score = torch.softmax(logits, dim=1)[0, 1].item()
        if not math.isfinite(score):
            raise ValueError("Non-finite model score.")
        return float(score)

    def analyze_file(self, path, display_name: Optional[str] = None) -> Dict[str, Any]:
        started = time.perf_counter()
        name = Path(display_name or str(path)).name
        base: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "input": {"filename": name},
            "model": self._model_block(),
        }

        try:
            audio = load_and_validate_audio(path, self.limits)
            plan = plan_windows(audio.num_samples)

            windows: List[Dict[str, Any]] = []
            scores: List[float] = []
            for index, (start, end) in enumerate(plan):
                segment = audio.waveform[start:end]
                score = self._score_window(pad_window(segment))
                scores.append(score)
                windows.append({
                    "index": index,
                    "start_sec": round(start / TARGET_SR, 6),
                    "end_sec": round(end / TARGET_SR, 6),
                    "audio_samples": int(end - start),
                    "padded_fraction": round(1.0 - (end - start) / WINDOW_SAMPLES, 6),
                    "spoof_score": score,
                })

            mean_score = float(np.mean(scores))
            max_score = float(np.max(scores))
            flagged = mean_score >= DECISION_THRESHOLD

            warnings = self._warnings(audio, windows)
            base["status"] = "ok"
            base["input"] = {
                "filename": name,
                "file_size_bytes": int(Path(path).stat().st_size),
                "input_sha256": sha256_file(Path(path)),
                "duration_sec": round(audio.duration_sec, 6),
                "num_samples_16k": audio.num_samples,
                "sample_rate": TARGET_SR,
            }
            base["policy"] = self._policy_block()
            base["windows"] = windows
            base["aggregate"] = {
                "num_windows": len(scores),
                "mean_spoof_score": mean_score,
                "max_spoof_score": max_score,
                "score_used_for_decision": "mean_spoof_score",
            }
            base["decision"] = {
                "label": "spoof" if flagged else "bonafide",
                "rule": f"mean_spoof_score >= {DECISION_THRESHOLD}",
                "display_text": (
                    "Flagged by the model as likely synthetic (score at or above "
                    f"{DECISION_THRESHOLD})."
                    if flagged else
                    f"Not flagged by the model (score below {DECISION_THRESHOLD}). "
                    "This does not verify that the audio is authentic."
                ),
                "score_semantics": "uncalibrated model score, not a probability",
            }
            base["warnings"] = warnings
            base["reference_performance"] = self.reference_performance
            base["limitations"] = list(LIMITATIONS)
            base["runtime"] = self._runtime_block()

        except AudioInputError as exc:
            base["status"] = "rejected"
            base["error"] = {"code": exc.code, "message": exc.message}
        except Exception as exc:  # unexpected: report type only, never crash caller
            base["status"] = "error"
            base["error"] = {
                "code": "inference_failed",
                "message": f"Unexpected failure ({type(exc).__name__}).",
            }

        base["elapsed_sec"] = round(time.perf_counter() - started, 4)
        return base

    @staticmethod
    def _warnings(audio: LoadedAudio, windows: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        if audio.duration_sec > EVAL_OBSERVED_MAX_DURATION_SEC:
            out.append({
                "code": "duration_above_evaluated_range",
                "message": (
                    f"Audio is longer than the longest evaluated utterance "
                    f"({EVAL_OBSERVED_MAX_DURATION_SEC:.2f} s). Window-mean "
                    "aggregation on longer audio has not been validated."
                ),
            })
        if audio.duration_sec < EVAL_OBSERVED_MIN_DURATION_SEC:
            out.append({
                "code": "duration_below_evaluated_range",
                "message": (
                    f"Audio is shorter than the shortest evaluated utterance "
                    f"({EVAL_OBSERVED_MIN_DURATION_SEC:.2f} s)."
                ),
            })
        if any(w["padded_fraction"] > 0 for w in windows):
            worst = max(w["padded_fraction"] for w in windows)
            out.append({
                "code": "zero_padded_window",
                "message": (
                    f"At least one window was zero-padded (up to {worst:.0%}), "
                    "matching the evaluation policy; padded windows count in the mean."
                ),
            })
        return out


# =========================================================
# Minimal CLI (read-only; for smoke tests, not for deployment)
# =========================================================

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="AEGIS audio inference (read-only).")
    parser.add_argument("audio", nargs="*", help="Audio file path(s) to analyse.")
    parser.add_argument("--verify-only", action="store_true",
                        help="Verify and load the checkpoint, print model info, exit.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--summary", type=Path, default=DEFAULT_EVAL_SUMMARY_PATH)
    args = parser.parse_args(argv)

    service = AudioInferenceService(
        checkpoint_path=args.checkpoint,
        evaluation_summary_path=args.summary,
    )
    if args.verify_only or not args.audio:
        print(json.dumps(service.model_info(), indent=2, allow_nan=False))
        return 0

    exit_code = 0
    for audio_path in args.audio:
        result = service.analyze_file(audio_path)
        print(json.dumps(result, indent=2, allow_nan=False))
        if result["status"] != "ok":
            exit_code = 2
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
