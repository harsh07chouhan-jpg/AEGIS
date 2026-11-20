# AEGIS — AI-Enabled Generative-media Integrity & Security

> An academic forensic media screening workstation for analyzing potentially manipulated **audio and video** with separate machine-learning inference pipelines.

AEGIS is a research prototype. Its outputs are **screening indicators, not proof** that a recording is authentic or manipulated. Model scores are not calibrated probabilities.

## Overview

AEGIS combines a Python FastAPI backend with a React + Vite frontend. It supports synthetic-speech screening using a spectral audio transformer and video-manipulation screening using an EfficientNet-B4 + BiLSTM pipeline. The interface uses a restrained editorial design with four areas: **Overview**, **New Analysis**, **Model Intelligence**, and **System Status**.

### Features

- **Audio forensics:** upload audio, view the overall spoof score, decision threshold, per-window scores, padding warnings, and evidence/checkpoint hashes when supplied by the API.
- **Video forensics:** upload video, review the REAL/FAKE screening label, fake-class score, model-reported confidence, sampling/face-detection metadata, and frame-level temporal attention when available.
- **Evidence export:** download Markdown (`.md`) dossiers and JSON (`.json`) audit payloads for completed audio and video analyses.
- **Model intelligence:** inspect the audio model architecture, checkpoint provenance, offline benchmark metrics, confusion matrix, and interpretation cautions.
- **System status:** inspect API connectivity and reported audio-service health, with an explicit limitation on video-readiness monitoring.
- **File validation:** reject unsupported or empty uploads and enforce separate audio/video upload-size limits in the frontend.

## Technology stack

| Layer | Technologies |
| --- | --- |
| Frontend | React, Vite, JavaScript, HTML/CSS |
| Backend | Python 3.11, FastAPI |
| Audio | PyTorch, Librosa, NumPy, spectral audio transformer |
| Video | PyTorch, EfficientNet-B4 + BiLSTM, OpenCV, face detection and temporal frame analysis |
| Reporting | Client-side Markdown and JSON generation |

## Architecture

```text
Browser — React / Vite
       |
       v
FastAPI (src/api.py)
       |
       +--- POST /analyze/audio ---> Audio inference service
       |                            -> decode/resample -> temporal windows
       |                            -> spectral features -> transformer
       |                            -> window scores + aggregate decision
       |
       +--- POST /analyze/video ---> Video detector
                                    -> frame sampling / face detection
                                    -> EfficientNet-B4 + BiLSTM
                                    -> classification + temporal metadata

Browser renders the appropriate report and can export .md or .json files.
```

**The audio and video engines are independent.** `GET /health` primarily establishes the audio service's state; it must **not** be interpreted as an independent video-engine readiness check.

## Audio model and checkpoint provenance

| Specification | Details |
| --- | --- |
| Model ID | `aegis_audio_transformer_reproducible_v1` |
| Checkpoint location (local) | `models/audio_transformer_reproducible_v1/best_audio_transformer.pt` |
| Verified checkpoint SHA-256 | `B937B6BB5C95650D361F75FB9FBE4E8BFFC010070F2853D91E36C64D9D8FD7A4` |
| Checkpoint epoch | 5 |
| Trainable parameters | 875,906 |
| Architecture | 4 transformer layers; 4 heads; 128-dimensional embeddings; GELU; 16×16 patch projection |
| Input | 16 kHz mono audio |
| Features | Log-Mel spectrogram + CQT, shape `(2, 128, 257)` |
| Windows | 2.56 s window / 1.28 s hop |
| Decision rule | `mean_spoof_score >= 0.50` |

The original project documentation describes SHA-256 verification before checkpoint deserialization. Checkpoint files are excluded from the Git repository; obtain them through an authorized project distribution channel and place them at the expected local paths. Do not substitute an unverified checkpoint.

### Audio benchmark: ASVspoof 2019 LA evaluation split

These are **offline results on a specific benchmark**, not estimates of real-world accuracy.

| Metric | Reported result |
| --- | ---: |
| Evaluated utterances | 71,237 (63,882 spoof; 7,355 bona fide) |
| Accuracy | 89.04% |
| Balanced accuracy | 74.74% |
| ROC AUC | 0.8846 |
| Approximate EER | 19.09% |
| Spoof precision | 94.90% |
| Spoof recall | 92.76% |
| Bona fide recall at 0.50 | 56.72% |
| False-alarm rate on bona fide recordings | **43.28%** |

**Confusion matrix** (rows = actual; columns = predicted):

| Actual class | Predicted bona fide | Predicted spoof |
| --- | ---: | ---: |
| Bona fide | 4,172 | 3,183 |
| Spoof | 4,628 | 59,254 |

The benchmark is approximately **89.68% spoof**. Its class imbalance strongly affects accuracy and precision. In particular, the large bona fide false-alarm rate means a flagged result is not sufficient evidence of synthesis, and an unflagged result is not proof of authenticity.

## Video model

| Specification | Details |
| --- | --- |
| Model version | `effnetb4-bilstm-v1.0.0` |
| Architecture | EfficientNet-B4 visual feature extraction + bidirectional LSTM temporal analysis |
| Processing | Frame sampling, face detection, temporal analysis |
| Example report fields | `prob_fake`, `verdict`, `confidence`, `model_version`, `metadata` |
| Temporal output | Optional `metadata.frame_predictions` timestamps and attention values |

The video result reports a **model score and screening label**, not authenticated provenance. Temporal attention values indicate relative model emphasis across sampled frames and are **not separate frame authenticity probabilities**. Video benchmark accuracy is not asserted here because a verified video evaluation summary has not been supplied for this README.

## Run locally

Commands below assume the repository is already cloned and its Python environment, dependencies and required checkpoints have been provisioned. **The checkpoints and large datasets are not included in Git.**

### Backend (Windows CMD)

From the repository root:

```cmd
.venv\Scripts\activate
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

API base: `http://127.0.0.1:8000`
Interactive API documentation: `http://127.0.0.1:8000/docs`

### Frontend (a separate CMD window)

```cmd
cd frontend
npm run dev
```

Frontend: `http://localhost:5173/`

If configuring a new machine rather than using the existing project environment, use the repository's dependency manifests and ensure the audio/video model assets are available separately. A Git clone alone does not provide the ignored checkpoints.

## API reference

| Endpoint | Method | Description |
| --- | --- | --- |
| `/health` | GET | Backend/audio inference health information; does not independently verify video readiness |
| `/model-info` | GET | Audio model and benchmark information |
| `/analyze/audio` | POST | Multipart upload (`file`) for audio screening |
| `/analyze/video` | POST | Multipart upload (`file`) for video screening |

The frontend currently accepts these filename extensions:

- **Audio:** WAV, FLAC, MP3, OGG, M4A, AIFF, AIF; frontend maximum **50 MiB**. The documented supported audio duration is **0.25–60 seconds**, though performance outside the benchmark's evaluated duration distribution is unvalidated.
- **Video:** MP4, MOV, MKV, AVI, WEBM, M4V; frontend maximum **200 MiB**.

Format acceptance in the UI does not guarantee that every codec/container combination will decode on every backend installation.

### Example video response shape

```json
{
  "prob_fake": 0.0724,
  "verdict": "REAL",
  "confidence": 0.8552,
  "model_version": "effnetb4-bilstm-v1.0.0",
  "metadata": {
    "duration_seconds": 15.95,
    "fps": 59.94,
    "frames_sampled": 16,
    "faces_detected": 16,
    "frame_predictions": [
      { "timestamp": 0, "attention": 0.0696 }
    ]
  }
}
```

Illustrative shape from a tested inference; not a claim about the authenticity of that file. Audio responses use the separate `aegis.audio_inference.v1` schema with `input`, `model`, `policy`, `windows`, `aggregate`, `decision`, and `warnings`.

## Evidence reports

Both report types support **Markdown evidence dossiers** and **JSON audit payloads**, generated in the browser after analysis.

| | Audio | Video |
| --- | --- | --- |
| Classification and scores | Spoof / bona fide and aggregate window scores | REAL / FAKE, fake-class score and model-reported confidence |
| Temporal detail | Per-window timestamps, spoof scores and padding | Sampled-frame timestamps and attention, when supplied |
| Metadata | Audio duration, sampling and digest | Video duration, FPS, sampled frames and face counts |
| Model reference | Audio model and checkpoint provenance | Video model version |
| Limitations | Calibration, benchmark imbalance, false alarms | Calibration and screening-only interpretation |

JSON exports retain the corresponding analysis payload; they are **not digitally signed evidence certificates**. Video export does not currently claim a source-file SHA-256 digest. The presence of an audio digest or checkpoint hash does not establish a legally sufficient chain of custody.

## Validation and tests

Frontend validation used during the project review:

```cmd
cd frontend
npm run build
npm run lint
```

Both commands passed on **20 November 2026**. Desktop smoke tests demonstrated completed audio and video analysis, report rendering, temporal details, and Markdown/JSON exports. Responsive/mobile behavior was **not** verified in this pass.

Existing regression test commands documented in the repository:

```cmd
python -m unittest tests/test_api_and_inference.py -v
node --test tests/test_evidence_report.mjs
```

These test commands are documented here; they were not rerun as part of the final frontend review. Refer to their actual output for pass/fail status.

## Repository layout

```text
AEGIS/
├── frontend/
│   └── src/
│       ├── App.jsx
│       ├── components/
│       │   ├── WorkspaceShell.jsx / .css
│       │   ├── UploadZone.jsx
│       │   ├── InferenceReport.jsx
│       │   └── VideoInferenceReport.jsx / .css
│       ├── services/apiClient.js
│       └── utils/
│           ├── evidenceReport.js
│           └── videoEvidenceReport.js
├── src/                      # FastAPI and audio inference implementation
├── detector_runtime/         # Detector support utilities
├── video_detector/
│   ├── src/                  # Video configuration and inference
│   └── training/             # Model implementation; checkpoints ignored
├── tests/
├── docs/
└── models/                   # Local checkpoints, ignored by Git
```

## Research limitations and responsible use

1. **Uncalibrated outputs:** Softmax-derived scores and model-reported confidence are not calibrated probabilities that a file is real or fake.
2. **Domain shift:** Performance may change substantially across new languages, devices, codecs, compression levels, noise conditions, and editing methods.
3. **Audio benchmark restrictions:** The published audio metrics concern ASVspoof 2019 LA and must not be generalized to arbitrary uploads.
4. **False alarms:** The reported 43.28% bona fide false-alarm rate is a substantial limitation at the default audio threshold.
5. **Video interpretability:** Temporal attention is an explanation aid, not a per-frame forensic verdict.
6. **Service monitoring:** The audio-oriented health check does not independently establish video-model readiness.
7. **Not definitive evidence:** Do not use an AEGIS label alone for legal, disciplinary, reputational, or other high-stakes decisions. Seek independent forensic and provenance checks.

## Future work

Potential research directions include calibrated scoring, broader cross-dataset testing, stronger provenance/integrity workflows, independent video-health reporting, responsive UI validation, and audio-video consistency research. These are **not** represented as completed features.

The original roadmap documents, where present, remain useful historical design references: [`docs/VIDEO_MULTIMODAL_ROADMAP.md`](docs/VIDEO_MULTIMODAL_ROADMAP.md) and [`docs/DATASET_STRATEGY.md`](docs/DATASET_STRATEGY.md).

## Attribution and licensing

AEGIS is intended for academic research and evaluation. ASVspoof 2019 data and protocols are attributed to their respective creators. **Do not infer an open-source license merely from a public GitHub repository**; consult the repository's `LICENSE` file, if any, for actual reuse permissions.
