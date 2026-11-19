# AEGIS — AI-Enabled Generative-media Integrity & Security

> A technical, reproducible, and explainable deepfake forensics platform for screening synthetic speech and acoustic media manipulations.

---

## 1. Project Overview

**AEGIS** is a full-stack media forensics workstation designed to inspect digital recordings for acoustic and synthetic manipulations. Developed as an applied AI research and digital evidence screening platform, AEGIS emphasizes **scientific rigor, checkpoint provenance, transparent limitations, and reproducible evaluation** over uncalibrated or exaggerated claims.

The system features:
- A standalone, reproducible **Spectral Audio Transformer** evaluated against the standard ASVspoof 2019 LA benchmark.
- A hardened **FastAPI backend** with in-memory streaming, strict payload boundaries, structured schema validation, and automatic temporary file cleanup.
- A dark, technical **Forensic Workstation UI** (React 19 + Vite 8) with drag-and-drop evidence ingest, uncalibrated score meters, temporal window breakdown, and live model telemetry.
- An automated **regression test suite** validating checkpoint bit-integrity, endpoint parity, error handling, and known-sample prediction reproducibility.

---

## 2. Verified Audio Model & Checkpoint Provenance

AEGIS uses a reproducible, self-contained Vision/Audio Transformer evaluated on the ASVspoof 2019 Logical Access (LA) evaluation protocol.

| Specification | Details |
| :--- | :--- |
| **Model Name** | `aegis_audio_transformer_reproducible_v1` |
| **Checkpoint Path** | `models/audio_transformer_reproducible_v1/best_audio_transformer.pt` |
| **Verified SHA-256** | `B937B6BB5C95650D361F75FB9FBE4E8BFFC010070F2853D91E36C64D9D8FD7A4` |
| **Training Epoch** | Epoch 5 (selected by validation loss) |
| **Trainable Parameters** | 875,906 |
| **Architecture** | 4 Transformer layers, 4 attention heads, 128 embedding dim, GELU activation, (16, 16) patch projection |
| **Target Sample Rate** | 16,000 Hz mono float32 |
| **Window Policy** | 2.56-second analysis window (40,960 samples), 1.28-second hop (20,480 samples) |
| **Acoustic Features** | Stacked Log-Mel Spectrogram + Constant-Q Transform (CQT) of shape `(2, 128, 257)` |
| **Decision Rule** | `mean_spoof_score >= 0.50` (uncalibrated softmax spoof score) |

> [!IMPORTANT]
> The checkpoint SHA-256 hash is computed on the exact byte stream prior to deserialization. The inference service refuses to load any checkpoint whose hash deviates from the verified signature.

---

## 3. Authoritative Offline Benchmark Performance

Benchmark evaluation was conducted offline across the complete ASVspoof 2019 LA evaluation split (`results/audio_transformer_reproducible_v1/eval/evaluation_summary.json`).

| Metric | Authoritative Value | Notes |
| :--- | :--- | :--- |
| **Evaluated Utterances** | **71,237** | 63,882 spoof + 7,355 bona fide utterances |
| **Overall Accuracy** | **89.04%** | Dominated by majority spoof class |
| **Balanced Accuracy** | **74.74%** | Unweighted mean of bona fide and spoof recall |
| **ROC AUC** | **0.8846** | Area under the Receiver Operating Characteristic curve |
| **Approximate EER** | **19.09%** | Equal Error Rate across evaluation split |
| **Spoof Precision** | **94.90%** | Proportion of flagged recordings that were synthetic |
| **Spoof Recall** | **92.76%** | Detection rate on synthetic recordings |
| **Bona Fide Recall @ 0.50** | **56.72%** | **43.28% false-alarm rate on authentic recordings** |

### Confusion Matrix (Rows: Ground Truth, Columns: Predicted)

| | Predicted Bona Fide | Predicted Spoof | Total |
| :--- | :--- | :--- | :--- |
| **Actual Bona Fide** | 4,172 (56.72%) | **3,183 (43.28%)** | 7,355 |
| **Actual Spoof** | 4,628 (7.24%) | 59,254 (92.76%) | 63,882 |

### Critical Scientific Notice on Class Imbalance
The evaluation split is heavily skewed towards synthetic media (**89.68% spoof**). Consequently:
1. High accuracy (89.04%) is heavily driven by spoof detection volume.
2. At the default 0.50 threshold, the model flagged **3,183 out of 7,355 genuine recordings** as synthetic (43.28% false-positive rate on human speech).
3. **Forensic Implication:** A score below 0.50 is NOT proof of authenticity, and a score above 0.50 should be treated as an investigative screening indicator rather than definitive legal proof.

---

## 4. System Architecture

```
                          [ Client Browser ]
                                  │
                          (React 19 + Vite 8)
                                  │
                         REST API (HTTP / CORS)
                                  │
                                  ▼
                     [ FastAPI Backend (src/api.py) ]
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
          [ Lifespan Service ]         [ Streaming Upload ]
          - SHA-256 verification       - 50 MiB limit
          - In-memory service state    - Decoded to 16 kHz
                                       - Auto temp-file unlink
                                                │
                                                ▼
                         [ Inference Service (src/audio_inference_service.py) ]
                          - Sliding window planner (2.56s / 1.28s hop)
                          - Feature extraction: Log-Mel + CQT
                          - PyTorch CPU inference (torch.inference_mode())
                          - Aggregate scores + structured warning flags
```

### Key Architectural Highlights
- **In-Memory / Zero-Retention Policy:** Uploaded audio chunks stream into temporary files that are unlinked immediately after inference finishes, guaranteeing that user evidence is not permanently retained.
- **Strict Error Modeling:** Unsupported file types, empty files, silent files, corrupt headers, and oversized payloads return structured JSON error payloads with stable machine-readable codes.
- **Temporal Window Breakdown:** The API returns per-window spoof scores and time intervals, allowing users to examine which portions of the audio triggered the classifier.

---

## 5. Quick Start & Local Execution

### 5.1 Prerequisites
- Python 3.11 with virtual environment support
- Node.js v18+ and npm
- Windows, macOS, or Linux

### 5.2 Backend Setup & Startup

```bash
# 1. Activate the Python virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

# 2. Run backend tests to verify environment and checkpoint integrity
python -m unittest tests/test_api_and_inference.py

# 3. Start the FastAPI development server
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

The API will be available at `http://127.0.0.1:8000` (API documentation at `http://127.0.0.1:8000/docs`).

### 5.3 Frontend Setup & Startup

```bash
# 1. Navigate to the frontend directory
cd frontend

# 2. Install dependencies (if not already installed)
npm install

# 3. Run linter and production build checks
npm run lint
npm run build

# 4. Start the Vite development server
npm run dev
```

Open `http://localhost:5173` in your browser.

---

## 6. API Reference

### `GET /health`
Returns service availability, model loading state, and any startup exceptions.
```json
{
  "status": "healthy",
  "model_loaded": true,
  "startup_error": null
}
```

### `GET /model-info`
Returns verified checkpoint identity, parameter count, architecture, windowing policy, and offline benchmark performance.

### `POST /analyze/audio`
Accepts `multipart/form-data` with key `file`.

**Successful Response (`HTTP 200`):**
```json
{
  "schema_version": "aegis.audio_inference.v1",
  "status": "ok",
  "input": {
    "filename": "LA_E_2834763.flac",
    "file_size_bytes": 32063,
    "input_sha256": "DCA19033BBCC6EBF52E071BAC8E9778CEAE7E4FF0F390EAFD03DCDB15D694D7E",
    "duration_sec": 1.426937,
    "sample_rate": 16000
  },
  "aggregate": {
    "num_windows": 1,
    "mean_spoof_score": 0.716103,
    "max_spoof_score": 0.716103,
    "score_used_for_decision": "mean_spoof_score"
  },
  "decision": {
    "label": "spoof",
    "rule": "mean_spoof_score >= 0.5",
    "display_text": "Flagged by the model as likely synthetic (score at or above 0.5)."
  },
  "windows": [
    {
      "index": 0,
      "start_sec": 0.0,
      "end_sec": 1.426937,
      "audio_samples": 22831,
      "padded_fraction": 0.442603,
      "spoof_score": 0.716103
    }
  ],
  "warnings": [
    {
      "code": "zero_padded_window",
      "message": "At least one window was zero-padded (up to 44%), matching the evaluation policy; padded windows count in the mean."
    }
  ],
  "elapsed_sec": 0.3541
}
```

**Rejection Response (`HTTP 200` with `status: "rejected"`):**
Returned when input is unreadable, corrupt, silent, or outside permissible duration limits.
```json
{
  "status": "rejected",
  "error": {
    "code": "unreadable_audio",
    "message": "Audio could not be decoded."
  }
}
```

---

## 7. Automated Test Suite

A multi-tiered regression test suite protects both backend inference integrity and frontend report generation:

### 7.1 Backend API & Checkpoint Tests
```bash
python -m unittest tests/test_api_and_inference.py -v
```
Verified test coverage:
- `test_checkpoint_sha256_integrity`: Bit-identical SHA-256 match on `best_audio_transformer.pt`.
- `test_evaluation_summary_parity`: Metric consistency with recorded evaluation runs (71,237 utterances).
- `test_root_endpoint`: Service status and documentation endpoints.
- `test_health_endpoint`: Active model loading verification.
- `test_model_info_endpoint`: Metadata schema conformity.
- `test_analyze_audio_known_sample_regression`: End-to-end regression check using `LA_E_2834763.flac` (verifying 0.7161 score, window count, duration, and hash).
- `test_analyze_empty_file_rejected`: HTTP 400 rejection for 0-byte uploads.
- `test_analyze_corrupt_file_rejected`: Structured rejection for non-audio data.
- `test_temporary_files_are_cleaned_up`: Zero disk leaks in temporary upload directories.

### 7.2 Evidence Export & Safety Tests
```bash
node --test tests/test_evidence_report.mjs
```
Verified test coverage:
- `sanitizeFilename`: Path traversal stripping and filesystem safety.
- `isValidInferenceResult`: Rejection of incomplete, failed, or malformed results.
- `generateMarkdownReport`: Verification of mandatory forensic fields, uncalibrated score notices, and benchmark caveats.
- `generateJsonReport`: Schema conformity of machine-readable forensic audit envelopes.

---

## 8. Downloadable Evidence Reports

Once an audio analysis completes with status `ok`, operators can export forensic artifacts directly from the interface:
1. **Forensic Evidence Dossier (`.md`)**: A structured, printable Markdown document detailing evidence file SHA-256, model checkpoint provenance, window breakdown table, warning flags, uncalibrated score semantics, and ASVspoof 2019 LA false-positive caveats.
2. **Machine-Readable Audit File (`.json`)**: A complete JSON audit payload containing the verbatim API response, export metadata envelope, and timestamped integrity signature.

> [!NOTE]
> Report export is strictly blocked when an analysis fails or is rejected by input validation, preventing the creation of misleading or corrupt evidence records.

---

## 8. Video & Multimodal Roadmap

AEGIS is designed to expand into video deepfake detection and cross-modal audio-visual consistency analysis. In accordance with our research transparency standards, video analysis is currently documented as a planned architecture rather than returning simulated results.

For the full 8-stage architectural specification, dataset prerequisites (FakeAVCeleb, FaceForensics++), and facial landmark alignment pipelines, see:
- [`docs/VIDEO_MULTIMODAL_ROADMAP.md`](docs/VIDEO_MULTIMODAL_ROADMAP.md)
- [`docs/DATASET_STRATEGY.md`](docs/DATASET_STRATEGY.md)

---

## 9. License & Ethical Attribution

This project is licensed for research, evaluation, and academic demonstration.
The ASVspoof 2019 dataset and protocol are the property of the ASVspoof consortium. Model outputs must be evaluated in context with independent provenance indicators.
