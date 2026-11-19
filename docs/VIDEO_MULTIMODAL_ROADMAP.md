# AEGIS — Video & Multimodal Deepfake Forensics Architecture Roadmap

## 1. Executive Summary & Scientific Scope

AEGIS is designed as a multimodal forensic platform for detecting synthetic and manipulated audio-visual media.

To maintain scientific integrity and prevent false assurance in forensic evaluations:
- **Implemented & Evaluated Capability:** Standalone audio deepfake screening using a 4-layer spectral transformer (`aegis_audio_transformer_reproducible_v1`, SHA-256: `B937B6BB5C95650D361F75FB9FBE4E8BFFC010070F2853D91E36C64D9D8FD7A4`), evaluated on 71,237 utterances from the ASVspoof 2019 LA evaluation dataset.
- **Roadmap Capability:** Video manipulation analysis and cross-modal audio-visual synchronization.
- **Transparency Policy:** No simulated, random, or heuristic video predictions are returned by the platform until a trained and rigorously evaluated PyTorch checkpoint is verified and integrated.

---

## 2. Dataset Strategy & Prerequisites

Before video inference is integrated into the production API, the following datasets must be formally acquired, audited, and benchmarked:

### 2.1 FakeAVCeleb (Primary Multimodal Dataset)
- **Status:** Access request submitted; pending institutional compliance review.
- **Purpose:** Multimodal deepfake analysis, audio-video consistency evaluation, and cross-modal fusion training.
- **Composition:** Pairs of real videos, face-swapped videos, cloned-audio videos, and dual-manipulated (video + audio) media across diverse demographic groups.

### 2.2 FaceForensics++ (Video Benchmark)
- **Status:** Planned acquisition for video-only benchmark.
- **Purpose:** Controlled evaluation across four distinct manipulation families:
  1. DeepFakes (autoencoder-based face replacement)
  2. Face2Face (facial expression reenactment)
  3. FaceSwap (graphics-based face swapping)
  4. NeuralTextures (neural rendering of facial regions)
- **Evaluation splits:** High quality (C23 compressed) and raw (lossless) evaluation to measure robustness against compression artifacts.

---

## 3. Staged Technical Architecture

The planned video and multimodal forensic pipeline is structured into eight decoupled stages:

```
                          [ Input Video File (MP4/MKV/MOV) ]
                                          │
                   ┌──────────────────────┴──────────────────────┐
                   ▼                                             ▼
        [ Demux: Audio Stream ]                       [ Demux: Video Stream ]
                   │                                             │
      (16 kHz Mono Float32)                       (Sample at 15–25 fps)
                   │                                             │
                   ▼                                             ▼
     [ Audio Ingest Pipeline ]                     [ Face Detection & Tracking ]
      - 2.56s / 1.28s windows                       - MTCNN / RetinaFace
      - Log-Mel + CQT (2, 128, 257)                 - Smooth BBox + Margin
      - Audio Transformer (v1)                      - 224×224 Aligned Crops
                   │                                             │
                   ▼                                             ▼
        [ Audio Spoof Score ]                     [ Spatial-Temporal Video ViT ]
       (Mean / Max / Window)                        - Spatio-temporal patches
                   │                                - Boundary seam detection
                   │                                - Temporal inconsistency
                   │                                             │
                   │                                             ▼
                   │                                   [ Video Spoof Score ]
                   │                                             │
                   └──────────────┬──────────────────────────────┘
                                  ▼
                    [ Cross-Modal Sync Analysis ]
                     - Visual lip landmark trajectory
                     - Acoustic phoneme spectral energy
                     - Cross-modal cosine distance / sync offset
                                  │
                                  ▼
                    [ Calibrated Fusion Layer ]
                     - Audio evidence weight
                     - Video evidence weight
                     - Sync mismatch penalty
                     - Platt scaling / Conformal calibration
                                  │
                                  ▼
                    [ Structured Forensic Report ]
```

### Stage 1: Video Ingest & Stream Decoupling
- Input validation: Supported video containers (MP4, MKV, MOV, WebM), maximum file size limit (100 MiB initial safety limit), container integrity check.
- Audio stream demuxing: In-memory or streaming extraction via `ffmpeg` / `torchaudio` resampled directly to 16 kHz mono.
- Video stream decoding: Uniform temporal frame sampling (e.g. 15 or 25 fps) to avoid decoding redundant identical frames while preserving temporal motion.

### Stage 2: Face Detection, Tracking & Landmark Extraction
- Landmark extraction: Multi-task Cascaded Convolutional Networks (MTCNN) or RetinaFace with temporal bounding-box smoothing (Kalman filtering) to eliminate bounding-box jitter across consecutive frames.
- Facial crop alignment: 224×224 normalized face crops with 1.3× margin expansion to ensure forehead, jawline, and ear-boundary blending seams are captured.

### Stage 3: Spatial-Temporal Feature Modeling
- Spatial feature extraction: Frame-level visual transformer (ViT-B/16 or Swin-Transformer) to capture subtle frequency artifacts (checkerboard patterns, color mismatch, edge blurring).
- Temporal feature extraction: Temporal transformer or 3D convolutional network (e.g. TimeSformer or VideoMAE) modeling inter-frame coherence and blinking/gaze continuity.

### Stage 4: Lip-Sync & Acoustic-Visual Alignment
- Visual phoneme tracking: Extracting 3D facial landmarks focused on lip aperture, corner displacement, and jaw velocity.
- Acoustic phoneme alignment: Aligning spectral formant transitions with visual mouth opening/closing cycles.
- Out-of-sync detection: Measuring temporal cross-correlation and time-lag offsets indicative of dubbing or audio replacement.

### Stage 5: Cross-Modal Bidirectional Attention
- Cross-attention transformer blocks where acoustic tokens attend to visual tokens and visual tokens attend to acoustic tokens.
- Cross-modal consistency metric: Flagging instances where video exhibits natural human characteristics while audio exhibits synthetic artifacts, or vice-versa.

### Stage 6: Temporal Window Aggregation
- Per-clip temporal aggregation: Reporting manipulation probability across 2-second temporal segments to localize exactly when and where manipulation occurred.

### Stage 7: Calibrated Decision Fusion
- Modality-aware weighting: Rather than a naive average, the fusion layer dynamically weights modalities based on evidence certainty and signal quality (e.g., low face resolution downweights visual confidence; background noise downweights acoustic confidence).
- Probability calibration: Isotonic regression or temperature scaling to ensure fused scores reflect true empirical risk.

### Stage 8: Independent Reporting & Attribution
- Explicit multi-part report:
  1. Audio Manipulation Score (with benchmark context and false-alarm disclaimer)
  2. Video Manipulation Score (with model attribution and detected artifact class)
  3. Audio-Video Sync Score (with estimated temporal offset in milliseconds)
  4. Overall Fused Assessment (investigative screening level)

---

## 4. Engineering & Safety Prerequisites for Video Deployment

1. **Model Checkpoint Integrity:** Checkpoints must be pinned by SHA-256 and verified at startup.
2. **Resource Constraints:** Video inference requires GPU acceleration (CUDA) or heavily optimized ONNX Runtime execution with batching to maintain acceptable latency (< 10 seconds for a 15-second clip).
3. **Data Protection:** No user video frames or audio may be cached persistently on server storage.
4. **Separate Benchmark Validation:** Video performance must be evaluated independently on FaceForensics++ and FakeAVCeleb, reporting accuracy, balanced accuracy, EER, and false-positive rates per manipulation family.
