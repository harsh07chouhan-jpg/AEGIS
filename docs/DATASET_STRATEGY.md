# AEGIS Dataset Strategy

## Primary Multimodal Dataset

### FakeAVCeleb

Purpose:
- Audio-video multimodal manipulation analysis
- Audio-video consistency experiments
- Multimodal fusion
- Manipulation-type analysis

Reason for selection:
- Contains manipulated video and synthesized cloned audio
- Supports multimodal deepfake research
- Provides fine-grained labeling

Status:
- Candidate primary dataset
- Access and licensing must be verified before download

---

## Video Benchmark

### FaceForensics++

Purpose:
- Controlled video manipulation experiments
- Video-only baseline
- Manipulation-family comparison
- Robustness experiments

Manipulation families include:
- DeepFakes
- Face2Face
- FaceSwap
- NeuralTextures

Status:
- Candidate video benchmark

---

## Audio Benchmark

### ASVspoof 2021 DF

Purpose:
- Audio-only synthetic speech detection
- Audio branch pretraining/benchmarking
- Attack-family analysis
- EER evaluation

Status:
- Candidate audio benchmark

---

## Large-Scale Video Generalization

### DFDC

Purpose:
- External video evaluation
- Generalization testing
- Robustness analysis

Status:
- Later-stage dataset due to scale

---

## Optional Audio Diversity

### WaveFake

Purpose:
- Additional synthetic speech diversity
- External audio generalization

Status:
- Optional

---

## Provenance Layer

### C2PA / Content Credentials

Purpose:
- Provenance verification
- Content history analysis
- Cryptographic validation

Important:
C2PA provenance is treated as a separate evidence source.
Absence of provenance must not automatically be interpreted as evidence
that media is manipulated.

---

## Dataset Selection Principles

AEGIS will prioritize:

1. Multimodal compatibility
2. Clear manipulation labels
3. Diversity of manipulation methods
4. Temporal information where available
5. Realistic compression and media transformations
6. Cross-dataset evaluation
7. Licensing/accessibility
8. Reproducibility
9. Avoidance of identity/source leakage
10. Suitability for transformer-based modelling

---

## Download Policy

Do not download every candidate dataset immediately.

The workflow is:

Dataset selection
→ Access verification
→ License verification
→ Small sample acquisition
→ Dataset audit
→ Storage planning
→ Full acquisition only when justified