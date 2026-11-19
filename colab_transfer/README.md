# AEGIS — Phase 2 Google Colab Transfer Package (`colab_transfer`)

This directory contains the small source, configuration, index, and baseline reference files needed to reproduce the **AEGIS Audio Spectrogram Transformer** (`AEGISAudioTransformer`, `875,906` parameters) on a Google Colab CUDA GPU without modifying any original artifacts.

---

## 1. Package Contents & Relative Layout

```text
colab_transfer/
├── README.md
├── SHA256SUMS.txt
├── configs/
│   ├── audio_transformer_reproducible.json          # Exact copy of original baseline config
│   └── audio_transformer_colab_v1.json              # Colab config with unique experiment/output paths
├── data/
│   └── audio_dev/
│       └── processed/
│           ├── asvspoof2019_la_window_index.csv     # 111,236 window rows (55,630 train / 55,606 dev)
│           └── audio_feature_cache/
│               └── cache_spec.json                  # Memmap specification for features.dat
├── results/
│   └── audio_transformer_reproducible_v1/           # Reference baseline metadata & metrics (read-only)
│       ├── run_config.json
│       ├── training_history.csv
│       ├── training_summary.json
│       └── eval/
│           ├── evaluation_summary.json
│           └── run_manifest.json
└── src/
    ├── audio_train_reproducible.py                  # Exact copy of original training script
    ├── audio_train_colab.py                         # Colab-adapted trainer (separate file)
    ├── audio_validate_checkpoint.py                 # Exact copy of original dev-validation script
    └── cached_audio_dataset.py                      # Exact copy of memmap dataset loader
```

*Not included in this small package (by design):*
* `data/audio_dev/processed/audio_feature_cache/features.dat` (`14,636,881,920` bytes / ~14.64 GB `float16` memmap)
* Raw ASVspoof 2019 LA `.flac` dataset files
* Existing model checkpoints (`models/audio_transformer_reproducible_v1/best_audio_transformer.pt`)

---

## 2. Output-Directory Safeguards & Unique Colab Experiment Name

Both `src/audio_train_reproducible.py` and `src/audio_train_colab.py` enforce two strict safeguards:
1. `config["safety"]["overwrite_existing"]` must be `false`.
2. If `model_dir` or `results_dir` already exists on disk, the script raises `FileExistsError("Output directory already exists; refusing to overwrite existing artifacts.")`.

To ensure existing baseline outputs (`models/audio_transformer_reproducible_v1` and `results/audio_transformer_reproducible_v1`) are never overwritten or reused, `configs/audio_transformer_colab_v1.json` defines a dedicated experiment identifier and output paths:
* **Experiment Name:** `aegis_audio_transformer_colab_v1`
* **Model Checkpoint Output Directory:** `models/audio_transformer_colab_v1/`
* **Training Results Output Directory:** `results/audio_transformer_colab_v1/`

All data splits (`train`: 55,630 windows, `dev`: 55,606 windows), labels (`bonafide: 0`, `spoof: 1`), feature shapes (`[2, 128, 257]`), class weights (`[5.0453472, 0.5550013]`), optimizer (`AdamW`, `lr=0.0003`, `weight_decay=0.0001`), scheduler (`ReduceLROnPlateau`), seed (`42`), and `AEGISAudioTransformer` architecture parameters remain 100% identical to the baseline.

---

## 3. Known Windows-Path & Colab Environment Issues Addressed

1. **Hardcoded Windows Path in `src/audio_validate_checkpoint.py`:**
   * The original `src/audio_validate_checkpoint.py` contains `ROOT = Path(r"D:\Projects\AEGIS")` at line 21, and `src/audio_train_reproducible.py` imports `AEGISAudioTransformer` from `src.audio_validate_checkpoint`.
   * **Mitigation:** `src/audio_train_colab.py` defines `AEGISAudioTransformer` directly and resolves all paths relative to `ROOT = Path(__file__).resolve().parents[1]` (or explicit CLI flags `--cache-dir` / `--index-csv`).
2. **Windows `file_path` Strings Inside `asvspoof2019_la_window_index.csv`:**
   * The `file_path` column in `asvspoof2019_la_window_index.csv` contains `D:\Projects\AEGIS\...` paths.
   * **Impact on Cached Training:** None. `src/cached_audio_dataset.py` only reads the `split` and `label` columns from `asvspoof2019_la_window_index.csv` and indexes `features.dat` by integer row position (`0..111235`).
   * **Impact if Regenerating Cache from Raw Audio:** If raw `.flac` files are used on Colab in a later step, replace the prefix `D:\Projects\AEGIS\` with the Colab root path and convert `\` to `/`.
3. **Google Drive FUSE Random-Read Bottleneck (`features.dat`):**
   * `features.dat` (`14.64 GB`) is accessed via `np.memmap` with `shuffle=True` during training. Reading random 128 KB pages over Google Drive FUSE (`/content/drive/...`) causes severe I/O throttling.
   * **Mitigation:** Copy `features.dat` + `cache_spec.json` onto local Colab NVMe storage (`/content/audio_feature_cache/`) before training and pass `--cache-dir /content/audio_feature_cache` to `src/audio_train_colab.py`.
4. **CUDA Memory Pinning:**
   * `src/audio_train_colab.py` automatically sets `pin_memory=True` when `torch.cuda.is_available()` is true.

---

## 4. Python Dependencies

Colab's default GPU runtime includes Python 3.10+, CUDA PyTorch, NumPy, Pandas, SciPy, and scikit-learn. Verify or install packages with:

```bash
pip install torch numpy pandas scikit-learn librosa soundfile scipy
```

---

## 5. Step-by-Step Colab Setup & Verification Instructions

### Step 1 — Upload & Verify Package Integrity
Upload `colab_transfer` (e.g., to `/content/AEGIS` or `/content/drive/MyDrive/AEGIS`) and verify every file against `SHA256SUMS.txt`:

```bash
cd /content/AEGIS
sha256sum -c SHA256SUMS.txt
```

On Windows PowerShell, verify `SHA256SUMS.txt` using:
```powershell
Get-Content "D:\Projects\AEGIS\colab_transfer\SHA256SUMS.txt" | ForEach-Object {
    $parts = $_ -split "  ", 2
    $expected = $parts[0].Trim().ToUpper()
    $relPath = $parts[1].Trim()
    $actual = (Get-FileHash -Algorithm SHA256 (Join-Path "D:\Projects\AEGIS\colab_transfer" $relPath)).Hash.ToUpper()
    if ($actual -ne $expected) { throw "SHA-256 mismatch for $relPath" }
    Write-Output "OK  $relPath"
}
```

### Step 2 — Stage `features.dat` on Local Colab NVMe Storage (When Approved)
Place `features.dat` (`14,636,881,920` bytes) alongside `cache_spec.json`, either at the default relative path:
```text
data/audio_dev/processed/audio_feature_cache/features.dat
data/audio_dev/processed/audio_feature_cache/cache_spec.json
```
or on local Colab NVMe disk (`/content/audio_feature_cache/`):
```bash
mkdir -p /content/audio_feature_cache
cp data/audio_dev/processed/audio_feature_cache/cache_spec.json /content/audio_feature_cache/
# Copy features.dat from Google Drive to local Colab SSD:
cp /content/drive/MyDrive/AEGIS_data/features.dat /content/audio_feature_cache/features.dat
```

### Step 3 — Run Read-Only Smoke Test (1 Train Batch + 1 Val Batch, No Outputs Written)
```bash
PYTHONPATH=/content/AEGIS python -m src.audio_train_colab \
  --config configs/audio_transformer_colab_v1.json \
  --cache-dir /content/audio_feature_cache \
  --smoke-test
```

### Step 4 — Run Reproducible Training (Only When Explicitly Approved)
```bash
PYTHONPATH=/content/AEGIS python -m src.audio_train_colab \
  --config configs/audio_transformer_colab_v1.json \
  --cache-dir /content/audio_feature_cache \
  --train
```
Outputs will be written exclusively to:
* `models/audio_transformer_colab_v1/best_audio_transformer.pt`
* `models/audio_transformer_colab_v1/last_audio_transformer.pt`
* `results/audio_transformer_colab_v1/run_config.json`
* `results/audio_transformer_colab_v1/training_history.csv`
* `results/audio_transformer_colab_v1/training_summary.json`
