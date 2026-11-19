# AEGIS Audio Baseline Report

**Report status:** Initial baseline documented
**Checkpoint epoch:** 1
**Evaluation device:** CPU
**Purpose:** Record the existing audio model, dataset configuration, metrics, and known limitations.

## 1. Existing checkpoint

- Path: `models/audio_transformer/best_audio_transformer.pt`
- Checkpoint epoch: 1
- Model: `AEGISAudioTransformer`
- Parameters: 875,906
- Input: two-channel Log-Mel + CQT features
- Input shape per window: `(2, 128, 257)`
- Classes: bonafide = 0, spoof = 1
- Optimizer recorded in checkpoint: AdamW
- Learning rate recorded in checkpoint: 0.0003
- Weight decay: 0.0001

The checkpoint is preserved. No retraining was performed for this report.

## 2. Feature and window configuration

- Sample rate: 16,000 Hz
- Window duration: 2.56 seconds
- Window hop: 1.28 seconds
- FFT size: 1,024
- Hop length: 160
- Mel bands: 128
- Mel frequency range: 20–8,000 Hz
- CQT minimum frequency: 65.4 Hz
- CQT bins: 128
- CQT bins per octave: 24
- Target time frames: 257
- Cached feature storage: float16
- Cache windows: 111,236
- Training windows: 55,630
- Development windows: 55,606

The full cache and window index were checked for expected sizes and alignment. The corrected validation used the original full-cache positions for development windows: 55,630–111,235.

## 3. Corrected development validation

Results below were recomputed from the saved window-level prediction CSV.

| Metric | Result |
|---|---:|
| Windows | 55,606 |
| Weighted cross-entropy loss | 0.665171 |
| Accuracy | 57.5208% |
| Balanced accuracy | 63.6549% |
| Spoof precision | 0.944991 |
| Spoof recall | 0.559466 |
| Spoof F1 | 0.702832 |
| ROC-AUC (spoof positive) | 0.696825 |
| Average precision (spoof positive) | 0.949187 |

### Confusion matrix

Rows are actual labels; columns are predicted labels. Label order is bonafide, spoof.

|  | Predicted bonafide | Predicted spoof |
|---|---:|---:|
| Actual bonafide | 4,052 | 1,626 |
| Actual spoof | 21,995 | 27,933 |

The prediction CSV was independently checked: 55,606 rows, 55,606 unique cache positions, no duplicate window IDs, and window IDs and labels matching the original index at those positions.

Average precision should be interpreted cautiously because spoof is the majority class in this development split. The model misses many spoof windows at its current classification threshold.

## 4. Separate evaluation split

Previously audited utterance-level evaluation results:

- Evaluation utterances: 71,237
- ROC-AUC (spoof positive): 0.606989
- Approximate equal error rate: 42.5947%
- Average precision: 0.927871

These results are from a separate evaluation split and must not be used to tune model choices or thresholds. Average precision is affected by the high spoof prevalence in this split.

The lower evaluation ROC-AUC relative to development ROC-AUC suggests a generalization gap that warrants investigation. The two evaluations use different aggregation levels (development windows versus evaluation utterances), so their metrics are not a strictly like-for-like comparison.

## 5. Notebook and cached-loader findings

The notebook contains a cached dataset implementation whose `__getitem__` uses the subset index directly to read the full feature cache. When the development index is reset to start at zero, that implementation pairs development metadata with features from the start of the cache. The cached loader therefore requires correction before it is used for training or validation.

The recorded execution counts show that the original training cell used `train_loader` and `val_loader`. Cell 37 creates these loaders from `AEGISAudioDataset`, which loads audio and extracts features directly. Cell 102 creates separate `cached_train_loader` and `cached_val_loader` without reassigning those names. Based on this recorded execution order, the cached indexing defect is not sufficient evidence that the original checkpoint was trained with misindexed cached validation data.

The original checkpoint's classification metrics match the corrected cached validation metrics at the displayed precision. This agreement does not prove the original pipeline was correct; the corrected validation and loader execution history are documented separately.

## 6. Configuration discrepancy

`configs/audio_baseline_config.json` records `AEGISAudioClassifier`, Adam, and learning rate 0.001. The actual checkpoint records `AEGISAudioTransformer` and was trained with AdamW at learning rate 0.0003. The JSON configuration should not be treated as an exact record of the checkpoint's training setup until reconciled.

## 7. Preserved artifacts

This validation and report did not retrain the model or overwrite the original checkpoint, notebooks, or existing evaluation predictions.

Corrected validation outputs:

- `results/audio_transformer_validation_corrected/dev_window_predictions.csv`
- `results/audio_transformer_validation_corrected/validation_summary.json`

Reference checkpoint:

- `models/audio_transformer/best_audio_transformer.pt`

## 8. Current conclusion and next work

The existing checkpoint provides a reproducible reference baseline, but current results do not establish deployment readiness. Spoof recall is limited, and the separate evaluation ROC-AUC is modest.

Before another training run:
1. Correct the cached dataset's full-cache position mapping.
2. Reconcile the baseline JSON configuration with the actual checkpoint configuration.
3. Create a reproducible training script/configuration with distinct checkpoint and result paths.
4. Use only the development split for model selection.
5. Keep the separate evaluation split reserved for final assessment.

No retraining or model selection was performed as part of this report.
