# Prosodic Gate — Experiment Results

## Purpose

The prosodic gate is an optimization layer that runs on eGeMAPSv02 acoustic features extracted by OpenSMILE and decides whether an utterance is likely **non-neutral** (emotional) before committing it to Whisper transcription and the downstream NLP model. Neutral utterances that would produce no useful signal are dropped early, saving the transcription and NLP cost entirely.

The gate is a binary MLP classifier (`MLP_gate_full`: 88 → 256 → 128 → 64 → 1) trained with:
- `StandardScaler` fit on the training split
- `BCEWithLogitsLoss` with `pos_weight` to correct for class imbalance
- `ReduceLROnPlateau` LR scheduling (initial lr = 3e-4, min = 1e-5)

The operating threshold is chosen to fix **non-neutral recall ≥ 0.95** — i.e. at most 5% of truly emotional utterances are incorrectly suppressed.

---

## Dataset Comparison

All models use the same architecture and hyperparameters. Three training configurations were evaluated.

| Dataset | Train samples | Test samples | ROC-AUC | Threshold | Non-neutral Recall | Neutral Recall |
|---|---|---|---|---|---|---|
| IEMOCAP only | 7,869 | 2,170 | 0.665 | 0.814 | 0.950 | 0.091 |
| MSP-PODCAST only | 137,168 | 36,769 | **0.738** | 0.707 | 0.950 | **0.159** |
| IEMOCAP + MSP combined | 145,037 | 38,939 | 0.732 | 0.740 | 0.950 | 0.156 |

### Classification Report at Operating Threshold

**IEMOCAP only** (threshold = 0.814)
```
              precision  recall    f1   support
non-neutral      0.59    0.95   0.73     1,266
neutral          0.57    0.09   0.16       904
accuracy                        0.59     2,170
```

**MSP-PODCAST only** (threshold = 0.707)
```
              precision  recall    f1   support
non-neutral      0.69    0.95   0.80    24,312
neutral          0.62    0.16   0.25    12,457
accuracy                        0.68    36,769
```

**IEMOCAP + MSP combined** (threshold = 0.740)
```
              precision  recall    f1   support
non-neutral      0.68    0.95   0.79    25,578
neutral          0.62    0.16   0.25    13,361
accuracy                        0.68    38,939
```

### ROC Curve (MSP-PODCAST only — recommended model)

| Threshold | Non-neutral Recall | Neutral Recall |
|---|---|---|
| 1.000 | 1.000 | 0.000 |
| 0.704 | 0.947 | 0.167 |
| 0.646 | 0.895 | 0.312 |
| 0.586 | 0.834 | 0.449 |
| 0.528 | 0.765 | 0.561 |
| 0.465 | 0.686 | 0.668 |
| 0.390 | 0.594 | 0.763 |
| 0.299 | 0.476 | 0.851 |
| 0.190 | 0.323 | 0.930 |
| 0.000 | 0.000 | 1.000 |

---

## Latency Benchmark

Measured on CPU (Apple Silicon, `torch` default device). Single-sample latency simulates real-time utterance-by-utterance gating. All three models share the same architecture so latency is identical.

| Dataset | n_test | mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | batch p50 (ms) | throughput (samples/s) |
|---|---|---|---|---|---|---|---|
| IEMOCAP only | 2,170 | 0.045 | 0.044 | 0.047 | 0.050 | 0.6 | 3,512,748 |
| MSP-PODCAST only | 36,769 | 0.045 | 0.044 | 0.046 | 0.048 | 11.9 | 3,081,099 |
| IEMOCAP + MSP combined | 38,939 | 0.045 | 0.045 | 0.047 | 0.051 | 12.8 | 3,038,341 |

**Single-sample p50: 0.044 ms** — the gate adds less than 50 µs of latency per utterance, making it effectively free compared to Whisper (typically 200–2000 ms per utterance depending on model size) and any NLP inference.

---

## Key Findings

1. **MSP-PODCAST is the best single dataset.** 17× more training data (137k vs 7.9k) raises ROC-AUC from 0.665 → 0.738 and doubles the fraction of true neutrals correctly suppressed (9% → 16%) at the same non-neutral recall target.

2. **Combining datasets does not help.** Adding IEMOCAP's 7.9k samples to 137k MSP examples makes no measurable difference (AUC 0.738 → 0.732). IEMOCAP is too small relative to MSP to add signal; it only slightly shifts the scaler and threshold.

3. **The gate is latency-negligible.** At 0.044 ms p50 per utterance, the gate consumes <0.02% of the time that a `tiny.en` Whisper call would take. The optimization benefit is entirely determined by how many neutrals it correctly suppresses.

4. **Practical suppression rate at 95% non-neutral recall:** the MSP model correctly drops ~16% of neutral utterances. If a session is 30% neutral speech, the gate would suppress roughly 5% of total utterances while passing 95% of emotional ones — a modest but real reduction in downstream workload.

5. **AUC ceiling is ~0.74 for prosodic features alone.** eGeMAPS features encode *how* speech sounds acoustically; neutral utterances can acoustically resemble emotional ones depending on context. Text and/or vision modalities would be needed to push AUC significantly higher.

---

## Recommendation

Use the **MSP-PODCAST only** model at threshold **0.707**. Save both the model weights and the fitted `StandardScaler` (required for inference). The scaler must be applied to raw 88-dim eGeMAPSv02 features before passing them to the network.
