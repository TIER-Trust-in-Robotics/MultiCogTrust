# IEMOCAP Baseline Findings

## Dataset Summary

- Dataset: `data/iemocap_aligned_features`
- Train examples: 7867
- Test examples: 2170
- Split: Sessions 1-4 train, Session 5 test
- Labels: anger, happiness, excitement, neutral, sadness, frustration, fear, surprise, disgust, undefined, other
- Feature source: aligned IEMOCAP metadata with token-aligned word prosody and temporal `audio_frame_features`.

## Headline Findings

The best model in this comparison is **Fine-tuned simple fusion** with macro F1 0.3190 and accuracy 0.4336. This is a small but measurable improvement over the fine-tuned DistilBERT text benchmark, which reached macro F1 0.3065 and accuracy 0.4346.

Macro F1 ranking: Fine-tuned simple fusion (0.3190), Fine-tuned text (0.3065), Cross-attention fusion (0.3026), Fine-tuned sliding fusion (0.3012), Sliding-frame fusion (0.2897), Simple fusion (0.2752), Text-only (0.2510), Audio-only (0.1866).

The new fine-tuned fusion runs initialize the DistilBERT text branch from `models/baselines/text_finetuned_distilbert.pt` and continue training with a smaller text learning rate than the audio/projection layers. They also use class-weighted cross entropy to target macro F1 on the imbalanced 11-label setup, so their train/test loss values are not directly comparable to the earlier unweighted frozen baselines.

The strongest result came from the simplest fine-tuned fusion architecture: pooled fine-tuned DistilBERT token embeddings plus pooled token-aligned prosody, concatenated into an MLP. The more temporal variants did not improve in this run. Fine-tuned sliding-frame fusion peaked early at macro F1 0.3012, and cross-attention reached macro F1 0.3026.

## Model Comparison

| Model | Test accuracy | Macro F1 | Best epoch | Train loss | Test loss | Metrics file |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Text-only | 0.3724 | 0.2510 | 5 | 1.5901 | 1.6419 | `/Users/magi/Code/MultiCogTrust/models/baselines/text_only_metrics.json` |
| Fine-tuned text | 0.4346 | 0.3065 | 3 | 1.0810 | 1.5700 | `/Users/magi/Code/MultiCogTrust/models/baselines/text_finetuned_metrics.json` |
| Audio-only | 0.3244 | 0.1866 | 20 | 1.6244 | 1.7185 | `/Users/magi/Code/MultiCogTrust/models/baselines/audio_only_metrics.json` |
| Simple fusion | 0.4290 | 0.2752 | 5 | 1.4286 | 1.5128 | `/Users/magi/Code/MultiCogTrust/models/baselines/simple_fusion_metrics.json` |
| Fine-tuned simple fusion | 0.4336 | 0.3190 | 3 | 0.7364 | 1.8890 | `/Users/magi/Code/MultiCogTrust/models/baselines/simple_fusion_finetuned_text_metrics.json` |
| Sliding-frame fusion | 0.4318 | 0.2897 | 5 | 1.3961 | 1.5098 | `/Users/magi/Code/MultiCogTrust/models/baselines/sliding_frame_fusion_metrics.json` |
| Fine-tuned sliding fusion | 0.4166 | 0.3012 | 1 | 1.0124 | 1.6976 | `/Users/magi/Code/MultiCogTrust/models/baselines/sliding_frame_fusion_finetuned_text_metrics.json` |
| Cross-attention fusion | 0.4111 | 0.3026 | 5 | 0.5142 | 2.1575 | `/Users/magi/Code/MultiCogTrust/models/baselines/cross_attention_fusion_metrics.json` |

## What Changed In This Run

- Added reusable class-weighted loss and differential learning-rate helpers in `src/iemocap_baseline_training.py`.
- Added `TextAudioCrossAttentionClassifier` in `src/models/baselines.py`.
- Added fine-tuned fusion CLIs:
  - `src/train_simple_fusion_finetuned_iemocap.py`
  - `src/train_sliding_frame_fusion_finetuned_iemocap.py`
  - `src/train_cross_attention_fusion_iemocap.py`
- Regenerated visual artifacts and notebook:
  - `models/baselines/baseline_metric_comparison.png`
  - `models/baselines/baseline_training_curves.png`
  - `models/baselines/baseline_per_class_f1.png`
  - `models/baselines/baseline_confusion_matrices.png`
  - `models/baselines/iemocap_baseline_visual_report.md`
  - `models/baselines/iemocap_baseline_visualizations.ipynb`

## Sliding-Window Embedding Check

The token-overlap reconstruction from sliding-window frames matched the existing word-aligned token features for 3/256 sampled records. The maximum absolute difference was 0.000488281, with 0 token-mask mismatches. Diagnostic conclusion: token-overlap pooling from sliding frames differs from existing `word_audio_features`, so the sliding-frame fusion runs are the comparison for the alternate sliding-window embedding path.

## Per-Model Notes

### Text-only

- Strongest class by F1: sadness (0.431)
- Weakest class by F1: disgust (0.000)
- Largest confusions: undefined -> frustration: 131; excitement -> undefined: 100; neutral -> frustration: 90; neutral -> undefined: 81; sadness -> frustration: 71

### Fine-tuned text

- Strongest class by F1: sadness (0.516)
- Weakest class by F1: disgust (0.000)
- Largest confusions: undefined -> frustration: 109; neutral -> undefined: 93; neutral -> frustration: 90; excitement -> undefined: 73; undefined -> neutral: 65

### Audio-only

- Strongest class by F1: sadness (0.450)
- Weakest class by F1: fear (0.000)
- Largest confusions: undefined -> neutral: 197; frustration -> undefined: 126; frustration -> neutral: 120; excitement -> neutral: 115; neutral -> undefined: 87

### Simple fusion

- Strongest class by F1: sadness (0.530)
- Weakest class by F1: disgust (0.000)
- Largest confusions: undefined -> frustration: 125; neutral -> undefined: 118; excitement -> undefined: 113; neutral -> frustration: 81; frustration -> undefined: 62

### Fine-tuned simple fusion

- Strongest class by F1: sadness (0.553)
- Weakest class by F1: disgust (0.000)
- Largest confusions: undefined -> happiness: 94; undefined -> neutral: 81; undefined -> sadness: 72; undefined -> frustration: 65; excitement -> happiness: 61

### Sliding-frame fusion

- Strongest class by F1: sadness (0.559)
- Weakest class by F1: disgust (0.000)
- Largest confusions: excitement -> undefined: 134; neutral -> undefined: 105; frustration -> undefined: 91; undefined -> frustration: 89; undefined -> neutral: 71

### Fine-tuned sliding fusion

- Strongest class by F1: sadness (0.541)
- Weakest class by F1: disgust (0.000)
- Largest confusions: undefined -> frustration: 76; undefined -> neutral: 68; undefined -> sadness: 65; frustration -> anger: 64; undefined -> anger: 60

### Cross-attention fusion

- Strongest class by F1: sadness (0.553)
- Weakest class by F1: disgust (0.000)
- Largest confusions: undefined -> neutral: 98; undefined -> sadness: 74; frustration -> neutral: 71; neutral -> undefined: 68; undefined -> frustration: 66

## Interpretation

Fine-tuned DistilBERT remains the strongest single-modality benchmark by accuracy, but the fine-tuned simple fusion model is now the best macro-F1 model. The gain is modest: +0.0125 macro F1 over fine-tuned text and -0.0009 accuracy. That makes the prosody signal real enough to keep, but not strong enough to justify a more complex temporal model without more targeted regularization or label handling.

The frozen fusion results show that prosody helps over frozen text-only (+0.0242 macro F1 for simple fusion, +0.0387 for sliding-frame fusion). The fine-tuned results narrow that story: once the text encoder is allowed to adapt, only the pooled token-aligned prosody model improves over text. The temporal GRU and cross-attention variants did not beat the simple pooled fusion in this run.

The low or zero F1 on disgust, other, and sometimes fear is partly a support problem. Disgust and other have no Session 5 support in the current test split, so macro F1 is harsh and should be interpreted with that label-distribution caveat. The large remaining errors concentrate among undefined, frustration, neutral, and excitement.

## Recommended Next Experiment

The next run should keep the fine-tuned text initialization and the class-weighted objective, but improve the multimodal setup before adding more architecture:

1. Filter or remap `xxx`/`oth` and rerun the benchmark on the common IEMOCAP emotion subset.
2. Add early stopping on validation macro F1 instead of using Session 5 as the only evaluation target.
3. Run class-balanced or focal loss for fine-tuned text and fine-tuned simple fusion under the same objective, so the comparison is controlled.
4. Then retry cross-modal attention with stronger regularization and a validation split. The current cross-attention result is close to fine-tuned text, but not better than pooled fusion.

