# MSP-PODCAST Sliding-Frame Fusion Findings

## Dataset Summary

- Dataset: `/Users/magi/Code/MultiCogTrust/data/msp_aligned_features`
- Processed examples: 159697 of 173808 discovered Train/Test1 candidates
- Train examples used: 123370
- Test examples used: 36327
- Labels: anger, sadness, happiness, surprise, fear, disgust, contempt, neutral
- Split counts in artifact: {'test': 36327, 'train': 123370}
- Train label counts: A: 20055, C: 2662, D: 1283, F: 775, H: 33092, N: 46317, S: 16123, U: 3063
- Test label counts: A: 6744, C: 1036, D: 744, F: 346, H: 10847, N: 12397, S: 3015, U: 1198
- Invalid alignment spans skipped during preprocessing: 14111
- Feature extraction failures: 0
- Words without audio overlap: 92
- Same-normalized transcript/alignment ratio: 0.6236247393501443

## Model Result

| Model | Test accuracy | Macro F1 | Best epoch | Train loss | Test loss | Metrics file |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Sliding-frame fusion | 0.5316 | 0.2650 | 5 | 1.0787 | 1.3347 | /Users/magi/Code/MultiCogTrust/models/baselines/msp_sliding_frame_fusion_metrics.json |

## Per-Class Notes

- Strongest class by F1: happiness (0.612)
- Weakest class by F1: surprise (0.000)
- Largest confusions: neutral -> sadness: 2289; neutral -> happiness: 2204; happiness -> neutral: 1789; happiness -> anger: 1596; neutral -> anger: 1458

## Conclusion

This run trains the MSP counterpart of the best IEMOCAP fusion variant: frozen DistilBERT pooled text plus a GRU over raw sliding-window `audio_frame_features`.

## Visualization Artifacts

- Training curves: `msp_training_curves.png`
- Per-class F1: `msp_per_class_f1.png`
- Normalized confusion matrix: `msp_confusion_matrix.png`
