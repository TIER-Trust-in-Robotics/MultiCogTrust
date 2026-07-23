# Audio-Only and Text-Only Baselines

The goal is to measure how much emotion signal each modality
has on its own, using the same train/test split and label mapping as the aligned
IEMOCAP artifacts.


## Dataset

Load them with:

```python
from src.iemocap_aligned_dataset import load_iemocap_aligned

train_ds, test_ds, num_labels, label_names = load_iemocap_aligned()
```

The split is already correct:

- Sessions 1-4: train
- Session 5: test

Keep the existing 11-label IEMOCAP mapping:

```text
ang, hap, exc, neu, sad, fru, fea, sur, dis, xxx, oth
```

Important dataset columns:

```text
text                         transcript string
label                        integer class id
label_code                   original IEMOCAP label
audio_frame_features         T x 88 sliding-window prosody features
audio_frame_start_sec        T frame start times
audio_frame_end_sec          T frame end times
word_audio_features          num_words x 88 word-aligned prosody features
word_audio_feature_mask      num_words mask
words                        forced-alignment word metadata
```

For the audio-only baseline, start with `audio_frame_features`, which uses the prosodic feature over the entire utterance. Eventually move onto `word_audio_features` which extracts features for each work in the utterance.

For the text-only baseline, use only `text` and `label`.

## Baseline 1: Text Only

### Purpose

Measure how well transcript text predicts the IEMOCAP emotion label without any
prosodic features.

### Model

Use DistilBERT as the text encoder:

```text
text -> tokenizer -> DistilBERT -> pooled text vector -> classifier
```

Recommended first implementation:

- Hugging Face model: `distilbert-base-uncased`
- Max sequence length: `128`
- Pooling: use the first token hidden state, or masked mean pooling
- Classifier: linear layer from DistilBERT hidden size to `num_labels`
- Loss: `torch.nn.CrossEntropyLoss`

Start by freezing DistilBERT and training only the classifier head. If this
works, add a flag to unfreeze DistilBERT for fine-tuning.

### Expected Batch Format

The dataloader should produce:

```python
{
    "input_ids": LongTensor[batch, max_tokens],
    "attention_mask": LongTensor[batch, max_tokens],
    "labels": LongTensor[batch],
}
```

### Metrics

Report at least:

- train loss
- test loss
- test accuracy
- macro F1
- per-class precision/recall/F1
- confusion matrix

Macro F1 matters because IEMOCAP labels are imbalanced.

## Baseline 2: Audio Only

### Purpose

Measure how well temporal prosodic features predict the IEMOCAP emotion label
without transcript text.

### Model

Use the temporal sequence of 88-dim OpenSMILE/eGeMAPS features:

```text
audio_frame_features: T x 88
```

Recommended first implementation:

```text
T x 88 -> input projection -> temporal encoder -> masked pooling -> classifier
```

Good first model:

- Input projection: `Linear(88, hidden_dim)`
- Temporal encoder: 1-layer GRU or 1-layer Transformer encoder
- Hidden size: `128` or `256`
- Dropout: `0.1` to `0.3`
- Pooling: masked mean over valid time steps
- Classifier: MLP or linear layer to `num_labels`
- Loss: `torch.nn.CrossEntropyLoss`

Start with a GRU because the sequence lengths are short and variable. A
Transformer encoder is a useful follow-up once the GRU version works.

### Padding and Masking

`audio_frame_features` is variable length. The collate function must pad each
batch to the longest sequence in that batch.

Expected batch tensors:

```python
{
    "audio_features": FloatTensor[batch, max_frames, 88],
    "audio_mask": BoolTensor[batch, max_frames],
    "labels": LongTensor[batch],
}
```

`audio_mask` should be `True` for real frames and `False` for padding.

Masked mean pooling example:

```python
mask = audio_mask.unsqueeze(-1).float()
pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
```

### Metrics

Report the same metrics as the text-only baseline:

- train loss
- test loss
- test accuracy
- macro F1
- per-class precision/recall/F1
- confusion matrix

## Suggested Files

Use a small number of focused files:

```text
src/models/baselines.py
src/train_text_baseline.py
src/train_audio_baseline.py
```

`src/models/baselines.py` should contain reusable PyTorch modules:

- `TextOnlyClassifier`
- `AudioOnlyGRUClassifier`

The training scripts should contain:

- argument parsing
- dataset loading
- tokenization or collation
- training loop
- evaluation loop
- checkpoint saving

## Suggested Commands

Text-only:

```bash
uv run python src/train_text_baseline.py \
  --model-name distilbert-base-uncased \
  --max-length 128 \
  --epochs 5 \
  --batch-size 16 \
  --device cpu
```

Audio-only:

```bash
uv run python src/train_audio_baseline.py \
  --hidden-dim 128 \
  --epochs 20 \
  --batch-size 64 \
  --device cpu
```

## Checkpoints and Outputs

Save outputs under:

```text
models/baselines/
```

Recommended outputs:

```text
models/baselines/text_only_distilbert.pt
models/baselines/text_only_metrics.json
```
and 
```text
models/baselines/audio_only_gru.pt
models/baselines/audio_only_metrics.json
```

The metrics JSON should include:

```json
{
  "model": "text_only_distilbert",
  "dataset": "iemocap_aligned",
  "num_labels": 11,
  "label_names": ["anger", "..."],
  "train_size": 7867,
  "test_size": 2170,
  "best_epoch": 3,
  "test_accuracy": 0.0,
  "test_macro_f1": 0.0,
  "classification_report": {},
  "confusion_matrix": []
}
```


## Out of Scope

Do not add these yet:

- audio/text concatenation
- gated fusion
- cross-modal attention
- self-attention over combined audio/text embeddings
- vision features
- MSP-PODCAST training

Those are the next stages after these two baselines are working.
