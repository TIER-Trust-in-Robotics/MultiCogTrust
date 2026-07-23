# CMU-MOSEI Benchmark — Distilled MemoCMT (audio + text)

Benchmarks the lightweight MemoCMT port (DistilBERT + DistilHuBERT, frozen
encoders, ~6M trainable fusion params) on CMU-MOSEI using only its native
audio + transcripts. This is the audio+text reference point to beat once the
visual branch is added.

## Data source

The official CMU server (`immortal.multicomp.cs.cmu.edu`) is dead. We use the
MMSA raw distribution mirrored on Hugging Face:

- `huggingface.co/datasets/tamb2203579/CMU-MOSEI`
  - `label.csv` — 22,856 segments: `video_id, clip_id, text, label (-3..3),
    annotation, mode` with the **standard folds** (16,326 train / 1,871 valid /
    4,659 test)
  - `Raw.zip` — 25.4 GB of per-segment video clips (`Raw/{video_id}/{clip_id}.mp4`);
    also the source material for the future visual branch

Place both under `data/MOSEI/` (gitignored).

## Pipeline

```bash
# 1. Extract 16 kHz mono wavs from the clips (idempotent, resumable)
uv run python src/mosei_dataset.py --extract-audio --write-metadata

# 2. Precompute frozen-encoder embeddings into fp16 shards (~once, resumable)
uv run python src/train_cross_modal_mosei.py --precompute

# 3. Train the fusion head and report test metrics
uv run python src/train_cross_modal_mosei.py --epochs 20
```

Artifacts land in `data/mosei_aligned_features/` (metadata + embedding shards)
and `models/mosei/` (checkpoint + metrics JSON).

## Protocol

- Standard MOSEI folds from `label.csv` (`mode` column) — no speaker overlap
  handling needed beyond the published split.
- 7-class cross-entropy on `round(sentiment) + 3`, batch 1, Adam 1e-4, `min`
  fusion aggregation (matching the IEMOCAP MemoCMT setup).
- Audio capped at 8 s per clip for the frozen embedding pass.
- Model selection on validation Acc-7.

## Metrics (MOSEI conventions)

| Metric | Meaning |
|---|---|
| Acc-7 | 7-class accuracy on rounded sentiment |
| Acc-2 / F1 (non0) | pos vs neg, clips with label = 0 excluded; pred positive iff class value > 0 |
| Acc-2 / F1 (has0) | non-negative vs negative, all clips |
| MAE | \|predicted class value − continuous label\| |

Published trimodal (audio+text+vision, fine-tuned) systems report Acc-2 in the
low-to-mid 80s on MOSEI; audio+text ablations typically run a couple of points
lower. Our frozen-encoder distilled setup is a deliberate lower bound — the
comparison to watch is *this number vs. the same model + visual branch*.
