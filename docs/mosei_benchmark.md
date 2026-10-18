# CMU-MOSEI benchmark and data audit

Updated 2026-10-12. MOSEI is the second model-quality benchmark and the dataset
for a three-modality video experiment. Its native task is **continuous sentiment
in [-3, 3]**, separate from the MVP's four IEMOCAP emotions. Positive sentiment
must not become happiness, or negative sentiment anger/sadness.

The historical seven-class audio/text checkpoint reported 83.04% test accuracy
including zero-valued labels, but 66.92% nonzero accuracy. Neither number alone
establishes competitiveness. Its test was already evaluated historically.
This quality pass uses development splits and runs **no new test predictions**.

## Local data and integrity

`data/MOSEI/` contains `label.csv`, per-segment `Raw/` videos and extracted
`wav/` audio, about 31 GB. Its 22,856 rows comprise 16,326 train, 1,871 valid
and 4,659 test. Every row has a nonempty video file and a readable, nonempty
16 kHz mono WAV header. All WAV files were hashed. This does not establish
full video decodability or semantic alignment.

The source was previously documented as an
[MMSA mirror on Hugging Face](https://huggingface.co/datasets/tamb2203579/CMU-MOSEI).
Comparison against the pinned
[CMU SDK folds](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK/blob/4f2eadcd7e7b9e20e83b868cdad385b41830285c/mmsdk/mmdatasdk/dataset/standard_datasets/CMU_MOSEI/cmu_mosei_std_folds.py)
found:

- Four training clips from `-NFrJFQijFE` are outside the official training list;
  official training video `HuIKyKkEL0Q` is absent. Validation's 300 video IDs
  match. Official test videos `7l3BNtSE0xc` and `dZFV0lyedX4` are absent locally.
- Three training WAVs are byte-identical to test WAVs under different video IDs.
  Quarantine covers all five clips from the affected training source videos,
  `VN1IPN1TL3Q` and `i5bmKlHv-SA`, including neighboring clips.
- The generated `mosei_clean_v2` view excludes **nine training clips**: four fold
  outsiders and five from duplicated source videos. It contains **16,317 train
  / 1,871 valid** examples. No raw files or validation/test examples are removed.
  This is a modified development protocol, not an exact published split.
- There are 37 cross-split groups of identical normalized transcripts. A
  targeted audio check of 19 long-transcript pairs found a **validation/test
  near-duplicate candidate**, `jrDqduyQrfo/1` versus `WbtsuXkaGeg/1`: normalized
  correlation 0.98464 with a 7.625 ms shift. Week-one video review confirms a
  shared prerecorded introduction. Sampled grayscale frame correlations are
  0.981–0.992; the other three duplicate pairs have identical sampled frames.
  The surrounding Expert Village clips cover different topics with the same
  visually matching presenter. This review does not prove other recordings
  or speakers are independent.

The new raw-data view, **`mosei_sdk_intersection_source_deduplicated_v3`**, retains
the nine training exclusions and conservatively excludes all four validation
clips from `jrDqduyQrfo`. Its development counts are **16,317 train / 1,867 valid**.
Test records and raw files are preserved. Decisions use source evidence only.
The report and manifest are under `week1_v1/source_resolution/`. The original
MMSA feature references retain **16,326 train / 1,871 valid**, with the known
source-overlap caveat explicitly attached.

A seeded audit decoded 12 train and 12 validation clips plus the four fold
outsiders. All 28 WAVs exactly matched audio decoded from their videos. Full-clip
tiny.en transcription had median word error rate 0.1625 on the random 24;
two random examples and all four outsiders exceeded 0.75. These flag review
candidates; ASR disagreement alone does not justify relabeling or exclusion.

The old eight-second audio cap truncates **740/1,871 validation clips (39.6%)**,
while text can describe the entire utterance. Compare full-utterance research
inputs and synchronized short deployment windows separately.

Generated evidence under `data/processed/benchmarks/`:

- `mosei_native/audit.json`: original failed integrity audit, preserved.
- `mosei_clean_v2/`: source-level exclusions, development manifest, checksums,
  MSP inventory and `near_duplicate_audit.json`.
- `mosei_media_sample_v1/`: sampled IDs, paired-media checks and ASR disagreement.
- `week1_v1/source_resolution/`: resolved recording groups, video/audio evidence,
  four additional validation exclusions and the v3 development contract.
- Earlier views and predictions remain preserved. Use the week-one v3 manifest
  for new raw-data development; use the unchanged released feature split for
  the separate MMSA references.

## Metrics and development results

`src/benchmark_metrics.py` follows the
[MMSA regression definitions](https://github.com/thuiar/MMSA/blob/a94e65d07fa1ae0d44e552390074b29b0898edfd/src/MMSA/utils/metricsTop.py).
Report all metrics and counts:

| Metric | Definition |
|---|---|
| MAE | Continuous-score absolute error; no rounding or clipping |
| Pearson r | Continuous-score correlation; null for a constant vector |
| Acc-7 | Predictions clipped to [-3, 3], then predictions/labels rounded with NumPy's tie convention |
| Acc-2 / weighted F1, non0 | Exactly-zero labels excluded; positive iff score > 0 |
| Acc-2 / weighted F1, has0 | All labels included; positive iff score >= 0 |

The lexical control fits a 30,000-feature TF-IDF vocabulary on training only,
then ridge regression on continuous sentiment. Minimum validation MAE selects
alpha 1 from `[0.1, 1, 10, 100]`. It uses the modified 16,317-example training
view; comparison with the old model is a development diagnostic, not a fully
matched architecture ablation.

| Full validation, 1,871 clips | MAE ↓ | Pearson r ↑ | Acc-7 | Acc-2 non0 | Weighted F1 non0 | Acc-2 has0 |
|---|---:|---:|---:|---:|---:|---:|
| Training-mean constant | 0.7778 | undefined | 44.63% | 64.81% | 0.5097 | 72.96% |
| TF-IDF + ridge, source-deduplicated training | 0.6406 | 0.5979 | 46.07% | 76.91% | 0.7683 | 75.09% |
| Legacy checkpoint, argmax class value | 0.5991 | 0.6553 | 53.71% | 63.49% | 0.6302 | 83.11% |
| Same checkpoint, expected sentiment score | 0.5476 | 0.7131 | 53.13% | 84.49% | 0.8468 | 79.48% |

The full legacy replay reports its original argmax class value and the expected
score over its seven probabilities. Expected-score projection is a diagnostic
of a CE-trained model, not a retrained regressor. See
`mosei_legacy_validation_v1/summary.json`. Encoder revisions are pinned for the
replay; historical revisions were not recorded, so agreement with the archived
validation metrics is checked rather than assumed. **All six archived validation
metrics reproduce exactly.** Its training used the unfiltered mirror; its
validation set is identical to the earlier v2 view. The expected-score gain in MAE
and nonzero classification comes with lower has0 accuracy and slightly lower
Acc-7. Do not describe it as an improvement on every metric or a new model.

## External comparison targets

These are **reported MOSEI test results from MMSA**, not local reproductions or
validation targets to rank against the table above. They use released feature
and training configurations and do not establish the latest SOTA. MAE and
correlation below are converted from the repository table's ×100 scale.

| Reference | Non0 Acc-2 | Non0 F1 | MAE | Pearson r | Role |
|---|---:|---:|---:|---:|---|
| LMF | 83.48% | 83.36% | 0.5757 | 0.7169 | Lightweight fusion comparator |
| MulT | 84.63% | 84.52% | 0.5593 | 0.7331 | Cross-attention comparator |
| Self-MM | 85.15% | 84.90% | 0.5309 | 0.7649 | Stronger established reference |

Source: [pinned MMSA results](https://github.com/thuiar/MMSA/blob/a94e65d07fa1ae0d44e552390074b29b0898edfd/results/result-stat.md).
Before a direct comparison, verify feature-file hashes, segment IDs, labels,
modalities, encoder training status and selection budgets. Separate published
feature reproduction from training comparators on our live-compatible features.
An official-feature result does not validate raw webcam/microphone inference.

## Week-one reference execution

The 13,652,131,313-byte `unaligned_50.pkl` has been downloaded and verified
against the [official MMSA checksum](https://github.com/thuiar/MMSA/wiki/Home/6d73e57a5c6af75737d719aac1e5e4f9a1c19dec):
`ad8b23d50557045e7d47959ce6c5b955d8d983f2979c7d9b7b9226f6dd6fec1f`.
The original public Drive download was quota-limited. A byte-identical public
mirror was pinned at commit `638362ecac5d7858e0bcd1e4129f2a8f56a857f3`;
`week1_v1/feature_download.json` records provenance.

Only training and validation arrays are exported to memory maps. The complete
release necessarily materializes test arrays during restricted NumPy pickle
deserialization; those arrays are immediately discarded without model inference.
All exported features are float32, with no quantization. Actual sequence lengths
are **50 text / 500 audio / 500 vision**, although the released config lists
375 vision steps. The upstream loader derives lengths from the arrays; these
runs preserve the complete arrays. All exported feature values are finite, and
all development IDs, labels and splits match the local CSV.

LMF and MulT use the pinned MMSA model code under `src/models/mmsa_reference/`,
with its MIT license and an explicit file/change manifest. MulT's future mask
is moved to the input device for MPS. Seeds are 42/43/44, one released config
per seed, stopping after eight epochs without improvement. Selection preserves
upstream shuffled-validation mean-batch MAE rounded to four decimals; ordered
full-set metrics are reported after checkpoint reload. The harness preserves
LMF's omitted fusion-weight/bias optimizer parameters and MulT's ignored weight
decay, summed eight-batch gradients and discarded incomplete accumulation group.
These are documented reference behaviors, not fixes or new improvements.

Training is **complete**; the checked three-seed results are appended below.
No superiority claim follows from validation versus published test numbers. Jobs, durable logs, epoch resume
states and predictions are under `week1_v1/mosei_lmf/` and `mosei_mult/`.
Run `uv run python src/summarize_week1.py` for completed results and explicit
missing runs. `--require-complete` fails until every planned run and replay passes.

On the new 1,867-example raw-data validation subset, rescoring existing
predictions gives expected-score MAE **0.5480**, r **0.7132**, versus lexical
MAE **0.6412**, r **0.5978**. This changes the evaluation subset only. Historical
model selection used the earlier 1,871-example validation set.

## Reproduce this quality pass

Audits and training reject overwriting existing outputs; choose a new output
path to repeat a completed run. Checkpoint replay resumes an unchanged protocol.

```bash
uv run python src/fetch_benchmark_sources.py
uv run python src/audit_benchmark_data.py --quarantine-training-conflicts \
  --output data/processed/benchmarks/mosei_clean_v2
uv run python src/audit_mosei_duplicates.py
uv run python src/audit_mosei_media.py
uv run python src/benchmark_controls.py mosei \
  --output data/processed/benchmarks/mosei_lexical_v3
uv run python src/evaluate_mosei_checkpoint.py --device mps \
  --manifest data/processed/benchmarks/mosei_clean_v2/development.jsonl \
  --output data/processed/benchmarks/mosei_legacy_validation_repeat
uv run python -m pytest -q tests/test_benchmark_protocols.py
```

`configs/benchmark_quality_v1.json` specifies tasks, tuning budgets, seeds and
encoder revisions. `configs/benchmark_sources.json` pins protocol references by
commit and SHA-256; those fold/config references are parsed as data. The
separately reviewed MMSA models are vendored with provenance and a license.
`configs/week1/protocol_matrix.json` freezes the new model matrix and test
exposure record; per-model/per-seed configs are alongside it.
No new dependencies were added for this quality pass.

Week-one reference commands (completed outputs are preserved; MMSA trainers
support `--resume` for an interrupted epoch-boundary state):

```bash
uv run python src/fetch_mmsa_features.py
uv run python src/mmsa_features.py
uv run python src/run_week1_experiments.py --group mosei_lmf --device cpu
uv run python src/run_week1_experiments.py --group mosei_mult --device mps \
  --after data/processed/benchmarks/week1_v1/mosei_lmf/job.json \
  --after data/processed/benchmarks/week1_v1/iemocap_asr/job.json
uv run python src/summarize_week1.py --require-complete
```

On this checkout the feature export and controllers already exist. Read each
group's `job.json` and log instead of starting a second live controller.
Feature export intentionally rejects an existing completed index. For isolated
reference repeats, call `src/train_mmsa_reference.py` with the frozen config and
a new `--output` directory. The same rule applies to IEMOCAP repeats with
`src/train_mvp_baseline.py`. The finalizer waits for all groups, checks complete
artifacts, runs the full test suite and appends computed results to these docs.

## Historical trainer and MSP-Podcast

`src/train_cross_modal_mosei.py` is retained for historical context. It rounds
continuous labels into seven classes, selects validation Acc-7, automatically
evaluates test and overwrites its output path. Its old fp16 embedding shards
are absent locally. Update its task, provenance and test boundary before using
that CLI for new development.

`docs/MSP-Podcast/` has release-2.0 metadata for 264,705 labeled/transcribed
examples: 169,190 Train, 34,399 Development, 46,294 Test1 and 14,822 Test2.
No audio is present there or under the repo's data folders. The
[official access page](https://www.lab-msp.com/MSP/MSP-Podcast.html) specifies an
institutional academic agreement; no public audio-download SDK was found.
Restore authorized audio access before acoustic evaluation. Preserve release-2.0
partitions and native labels. Older local MSP scores used a different processed
subset and cannot stand in for a current reproduction.

<!-- WEEK1_RESULTS_START -->
## Completed week-one original-feature references

All rows use the original released 16,326-example training / 1,871-example validation split and seeds 42/43/44. Values are mean ± sample SD. The known recording-overlap caveat remains attached; no new test predictions were made.

| Reference | MAE | Pearson r | Acc-7 | Acc-2 non0 | F1 non0 | Acc-2 has0 |
|---|---:|---:|---:|---:|---:|---:|
| MMSA LMF | 0.5390 ± 0.0066 | 0.7147 ± 0.0081 | 53.79 ± 0.12% | 83.29 ± 1.55% | 0.8336 ± 0.0137 | 78.78 ± 3.33% |
| MMSA MulT | 0.5371 ± 0.0046 | 0.7163 ± 0.0069 | 53.64 ± 0.43% | 84.10 ± 0.41% | 0.8403 ± 0.0029 | 79.99 ± 0.30% |

Every selected reference checkpoint reproduces its saved validation predictions after reload. Validation-only selection follows the pinned release behavior; original-feature scores are separate from the modified raw-data results and published test scores. These runs establish reproducible references, without a latest-SOTA or live-camera quality claim.

Checked results: `data/processed/benchmarks/week1_v1/summary.json`. Final test-suite evidence and source hashes: `week1_v1/finalization/verification.json`.
<!-- WEEK1_RESULTS_END -->
