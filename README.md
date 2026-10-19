# MultiCogTrust

MultiCogTrust is a prototype for affect sensing during human–robot interaction.
The current MVP predicts **anger, happiness, sadness, and neutral** from speech
and its transcript, with trained visual branches available for comparison. The
Mac runtime captures microphone/webcam input and has a local processing inspector.
Actual-ASR validation currently selects **audio/text for the final prediction**;
the visual branch is computed and displayed separately. Trust
prediction remains a later research task requiring trust-specific supervision;
the runtime does not turn emotion probabilities into a trust score.

## Environment

Python 3.11 or newer, managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked
uv run python -m pytest -q
```

`pyproject.toml` and `uv.lock` define the environment, including data-processing
packages and pytest. The raw-data audit also needs `ffprobe` on PATH. Model
assets are downloaded from Hugging Face on first use; `--offline` requires
those assets to have been cached already. The [updated two-week roadmap](docs/async_multimodal_pipeline_roadmap.md)
starts from the completed original days 1–7 work and now prioritizes model
quality, comparable baselines and dataset integrity. The
[MOSEI benchmark audit](docs/mosei_benchmark.md) documents the second dataset,
continuous sentiment metrics and the historical checkpoint replay.

To launch the current local demo with existing checkpoints:

```bash
uv run python src/pipeline_demo.py --device mps --offline \
  --visual-checkpoint models/mvp/visual_day5/deployment.pt \
  --dashboard --duration 60
```

Open `http://127.0.0.1:8765`. Camera and microphone access must already be
available. The Stop button ends capture; `--duration` provides a bounded run.
For repeatable recorded input, add `--replay-dialog Ses01F_impro01 --speaker F`.
Licensed IEMOCAP data and locally generated checkpoints are not bundled.

### HRI session demo

`src/live_demo.py` runs a scripted human–robot interaction session. A robot avatar
speaks short scenario lines (macOS `say`), and the participant answers naturally.
For each answer the page shows what each signal says on its own, then the combined reading:
- **Words only**, **Voice only** and **Face only**: separate comparison models, one signal each.
- **B · Words + voice**: the headline model. It also reads the participant's previous line. The face is shown but not used.
- **Compare with A** (switch, or open the page with `?compare`): adds the deployed model **A**, which reads only the current line.

A one-line readout says where the signals agree or differ. Both models and all three signals
are logged for every answer, whatever is shown. Speech that overlaps the robot's voice is ignored.

```bash
brew install portaudio           # needed to build pyaudio (microphone capture)
uv sync --locked
uv run python src/live_demo.py   # first run online; later runs can add --offline
```

The first online run downloads DistilBERT, DistilHuBERT, Whisper tiny.en and
MobileNetV3. Open `http://127.0.0.1:8765` and allow microphone and camera access.
Then enter a participant ID, confirm consent and start trials. Scenarios live in
`src/hri_scenarios.json`.

Useful flags:
- `--no-camera`: audio and text only.
- `--robot-voice Fred`: a different robot voice.
- `--deployed-asr`: the Whisper settings of the Session 4 evaluation. The default decodes faster.
- `--replay-dialog Ses04F_impro05 --speaker F --silent-robot`: replay a recorded scene instead of the devices. Needs IEMOCAP.

The demo needs these model files, which are **not in git** because they are trained
on licensed IEMOCAP data. Get them from the project team, or rebuild them with
IEMOCAP access:

| Path | Model | Rebuild |
|---|---|---|
| `models/mvp/audio_text_baseline.pt` | A | `src/train_mvp_baseline.py` (`configs/mvp_day3_baseline.json`) |
| `models/mvp/context_baseline.pt` | B | `src/train_mvp_context.py` (`configs/mvp_context.json`) |
| `models/mvp/visual_day5/deployment.pt` | Face head | `src/train_mvp_visual.py`, then `src/select_deployment_fusion.py` |
| `models/mvp/controls/{text,audio}_{42,43,44}.pt` | Words-only and voice-only rows (6 files) | `src/benchmark_controls.py`, then copy its `text_*` and `audio_*` heads here |

Copy the files byte-for-byte: the face head checks A's checksum. A rebuilt A
needs a rebuilt face head; otherwise run with `--no-camera`. Without the six control files
the demo still runs, and the words-only and voice-only rows stay hidden.

Start the demo from a terminal app. macOS grants camera access per app and asks the first
time; a process launched by an IDE or background tool may be refused the camera silently.

Sessions are recorded only after consent, to `data/processed/hri_sessions/`
(gitignored). Each holds an event log, one WAV per response, the model outputs
and the self-reports. The demo is macOS-only for now because the robot voice uses `say`.

## Days 1–2: restore and verify the inference foundation

Place the licensed IEMOCAP release at `data/IEMOCAP_full_release/`, then run:

```bash
uv run python src/prepare_iemocap.py
```

This audits WAV headers, transcripts, emotion annotations, forced-alignment
word times and dialogue video streams. It writes:

- `data/processed/iemocap_manifest/metadata.jsonl`
- `data/processed/iemocap_manifest/validation_report.json`

The manifest preserves the original **11 labels** and **Sessions 1–4 train /
Session 5 test** order. `emotion4_label` is an additional experiment view:
`ang → anger`, `hap/exc → happiness`, `sad → sadness`, `neu → neutral`.
Other labels remain in the manifest with no four-class target.

This is a metadata-only artifact. It does not regenerate temporal eGeMAPS
arrays or extract visual features. Existing prosodic feature code remains in
`src/iemocap_aligned_dataset.py`. Missing word alignment and incomplete video
coverage are recorded explicitly. `video_span_valid` describes the full source
utterance; later visual extraction masks unavailable frames within the chosen
six-second window. Dialogue timestamps refer to the full two-person video,
so speaker selection and visual feature alignment are still required.

To build a small checkpoint for integration testing:

```bash
uv run python src/train_cross_modal_finetune_iemocap.py \
  --freeze-audio --max-samples 128 --epochs 2 --batch-size 4 \
  --num-workers 0 --device auto --skip-test \
  --out models/mvp/day2_smoke.pt
```

This fits 128 examples from Sessions 1–3 and selects on 32 examples from
Session 4. It reloads the best validation checkpoint. Session 5 is not evaluated.
The artifact is marked `development_smoke`; it is not a full emotion benchmark.
For later baseline work, remove the sample cap and choose a training budget.
Keep `--skip-test` during development.

## Days 3–4: full development baseline and visual sequences

The fixed [baseline configuration](configs/mvp_day3_baseline.json) uses every
four-class example from Sessions 1–3 (3,259 training utterances) and Session 4
(1,031 validation utterances). It fits the fusion head for 20 epochs with both
encoders frozen, then reloads the checkpoint with the highest validation macro
F1. Session 5 is never encoded or evaluated by this command:

```bash
uv run python src/train_mvp_baseline.py --device mps --offline
```

The float32 encoder cache under `data/processed/iemocap_encoder_cache/` records
source and feature checksums, encoder revisions, and preprocessing. Reusing a
complete cache preserves its fingerprint. This accelerates training; runtime
inference still runs the encoders. The trainer preserves existing checkpoint
files; use `--output` with a new filename for another run.

Results are saved in `models/mvp/audio_text_baseline.metrics.json`, with
per-utterance validation probabilities in `audio_text_baseline.validation.jsonl`.
These are **reference-transcript validation results**, with the first six seconds
of audio and up to 297 text tokens. They do not measure Whisper transcript errors
or held-out test performance.

The completed run selected epoch 6: **macro F1 0.7469, UAR 0.7444, accuracy
74.88%** on all 1,031 Session 4 examples. The training-majority baseline reaches
25.02% accuracy and 0.1001 macro F1. This is one fixed development run, not a
reproduction of MemoCMT's published benchmark protocol. Render its learning
curve and confusion matrix with:

```bash
uv run python src/plot_mvp_validation.py
```

Evaluate the same checkpoint with actual `tiny.en` transcripts separately:

```bash
uv run python src/evaluate_asr_baseline.py --device mps
```

This diagnostic uses every Session 4 utterance and does not select the audio/text checkpoint.
Its transcript cache, predictions and metrics are saved under
`data/processed/mvp/asr_validation/`. Empty transcripts count as abstentions.
The completed diagnostic reaches **macro F1 0.7229, UAR 0.7202, accuracy
72.36%**, including four abstentions among 1,031 examples.
Both ASR and acoustic inference receive the first six seconds; the reference-text
baseline can see words later in a long utterance. The report separates short and
truncated utterances. It does not measure live VAD segmentation errors.
Days 5–7 subsequently used these ASR predictions to select a visual deployment
weight; the resulting deployment score is a selection score, not independent evaluation.

Extract the paired visual input and audit its timing:

```bash
uv run python src/audit_iemocap_video_alignment.py
uv run python src/extract_iemocap_visual.py --device mps
uv run python src/inspect_visual_alignment.py
```

The visual store at `data/processed/iemocap_visual_features_v2/` covers the same
4,290 development utterances, samples at 5 Hz,
and uses the same first-six-second window. Each record contains timestamped
52-dimensional MediaPipe blendshapes and 576-dimensional MobileNetV3-small
image embeddings from the selected face crop. The image encoder starts from
ImageNet weights; it has not been trained for facial emotion here. Blendshapes
are not FACS action units. The separate AU and external-data comparisons in the
[visual research roadmap](docs/visual_representation_research_roadmap.md) remain
pending.

The dialogue's marker participant occupies the left panel; the utterance speaker
selects the corresponding panel. YuNet detects small faces in the wide panels;
MediaPipe then processes a tight face crop. A small or ambiguous face is masked. Every
utterance is retained, including those with no usable face. The shared loader
in `src/visual_feature_dataset.py` distinguishes image availability, blendshape
availability, undecoded frames, and batch padding. Image features remain available
when dense landmarking fails on a detected face. No feature normalization is
fitted on validation data. Join these records to audio/text by `utterance_id`,
and retain utterances with missing vision for the audio/text fallback.
The earlier `iemocap_visual_features/` directory is
an incomplete diagnostic of direct panel landmarking, with poor tracking coverage;
it is not the store for fusion training.

For CPU-only extraction, use a separate output directory and optionally
`--workers 2`. Model assets and their checksums are recorded in `config.json`;
YuNet's source and license are retained under `src/weights/`.

The AVI decoder uses FFmpeg display timestamps: raw DivX B-frame PTS can move
backward. Untimed final frames remain unavailable. The independent timing audit
matches utterance waveforms against the audio muxed into each source video.
Contact sheets and word/audio/blendshape timelines are written under
`data/processed/mvp/day4_alignment/visual_review/` for inspection. After inspecting
newly rendered images, record the reviewer and findings with
`--record-review "your findings" --reviewer "your name"` on the inspection command.
Default samples cover both speaker panels for both marker participants in each
development session.

The completed v2 extraction contains **83,946 requested frames across 4,290
utterances and 120 dialogues**. Image features are available for 94.49% of frames;
blendshapes for 80.95%. All records are retained, including 37 utterances without
any image features and 105 without any blendshapes. The coverage report breaks
these counts down by partition and class. Every validation utterance has at least
one usable image, so later missing-vision evaluation also needs controlled
dropouts. A review of 22 contact-sheet frames and 16 timelines confirmed the
sampled speaker selection and inspected timing/mask behavior. Weak or noisy
blendshape traces remain possible even when a face is detected.

The independent waveform audit matched 710 utterances across all 120 development
dialogues to their video's embedded audio, with a maximum absolute offset of
0.1 ms. This checks the dataset's audio timestamp mapping; it does not measure
physical camera/audio synchronization to that precision.

```bash
uv run python src/verify_day4.py --device mps
uv run python src/benchmark_visual_replay.py --device mps
```

The verifier checks full sample coverage, checkpoint selection, all validation
predictions, ASR diagnostic counts and metrics, raw/cached inference parity,
visual masks and timestamps, and a
fresh visual re-extraction. The paced visual benchmark includes decoding,
face detection, image preprocessing and uncached image inference at 10 updates
per second. Run it without concurrent training for standalone measurements.
Its results cover visual processing only, not webcam/audio/fusion latency.
On the Apple M3 Pro, the 30-second MPS replay processed 300 updates at 10 Hz.
After 20 warm-up updates, p50 was **32.95 ms** and p95 **38.30 ms**, with no
accumulated backlog and peak process RSS of 655 MiB. All 280 measured updates
contained a detected face. See `data/processed/mvp/day4_visual_latency.json`;
the earlier concurrent-load v1 measurement is a separate diagnostic.

## Days 5–7: trained visual branches and synchronized inference

The [fixed visual protocol](configs/mvp_day5_visual.json) compares blendshapes,
image embeddings and their combination using three seeds and 20 epochs each.
The audio/text model stays frozen. Feature normalization uses only Sessions 1–3;
all 1,031 Session 4 examples remain in each comparison. A small GRU head learns
visual emotion probabilities, then a validation-selected probability mixture
joins them to audio/text. This is late fusion, not a new visual cross-attention model.

```bash
uv run python src/train_mvp_visual.py --base-device mps
uv run python src/verify_mvp_visual.py
uv run python src/plot_day7.py
```

The trainer reuses verified completed runs. It refuses incomplete existing
checkpoints. Full metrics, histories and per-utterance probabilities are in
`models/mvp/visual_day5/`; verification and the comparison plot are under
`data/processed/mvp/day5_7/`.

| Added visual input, reference transcripts | Macro F1, mean ± sample SD | Trainable visual parameters |
|---|---:|---:|
| None, fixed audio/text model | 0.7469 | — |
| Blendshapes | 0.7518 ± 0.0013 | 37,188 |
| Image embeddings | 0.7512 ± 0.0015 | 70,724 |
| Both | 0.7545 ± 0.0026 | 74,052 |

`selected.pt` preserves the reference-text choice: combined features, seed 42,
epoch 5, visual probability weight 0.25, macro F1 **0.7553**. Controlled missing
vision returns the exact audio/text prediction. Time shuffling scored 0.7562,
so temporal-order benefit is not established. AU and external-data adaptation
experiments remain pending.

With actual Whisper transcripts, that selected mixture scores **0.7215**, below
audio/text's **0.7229**. The separate [deployment protocol](configs/mvp_day7_deployment.json)
holds the head fixed and selects only its mixture weight from the same grid.
It chooses **0**: `deployment.pt` runs vision for inspection while the final
probabilities remain audio/text. The original reference selection is preserved.
`deployment.selection.json` records every candidate and the ASR provenance.
For a fresh artifact, `uv run python src/select_deployment_fusion.py` creates
this selection; it deliberately refuses to overwrite an existing deployment.

Use these commands to reproduce the current integration checks, with separate
output names for new evidence you want to preserve:

```bash
# Raw development dialogue, paced through real VAD, ASR and video processing.
uv run python src/pipeline_demo.py --replay-dialog Ses01F_impro01 --speaker F \
  --duration 35 --visual-checkpoint models/mvp/visual_day5/deployment.pt \
  --device mps --offline --dashboard --dashboard-hold 120 \
  --output data/processed/mvp/day5_7/dialogue_replay_final.jsonl

# In another terminal while the finished inspector is held open:
uv run python src/verify_inspector.py --stop

# Real microphone and webcam; optionally specify device indices for this machine.
uv run python src/pipeline_demo.py --duration 30 \
  --visual-checkpoint models/mvp/visual_day5/deployment.pt \
  --device mps --offline \
  --output data/processed/mvp/day5_7/live_capture_final.jsonl

# Exercise missing vision even with the nonzero-weight research checkpoint.
uv run python src/pipeline_demo.py --replay-utterance Ses01F_impro01_F000 \
  --stream-replay --no-vision --visual-checkpoint models/mvp/visual_day5/selected.pt \
  --device mps --offline \
  --output data/processed/mvp/day5_7/fallback_replay.jsonl

uv run python src/verify_day7.py --device mps
```

The inspector verifier checks its HTTP contract; it does not inspect browser
appearance. The day-7 verifier reproduces the complete ASR weight selection, checks that deployment
did not retrain the head, re-extracts four clips covering every class and both
speakers, and checks raw/cached visual-feature and fusion parity. It audits saved
replay/device logs, exact fallback, causal frame selection, bounded queues and
clean shutdown. Runtime source/checkpoint hashes accompany each new run.
It does not activate devices or evaluate Session 5 itself.

The final **35-second full replay** emitted 11 predictions with no queue drops:
speech-end-to-result p50 **0.867 s**, p95 **1.474 s**, maximum **1.561 s**;
visual p95 **67.51 ms**, peak process RSS **895 MiB**. These 11 development
segments are an integration measurement, not a broad latency guarantee.
The **30-second device check** captured 933 audio chunks and 139 video frames,
including 138 usable faces, with no queue drops and a clean exit at 30.27 seconds.
One audio discontinuity was reported and reset. No speech was detected during
that check; it does not validate live spoken emotion accuracy.

Audio ADC and video receipt timestamps share a monotonic clock. Camera sensor
delay remains uncalibrated. Vision samples at 5 Hz and uses only the first six
seconds inside each speech interval. Missing features stay masked. Queue
capacities are fixed; old/stale work is dropped observably, long speech is
segmented, and accepted speech is flushed at end of input. Raw microphone audio
and camera video are not saved by this runtime. An initial microphone shutdown
hang was fixed and retained as a regression report.

The inspector presents the transcript, face availability, blendshape coefficients,
audio/text and visual probabilities, final prediction route, stage timings and
utterance history. Its HTTP/state/preview/Stop behavior and JavaScript syntax
were checked. **Browser appearance review is pending** because the computer-use
tool denied Firefox access. The revised roadmap prioritizes benchmark quality
and retains this review and longer spoken sessions before release. The inspector is bound to localhost, with preview frames
held only in memory.

## Audio/text-only inference

Run the actual ASR and emotion model on one recorded utterance:

```bash
uv run python src/pipeline_demo.py \
  --audio-file data/IEMOCAP_full_release/Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav \
  --emotion-checkpoint models/mvp/audio_text_baseline.pt \
  --device mps --offline \
  --output data/processed/mvp/day4_baseline_replay.jsonl
```

Use `--device cpu` for CPU inference. To inspect reference-transcript behavior,
add `--text "Excuse me."`; output labels this as a supplied transcript.
The audio-file mode accepts an already segmented utterance. The reproducible
verification command below additionally replays it through the real Silero VAD:

```bash
uv run python src/verify_day2.py --device mps
```

It verifies the full manifest against the original loader, checkpoint split
provenance, real ASR/emotion inference, VAD replay, and prediction invariance to
batch padding. Results go to `data/processed/mvp/day2_verification.json`.

For microphone input, omit `--audio-file`:

```bash
uv run python src/pipeline_demo.py --model tiny.en --threshold 0.4 \
  --emotion-checkpoint models/mvp/audio_text_baseline.pt --device mps --offline
```

Microphone capture requires local device permission. `--duration 30` bounds a
capture session. `src/workers_demo.py` is a compatibility entry point to this
same pipeline. Audio failures propagate through the async task group; camera
failure is reported and allows audio/text processing to continue.

Both modes use `SpeechProcessor` and `CrossModalEmotionPredictor`, return
structured JSON, and share 16 kHz float-audio preparation and tokenization.
The checkpoint records its maximum audio duration (6 seconds by default);
long inputs keep only the first 6 seconds and output includes `audio_truncated`.
Streaming inference warms the audio/text model before starting its capture clock.
Direct single-utterance replay includes cold processing and is not a p95 measurement.

## Model and checkpoint contract

The current MemoCMT-style model uses DistilBERT and DistilHuBERT, approximately
96 million total parameters including a 6-million-parameter fusion head.
Waveform length masks are applied before audio encoding; encoder-frame and
text masks also exclude padded queries from fusion pooling. Frozen encoders
remain in evaluation mode while the head trains.

Version-1 checkpoints record the task, label mapping, encoder names and
revisions, architecture, preprocessing, weight scope, and training provenance.
The predictor checks these before loading weights. A MOSEI seven-class
sentiment checkpoint is not a four-class IEMOCAP emotion checkpoint. Legacy
checkpoint formats are rejected with an explicit error; existing artifacts
are preserved, and new IEMOCAP runs use the versioned format.

See [the roadmap](docs/async_multimodal_pipeline_roadmap.md) for the remaining
MVP milestones. Historical 11-class/eGeMAPS results and the MOSEI benchmark
remain separate experiments, not results for this MVP.
