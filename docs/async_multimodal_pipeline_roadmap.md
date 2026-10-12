# Async Multimodal Pipeline Roadmap

Updated 2026-10-12: week-one comparisons complete; prioritize model quality and comparable benchmarks.
The original days 1–7 implementation is complete. The revised **ten-working-day
plan** below allocates most of the next two weeks to evaluation and model work;
interface polish and robot integration are secondary.

## Scope and current decision

Build an inspectable local Mac demo for one observed human with a microphone and
webcam. Report **anger, happiness, sadness, neutral** per utterance. Robot
integration is optional. These are emotion estimates; trust prediction requires
separate supervision and a later HRI study.

The audio/text model is a distilled MemoCMT-style baseline, not a reproduction
of the paper's benchmark. Vision currently joins it through probability fusion,
not a new three-modality cross-attention architecture. The transcript is shown
as words and encoded by DistilBERT; these are two representations of text,
not independent modalities. Explicit eGeMAPS/prosody is not in this baseline.

**Default deployment decision:** use the audio/text probabilities while computing
and displaying the visual branch separately. Actual Whisper validation selected
visual weight **0**. The reference-text experiment's 0.25 visual weight remains
available in `selected.pt` for research comparisons. Do not describe the default
demo as an established accuracy improvement from vision.

## Completed foundation and results

| Original milestone | Current evidence |
|---|---|
| Days 1–2: data and inference contracts | Original 11 labels retained; four-class experiment view; versioned checkpoints; one shared utterance processor; audited Session 5 boundary |
| Days 3–4: full audio/text baseline and visual data | All 3,259 Session 1–3 training and 1,031 Session 4 validation utterances; complete 4,290-utterance visual cache; audited speaker/time alignment |
| Days 5–6: visual representation comparison | Three representations × three seeds × 20 epochs; frozen audio/text model; training-only normalization; full-set comparison and controlled missing-input checks |
| Day 7: synchronized runtime and inspector | Paced raw audio/video replay; real camera/mic capture; bounded queues and history; final speech draining; fallback; localhost inspector with HTTP checks |
| Still open from the original inspector milestone | Browser appearance/responsive-layout review: the computer-use tool denied Firefox access; no visual QA is claimed |

Current four-class MVP selection uses Session 4. **Session 5 has not been
evaluated by this four-class experiment.** It is not a historically untouched
project holdout: older 11-class experiments evaluated it and selected epochs
on test macro F1 (`src/iemocap_baseline_training.py`). Keep that exposure in
the final report. Session 4 ASR deployment-weight selection is also a development
result, not independent validation of that choice.

| Session 4 input / added visual features | Macro F1 | Interpretation |
|---|---:|---|
| Reference text, audio/text baseline | 0.7469 | Fixed baseline, epoch 6; UAR 0.7444; accuracy 74.88% |
| Reference text + blendshapes | 0.7518 ± 0.0013 | Mean ± sample SD over seeds 42, 43, 44 |
| Reference text + image embeddings | 0.7512 ± 0.0015 | Same split, examples, epoch and fusion-weight budget |
| Reference text + both visual streams | 0.7545 ± 0.0026 | 74,052 trainable visual-head parameters; selected representative seed 42 scores 0.7553 |
| Actual tiny.en, audio/text baseline | 0.7229 | Four empty-transcript abstentions included; UAR 0.7202; accuracy 72.36% |
| Actual tiny.en + reference-selected visual weight 0.25 | 0.7215 | No macro-F1 improvement; same combined visual head |
| Actual tiny.en, selected deployment weight 0 | 0.7229 | Exact audio/text output; visual branch remains observable |

The visual store samples the first six seconds at 5 Hz: YuNet face crop,
52 MediaPipe blendshapes and frozen 576-dimensional MobileNetV3-small ImageNet
features. Image coverage is 94.49%, blendshape coverage 80.95%; masks stay
separate. Blendshapes are not facial action units, and ImageNet pretraining is
not facial-emotion pretraining. The raw/cached parity checks preserve that
same extraction contract at runtime.

The full 1,031-utterance missing-vision intervention returns the exact baseline.
Removing only image or blendshape input is supported by the combined head's
training dropout, but can lower its score. Time shuffling scored **0.7562**
versus the selected ordered model's **0.7553**: there is currently no evidence
that temporal order adds value. Seed variation here is not a population-level
confidence interval. AU estimation and MEAD adaptation remain unrun.

Reference transcripts can include words after the six-second audio/video crop;
Whisper receives only that crop. The next comparison must separate this input
mismatch from transcription errors and add a simple temporal-pooling control.

## Verified runtime, with limits

`src/pipeline_demo.py` is the canonical entry point. `src/stream_runtime.py`
coordinates live devices or paced recordings through the same VAD, ASR and
inference path. Audio, frames, speech segments and visual history have fixed
capacities of 32, 2, 4 and 450. Overflow drops old work and records the loss;
audio discontinuities discard partial speech; stale segments are not processed.
Speech is capped at approximately six seconds, and accepted final speech is
flushed on source completion. Latency summaries retain at most 2,000 samples.

Audio ADC timestamps are mapped to monotonic time. Camera frames use monotonic
receipt timestamps, so physical sensor delay is still uncalibrated. Fusion
selects only frames inside the speech interval and its first six seconds;
missing frames stay masked. Global live sampling can differ from the
utterance-relative training cadence by a fraction of the 200 ms interval.

| M3 Pro check | Observed result |
|---|---|
| Final 35-second paced dialogue, real VAD/ASR and uncached vision | 11 predictions; p50 response 0.867 s, p95 1.474 s, max 1.561 s; 175 visual frames; no queue drops or audio gaps |
| Visual cost during that full replay | p50 36.31 ms, p95 67.51 ms; peak process RSS 895 MiB; warm-up/outliers included |
| Final 30-second real microphone/webcam check | 933 audio chunks, 139 frames, 138 with image and blendshapes; no queue drops; one reported audio discontinuity; clean exit at 30.27 s |
| Live visual cost | p50 39.61 ms, p95 50.69 ms; peak process RSS 835 MiB |

The live device check detected no speech; it verifies capture, features and
shutdown, not spoken prediction quality. The response figures above come from
11 replay segments, not a broad live latency benchmark. The earlier standalone
10 Hz visual-only benchmark remains separate from the runtime's 5 Hz contract.
An initial device run exposed a CoreAudio shutdown hang; the repaired single-owner
stop path passed three microphone stop cycles and the combined capture check.

The inspector shows face availability, blendshape values, transcript, audio/text
and visual probabilities, the final output route, timing and utterance history.
It binds to localhost and keeps preview frames in memory. Its state, preview,
stop endpoint and JavaScript syntax were checked. Browser appearance is pending.
See [README commands](../README.md), `data/processed/mvp/day5_7/`, and
`models/mvp/visual_day5/comparison.json` for reproducible evidence.

## Model-quality evidence from the new benchmark pass

All IEMOCAP controls use the same 3,259 training and 1,031 validation utterances,
pinned frozen DistilBERT/DistilHuBERT features, six-second audio cap, reference
transcripts, 20 epochs, Adam 1e-4 and batch size 16. Each head uses seeds 42,
43 and 44; select epochs by validation macro F1. Pooled controls standardize
features using training statistics only. Head capacity and pooling deliberately
differ; this is a same-input/budget comparison, not a parameter-matched experiment.

| IEMOCAP development model | Macro F1, mean ± sample SD | UAR, mean | Trainable head parameters |
|---|---:|---:|---:|
| Mean-pooled text + MLP | 0.6448 ± 0.0053 | 0.6517 | 98,948 |
| Mean-pooled audio + MLP | 0.6307 ± 0.0103 | 0.6275 | 98,948 |
| Concatenated pooled audio/text + MLP | 0.7342 ± 0.0097 | 0.7249 | 197,252 |
| Current cross-attention | 0.7408 ± 0.0066 | 0.7346 | 6,007,940 |

Cross-attention's mean margin over concatenation is **0.0065 macro F1**,
with a loss to concatenation on seed 44. It uses about **30.5 times as many
trainable head parameters**; that is not a measured end-to-end speed ratio.
The result supports using both audio and text, but does not establish a robust
advantage for this fusion architecture. Across the cross-attention runs,
neutral recall averages 0.6098, below anger 0.8787, happiness 0.7228 and sadness
0.7273. Inspect neutral/confusable-class errors before expanding the architecture.
Three seeds measure training variation, not uncertainty across human populations.

The original seed-42 deployed checkpoint is unchanged. The two new cross-attention
runs select epochs 9 and 20. Training reloads each selected checkpoint and checks
its full validation score. All nine pooled controls were also reloaded; their
1,031-example probabilities and metrics reproduce their saved reports.

Artifacts: `data/processed/benchmarks/iemocap_controls_v1/` and
`iemocap_cmt_v1/`; exact repeat configs are under `configs/benchmark_iemocap_*`.
The consolidated, checked results are in
`data/processed/benchmarks/quality_summary.json`. The full repository suite
passes **54 tests**; the new benchmark modules pass Ruff. No new test-model
inference or deployment-checkpoint replacement was performed in this pass.
The control runner is `src/benchmark_controls.py iemocap --output <new-directory>`.
Use `src/train_mvp_baseline.py --config configs/benchmark_iemocap_cmt_seed43.json
--output <new-checkpoint> --device mps --offline` for the corresponding repeat.

## Dataset roles and comparison boundaries

| Dataset | Local readiness | Role in the next experiments |
|---|---|---|
| IEMOCAP | Raw audio/video, annotations and verified development features | Primary anger/happiness/sadness/neutral benchmark; excitement merges into happiness in the experiment view only |
| CMU-MOSEI | 22,856 audio/video clips, continuous sentiment labels, checksum-verified released features | Second native-task benchmark; reviewed raw-data view excludes nine training and four validation clips from flagged source groups; original-feature reference retains the original split with overlap caveats |
| MSP-Podcast 2.0 | Labels, transcripts, forced alignment and partitions; no local audio | Later naturalistic emotion evaluation after audio access and release matching are restored; do not block the two-week MVP on it |

MOSEI's historical checkpoint now reproduces every archived validation metric
exactly. Using its expected sentiment score instead of its argmax class value
changes validation MAE **0.5991 → 0.5476**, Pearson r **0.6553 → 0.7131**, and
nonzero accuracy **63.49% → 84.49%**. Including zero labels changes accuracy
**83.11% → 79.48%**. This is a scoring projection, not a training or architecture
gain; keep both definitions visible. A lexical ridge control scores MAE 0.6406
and nonzero accuracy 76.91% on the same validation IDs, with nine fewer training
examples than the historical model. No new MOSEI test inference was run.

See [the MOSEI benchmark audit](mosei_benchmark.md) for data hashes, exclusions,
full metric conventions, media checks, reported LMF/MulT/Self-MM reference
numbers and reproducible commands. Validation numbers must not be ranked against
published test numbers, or modified raw-data folds against a different feature
release. Sentiment and emotion retain separate targets and output heads.

MemoCMT's released IEMOCAP preprocessing randomly splits all five sessions in
two stages (approximately 81/9/10). Its released configuration uses frozen
BERT/HuBERT-base, eight-second audio and 100 epochs. Our speaker-held-out,
distilled, six-second/20-epoch experiment is different. The older local
`compare_iemocap_protocols.py` used 80/10/10 and incorrectly called it a direct
paper reproduction; that description is now corrected. It remains a legacy
command that consumes Session 5 and is not used by this development pass.
Sources: [released preprocessing](https://github.com/tpnam0901/MemoCMT/blob/d3248703e755319736b805fd6b0c2e8ab6089160/scripts/preprocess.py),
[released configuration](https://github.com/tpnam0901/MemoCMT/blob/d3248703e755319736b805fd6b0c2e8ab6089160/src/configs/hubert_base.py).

## The next two weeks: benchmark quality first

Week one is complete under `configs/week1/protocol_matrix.json`. Known MOSEI
recording groups are reviewed, a separate v3 raw-data manifest is frozen, and
the official feature bytes are verified and exported for development only.
The larger BERT/HuBERT-base, LMF and MulT reference runs are complete. The
first improvement is preregistered **label smoothing 0.1** on the distilled
model, keeping the split, input crop, frozen encoders and 20-epoch budget fixed.
All groups use seeds 42/43/44. Matched reference and actual tiny.en ASR replay
is complete for all nine IEMOCAP heads. LMF and MulT each completed three seeds.
Current results and missing runs are checked by `src/summarize_week1.py` and
saved under `data/processed/benchmarks/week1_v1/summary.json`. The checked comparisons are appended below. The deployed baseline is preserved.

The deliverable is a reproducible model comparison plus the working local demo.
“Comparable” needs a named model, the same task/split/inputs, uncertainty over
runs and measured inference cost. As an engineering target, aim to stay within
**two percentage points of the matched larger reference's macro F1/UAR** while
reducing deployment cost. This is a proposed tolerance, not an achieved result
or a statistical equivalence claim. For MOSEI, report MAE, correlation and both
binary definitions; never select a convenient headline metric after testing.

| Working days | Work | Exit condition |
|---|---|---|
| 1–2 | Finish the benchmark contract and data review. Current controls, three cross-attention seeds and the legacy MOSEI replay are complete. Resolve MOSEI's flagged source identities and select explicit original-feature versus modified-raw-data protocols. Fix label maps, splits, seed lists, selection metrics and allowed tuning before further model selection. | Versioned manifests, test-exposure record, bounded model matrix; no claim of independent testing while near-duplicate candidates remain unresolved |
| 3–4 | Run a larger BERT/HuBERT-base MemoCMT-style reference on our exact IEMOCAP split and crop, against the distilled model and pooled fusion. Three seeds, matched tuning budget. Separately establish a reproducible MOSEI reference using verified MMSA feature files: LMF first, then MulT if the first run passes. | Same-protocol comparison distinguishes encoder compression from fusion effects; original-feature reproduction is reported separately from live-compatible features |
| 5–6 | Address one demonstrated quality gap. For IEMOCAP, prioritize crop-matched reference/ASR inputs and neutral-class errors; test one bounded encoder-adaptation or loss/regularization change. For MOSEI, compare a native continuous regression objective against the fixed expected-score baseline on the same cleaned training view. | Same-budget, three-seed development results, per-class/residual analysis and full sample coverage; retain baselines when improvement is absent |
| 7 | Test the visual contribution using one justified representation change, with pooled/temporal controls and missing-vision evaluation. Reuse matched audio/text inputs. Default to audio/text if actual-ASR quality does not improve. | Paired T / A / T+A / T+A+V evidence; explicit visual coverage and cost; no temporal or visual novelty claim from the stream count alone |
| 8 | Freeze the selected models, preprocessing, metrics, missing-input behavior and evaluation subsets. Profile the quality finalists on the Mac; optimize only a measured bottleneck. | A quality/cost table, frozen experiment configuration, no growing queue, and a fixed final-evaluation command; target p95 <2 s on stated 1–6 s speech segments |
| 9 | Run the frozen four-class Session 5 evaluation once, including actual ASR and per-class errors, while disclosing prior project exposure. Run MOSEI final testing only if its protocol/source-identity gate is resolved. Do not tune afterward. | Held-out reports with limitations, identical comparator sample sets and no post-test selection; defer a failed gate rather than using test data to debug |
| 10 | Package the evidence and demo. Complete the pending browser review and three spoken-session rehearsals, with at least 100 total speech segments for the latency check. Robot integration is optional. | Reproducible configs/checkpoints/results; explicit benchmark versus live-demo claims; reliable startup, Stop and replay fallback |

Do not train one shared emotion head by coercing MOSEI sentiment labels into
IEMOCAP categories. A later shared encoder with separate task heads is a
research option after the native baselines are sound. MSP training waits for
its authorized audio and a version/partition audit. Full five-session CV or
exact author-split reproduction is a separately frozen study, not a reason
to expose Session 5 during current development.

**Deployment gate:** improve actual-ASR macro F1, inspect class regressions,
and measure full-system cost. A small reference-text gain alone does not enable
vision. Maintain the current audio/text fallback and inspector throughout.
Additional UI features, robot behavior, a trained trust head, large new datasets
and a broad prosody/AU architecture search remain outside this two-week scope.
The [visual research roadmap](visual_representation_research_roadmap.md) retains
those later experiments.

## Archived worker design notes

The sections below retain the earlier architecture proposal. Names and status
are historical; follow the runtime and commands above for the working system.

## Goal

Build a real-time multimodal sentiment/trust pipeline where audio, vision, text,
and fusion run as self-contained workers managed by one async orchestrator.

The important design point is not that every function becomes `async`. Instead:

- Blocking device and model calls stay normal Python internally.
- Worker loops are async.
- Workers communicate through `asyncio.Queue`.
- Every emitted event is timestamped.
- The main pipeline starts, stops, and routes workers without owning modality logic.

Target flow:

```text
mic -> audio capture worker -> VAD worker -> transcription worker -> text worker
cam -> vision worker
text/audio/vision events -> fusion worker -> sentiment/trust prediction
```

## Libraries

The repo already has most of what is needed:

- `asyncio`: orchestration, queues, cancellation
- `dataclasses`: typed event records
- `numpy`: feature arrays
- `torch`: model inference
- `pyaudio`: microphone input
- `opencv-python`: webcam input
- `mediapipe`: face landmarks and gaze
- `faster-whisper`: transcription path used by `pipeline_demo.py`
- `transformers`: text embeddings/sentiment
- `opensmile`, `librosa`, `soundfile`: audio/prosody features

Recommended addition:

```bash
uv add --dev pytest
```

Optional later:

- `pydantic`: runtime validation for event payloads
- `rich`: nicer terminal logging
- `typer`: cleaner CLI commands

## Proposed File Structure

```text
src/
  pipeline_demo.py              # async orchestrator / demo entry point

  core/
    __init__.py
    events.py                   # shared dataclasses
    clock.py                    # optional shared clock helpers
    shutdown.py                 # optional cancellation helpers

  workers/
    __init__.py
    audio.py                    # public audio worker imports
    vad.py                      # AudioChunk -> SpeechSegment
    transcription.py            # SpeechSegment -> TranscriptEvent
    vision.py                   # webcam -> VisionEvent
    text.py                     # TranscriptEvent -> TextEmbeddingEvent
    fusion.py                   # align modality events -> model input

  audio/
    __init__.py
    capture.py                  # PyAudio -> AudioChunk
    vad.py                      # eventual home for SileroVAD
    prosody.py                  # eventual home for prosody extraction

  vision/
    __init__.py
    gaze.py                     # eventual home for GazeClassifier

  text/
    __init__.py
    encoder.py                  # eventual home for TextHiddenStateEncoder

  models/
    fusion.py                   # eventual multimodal model
```

Do not move everything at once. Add wrappers around the current modules first,
then move code once the interfaces are stable.

## Step 1: Shared Events

Keep event classes in `src/core/events.py`.

Example event types:

```python
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class AudioChunk:
    timestamp: float
    sample_rate: int
    samples: np.ndarray

@dataclass(frozen=True)
class SpeechSegment:
    start_time: float
    end_time: float
    sample_rate: int
    samples: np.ndarray

@dataclass(frozen=True)
class TranscriptEvent:
    start_time: float
    end_time: float
    text: str

@dataclass(frozen=True)
class VisionEvent:
    timestamp: float
    face_detected: bool
    is_facing: bool
    is_looking: bool
    head_offset: float | None
    gaze_offset: float | None
```

Every event should carry enough timestamp information for later fusion.

## Step 2: Audio Capture Worker

This is now started in:

```text
src/audio/capture.py
```

Responsibility:

```text
PyAudio mic stream -> AudioChunk events
```

Interface:

```python
async def run_audio_capture(output_queue, stop_event, config): ...
```

Internally, PyAudio's blocking `stream.read()` should be wrapped with:

```python
raw = await asyncio.to_thread(
    stream.read,
    frames_per_buffer,
    exception_on_overflow=False,
)
```

This keeps the event loop responsive while the microphone waits for data.

## Step 3: VAD Worker

Create:

```text
src/workers/vad.py
```

Responsibility:

```text
AudioChunk -> SpeechSegment
```

The worker should own a `SileroVAD` instance and call `vad.process_chunk(...)`.

Interface:

```python
async def run_vad_worker(input_queue, output_queue, stop_event, config): ...
```

Use `AudioChunk.timestamp` as the chunk start time. The VAD output should retain
`start_time` and `end_time` so transcription and fusion can align it.

## Step 4: Transcription Worker

Create:

```text
src/workers/transcription.py
```

Responsibility:

```text
SpeechSegment -> TranscriptEvent
```

Use `faster-whisper` first because `pipeline_demo.py` already works that way.

Blocking model inference should be run through `asyncio.to_thread()`:

```python
segments, _ = await asyncio.to_thread(
    model.transcribe,
    speech.samples,
    language="en",
    beam_size=1,
    vad_filter=False,
)
```

Do not merge transcription into the VAD module. Transcription is a separate
pipeline stage.

## Step 5: Vision Worker

Create:

```text
src/workers/vision.py
```

Responsibility:

```text
OpenCV webcam frames -> VisionEvent
```

Use OpenCV for frame capture and `GazeClassifier.classify(frame)` for gaze.

Before relying on MediaPipe, fix the color conversion:

```python
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```

OpenCV gives BGR frames, but MediaPipe expects RGB.

## Step 6: Text Worker

Create:

```text
src/workers/text.py
```

Responsibility:

```text
TranscriptEvent -> text embedding/sentiment event
```

This worker should wrap `TextHiddenStateEncoder.encode(text)`.

Text embedding can lag behind transcription, so it should have its own queue.
Batching can be added later if needed.

## Step 7: Fusion Worker

Create:

```text
src/workers/fusion.py
```

Start simple. The first version should align and print events before trying to
run a learned model.

Responsibility:

```text
collect recent audio/text/vision events by timestamp
emit one combined prediction window
```

Initial fusion policy:

- Keep a rolling buffer of recent `VisionEvent`s.
- When a `TranscriptEvent` arrives, find vision events near or inside
  `[start_time, end_time]`.
- Aggregate gaze as mean, count, or majority.
- Pair transcript/text embedding with audio and vision features.
- Print the aligned record for inspection.

Later, replace the print with the actual multimodal sentiment/trust model.

## Step 8: Async Orchestrator

Eventually rewrite `pipeline_demo.py` as a thin orchestrator:

```python
async def main():
    stop_event = asyncio.Event()

    audio_chunks = asyncio.Queue(maxsize=32)
    speech_segments = asyncio.Queue(maxsize=16)
    transcripts = asyncio.Queue(maxsize=16)
    vision_events = asyncio.Queue(maxsize=64)
    text_events = asyncio.Queue(maxsize=16)

    tasks = [
        asyncio.create_task(run_audio_capture(audio_chunks, stop_event, audio_config)),
        asyncio.create_task(run_vad_worker(audio_chunks, speech_segments, stop_event, vad_config)),
        asyncio.create_task(run_transcription_worker(speech_segments, transcripts, stop_event, whisper_config)),
        asyncio.create_task(run_text_worker(transcripts, text_events, stop_event, text_config)),
        asyncio.create_task(run_vision_worker(vision_events, stop_event, vision_config)),
        asyncio.create_task(run_fusion_worker(text_events, vision_events, stop_event, fusion_config)),
    ]

    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        for task in tasks:
            task.cancel()
```

Production-quality orchestration should also drain exceptions from tasks and
wait for cleanup.

## Step 9: Tests

Start with pure tests that do not need a microphone, webcam, or model download.

Good first tests:

- `int2float()` converts int16 audio correctly.
- `AudioChunk.duration` and `AudioChunk.end_time` are correct.
- VAD worker emits a segment when given fake chunk probabilities.
- Fusion buffer selects the right vision events for a transcript interval.
- Alignment code handles empty or missing events.

Run with:

```bash
uv run python -m pytest -q
```

## Worker Contract

Every worker should follow this contract:

```text
owns its model/device
reads from zero or more queues
writes structured timestamped events
exits when stop_event is set
cleans up its own resources
does not know about the full app
```

This keeps each module understandable and makes `pipeline_demo.py` a manager
rather than a pile of modality-specific logic.

<!-- WEEK1_RESULTS_START -->
## Completed week-one development comparisons

Computed from the checked three-seed runs (42/43/44). Values are mean ± sample SD; all IEMOCAP rows use the same 1,031 Session 4 examples.

| IEMOCAP model | Reference macro F1 | Reference UAR | Actual tiny.en ASR macro F1 |
|---|---:|---:|---:|
| Distilled CMT baseline | 0.7408 ± 0.0066 | 0.7346 ± 0.0100 | 0.7072 ± 0.0205 |
| BERT/HuBERT-base CMT | 0.7490 ± 0.0074 | 0.7474 ± 0.0079 | 0.7168 ± 0.0087 |
| Distilled CMT, smoothing 0.1 | 0.7423 ± 0.0035 | 0.7333 ± 0.0048 | 0.7071 ± 0.0020 |

The larger-minus-distilled mean gaps are **0.83 macro-F1 points** and **1.28 UAR points**. The baseline meets the proposed two-point engineering tolerance on both metrics.

Smoothing changes mean reference macro F1 by **+0.15 points** and actual-ASR macro F1 by **-0.01 points**. It does not pass the preregistered mean-quality gate; retain the existing baseline.

All nine selected IEMOCAP heads pass independent reference replay and actual-ASR diagnostics. Four empty-ASR utterances remain as abstentions in every 1,031-example ASR score. No deployment checkpoint was replaced.

These are development comparisons, with the existing session split and six-second crop. They are not MemoCMT author-protocol reproduction, statistical equivalence, population-generalization evidence or trust validation. See `data/processed/benchmarks/week1_v1/summary.json` for seed results, paired deltas and provenance.
<!-- WEEK1_RESULTS_END -->
