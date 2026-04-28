# Async Multimodal Pipeline Roadmap

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

