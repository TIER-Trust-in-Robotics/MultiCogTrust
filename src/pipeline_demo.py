"""Microphone demo and recorded-utterance replay using the same inference path."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="tiny.en", help="faster-whisper ASR model")
    parser.add_argument("--threshold", type=float, default=0.4, help="VAD threshold")
    parser.add_argument(
        "--emotion-checkpoint",
        type=Path,
        help="Versioned four-class IEMOCAP checkpoint",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "mps", "cuda"),
        default="cpu",
        help="Emotion inference device; ASR uses CPU int8",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--audio-file", type=Path, help="Replay one already segmented utterance WAV"
    )
    source.add_argument(
        "--replay-utterance",
        help="Development IEMOCAP utterance with its synchronized speaker video",
    )
    source.add_argument(
        "--replay-dialog",
        help="Paced development dialogue through real VAD, ASR and video extraction",
    )
    parser.add_argument("--speaker", choices=("F", "M"), default="F")
    parser.add_argument(
        "--stream-replay",
        action="store_true",
        help="Pace a recorded utterance through the live VAD/queue path",
    )
    parser.add_argument(
        "--visual-checkpoint",
        type=Path,
        help="Versioned compact visual head; defaults the audio/text reference to the MVP baseline",
    )
    parser.add_argument(
        "--context",
        choices=("off", "balanced", "max_f1"),
        default="off",
        help="Causal smoothing of emotion over the speaker's previous utterances (src/emotion_context.py)",
    )
    parser.add_argument("--camera-index", type=int)
    parser.add_argument("--microphone-index", type=int)
    parser.add_argument(
        "--no-vision",
        action="store_true",
        help="Exercise the trained audio/text fallback",
    )
    parser.add_argument(
        "--dashboard", action="store_true", help="Serve the local processing inspector"
    )
    parser.add_argument("--dashboard-port", type=int, default=8765)
    parser.add_argument(
        "--dashboard-hold",
        type=float,
        default=0,
        help="Keep the finished inspector open for this many seconds",
    )
    parser.add_argument(
        "--text", help="Optional reference transcript for replay; default runs ASR"
    )
    parser.add_argument("--output", type=Path, help="Write structured JSONL results")
    parser.add_argument(
        "--offline", action="store_true", help="Require locally cached model assets"
    )
    parser.add_argument(
        "--duration", type=float, help="Stop microphone capture after this many seconds"
    )
    args = parser.parse_args(argv)
    if args.text is not None and not (args.audio_file or args.replay_utterance):
        parser.error("--text requires --audio-file or --replay-utterance")
    if args.text is not None and args.stream_replay:
        parser.error("Streaming replay uses ASR rather than a supplied full transcript")
    if args.stream_replay and not (args.audio_file or args.replay_utterance):
        parser.error("--stream-replay requires a recorded utterance")
    if args.camera_index is not None and not args.visual_checkpoint:
        parser.error("--camera-index requires --visual-checkpoint")
    if args.visual_checkpoint and not args.emotion_checkpoint:
        args.emotion_checkpoint = ROOT / "models/mvp/audio_text_baseline.pt"
    if args.dashboard_hold < 0:
        parser.error("--dashboard-hold must not be negative")
    if not 0 < args.threshold < 1:
        parser.error("--threshold must be between 0 and 1")
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be positive")
    return args


async def run_microphone(processor, emit, args):
    from src.stream_runtime import run_stream

    return await run_stream(
        processor,
        emit,
        microphone_index=args.microphone_index,
        duration=args.duration,
        threshold=args.threshold,
    )


def replay_source(args):
    import numpy as np
    import soundfile as sf
    from src.stream_runtime import ReplaySource
    from src.prepare_iemocap import DEFAULT_METADATA
    from src.visual_features import speaker_side

    if args.audio_file:
        audio, sr = sf.read(args.audio_file, dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(1)
        if args.stream_replay and sr != 16000:
            from scipy.signal import resample_poly
            from math import gcd

            divisor = gcd(sr, 16000)
            audio = resample_poly(audio, 16000 // divisor, sr // divisor).astype(
                np.float32
            )
            sr = 16000
        return ReplaySource(audio, sr, None), None
    if not (args.replay_utterance or args.replay_dialog):
        return None, None
    metadata = [json.loads(line) for line in DEFAULT_METADATA.read_text().splitlines()]
    if args.replay_utterance:
        matches = [r for r in metadata if r["utterance_id"] == args.replay_utterance]
        if len(matches) != 1 or matches[0]["session"] == 5:
            raise ValueError(
                "Choose one development utterance; Session 5 remains reserved"
            )
        row = matches[0]
        audio, sr = sf.read(row["audio_path"], dtype="float32")
        return ReplaySource(
            audio,
            sr,
            row["video_path"],
            row["dialog_start_sec"],
            speaker_side(row["dialog_id"], row["speaker"]),
        ), row
    matches = [
        r
        for r in metadata
        if r["dialog_id"] == args.replay_dialog and r["speaker"] == args.speaker
    ]
    if not matches or matches[0]["session"] == 5:
        raise ValueError(
            "Choose a development dialogue/speaker; Session 5 remains reserved"
        )
    row = matches[0]
    path = (
        ROOT
        / f"data/IEMOCAP_full_release/Session{row['session']}/dialog/wav/{args.replay_dialog}.wav"
    )
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim != 2 or sr != 16000:
        raise ValueError("Expected original stereo 16 kHz dialogue audio")
    from src.audit_iemocap_video_alignment import match_audio

    reference = next(r for r in matches if r["duration_sec"] >= 0.75)
    snippet, _ = sf.read(reference["audio_path"], dtype="float32")
    match = match_audio(snippet, audio, reference["dialog_start_sec"])
    if match.get("correlation", 0) < 0.8:
        raise ValueError("Cannot verify the requested speaker's dialogue audio channel")
    selected = np.array(audio[:, match["channel"]], copy=True)
    return ReplaySource(
        selected,
        sr,
        row["video_path"],
        0.0,
        speaker_side(row["dialog_id"], args.speaker),
    ), {
        "dialog_id": args.replay_dialog,
        "speaker": args.speaker,
        "audio_channel_match": match,
    }


def main(argv=None):
    args = parse_args(argv)
    from src.inference_pipeline import SpeechProcessor
    import torch

    torch.set_num_threads(2)
    dashboard, extractor = None, None
    output = None
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output = args.output.open("w")

    context = None
    if args.context != "off":
        from src.emotion_context import SpeakerContext

        context = SpeakerContext.preset(args.context)

    def emit(result):
        if context and result.get("emotion"):
            # One speaker per run, so one context state. The dashboard records
            # this same dict after emit, so it shows the smoothed emotion too.
            result["utterance_emotion"] = result["emotion"]
            probabilities = context.update(result["emotion"]["probabilities"])
            label = max(probabilities, key=probabilities.get)
            result["emotion"] = {
                **result["emotion"],
                "label": label,
                "label_index": list(probabilities).index(label),
                "probabilities": probabilities,
                "context": args.context,
            }
        line = json.dumps(result, allow_nan=False)
        print(line, flush=True)
        if output:
            output.write(line + "\n")
            output.flush()

    try:
        if args.dashboard:
            from src.inspector_dashboard import InspectorDashboard

            dashboard = InspectorDashboard(args.dashboard_port)
            print(f"Inspector: {dashboard.url}", file=sys.stderr, flush=True)
        predictor = None
        if args.emotion_checkpoint:
            from src.emotion_fusion_infer import CrossModalEmotionPredictor

            predictor = CrossModalEmotionPredictor(
                args.emotion_checkpoint, args.device, args.offline
            )
        transcriber = None
        if args.text is None:
            from faster_whisper import WhisperModel

            transcriber = WhisperModel(
                args.model,
                device="cpu",
                compute_type="int8",
                cpu_threads=2,
                local_files_only=args.offline,
            )
        processor = SpeechProcessor(transcriber, predictor)
        visual_predictor = None
        if args.visual_checkpoint:
            from src.visual_fusion import VisualFusionPredictor

            visual_predictor = VisualFusionPredictor(
                args.visual_checkpoint, args.emotion_checkpoint
            )
        source, source_metadata = replay_source(args)
        from src.iemocap_embedding_cache import sha256

        emit(
            {
                "event": "run_config",
                "arguments": {
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in vars(args).items()
                },
                "replay_source": source_metadata,
                "base_checkpoint_sha256": sha256(args.emotion_checkpoint)
                if args.emotion_checkpoint
                else None,
                "visual_checkpoint_sha256": visual_predictor.checkpoint_sha256
                if visual_predictor
                else None,
                "runtime_sha256": {
                    name: sha256(ROOT / "src" / name)
                    for name in (
                        "pipeline_demo.py",
                        "stream_runtime.py",
                        "visual_stream.py",
                        "visual_fusion.py",
                        "inference_pipeline.py",
                        "segmentAudioTorch.py",
                        "audio/capture.py",
                        "inspector_dashboard.py",
                        "inspector.html",
                    )
                },
            }
        )
        streaming = (
            source is None or args.stream_replay or args.replay_dialog is not None
        )
        has_video = source.video_path is not None if source else True
        if visual_predictor and has_video and not args.no_vision:
            from src.visual_stream import VisualExtractor

            extractor = VisualExtractor(
                visual_predictor.payload["feature_config"], args.device
            )
        if dashboard and dashboard.stop_requested.is_set():
            return
        if streaming:
            # Warm the audio/text graph before acquisition and its latency clock.
            if predictor:
                import numpy as np

                for seconds in (1, 6):
                    predictor.predict(
                        "Ready for the next utterance.",
                        np.zeros(16000 * seconds, np.float32),
                        16000,
                    )
            from src.stream_runtime import run_stream

            if source is None:
                print(
                    "Capturing microphone"
                    + (" and camera" if extractor else "")
                    + ". Ctrl+C to stop.",
                    file=sys.stderr,
                    flush=True,
                )
            asyncio.run(
                run_stream(
                    processor,
                    emit,
                    visual_predictor=visual_predictor,
                    extractor=extractor,
                    replay=source,
                    camera_index=(
                        args.camera_index if args.camera_index is not None else 0
                    )
                    if source is None and extractor
                    else None,
                    microphone_index=args.microphone_index,
                    duration=args.duration,
                    threshold=args.threshold,
                    dashboard=dashboard,
                )
            )
        else:
            from src.core.events import SpeechSegment

            processing_started = time.perf_counter()
            start = source.video_start_sec
            end = start + len(source.samples) / source.sample_rate
            segment = SpeechSegment(start, end, source.sample_rate, source.samples)
            if dashboard:
                dashboard.update(status="processing", source="recorded_utterance")
            row, evidence = None, None
            if visual_predictor:
                from src.visual_stream import VisualBuffer, extract_interval

                if extractor:
                    row, evidence = extract_interval(
                        extractor,
                        source.video_path,
                        source.side,
                        start,
                        end,
                        dashboard.vision if dashboard else None,
                    )
                else:
                    row, evidence = VisualBuffer().select(start, end)
            result = processor.process(segment, args.text)
            result.update(start_sec=0.0, end_sec=end - start)
            if visual_predictor:
                result["visual"] = evidence
                if result["emotion"]:
                    result["audio_text_emotion"] = result["emotion"]
                    before = time.perf_counter()
                    result["emotion"] = visual_predictor.predict(
                        result["emotion"]["probabilities"], row
                    )
                    result["timings_ms"]["visual_fusion"] = 1000 * (
                        time.perf_counter() - before
                    )
            if source_metadata:
                result["replay_source"] = {
                    k: source_metadata[k]
                    for k in ("utterance_id", "dialog_id", "speaker")
                    if k in source_metadata
                }
            result["timings_ms"]["recorded_processing_total"] = 1000 * (
                time.perf_counter() - processing_started
            )
            emit(result)
            if dashboard:
                dashboard.record(result)
                dashboard.update(status="finished")
        if dashboard and args.dashboard_hold:
            dashboard.hold(args.dashboard_hold)
    except KeyboardInterrupt:
        pass
    except Exception as error:
        if dashboard:
            dashboard.update(status="error", error=str(error))
        raise
    finally:
        if extractor:
            extractor.close()
        if dashboard:
            dashboard.close()
        if output:
            output.close()


if __name__ == "__main__":
    main()
