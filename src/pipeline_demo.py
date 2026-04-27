"""
Usage:
    uv run python src/pipeline_demo.py [--model tiny.en] [--threshold 0.3]
"""

import threading
import queue
import sys
import traceback
import numpy as np
import pyaudio
from faster_whisper import WhisperModel

from segmentAudioTorch import SileroVAD, audioSegment, speachSegment, int2float
from textSentiment import TextHiddenStateEncoder, nlp_worker

# Audio constants (must match SileroVAD expectations) FIX: put audio consts in a .env file?
SAMPLE_RATE = 16000
N_SAMPLES = 512  # Silero VAD expects 512 samples per chunk at 16kHz
CHUNK_MS = N_SAMPLES / SAMPLE_RATE * 1000


def _record_worker_error(
    error_queue: queue.Queue | None,
    worker_name: str,
    exc: BaseException,
) -> None:
    if error_queue is not None:
        error_queue.put((worker_name, exc, traceback.format_exc()))


def vad_worker(
    vad: SileroVAD,
    stream: pyaudio.Stream,
    seg_queue: queue.Queue,
    stop_event: threading.Event,
    error_queue: queue.Queue | None = None,
):
    """
    Capture mic audio, run VAD, push speech segments onto the queue.
    """
    try:
        cur_t = 0.0
        while not stop_event.is_set():
            try:
                raw = stream.read(N_SAMPLES, exception_on_overflow=False)
            except OSError as exc:
                if not stop_event.is_set():
                    _record_worker_error(error_queue, "VAD", exc)
                    stop_event.set()
                break

            audio_int16 = np.frombuffer(raw, dtype=np.int16)
            audio_float32 = int2float(audio_int16)

            chunk = audioSegment(
                samples=audio_float32,
                timestamp=cur_t,
                sample_rate=SAMPLE_RATE,
            )
            cur_t += CHUNK_MS / 1000.0

            prob, segment = vad.process_chunk(chunk)

            # Live VAD indicator
            bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
            state = "SPEAKING" if vad._speaking else "IDLE"
            sys.stdout.write(f"\r  VAD [{state:>8s}] {bar} {prob:.2f}  ")
            sys.stdout.flush()

            if segment is not None:
                seg_queue.put(segment)
    except Exception as exc:
        _record_worker_error(error_queue, "VAD", exc)
        stop_event.set()
    finally:
        # Sentinel to tell transcription thread to exit.
        seg_queue.put(None)


def transcribe_worker(
    model: WhisperModel,
    seg_queue: queue.Queue,
    stop_event: threading.Event,
    text_queue: queue.Queue | None = None,
    error_queue: queue.Queue | None = None,
):
    """Pull speech segments from the queue and transcribe them."""
    try:
        while True:
            try:
                segment: speachSegment = seg_queue.get(timeout=0.5)
            except queue.Empty:
                if stop_event.is_set():
                    break
                continue

            if segment is None:
                break

            segments, _ = model.transcribe(
                segment.samples,
                language="en",
                beam_size=1,
                vad_filter=False,  # we already did VAD
            )

            text = " ".join(s.text.strip() for s in segments)
            if text:
                # Clear the VAD status line and print the transcription.
                sys.stdout.write("\r\033[K")
                print(
                    f"  [{segment.start_time:6.1f}s - {segment.end_time:6.1f}s]  {text}"
                )
                if text_queue is not None:
                    text_queue.put(
                        {
                            "text": text,
                            "start_time": segment.start_time,
                            "end_time": segment.end_time,
                        }
                    )
    except Exception as exc:
        _record_worker_error(error_queue, "transcription", exc)
        stop_event.set()
    finally:
        if text_queue is not None:
            text_queue.put(None)


def text_worker(
    text_queue: queue.Queue,
    stop_event: threading.Event,
    error_queue: queue.Queue | None = None,
    encoder: TextHiddenStateEncoder | None = None,
) -> None:
    try:
        nlp_worker(text_queue, stop_event, encoder=encoder)
    except Exception as exc:
        _record_worker_error(error_queue, "text", exc)
        stop_event.set()


def _close_audio(stream: pyaudio.Stream | None, pa: pyaudio.PyAudio | None) -> None:
    if stream is not None:
        try:
            if stream.is_active():
                stream.stop_stream()
        except OSError:
            pass
        finally:
            try:
                stream.close()
            except OSError:
                pass

    if pa is not None:
        try:
            pa.terminate()
        except OSError:
            pass


def _drain_worker_errors(
    error_queue: queue.Queue,
) -> list[tuple[str, BaseException, str]]:
    errors = []
    while True:
        try:
            errors.append(error_queue.get_nowait())
        except queue.Empty:
            return errors


def main(
    model_name: str = "tiny.en",
    threshold: float = 0.5,
    text_model_name: str = "distilbert-base-uncased",
):
    print(f"Loading text model '{text_model_name}'...")
    text_encoder = TextHiddenStateEncoder(model_name=text_model_name)

    print(f"Loading Whisper model '{model_name}'...")
    whisper = WhisperModel(model_name, device="cpu", compute_type="int8")

    print("Initialising SileroVAD...")
    vad = SileroVAD(threshold=threshold)

    pa = None
    stream = None

    seg_queue = queue.Queue()
    text_queue = queue.Queue()
    error_queue = queue.Queue()
    stop_event = threading.Event()

    pa = pyaudio.PyAudio()
    try:
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=N_SAMPLES,
        )
    except Exception:
        _close_audio(stream, pa)
        raise

    vad_thread = threading.Thread(
        target=vad_worker,
        args=(vad, stream, seg_queue, stop_event, error_queue),
        name="vad-worker",
    )
    transcribe_thread = threading.Thread(
        target=transcribe_worker,
        args=(whisper, seg_queue, stop_event, text_queue, error_queue),
        name="transcribe-worker",
    )
    text_thread = threading.Thread(
        target=text_worker,
        args=(text_queue, stop_event, error_queue, text_encoder),
        name="text-worker",
    )

    print("Listening... speak into the mic. Ctrl+C to stop.\n")

    vad_thread.start()
    transcribe_thread.start()
    text_thread.start()

    threads = [vad_thread, transcribe_thread, text_thread]
    exit_code = 0

    try:
        while any(thread.is_alive() for thread in threads):
            for thread in threads:
                thread.join(timeout=0.2)

            errors = _drain_worker_errors(error_queue)
            if errors:
                stop_event.set()
                exit_code = 1
                sys.stdout.write("\r\033[K")
                for worker_name, exc, tb in errors:
                    print(f"\n{worker_name} worker failed: {exc}", file=sys.stderr)
                    print(tb, file=sys.stderr)
                break
    except KeyboardInterrupt:
        sys.stdout.write("\r\033[K")
        print("\nStopping...")
        stop_event.set()
    finally:
        stop_event.set()
        _close_audio(stream, pa)
        seg_queue.put(None)
        text_queue.put(None)

        for thread in threads:
            thread.join(timeout=2.0)

        for worker_name, exc, tb in _drain_worker_errors(error_queue):
            exit_code = 1
            print(
                f"\n{worker_name} worker failed during shutdown: {exc}",
                file=sys.stderr,
            )
            print(tb, file=sys.stderr)

    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="tiny.en", help="Whisper model name")
    p.add_argument("--threshold", type=float, default=0.5, help="VAD threshold")
    p.add_argument(
        "--text-model",
        default="distilbert-base-uncased",
        help="Hugging Face encoder model used for token and word hidden states",
    )
    args = p.parse_args()

    main(
        model_name=args.model, threshold=args.threshold, text_model_name=args.text_model
    )
