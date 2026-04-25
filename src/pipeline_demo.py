"""
Usage:
    python src/pipeline_demo.py [--model tiny.en] [--threshold 0.3]
"""

import threading
import queue
import sys
import numpy as np
import pyaudio
from faster_whisper import WhisperModel

from segmentAudioTorch import SileroVAD, audioSegment, speachSegment, int2float
from textSentiment import TextHiddenStateEncoder, nlp_worker

# Audio constants (must match SileroVAD expectations) FIX: put audio consts in a .env file?
SAMPLE_RATE = 16000
N_SAMPLES = 512  # Silero VAD expects 512 samples per chunk at 16kHz
CHUNK_MS = N_SAMPLES / SAMPLE_RATE * 1000


def vad_worker(
    vad: SileroVAD,
    stream: pyaudio.Stream,
    seg_queue: queue.Queue,
    stop_event: threading.Event,
):
    """
    Capture mic audio, run VAD, push speech segments onto the queue.
    """
    cur_t = 0.0
    while not stop_event.is_set():
        try:
            raw = stream.read(N_SAMPLES, exception_on_overflow=False)
        except OSError:
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

    # Sentinel to tell transcription thread to exit
    seg_queue.put(None)


def transcribe_worker(
    model: WhisperModel,
    seg_queue: queue.Queue,
    stop_event: threading.Event,
    text_queue: queue.Queue | None = None,
):
    """Pull speech segments from the queue and transcribe them."""
    seg_count = 0
    while not stop_event.is_set():
        try:
            segment: speachSegment = seg_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        if segment is None:
            break

        seg_count += 1
        segments, _ = model.transcribe(
            segment.samples,
            language="en",
            beam_size=1,
            vad_filter=False,  # we already did VAD
        )

        text = " ".join(s.text.strip() for s in segments)
        if text:
            # Clear the VAD status line and print the transcription
            sys.stdout.write(f"\r\033[K")
            print(f"  [{segment.start_time:6.1f}s - {segment.end_time:6.1f}s]  {text}")
            if text_queue is not None:
                text_queue.put(
                    {
                        "text": text,
                        "start_time": segment.start_time,
                        "end_time": segment.end_time,
                    }
                )

    if text_queue is not None:
        text_queue.put(None)


def main(
    model_name: str = "tiny.en",
    threshold: float = 0.5,
    text_model_name: str = "distilbert-base-uncased",
):
    print(f"Loading Whisper model '{model_name}'...")
    whisper = WhisperModel(model_name, device="cpu", compute_type="int8")

    print(f"Loading text model '{text_model_name}'...")
    text_encoder = TextHiddenStateEncoder(model_name=text_model_name)

    print("Initialising SileroVAD...")
    vad = SileroVAD(threshold=threshold)

    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=N_SAMPLES,
    )

    seg_queue = queue.Queue()
    text_queue = queue.Queue()
    stop_event = threading.Event()

    vad_thread = threading.Thread(
        target=vad_worker,
        args=(vad, stream, seg_queue, stop_event),
        daemon=True,
    )
    transcribe_thread = threading.Thread(
        target=transcribe_worker,
        args=(whisper, seg_queue, stop_event, text_queue),
        daemon=True,
    )
    text_thread = threading.Thread(
        target=nlp_worker,
        args=(text_queue, stop_event),
        kwargs={"encoder": text_encoder},
        daemon=True,
    )

    print("Listening... speak into the mic. Ctrl+C to stop.\n")

    vad_thread.start()
    transcribe_thread.start()
    text_thread.start()

    try:
        vad_thread.join()
    except KeyboardInterrupt:
        print("\n\nStopping...")
        stop_event.set()

    stream.stop_stream()
    stream.close()
    pa.terminate()


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
