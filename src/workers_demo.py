from __future__ import annotations
import asyncio

from src.audio.capture import AudioCaptureConfig, run_audio_capture
from src.audio.vad import VADConfig, run_vad_worker
from src.text.nlp import NLPConfig, run_nlp_worker
from src.workers.transcribe import WhisperConfig, run_transcribe_worker
from src.core.shutdown import cancel_tasks


async def main() -> None:
    stop_event = asyncio.Event()

    audio_chunks = asyncio.Queue(maxsize=32)
    speech_segments = asyncio.Queue(maxsize=16)
    transcripts = asyncio.Queue(maxsize=16)
    text_emebeddings = asyncio.Queue(maxsize=16)

    tasks = [
        asyncio.create_task(
            run_audio_capture(
                audio_chunks,
                stop_event,
                AudioCaptureConfig,
            ),
            name="audio_capture",
        ),
        asyncio.create_task(
            run_vad_worker(
                audio_chunks,
                speech_segments,
                stop_event,
                VADConfig,
            ),
            name="vad",
        ),
        asyncio.create_task(
            run_transcribe_worker(
                speech_segments,
                transcripts,
                stop_event,
                WhisperConfig,
            ),
            name="transcription",
        ),
        asyncio.create_task(
            run_nlp_worker(
                transcripts,
                text_emebeddings,
                stop_event,
                NLPConfig,
            ),
            name="nlp",
        ),
    ]

    try:
        while not stop_event.is_set():
            event = await text_emebeddings.get()
            print(f"[{event.start_time:.2f}s - {event.end_time:.2f}s]{event.text}")
    except KeyboardInterrupt:
        print("\nStopping...")
        stop_event.set()

    finally:
        stop_event.set()
        await cancel_tasks(tasks)


if __name__ == "__main__":
    asyncio.run(main())
