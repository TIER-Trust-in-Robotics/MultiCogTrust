import asyncio
import numpy as np
import time
import pyaudio
import sys
from pprint import pprint

# Refer to: https://github.com/QuentinFuxa/WhisperLiveKit
# For document on WhisperLive
from whisperlivekit import TranscriptionEngine, AudioProcessor

# Consider also using the mini backend version:  https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602, works well with metal MLX!


# Audio Const
SAMPLE_RATE = 16000
CHANELS = 1
CHUNCKS_MS = 100
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNCKS_MS / 1000)


# should default to 'tiny.en'
def createEngine(model: str) -> TranscriptionEngine:
    # basic verison
    engine = TranscriptionEngine(
        model=model,
        lan="en",  # always using english for our use case, but this model supports other langauge
        pcm_input=True,  # to skp FFmpeg and feed raw s16le bytes, the same kind that is used by pyaudio
    )

    return engine

    # advanced version to use cli keywords??


# see: https://github.com/QuentinFuxa/WhisperLiveKit/blob/main/docs/API.md
def display(response: dict):
    pass  # working on


# List of supported model can be found here: https://github.com/QuentinFuxa/WhisperLiveKit/blob/main/docs/default_and_custom_models.md
# Using "tiny.en" as default as it is the lightest weight
async def run(model: str):
    engine = createEngine(model)

    processor = AudioProcessor(transcription_engine=engine)
    results = await processor.create_tasks()

    # pyAudio
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=CHANELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SAMPLES,
    )

    print("Listening... Ctrl+C to stop.\n")

    loop = asyncio.get_event_loop()

    """
    Brief explaination on using 'async' keyword in python with the 'asyncio' library.

    Note that full asyncronis programming requires releasing interpretor lock.

    """

    # Explain
    async def feed():
        try:
            while True:
                pcm = await loop.run_in_executor(
                    None,
                    lambda: stream.read(CHUNK_SAMPLES, exception_on_overflow=False),
                )
                await processor.process_audio(pcm)
        except asyncio.CancelledError:
            pass

    # Explain
    async def consume():
        try:
            async for r in results:
                display(r)
        except asyncio.CancelledError:
            pass

    feed_task = asyncio.create_task(feed())
    consume_task = asyncio.create_task(consume())

    # Explain
    try:
        await asyncio.gather(feed_task, consume_task)  #
    except KeyboardInterrupt:
        pass
    finally:
        feed_task.cancel()
        consume_task.cancel()
        stream.stop_stream()
        stream.close()
        audio.terminate()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default="tiny.en",
        help="Whisper model (tiny.en, base, small, medium, large-v3)",
    )
    args = p.parse_args()

    asyncio.run(run(args.model))
