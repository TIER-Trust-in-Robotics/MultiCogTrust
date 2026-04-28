import asyncio
import numpy as np
import pyaudio

# Refer to: https://github.com/QuentinFuxa/WhisperLiveKit for document on WhisperLive
from whisperlivekit import TranscriptionEngine, AudioProcessor

# Audio Const
SAMPLE_RATE = 16000
CHANELS = 1
CHUNCKS_MS = 100
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNCKS_MS / 1000)


def createEngine(model: str) -> TranscriptionEngine:
    engine = TranscriptionEngine(
        model=model,
        lan=model,
        pcm_input=True,  # to skp FFmpeg and feed raw s16le bytes, the same kind that is used by pyaudio
    )
    return engine


# List of supported model can be found here: https://github.com/QuentinFuxa/WhisperLiveKit/blob/main/docs/default_and_custom_models.md
# Using the lightest weight model, "tiny.en". Works surprisingly well.
async def run(model: str):
    engine = createEngine(model)

    processor = AudioProcessor(transcription_engine=engine)
    results = await processor.create_tasks()

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

    async def consume():
        try:
            async for r in results:
                pass
        except asyncio.CancelledError:
            pass

    feed_task = asyncio.create_task(feed())
    consume_task = asyncio.create_task(consume())

    try:
        await asyncio.gather(feed_task, consume_task)
    except KeyboardInterrupt:
        pass
    finally:
        feed_task.cancel()
        consume_task.cancel()
        stream.stop_stream()
        stream.close()
        audio.terminate()
