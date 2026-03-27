import torch
import numpy as np
from collections import deque
from silero_vad import load_silero_vad


# Custom DataTypes for audio.


class speachSegment:
    """
    Segment of audio corresponding to speach.
    """

    def __init__(
        self,
        samples: np.ndarray,
        start_time: float,
        end_time: float,
        sample_rate: int = 16000,
    ):
        self.samples = samples
        self.start_time = start_time
        self.end_time = end_time
        self.sample_rate = sample_rate

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class audioSegment:
    """
    Segment of audio picked up by SileroVad, possibly background noise, or actual speech.
    """

    def __init__(
        self,
        samples: np.ndarray,
        timestamp: float,
        sample_rate: int = 16000,
    ):
        self.samples = samples
        self.timestamp = timestamp
        self.sample_rate = sample_rate

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate


# Main function


class SileroVAD:
    """
    Wrapper for Silero VAD model (PyTorch JIT version).

    How it work:
    Silero VAD has two states, IDLE and SPEAKING (self._speaking).


    IDLE state:

    AudioSample (audio samples from pyAudio) are continuously added to a pre buffer in Silero VAD (self._pre_buffer). These audioSegments are passed through


    Silero VAD process each audioSegment sample to determine if it contains spoken words, adding them to the pre buffer in positive cases. Once the pre buffer reaches a certain size, the model enters the SPEAKING STATE and the pre buffer is dumped into the speach chunk (self.speach_chunk)


    SPEAKING state:

    All incoming audioBuffers are then added to the building speachChunk.

    When a certain number of audioChunks with no spoken words are detected sequentially, the model returns to the IDLE state. This transition also has the speach chunks concatenated to form a speachChunk.

    """

    def __init__(
        self,
        threshold: float = 0.5,
        pre_buffer_size: int = 10,
        sample_rate: int = 16000,
        chunk_ms: int = 32,
        min_speech_ms: int = 300,  # duration of detected speach before switching from IDLE to SPEAKING
        min_silence_ms: int = 300,  # duration of detected silence before switching from SPEAKING to IDLE
    ):
        self.model = load_silero_vad()
        """
        The torch JIT version of Silero VAD manages its own internal state,
        so we don't need to manually track state tensors like the ONNX version.
        Call model.reset_states() to clear the internal state.
        """

        self._threshold = threshold
        self._sample_rate = sample_rate
        self._chunk_ms = chunk_ms

        self._speaking = False  # used for connecting speech chunks together in an utterance. i.e. a sentence.
        self._speech_count = 0
        self._silence_count = 0
        self.min_speech_chunks = max(
            1, int(min_speech_ms / chunk_ms)
        )  # threshold for self._speaking to be True (speaking mode)
        self.min_silence_chunks = max(
            1, int(min_silence_ms / chunk_ms)
        )  # threshold for self._speaking to be False (idle mode)

        self._speech_chunks: list[speachSegment] = []
        self._pre_buffer: deque = deque(maxlen=pre_buffer_size)

    def _finish_segment(self) -> audioSegment:
        utterance = np.concatenate([chunk.samples for chunk in self._speech_chunks])

        return speachSegment(
            samples=utterance,
            start_time=self._speech_chunks[0].timestamp,
            end_time=self._speech_chunks[-1].timestamp
            + self._speech_chunks[-1].duration,
            sample_rate=self._sample_rate,
        )

    def _speach_prob(self, samples: np.ndarray) -> float:
        """
        The torch JIT model accepts a 1D tensor of float32 samples
        and the sampling rate. It handles batching and state internally.
        """
        audio_tensor = torch.from_numpy(samples.astype(np.float32))
        prob = self.model(audio_tensor, self._sample_rate)
        return float(prob)

    def process_chunk(self, chunk: audioSegment):
        """
        Responsible for adding audioSegments to the buffer and determining if buffer is background noise or spoken words. Output is a speechSegment in the later case, None in the former.
        """

        prob = self._speach_prob(chunk.samples)
        is_speech = prob > self._threshold

        if self._speaking:  # SPEAKING
            self._speech_chunks.append(chunk)

            if not is_speech:
                self._silence_count += 1

                if self._silence_count >= self.min_silence_chunks:
                    segment = self._finish_segment()
                    self._reset_segment()
                    return prob, segment
            else:
                self._silence_count = 0
        else:  # IDLE
            if is_speech:
                self._speech_count += 1
                self._pre_buffer.append(chunk)

                if self._speech_count >= self.min_speech_chunks:
                    self._speaking = True
                    self._silence_count = 0
                    self._speech_chunks = list(self._pre_buffer)
            else:
                self._speech_count = 0
                self._pre_buffer.append(chunk)

        return prob, None

    def _reset_segment(self):
        self._pre_buffer.clear()
        self._speech_chunks = []
        self._speaking = False
        self._silence_count = 0
        self._speech_count = 0

    def _reset_all(self):
        self.model.reset_states()
        self._reset_segment()


# from: https://github.com/snakers4/silero-vad/blob/master/examples/pyaudio-streaming/pyaudio-streaming-examples.ipynb (the Siler VAD input must be of type float32)
def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype("float32")
    if abs_max > 0:
        sound *= 1 / 32768
    sound = sound.squeeze()  # depends on the use case
    return sound


if __name__ == "__main__":
    import pyaudio
    import wave
    import sys

    SAMPLE_RATE = 16000
    N_SAMPLES = (
        512  # newer versions of Silero VAD expect 512 samples per chunk at 16kHz
    )
    CHUNK_MS = N_SAMPLES / SAMPLE_RATE * 1000  # 32 ms

    VAD = SileroVAD(threshold=0.2)
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        input=True,
        rate=SAMPLE_RATE,
        frames_per_buffer=N_SAMPLES,
    )

    frames = []
    probabilities = []
    speach_segments = []
    cur_t = 0.0

    segment_count = 0
    try:
        while True:
            raw = stream.read(N_SAMPLES, exception_on_overflow=False)
            frames.append(raw)

            audio_int16 = np.frombuffer(raw, dtype=np.int16)
            audio_float32 = int2float(audio_int16)

            timestamp = cur_t
            cur_t += CHUNK_MS / 1000.0

            aSegment = audioSegment(
                samples=audio_float32,
                timestamp=timestamp,
                sample_rate=SAMPLE_RATE,
            )

            prob, result = VAD.process_chunk(aSegment)

            if result is not None:
                segment_count += 1
                speach_segments.append(result)

            bar = "█" * int(prob * 30) + "░" * (30 - int(prob * 30))
            marker = " ◄ SPEECH" if prob > VAD._threshold else "   IDLE"

            state = "SPEAKING" if VAD._speaking else "IDLE"

            line = f"[{timestamp:7.2f}s] [{state:>8s}] {bar} {prob:.2f}{marker}"

            sys.stdout.write(f"\r{line}    ")
            sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n\nStopping...")
        print(f"Detected {segment_count} speech segment(s)")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    wf = wave.open("out.wav", "wb")
    wf.setnchannels(1)
    wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(b"".join(frames))
    wf.close()
