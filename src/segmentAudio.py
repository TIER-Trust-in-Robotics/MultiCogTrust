import numpy as np
import onnxruntime as ort
from collections import deque


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
    Wrapper for Silero VAD model.

    How it work:
    Silero VAD has two states, IDLE and SPEAKING (self._speaking).


    IDLE state:

    AudioSample (audio samples from pyAudio) are continously addekkd to a pre buffer in Silero VAD (self._pre_buffer). These audioSegments are passed through


    Silero VAD process each audioSegment sample to determin if it contains spoken words, adding them to the pre buffer in positive cases. Once the pre buffer reaches a certain size, the model enters the SPEAKING STATE and the pre buffer is dumped into the speach chunk (self.speach_chunk)


    SPEAKING state:

    All incoming audioBuffers are then added to the building speachChunk.

    When a certain number of audioChunks will no spoken words are detected sequentially, the model returns to the IDLE state. This transition also has the speach chunks concatonated to form a speachChunk.

    """

    def __init__(
        self,
        model_path: str = "weights/silero_vad.onnx",
        threshold: float = 0.5,
        pre_buffer_size: int = 10,
        sample_rate: int = 16000,
        chunk_ms: int = 30,
        min_speech_ms: int = 300,  # durration of detected speach before switching from IDLE to SPEAKING
        min_silence_ms: int = 300,  # duriaton of detected silence before switching from SPEAKING to IDLE
    ):
        self.model = ort.InferenceSession(
            model_path,
            providers=[
                "CPUExecutionProvider"
            ],  # change to CUDA when deployed on Jetson Orin Nano
        )
        """
        The ONNX version of silero VAD expects batched inputs (batch_size, num_samples) which requires an batch_size axis be added to the audio before passing.
        """
        self._sr = np.array(sample_rate, dtype=np.int64)
        # self._h = np.zeros((2, 1, 16), dtype=np.float32)
        # self._c = np.zeros((2, 1, 16), dtype=np.float32)
        self._state = np.zeros((2, 1, 128), dtype=np.float32)

        self._threshold = threshold
        self._sample_rate = sample_rate
        self._chunk_ms = chunk_ms

        self._speaking = False  # used for connecting speech chunks together in an utterance. i.e. a sentence.
        self._speech_count = 0  #
        self._silence_count = 0  #
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
        Silero on Onnx expects inputs to be of the form (batch_size, chunk_samples)
        """
        input_data = samples.astype(np.float32).reshape(1, -1)

        out, self._state = self.model.run(
            None,
            {
                "input": input_data,
                "state": self._state,
                "sr": self._sr,
            },
        )

        return float(out[0][0])  # out is probability in the range [0.0, 1.0]

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

        return None, prob

    def _reset_segment(self):
        self._pre_buffer.clear()
        self._speech_chunks = []
        self._speaking = False
        self._silence_count = 0
        self._speech_count = 0

    def _reset_all(self):
        self._state = np.zeros_like(self._state)
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

    # import time
    import sys

    SAMPLE_RATE = 16000
    N_SAMPLES = (
        512  # newer versions of Silero VAD expect 512 samples per chunk at 16kHz
    )
    CHUNK_MS = N_SAMPLES / SAMPLE_RATE * 1000  # 32 ms

    VAD = SileroVAD()
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
    speach_segments: []
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

            result, prob = VAD.process_chunk(aSegment)

            bar = "█" * int(prob * 30) + "░" * (30 - int(prob * 30))
            marker = " ◄ SPEECH" if prob > VAD._threshold else ""

            state = "SPEAKING" if VAD._speaking else "IDLE"

            # use \r to overwrite line in IDLE, newline on state transitions
            line = f"[{timestamp:7.2f}s] [{state:>8s}] {bar} {prob:.3f}{marker}"
            sys.stdout.write(f"\r{line}    ")
            sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n\nStopping...")
        print(f"Detected {segment_count} speech segment(s)")

    stream.stop_stream()
    stream.close()
    audio.terminate()
