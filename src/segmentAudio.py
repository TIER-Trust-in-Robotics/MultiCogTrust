import silero_vad
import numpy as np
import onnxruntime as ort
from collections import deque


# Custom DataTypes


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


#
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


class SileroVAD:
    """
    Wrapper for Silero VAD model

    How it work:
    Silero VAD has two states, IDLE and SPEAKING (self._speaking).


    ## IDLE state

    AudioSample (audio samples from pyAudio) are continously addekkd to a pre buffer in Silero VAD (self._pre_buffer). These audioSegments are passed through


    Silero VAD process each audioSegment sample to determin if it contains spoken words, adding them to the pre buffer in positive cases. Once the pre buffer reaches a certain size, the model enters the SPEAKING STATE and the pre buffer is dumped into the speach chunk (self.speach_chunk)


    ## SPEAKING state
    All incoming audioBuffers are then added to the


    When a certain number of audioChunks will no spoken words are detected sequentially, the model returns to the IDLE state. This transition also has the speach chunks concatonated to form a speachChunk.

    """

    def __init__(
        self,
        model_path: str = "weights/silero_vad.onnx",
        threshold: float = 0.5,
        pre_buffer_size: int = 10,
        sample_rate: int = 16000,
        chunk_ms: int = 30,
        min_speech_ms: int = 300,
        min_silence_ms: int = 300,
    ):
        self.model = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        """
        The ONNX version of silero VAD expects batched inputs (batch_size, num_samples) which requires an batch_size axis be added to the audio before passing.
        """

        self._state = np.zeros(
            (2, 1, 128), dtype=np.float32
        )  # unified LSTM state (v6 API)
        self._sr = np.array(
            [sample_rate], dtype=np.int64
        )  # sample rate as array to make reshaping inputs easier

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
        utterance = np.concat([chunk.sampes for chunk in self._speech_chunks])

        return audioSegment(
            samples=utterance,
            start_time=self._speech_chunks[0].timestamp,
            end_time=self._speech_chunks[-1].timestamp,
            sample_rate=self._sample_rate,
        )

    def _speach_prob(self, samples: np.ndarray) -> float:
        audio = samples[np.newaxis, :].astype(
            np.float32
        )  # adding batch dim, shape is now (1, n_samples)

        out, self._state = self.model.run(
            None,
            {
                "input": audio,
                "state": self._state,
                "sr": self._sr,
            },
        )

        return float(out[0][0])  # out is probability in the range [0.0, 1.0]

    def process_chunk(self, chunk: audioSegment) -> speachSegment | None:
        """
        Responsible for adding audioSegments to the buffer and determining if buffer is background noise or spoken words. Output is a speechSegment in the later case, None in the former.
        """

        prob = self._speach_prob(chunk.samples)
        is_speech = prob > self._threshold

        if self._speaking:  # model is in SPEAKING state
            self._speech_chunks.append(chunk)

            if not is_speech:  # speaker is no longer speakign
                self._silence_count += 1

                if (
                    self._silence_count >= self.min_silence_chunks
                ):  # dump audioSegments into speechSegment (speaker is finished)
                    segment = self._finish_segment()
                    self.reset()
                    return segment
            else:  # speaker resumed, reset silence counter
                self._silence_count = 0
        else:  # model is in IDLE state
            if is_speech:
                self._speech_count += 1
                if self._speech_count >= self.min_speech_chunks:
                    self._speaking = True
                    self._silence_count = 0
                    self._speech_chunks = list(self._pre_buffer)
                    self._speech_chunks.append(chunk)
            else:
                self._pre_buffer.append(chunk)

        return None

    def reset(self):
        self._pre_buffer.clear()
        self._speech_chunks = []
        self._speaking = False
        self._silence_count = 0
        self._speech_count = 0


# PyAudio Wrapper that turns raw audio from PyAudio into audioSegment objects

# Demo


# from: https://github.com/snakers4/silero-vad/blob/master/examples/pyaudio-streaming/pyaudio-streaming-examples.ipynb
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
    import cv2
    import time
    import matplotlib.pyplot as plt

    SAMPLE_RATE = 16000
    CHUNK_MS = 30
    CHUNK = int(SAMPLE_RATE / 10)

    # VAD = SileroVAD(sample_rate=SAMPLE_RATE, chunk_ms=CHUNK_MS)
    VAD = SileroVAD()
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        input=True,
        rate=SAMPLE_RATE,
        frames_per_buffer=CHUNK,
    )
    # cap = cv2.VideoCapture(0)

    # paramets not understood

    n_samples = 512

    probabilities = []

    data = []

    print("Listening...")
    for i in range(0, 100):
        audio_chunks = stream.read(n_samples)
        data.append(audio_chunks)

        audio_chunks_int16 = np.frombuffer(audio_chunks, np.int16)
        audio_chunks_float32 = int2float(audio_chunks_int16)

        speach_prob = VAD._speach_prob(audio_chunks_float32)
        probabilities.append(speach_prob)
        print(speach_prob)

        if speach_prob > VAD._threshold:
            print("Speaking")

    # Cleanup
    print("Done")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    X = np.arange(len(probabilities))

    wf = wave.open("outvoice.wav", "wb")
    wf.setnchannels(1)
    wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(b"".join(data))
    wf.close()

    plt.plot(X, probabilities)
    # print(max())
    plt.show()
