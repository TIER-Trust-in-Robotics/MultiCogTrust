import silero_vad
import numpy as np
import onnxruntime as ort
from collections import deque


# datatype for storing audio segments
class audioSegment:
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


class speachSegment:
    samples: np.ndarray
    timestamp: float
    sample_rate: int = 16000

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate


class SileroVAD:
    """
    Wrapper for Silero VAD model

    How it work:
    Silero VAD has two states, IDLE and SPEAKING (self._speaking)
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
        The ONNX version expects batched inputs (batch_size, num_samples) which requires an batch_size axis be added to the audio before passing.
        """

        self._h = np.zeros((2, 1, 64), dtype=np.float32)  # hidden layer of SileroVAD
        self._c = np.zeros((2, 1, 64), dtype=np.float32)  # LSTM cell of SileroVAD
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

        self._speech_chunks = []
        self._pre_buffer: deque = deque(maxlen=pre_buffer_size)

    def _speach_prob(self, samples: np.ndarray) -> float:
        audio = samples[np.newaxis, :].astype(
            np.float32
        )  # adding batch dim, shape is now (1,480)

        out, self._h, self._c = self.model.run(
            None,  # output_names, options 'output', 'hn', or 'cn'
            {
                "input": audio,
                "h": self._h,
                "c": self._c,
                "sr": self._sr,
            },  # input_feed
        )

        return float(out[0][0])  # out is probability in the range [0.0, 1.0]
