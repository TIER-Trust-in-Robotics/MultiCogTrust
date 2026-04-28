import torch
import numpy as np
from collections import deque
from silero_vad import load_silero_vad

from src.core.events import AudioChunk, SpeechSegment


# Main function


class SileroVAD:
    """
    Wrapper for Silero VAD model (PyTorch JIT version).

    How it work:
    Silero VAD has two states, IDLE and SPEAKING (self._speaking).

    IDLE state:

    SileroVAD processes each audio sample (~32ms) directly from PyAudio (audioSegment) and calculates the probability of it containing spooken word. All segments that pass the threshold (tunable with self.threshold) are added to a buffer. Once the buffer reaches sufficient size (tunable with self.min_speAech_ms) within an certain time (clearning every??? ) interval the model enters the SPEAKING state.


    SPEAKING state:

    SilerVAD starts to conncate all speech positive audioSegments into a speechSegment. If a continous stream of speech negative audio segments are detected (tunable by self.min_silence_ms), the speechSegmenet is constructued and returned.

    """

    def __init__(
        self,
        threshold: float = 0.5,
        pre_buffer_size: int = 10,
        sample_rate: int = 16000,
        chunk_ms: float = 32.0,
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

        self._speech_chunks: list[AudioChunk] = []
        self._pre_buffer: deque = deque(maxlen=pre_buffer_size)

    def _finish_segment(self) -> SpeechSegment:
        utterance = np.concatenate([chunk.samples for chunk in self._speech_chunks])

        return SpeechSegment(
            samples=utterance,
            start_time=self._speech_chunks[0].timestamp,
            end_time=self._speech_chunks[-1].end_time,
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

    def process_chunk(self, chunk: AudioChunk) -> tuple[float, SpeechSegment | None]:
        """
        Responsible for adding audioSegments to the buffer and determining if buffer is background noise or spoken words. Output is a speechSegment in the later case, None in the former.
        """

        chunk = AudioChunk(
            samples=int2float(chunk.samples),
            timestamp=chunk.timestamp,
            sample_rate=chunk.sample_rate,
        )

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
    if np.issubdtype(sound.dtype, np.floating):
        return sound.astype("float32").squeeze()

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

    def close_audio(stream, audio):
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

        if audio is not None:
            try:
                audio.terminate()
            except OSError:
                pass

    def write_recording(path: str, frames: list[bytes], sample_width: int) -> None:
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(sample_width)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b"".join(frames))

    def main() -> None:
        vad = SileroVAD(threshold=0.2)
        audio = None
        stream = None
        frames: list[bytes] = []
        cur_t = 0.0
        segment_count = 0
        sample_width = pyaudio.get_sample_size(pyaudio.paInt16)

        try:
            audio = pyaudio.PyAudio()
            sample_width = audio.get_sample_size(pyaudio.paInt16)
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                input=True,
                rate=SAMPLE_RATE,
                frames_per_buffer=N_SAMPLES,
            )

            while True:
                raw = stream.read(N_SAMPLES, exception_on_overflow=False)
                frames.append(raw)

                timestamp = cur_t
                cur_t += CHUNK_MS / 1000.0

                chunk = AudioChunk(
                    samples=np.frombuffer(raw, dtype=np.int16).copy(),
                    timestamp=timestamp,
                    sample_rate=SAMPLE_RATE,
                )

                prob, result = vad.process_chunk(chunk)

                if result is not None:
                    segment_count += 1

                bar = "█" * int(prob * 30) + "░" * (30 - int(prob * 30))
                marker = " ◄ SPEECH" if prob > vad._threshold else "   IDLE"
                state = "SPEAKING" if vad._speaking else "IDLE"
                line = f"[{timestamp:7.2f}s] [{state:>8s}] {bar} {prob:.2f}{marker}"

                sys.stdout.write(f"\r{line}    ")
                sys.stdout.flush()

        except KeyboardInterrupt:
            sys.stdout.write("\r\033[K")
            print("\nStopping...")
        finally:
            print(f"Detected {segment_count} speech segment(s)")
            close_audio(stream, audio)
            write_recording("out.wav", frames, sample_width)

    main()
