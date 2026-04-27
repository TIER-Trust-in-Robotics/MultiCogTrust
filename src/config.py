"""audio configuration"""

# General
SAMPLE_RATE = 16_000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2

# Silero VAD (Speech Detection)
VAD_CHUNK_SAMPLES = 512  # Silero VAD expects 512 samples per chunk at 16 kHz.
VAD_CHUNK_MS = VAD_CHUNK_SAMPLES / SAMPLE_RATE * 1000
VAD_PRE_BUFFER_SIZE = 10  # number of audio samples saved in
VAD_MIN_SPEECH_MS = 300
VAD_MIN_SILENCE_MS = 300
VAD_THRESHOLD = 0.5

# WhisperLive (Speech Transcription)
TRANSCRIPTION_CHUNK_MS = 100
TRANSCRIPTION_CHUNK_SAMPLES = int(SAMPLE_RATE * TRANSCRIPTION_CHUNK_MS / 1000)

# Transcriber (must support fast_fast)
MODEL_NAME = "distilbert-base-uncased"
