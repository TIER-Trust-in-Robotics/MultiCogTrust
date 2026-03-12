# Package manager
We will be using the [uv](https://docs.astral.sh/uv/) package manager for dependencies.

Once installed, create a virutal environment with uv, then install all libraries using `uv sync`.

# Structure

| Modality | input | Libraries | Output | 
| ----- | ----- | ----- | ----- |
| Visual | OpenCV | ??? | AU activations
| Visual | OpenCv | MediaPipe | Head and eye gaze classification | 
| Audio | PyAudio | Silero VAD | speachSegment object |
| Audio | speachSegment objects (senteice) | OpenSmile | Utterance Signal analysis |
| Audio | speachSegment | Whisper-faster or WhisperLiveKit | Transcription of Utterance
| Text | Transcription | tiny/distilBERT | Emotions (multi-)classification |
| Audio + Text | combined tensor using some embedding |  ???? | Sentiment analysis |

# Proposed Architecture
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  Vision     │  │  Audio      │  │    Text     │
│  Pipeline   │  │  Pipeline   │  │  Pipeline   │
│             │  │             │  │             │
│ Gaze Track  │  │ VAD/Feature │  │  ASR Output │
│ + FAU Det.  │  │ Extraction  │  │  + NLU      │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │
   [emb_v]          [emb_a]          [emb_t]
       │                │                │
       └────────┬───────┴────────┬───────┘
                │                │
         ┌──────▼──────┐  ┌─────▼──────┐
         │ Cross-Modal │  │ Temporal   │
         │ Attention   │  │ Aggregator │
         │ (optional)  │  │ (GRU/LSTM) │
         └──────┬──────┘  └─────┬──────┘
                └───────┬───────┘
                   ┌────▼────┐
                   │ Trust   │
                   │ Head    │
                   │ (MLP)   │
                   └─────────┘
