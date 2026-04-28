# IEMOCAP Alignment And Feature Space

This note explains how the IEMOCAP data in this repository is converted into
word-aligned text and prosodic features for multimodal sentiment or emotion
models. It is written for developers who need to understand the data shape
before working on cross-modal attention.

## Big Picture

For each IEMOCAP utterance, the pipeline tries to build one training example
with:

- transcript text
- emotion label
- valence, activation, dominance scores
- utterance WAV path
- word-level start and end times
- temporal acoustic/prosodic features
- word-aligned acoustic/prosodic features

The important idea is:

```text
audio time spans -> aligned words -> DistilBERT tokens
```

IEMOCAP is useful for this because it already includes forced-alignment files
that tell us when each word occurs in the audio.

## Relevant Files

Raw IEMOCAP files live under:

```text
data/IEMOCAP_full_release/
```

The important source files are:

```text
Session*/sentences/wav/<dialog>/<utterance_id>.wav
Session*/dialog/transcriptions/<dialog>.txt
Session*/dialog/EmoEvaluation/<dialog>.txt
Session*/sentences/ForcedAlignment/<dialog>/<utterance_id>.wdseg
```

The generated aligned artifacts live under:

```text
data/processed/iemocap_aligned_features/
```

The main generated files are:

```text
metadata.jsonl       one JSON record per utterance
audio_frames.npz     temporal audio feature arrays
word_features.npz    word-aligned audio feature arrays
feature_names.json   ordered names for the 88 acoustic features
alignment_report.json
```

The implementation is in:

```text
src/iemocap_aligned_dataset.py
```

## Train/Test Split

The repository uses the standard session split:

```text
Sessions 1-4 -> train
Session 5    -> test
```

The current aligned artifact contains:

```text
10037 utterances total
7867 train utterances
2170 test utterances
```

## Label Space

The current IEMOCAP label mapping keeps the 11 local labels:

```text
ang  anger
hap  happiness
exc  excitement
neu  neutral
sad  sadness
fru  frustration
fea  fear
sur  surprise
dis  disgust
xxx  undefined / other
oth  other
```

Each metadata record stores both:

```python
"label_code": "ang"
"label": 0
```

It also stores VAD scores:

```python
"vad": [valence, activation, dominance]
```

## Word Alignment

IEMOCAP forced alignment comes from `.wdseg` files. These files contain word
start and end frame indices.

Example:

```text
SFrm  EFrm    SegAScr Word
41    72      -17530  WHAT(2)
133   146     -258661 WELL
147   156     -308615 DO(2)
```

The repository treats each alignment frame as 10 ms:

```python
start_sec = SFrm * 0.01
end_sec = (EFrm + 1) * 0.01
```

The `+ 1` on the end frame makes the end time inclusive of the final frame.

The parser removes alignment-control tokens:

```text
<s>
</s>
<sil>
```

It also lightly normalizes pronunciation alternatives:

```text
EXCUSE(8)  -> excuse
LICENSE(2) -> license
```

The resulting word records look like:

```python
{
    "word": "excuse",
    "raw_word": "EXCUSE(8)",
    "start_sec": 0.45,
    "end_sec": 0.80,
}
```

This gives one time span per aligned word:

```text
word_i -> [start_sec, end_sec]
```

## Utterance Metadata Shape

Each line in `metadata.jsonl` is one utterance-level JSON object. Important
fields include:

```python
{
    "utterance_id": str,
    "dialog_id": str,
    "session": int,
    "split": "train" | "test",
    "text": str,
    "label_code": str,
    "label": int,
    "vad": [float, float, float],
    "audio_path": str,
    "duration_sec": float,
    "words": [
        {
            "word": str,
            "raw_word": str,
            "start_sec": float,
            "end_sec": float,
        }
    ],
    "audio_frame_count": int,
    "word_count": int,
    "word_audio_feature_mask": list[int],
}
```

## Prosodic Feature Space

The aligned pipeline extracts OpenSMILE `eGeMAPSv02` functionals.

Each acoustic/prosodic vector has:

```text
88 features
```

The exact feature order is stored in:

```text
data/processed/iemocap_aligned_features/feature_names.json
```

The feature groups are roughly:

```text
Pitch / F0:
  F0 mean, normalized std, percentiles, range, rising/falling slopes

Loudness:
  loudness mean, normalized std, percentiles, range, rising/falling slopes

Spectral:
  spectral flux, alpha ratio, Hammarberg index, spectral slopes

Cepstral:
  MFCC1-MFCC4

Voice quality:
  jitter, shimmer, harmonics-to-noise ratio, H1-H2, H1-A3

Formants:
  F1/F2/F3 frequency, bandwidth, and amplitude relative to F0

Voicing / rhythm:
  loudness peaks per second, voiced segments per second,
  voiced/unvoiced segment length statistics

Energy:
  equivalent sound level in dB
```

The first few dimensions are:

```text
0  F0semitoneFrom27.5Hz_sma3nz_amean
1  F0semitoneFrom27.5Hz_sma3nz_stddevNorm
2  F0semitoneFrom27.5Hz_sma3nz_percentile20.0
3  F0semitoneFrom27.5Hz_sma3nz_percentile50.0
4  F0semitoneFrom27.5Hz_sma3nz_percentile80.0
10 loudness_sma3_amean
30 jitterLocal_sma3nz_amean
32 shimmerLocaldB_sma3nz_amean
34 HNRdBACF_sma3nz_amean
40 F1frequency_sma3nz_amean
46 F2frequency_sma3nz_amean
52 F3frequency_sma3nz_amean
87 equivalentSoundLevel_dBp
```

## Temporal Audio Frames

The audio extractor does not produce only one vector for the whole utterance.
It extracts features over sliding windows:

```text
window size: 1.0 second
hop size:    0.5 second
```

For each utterance:

```text
audio_frame_features:  T x 88
audio_frame_start_sec: T
audio_frame_end_sec:   T
```

`T` depends on the utterance duration. In the current artifact:

```text
minimum T: 1
maximum T: 68
mean T:    about 8.41
```

These arrays are stored in `audio_frames.npz`, keyed by utterance id:

```text
<utterance_id>              -> T x 88 feature matrix
<utterance_id>__start_sec   -> T start times
<utterance_id>__end_sec     -> T end times
```

## Word-Aligned Audio Features

The model usually needs audio features at the word or token level. To get
word-level features, the pipeline aggregates the sliding-window audio features
over each aligned word span.

For each word:

1. Find every audio window that overlaps the word time span.
2. Compute the amount of overlap in seconds.
3. Use the overlap durations as weights.
4. Average the overlapping `88`-dimensional audio vectors.

The output shape is:

```text
word_audio_features: W x 88
word_audio_mask:     W
```

where `W` is the number of aligned words in the utterance.

Example:

```python
words = [
    {"word": "excuse", "start_sec": 0.45, "end_sec": 0.80},
    {"word": "me", "start_sec": 0.80, "end_sec": 1.06},
]

word_audio_features.shape == (2, 88)
word_audio_mask.shape == (2,)
```

The mask is `1` when the word received a valid audio vector and `0` when no
audio window overlapped that word. In the current IEMOCAP artifact, all words
received overlap:

```text
words_without_audio_overlap: 0
```

## Mapping To DistilBERT Tokens

DistilBERT operates on subword tokens, not necessarily whole words. The aligned
IEMOCAP features are word-level, so there is one extra mapping step.

Use a Hugging Face fast tokenizer with:

```python
tokenizer(
    words,
    is_split_into_words=True,
    padding="max_length",
    truncation=True,
    max_length=max_length,
)
```

Then use:

```python
encoding.word_ids()
```

to find which source word each token came from.

For each token:

- if it maps to word `i`, copy `word_audio_features[i]`
- if it is `[CLS]`, `[SEP]`, `[PAD]`, or otherwise has no word id, use zeros
- set `audio_token_mask` to `1` for real word tokens and `0` for special/pad tokens

The result is:

```text
DistilBERT hidden states: L x 768
audio token features:    L x 88
audio token mask:        L
```

For cross-modal attention, project audio features into the same hidden size:

```python
audio_hidden = Linear(88, 768)(audio_token_features)
```

Then attention can operate over:

```text
text_hidden:  L x 768
audio_hidden: L x 768
```

## Important Caveat

The current acoustic features use `1.0s` windows with a `0.5s` hop. This gives
stable prosodic summaries, but it is coarse for very short words. A short word
may receive an audio vector influenced by nearby words in the same one-second
window.

This is acceptable for a first cross-modal model, but it is not perfect
word-local prosody.

If tighter alignment becomes important, consider:

```text
0.25s window / 0.10s hop
```

or extract lower-level OpenSMILE descriptors at a finer frame rate and aggregate
those directly over word spans.

## Current Alignment Quality

The generated IEMOCAP alignment report shows:

```text
included utterances: 10037
missing wdseg files: 2
invalid word spans: 0
same normalized transcript/alignment ratio: about 96.5%
mean exact-prefix token match ratio: about 99.0%
```

This means IEMOCAP is a good first dataset for word-level and token-level
audio-text fusion in this repository.

