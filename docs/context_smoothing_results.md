# Speaker Context Smoothing — Experiment Results

## Purpose

The deployed audio/text model (`models/mvp/audio_text_baseline.pt`) scores each utterance on its own. IEMOCAP speakers tend to hold an emotion across consecutive lines, so `src/emotion_context.py` combines each new prediction with the same speaker's previous smoothed state. It is causal, so it works live:

```
q_t ∝ p_t · q_{t-1} ** (strength · (1 - max p_t) ** gate)
```

`gate = 0` is plain log-linear smoothing. With `gate > 0`, a confident new reading overrides the context. Enable it in the live pipeline with `uv run python src/pipeline_demo.py --context {off,balanced,max_f1}` (default `off`). The raw per-utterance result is kept as `utterance_emotion`.

| Preset | strength | gate |
|---|---|---|
| `max_f1` | 0.6 | 0 |
| `balanced` | 3.0 | 2 |

---

## Protocol

- **Selection:** all settings were chosen out-of-fold on Sessions 1-3. Fusion heads were trained on two sessions and predicted the third: 2 seeds, epochs 3-8 averaged, on cached frozen-encoder sequences.
- **Session 4:** evaluated once per stage, with Whisper tiny.en transcripts (deployment settings). The 4 empty transcripts count as wrong.
- **Session 5:** never encoded or evaluated.
- **Context chain:** includes each speaker's unscored lines (frustration, no majority) as it would live. Skipping them would leak label information.
- **"Accuracy on emotion changes":** accuracy on lines whose label differs from the speaker's previous scored line. It measures how quickly a real change is caught.

---

## Results

### Session 4, deployed checkpoint, Whisper text

| | Macro F1 | Accuracy | Accuracy on 165 emotion changes |
|---|---|---|---|
| Deployed | 0.723 | 0.724 | 0.648 |
| + `balanced` | 0.735 | 0.736 | 0.618 |
| + `max_f1` | **0.761** | **0.760** | 0.539 |

A dialog-level bootstrap (30 dialogs, 2,000 resamples) gives these changes in macro F1 against the deployed model:
- `max_f1`: +0.038, 95% CI [+0.005, +0.071].
- `balanced`: +0.012, 95% CI [-0.007, +0.031].

### Consistency (macro F1, Whisper text)

| Held-out data | None | `balanced` | `max_f1` |
|---|---|---|---|
| Session 1 (out-of-fold) | 0.687 | 0.722 | 0.746 |
| Session 2 (out-of-fold) | 0.696 | 0.723 | 0.736 |
| Session 3 (out-of-fold) | 0.691 | 0.721 | 0.737 |
| Session 4 female actor | 0.745 | 0.767 | 0.797 |
| Session 4 male actor | 0.694 | 0.700 | 0.727 |
| Session 4 improvised | 0.726 | 0.739 | 0.781 |
| Session 4 scripted | 0.691 | 0.698 | 0.716 |

Each IEMOCAP improvisation targets one emotion per scenario, so real conversations may shift more often than this data does. On the demo clip `what_the_hell` (neutral → anger), both presets detect anger one line late.

---

## Previous-Line Text Context (Needs Retraining)

The text encoder sees `[CLS] previous same-speaker line [SEP] current line [SEP]`. If the pair exceeds the 297-token limit, the start of the previous line is cut first and the current line is kept (`tokenize_text` in `src/emotion_checkpoint.py`). No IEMOCAP pair comes close; the longest is 162 tokens. The previous line has already been transcribed, so this works live. It needs a fusion head retrained on that input; the encoders are unchanged. Scores below are 2-seed ensembles (42, 43), with epochs 3-8 averaged. Cells show macro F1 / accuracy on emotion changes.

| | Current line only | Previous + current line |
|---|---|---|
| Out-of-fold Sessions 1-3, reference text | 0.716 / 0.610 | 0.742 / 0.620 |
| Session 4, Whisper text | 0.701 / 0.667 | **0.740 / 0.703** |
| Session 4, Whisper text + `balanced` | 0.748 / 0.655 | 0.752 / 0.648 |
| Session 4, Whisper text + `max_f1` | 0.748 / 0.533 | 0.762 / 0.527 |

- **Consistency:** each seed gains about 2.2-2.3 points out-of-fold. By held-out session (seed 42): Session 1 flat, Sessions 2 and 3 about +3.
- **Unlike smoothing,** it also improves accuracy on emotion changes here. Without a smoother, it is the only change that beats the deployed model (0.723 / 0.648) on both measures. A retrained current-line model also does once `balanced` smoothing is added: 0.748 / 0.655, or 0.751 / 0.661 as a 3-seed ensemble. A later 5-seed check (`docs/overnight_research.md`, Phase 1) confirms the F1 gain on every seed. It does not confirm the gain on emotion changes: out-of-fold the difference is −0.5 points, and on Session 4 its interval includes zero.
- **Combined with smoothing,** the gains mostly overlap, since both use earlier lines.

---

## A/B: Deployed Model vs Retrained Context Model

| | A · Deployed | B · Context |
|---|---|---|
| Checkpoint | `models/mvp/audio_text_baseline.pt` | `models/mvp/context_baseline.pt` |
| Text input | current line | previous same-speaker line `[SEP]` current line |
| Selection | best epoch on Session 4 (optimistic) | weights averaged over epochs 3-8, chosen out-of-fold on Sessions 1-3 |
| Reproduce | `src/train_mvp_baseline.py` | `src/train_mvp_context.py` (`configs/mvp_context.json`) |

Both use the same architecture, frozen DistilBERT/DistilHuBERT encoders and training data (Sessions 1-3, reference transcripts). The benchmark below ran both through the real `CrossModalEmotionPredictor` on every Session 4 line in speaker order, with Whisper tiny.en text. B's context is the previous Whisper line.

| Session 4, Whisper text | A | B |
|---|---|---|
| Macro F1 (1,031 scored lines) | 0.723 | **0.728** |
| Accuracy | 0.724 | **0.731** |
| Accuracy on emotion changes (165 lines) | 0.648 | **0.703** |
| Macro F1, reference transcripts | 0.747 | **0.779** |
| Text tokens per line (mean) | 15.8 | 30.2 |
| Emotion step, CPU median / p95 (300 lines) | 49.5 / 73.1 ms | 50.9 / 74.7 ms |
| Emotion step, Apple GPU median / p95 (2,098 lines) | 44.3 / 380 ms | 50.1 / 372 ms |

- **Speed:** doubling the text tokens costs about 1.5 ms per line on CPU (+3%), because DistilHuBERT audio encoding dominates. Whisper (about 300 ms) is shared by both models.
- **Apple GPU timing:** each new audio length triggers a one-off graph compile, so whichever model runs first on a line pays it. The benchmark alternates the order, and the comparison videos use CPU timings.
- **Variance:** a second single checkpoint with the same recipe (seed 43) scores F1 0.743 on Session 4 with Whisper text, against 0.728 for the shipped seed 42. Both are above A's S4-selected 0.723. Accuracy on emotion changes is 0.703 for both seeds, while F1 moves about 1.5 points with the seed alone. The later five-seed recipe comparison in [overnight_research.md](overnight_research.md) supports a consistent macro F1 gain, but its paired interval for accuracy on emotion changes includes zero; the two-seed observation does not establish a robust change-accuracy gain. Seed 42 stays shipped, because choosing seed 43 on its Session 4 score would be selection on Session 4.
- **Where B fails:** in `wedding_plans`, B reads seven excited lines as anger. On all of Session 4 with Whisper text, B misreads 28% of happiness as anger, against the deployed A's 7%. This bias comes from the training recipe, not the context. A model retrained the same way on the current line only (seed 42, epochs 3-8) is worse:

  | Session 4: happiness read as anger | Reference text | Whisper text |
  |---|---|---|
  | Retrained, current line only | 30% | 37% |
  | B, previous + current line | 22% | 28% |

  Out-of-fold on Sessions 1-3 neither model shows it (7%), so it is specific to Session 4's actors. The deployed A avoids it because its epoch was picked on Session 4. A fair comparison therefore shows context reducing the bias. Per-class offsets chosen out-of-fold did not transfer.

  **Per-speaker audio normalization** subtracts each speaker's mean DistilHuBERT frame and divides by its std, using unlabelled audio. Trained and evaluated with each speaker's statistics over all of their sessions, it fixes the bias. Out-of-fold F1 rises 0.735 → 0.743; on Session 4 with Whisper text, F1 is 0.732, happiness-to-anger falls 28% → 8%, and anger precision rises 0.71 → 0.85.

  It does not survive a causal, live-style test. Running statistics within one conversation reach only F1 0.58-0.71 depending on the prior, below plain B (0.728). A single conversation's average voice carries that conversation's emotion, so normalizing removes the signal. The gain needs a per-user voice profile built over many varied sessions, so it is not shipped.

  A returning-user simulation normalizes each Session 4 conversation with the speaker's statistics from their other conversations only. Using all of them (about 14) matches the ideal: F1 0.734, happiness-to-anger 8%. With 1/2/4/8 earlier conversations (mean of three random draws), F1 is 0.698/0.712/0.721/0.731. The anger bias stays fixed (about 8-9%) from the first conversation, but F1 only passes plain B (0.728) at about 8. A voice profile is worth adding for returning users once roughly 8 conversations are stored.

**Through the live loop** (`src/live_demo.py`, CPU, 21 utterances replayed from Ses04F_impro05), medians and 95th percentiles:
- Whisper: 216 / 640 ms (shared).
- A emotion step: 48 / 79 ms.
- B emotion step: 40 / 82 ms.
- End of speech to both results: 558 / 992 ms. That includes the voice detector's 300 ms silence wait.

Running both models live adds about 40 ms per utterance.

**Live-loop optimizations** (`src/live_demo.py`):
- **Shared encoders.** A and B use identical frozen encoders, so `SharedEncoders` encodes audio once and each distinct text once, then runs both fusion heads. Outputs are bit-identical (60 lines, maximum difference 0.0). Both models take 64 ms median on CPU instead of 108 (p95 87 vs 150), and 343 MiB of duplicate encoder weights is freed.
- **Faster Whisper decoding** (`FAST_ASR` in `src/inference_pipeline.py`: no timestamp tokens, no temperature fallback, 4 threads). On 300 Session 4 lines the median drops from 196 to 135 ms and the worst case from 1,258 to 213 ms. Hallucination retries caused those spikes. WER improves (0.333 → 0.300).
  - On all of Session 4, macro F1 changes within noise: A 0.723 → 0.730, B 0.728 → 0.725. Accuracy on emotion changes moves 0.648 → 0.642 for A and stays 0.703 for B. This was a no-regression check: the setting was chosen on speed and WER, not on Session 4 emotion scores.
  - It is opt-in. The deployed pipeline and benchmarks keep the original settings; `live_demo.py --deployed-asr` restores them.
- **Together**, replaying Ses04F_impro05 through the live loop (21 utterances, face model on): end of speech to results goes from 558 to 481 ms median, and from 992 to 545 ms at p95. About 300 ms of the rest is the voice detector's end-of-speech pause.

**Live use:** `SpeechProcessor` (`src/inference_pipeline.py`) reads `text_context` from the checkpoint's provenance. For B it keeps the speaker's last transcript and adds `context_text` to each result; A and other checkpoints are unchanged.

**Comparison videos:** `uv run python src/render_inference_demo.py --compare --clip all --device cpu` writes `data/processed/demo/compare_<clip>.mp4`. That covers the seven demo scenes plus `veiled_anger` and `quiet_grief` (B helps most) and `wedding_plans` (B hurts most).

---

## Other Dataset: CMU-MOSEI (Validation Only)

MOSEI scores sentiment from -3 to +3 on single-speaker YouTube monologues. Here the same standardized ridge regressor is fitted on time-pooled released MMSA features for every modality subset: text BERT, audio COVAREP, vision FACET. Only the inputs differ. It is trained on the released train split and scored on validation (1,871 clips); the test split was not used.

| Inputs | MAE ↓ | Pearson r | Acc-2 (non-zero) | Acc-7 |
|---|---|---|---|---|
| Text | 0.589 | 0.667 | **0.841** | 0.489 |
| Audio | 0.777 | 0.241 | 0.644 | 0.424 |
| Vision | 0.770 | 0.246 | 0.643 | 0.424 |
| Text + audio | 0.588 | 0.666 | 0.832 | 0.498 |
| Text + vision | **0.580** | **0.673** | 0.833 | 0.497 |
| Text + audio + vision | 0.583 | 0.670 | 0.830 | **0.501** |

On the 89 clear-sentiment clips (|label| ≥ 1) where the text model reads near-neutral (|prediction| < 0.25), sign accuracy rises from 0.562 (text) to 0.629 (text + audio) and 0.652 (all three). Across all 720 clear clips, adding audio and vision fixes 16 and breaks 12. MOSEI is text-dominated, and this simple pooled fusion gains only where the words read neutral. That matches the IEMOCAP pattern; a stronger fusion model would be needed to turn it into an aggregate gain.

**With our encoders.** The existing legacy MOSEI audio+text checkpoint (`models/mosei/mosei_cross_modal.pt`, DistilBERT + DistilHuBERT cross-modal fusion) was compared against a DistilBERT text-only ridge using the same pinned encoder.

| Validation, 1,871 clips | DistilBERT text only | Audio + text (legacy model) |
|---|---|---|
| Acc-2 (non-zero) | 0.837 | **0.845** |
| MAE | 0.594 | **0.548** |
| Pearson r | 0.668 | **0.713** |
| MAE on strong clips (\|label\| ≥ 2, n = 188) | 1.369 | **1.126** |
| Mean \|prediction\| on strong clips (true mean 2.27) | 0.96 | **1.23** |
| Sign correct where the text reads neutral (95 clips) | 0.737 | 0.737 |

Here the voice adds intensity rather than direction. For example, "Pretty, pretty good" is labelled +2.33: text only says +0.14, audio+text +1.81.

This comparison is not perfectly matched. The legacy model is a 7-class classifier read out as an expected score, trained on the unfiltered training mirror, while the text baseline is a ridge regressor, which shrinks extreme predictions. With the same ridge on every subset of the pooled MMSA features (table above), adding audio and vision barely changes strong-clip MAE (1.387 → 1.357). So part of the intensity gain is likely the learned fusion or estimator rather than the audio alone.

A fully matched check confirms this. It uses the same frozen DistilBERT and DistilHuBERT pooled features (mean + std over the first 8 s of audio) and the same ridge for every input set.

| Validation | Text | Audio | Text + audio |
|---|---|---|---|
| Acc-2 (non-zero) | **0.837** | 0.692 | 0.828 |
| MAE | 0.594 | 0.744 | **0.592** |
| Strong-clip MAE | **1.369** | 1.909 | 1.371 |
| Sign correct where the text reads neutral (95 clips) | **0.737** | — | 0.716 |

Linear fusion of pooled audio adds nothing on MOSEI: on clear clips it fixes 10 signs and breaks 15.

**Demo video:** `uv run python src/render_mosei_demo.py` writes `data/processed/demo/mosei_voice_intensity.mp4` (61 s, kept local because it contains YouTube footage). It plays six validation clips, each followed by diverging −3..+3 bars for annotators, text only and audio + text:
- one with clear words, where both models agree;
- four where the words read near-neutral and the voice carries the sentiment;
- one counterexample, where a flat delivery misleads the audio+text model. Any audio benefit here needs learned cross-modal fusion. MOSEI's opinion monologues are mostly lexical; the case for multiple modalities is much clearer on IEMOCAP's acted emotional dialogue, where the voice-only model overturns neutral-reading words (126 Session 4 lines).

---

## Other Dataset: MSP-Podcast 2.0 (Text Only)

The local copy has transcripts, forced alignments, labels, partitions and speaker IDs. It has no audio, since the 409-hour "Audios" folder (about 47 GB) does not fit on this machine, so only the text side can be studied.

The labelled 4-class turns (anger/happiness/sadness/neutral) are:
- Train: 129,064 turns, 2,526 speakers.
- Development: 23,885 turns, 687 speakers.
- Test1 and Test2: 33,431 and 10,712 turns, left untouched.
- Test3: labels are hidden.

| Text only (frozen DistilBERT) | Evaluated on | Macro F1 | Emotional turns read as neutral |
|---|---|---|---|
| IEMOCAP-trained control | IEMOCAP Session 4 (acted) | 0.648 | 22.0% |
| MSP-trained (logistic regression, 129k turns) | MSP Development (podcasts) | 0.508 | 37.8% |
| IEMOCAP-trained control | MSP Development | 0.397 | 37.6% |

- **Why audio matters more for natural speech:** in natural podcast speech, words carry much less of the emotion. Even with 40× IEMOCAP's training data, 38% of anger, happiness and sadness turns read neutral from their words (sadness 44%), against 22% on IEMOCAP. MSP annotators listened to the audio, so these labels often rest on the voice. Some of the gap is label noise: annotators agree on the consensus class about 68% of the time.
- **Domain shift:** the IEMOCAP-trained text model falls from 0.648 to 0.397 on podcast speech. The deployed models are probably much weaker on natural speech than Session 4 suggests. The MSP audio (Development as validation, Test1/Test2 sealed) is the most important missing piece for an HRI system.

---

## Comparison Videos: Per-Clip Results

Scored lines correct per clip; A and B use the same Whisper transcripts.

| Clip | Scored | A · Deployed | B · Context |
|---|---|---|---|
| `veiled_anger` | 11 | 5 | **10** |
| `quiet_grief` | 9 | 5 | **9** |
| `complaint_call`, `what_the_hell`, `kiss_then_confession`, `supervisor`, `thanks_a_lot` | 30 | 30 | 30 |
| `condolences` | 6 | **6** | 5 |
| `dorky` | 9 | **7** | 5 |
| `wedding_plans` | 14 | **13** | 6 |

Live Whisper output differed from the cached transcripts on 2 of these lines: a laughter-only line in `dorky` and the one-word "Line?" in `veiled_anger`. When that happens, B's context differs from the benchmark's.

---

## Tried and Rejected

| Idea | Result |
|---|---|
| Retrained 3-seed ensemble (epochs 3-8 averaged) | Session 4 0.696 alone; 0.751 with `balanced` (accuracy on changes 0.661). Seed and epoch averaging adds about 1 point out-of-fold. |
| Per-class log-prior offsets | Out-of-fold 0.744 → 0.750, but Session 4 fell 0.749 → 0.737 |
| Training on reference plus Whisper transcripts | Out-of-fold 0.682 vs 0.682; worse with smoothing |
| Cross-speaker context, time decay | No gain, or within noise |
| Learned logistic context combiner | Nested out-of-fold 0.704, below the fixed smoother's 0.744 |
| Class-balanced loss, label smoothing 0.1 | Mean single-epoch out-of-fold F1 0.655 / 0.664 vs 0.662 (partial runs) |

The deployed checkpoint's epoch (6) was selected on Session 4, so its 0.723 is optimistic. Out-of-fold, epoch 6 was the weakest single epoch.
