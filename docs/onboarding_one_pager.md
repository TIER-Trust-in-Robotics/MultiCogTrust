# MultiCogTrust: Onboarding One-Pager

## The flan

It is no longer a question as to if robots will roam among us, but a matter of how long until, and how should them behave amonst us. For fields where robotics (especally those of the the humanoid variety) are interacting face-to-face with humans, building rapport and trust are paramount for sucessful intergration. People, however, rarely say "I don't trust you." They show it through **what they say**, **how they say it** and **their face**. MultiCogTrust seek to read those three channels (text, audio and video) and estimates the person's emotional state in live environments.  

While there has been major advacements in the field of natural language, it's scope is largely focused on its written form. Human-robot-Interactins (HRI) is not exclusive to messaging over text; We *speak* with our body as well, the way we pronounce things also carries semantics that are lost in the written medium.

Enter MultiCog(Trust), a **multi-modal** modal that 



A major caviat: **emotion is not trust.** Today the system predicts four emotions: **anger, happiness, sadness and neutral**. Trust is an vague construct; we are not here to define it. Predicting it needs trust-labeled data from a human–robot study we have not run yet. Though we can start working towards identifiying the building blocks of trust.

## Background

**Affective computing** builds systems that recognize human emotion. The usual approach is supervised learning on recordings of speakers (often actors) whose utterances annotators have labeled with emotions. Our main dataset is **IEMOCAP**: about 12 hours of 10 actors in 5 two-person sessions, with audio, video, transcripts and emotion labels. **CMU-MOSEI** (YouTube clips with sentiment scored from −3 to +3) is our second benchmark.

**Multimodal** models combine signals because each one is ambiguous on its own. "That's fine" may reads as neutral in text, but a tense voice or a frown completly changes its meaning. The research questions are *how* to combine the signals (**fusion**) and whether each added modality is worth its cost.

**Real-time** use adds constraints that offline benchmarks ignore:

- The model hears audio only up to the current moment.
- Transcripts come from imperfect speech recognition, not human typists.
- The face can leave the frame.
- The answer has to arrive within about 1–2 seconds to be useful in conversation.

## How the system works

```
microphone ─► VAD ─► speech segment ─┬─► Whisper ─► transcript ─► DistilBERT ──┐
                     (≤ 6 s)         └─────────────────────────► DistilHuBERT ─┴─► cross-attention ─► 4 emotion probabilities
webcam ─► face crop ─► blendshapes + image embedding ─► visual head ─► 4 probabilities (displayed, not yet used)
```

1. **Segment speech.** Silero VAD scores 32 ms chunks of audio. A segment ends after about 300 ms of silence and is capped at about 6 s.
2. **Transcribe.** faster-whisper (`tiny.en`) converts the segment to text.
3. **Encode.** Pretrained encoders, which stay *frozen* during training, turn the audio (DistilHuBERT) and the text (DistilBERT) into embeddings.
4. **Fuse and classify.** A small cross-attention head, a lightweight adaptation of the MemoCMT paper, lets each modality attend to the other. It outputs the four class probabilities.
5. **Vision, in parallel.** YuNet finds the face. From that crop, MediaPipe extracts 52 blendshape coefficients and MobileNetV3 produces an image embedding, sampled 5 times per second. The visual probabilities can be blended with the audio/text output (late fusion). The blend weight is currently **0** because vision has not improved results with real Whisper transcripts.
6. **Inspect.** A dashboard on your own machine shows the transcript, probabilities, face availability and timings for each utterance.

## Where things stand (October 2026)

| IEMOCAP Session 4 validation            | Macro F1 |
| --------------------------------------- | --------:|
| Audio + text, human-written transcripts | 0.747    |
| Audio + text, Whisper transcripts       | 0.723    |
| + vision, human-written transcripts     | ≈ 0.75   |
| + vision, Whisper transcripts           | no gain  |

- **Neutral** is the hardest class to recognize.
- A plain concatenate-and-MLP fusion scores almost as well as cross-attention while training about 30× fewer parameters. The more complex architecture has not yet proven its worth.
- On recorded dialogues replayed through the live system, the median time from end of speech to prediction is about 0.9 s.
- Current work focuses on benchmark quality: a larger reference model, the MOSEI benchmark and one well-justified visual experiment. See the roadmap for details.

## Ground rules

- **Session 5 is the held-out test set.** Train on Sessions 1–3 and tune on Session 4. Evaluate on Session 5 only once, at the end, with a frozen configuration. Random splits put the same speakers in training and test data, which inflates scores by about 6–7 points.
- **Compare like with like.** Use the same split, inputs and epoch budget, with three seeds (42/43/44). Report per-class results, not only the headline number.
- **Missing is not neutral.** If the face is lost, mask the visual input and fall back to audio/text. Never substitute a "neutral" face.
- **Don't overwrite checkpoints.** Pass a new filename with `--output`.
- **Datasets live outside git.** IEMOCAP and MOSEI are licensed and should live under `data/`. You can download them on [Google Drive](https://drive.google.com/drive/folders/14APd8Rv0-pNTcl307BP-vCJb71XZgpGJ?usp=drive_link)
- **Name branches `{modality}/{feature}`**, for example `visual/au-probe`.

## Your first week

1. Install and test: run `uv sync --locked`, then `uv run python -m pytest -q`.
2. Read `README.md` and the top half of `docs/async_multimodal_pipeline_roadmap.md`. The bottom half holds archived design notes.
3. Once the data is in place, replay a recorded dialogue through the live system:
   `uv run python src/pipeline_demo.py --device mps --visual-checkpoint models/mvp/visual_day5/deployment.pt --replay-dialog Ses01F_impro01 --speaker F --dashboard`
   Then open `http://127.0.0.1:8765`.
4. Trace one utterance through the code: `src/stream_runtime.py` → `src/inference_pipeline.py` → `src/emotion_fusion_infer.py` → `src/models/cross_modal_transformer.py`.

## Terms and ideas to learn

**Research context**

- **Human–robot interaction (HRI)**; **trust in automation**; **calibrated trust** (avoiding both over-trust and under-trust); **trust repair** after a failure
- **Categorical vs. dimensional emotion**: discrete labels vs. continuous scales such as valence (pleasant–unpleasant) and arousal (calm–excited)
- **Sentiment vs. emotion**: a positive/negative score (MOSEI) vs. emotion categories (IEMOCAP)

**Speech and audio**

- **Sample rate** (we use 16 kHz), waveform, **(mel) spectrogram**
- **Voice activity detection (VAD)**: detects *whether* someone is speaking, not *what* they say
- **Automatic speech recognition (ASR)**; **word error rate (WER)**
- **Prosody**: pitch, loudness and speaking rate; eGeMAPS/openSMILE are standard feature sets for it
- **Self-supervised speech models**: wav2vec 2.0 → HuBERT → DistilHuBERT

**Language**

- **Tokenization and subwords** (WordPiece); **Transformer** and **self-attention**; **BERT / DistilBERT**
- **Embeddings and hidden states**; **pooling** (mean pooling, the [CLS] token)

**Vision**

- **Face detection vs. facial landmarks** (YuNet; MediaPipe Face Landmarker)
- **Blendshapes vs. FACS action units (AUs)**: related ideas but different measurements, so don't use the terms interchangeably
- **CNN image embeddings**; **ImageNet pretraining** (general object images, not trained on facial emotion); MobileNetV3

**Modeling**

- **Transfer learning**; **frozen encoders vs. fine-tuning**; **knowledge distillation** (how the "Distil" models were made)
- **Early vs. late vs. feature-level fusion**; **cross-attention** and cross-modal Transformers (MulT)
- **Masking for missing modalities**; softmax, cross-entropy loss, **label smoothing**
- **Feature caching**: precompute frozen-encoder outputs once so each training run is faster

**Evaluation**

- **Train/validation/test splits**; **data leakage**; **speaker-independent (leave-one-session-out) evaluation**
- **Class imbalance**; **accuracy vs. macro F1 vs. UAR** (unweighted average recall; papers often write WA/UA); **confusion matrix**
- **Random seeds and run-to-run variance**; **baselines, controls and ablations**
- MOSEI metrics: **MAE**, **Pearson correlation**, **Acc-2 / Acc-7**

**Real-time systems**

- **Latency (p50/p95) vs. throughput**; **producer/consumer queues**, bounded queues, **backpressure**
- Python **asyncio**, threads and the **GIL**; **timestamp alignment** across sensors

**Tools**

- `uv`, **PyTorch** (including **MPS**, its GPU backend for Apple Silicon), **Hugging Face Transformers**, **faster-whisper**, **OpenCV**, **MediaPipe**, **pytest**

**Key papers** (on a first pass, the abstract and figures are enough)

- Busso et al., 2008: *IEMOCAP: Interactive emotional dyadic motion capture database*
- Lee & See, 2004: *Trust in automation: Designing for appropriate reliance*
- Hancock et al., 2011: *A meta-analysis of factors affecting trust in human-robot interaction*
- Tsai et al., 2019: *Multimodal Transformer for unaligned multimodal language sequences* (MulT)
- Radford et al., 2022: *Robust speech recognition via large-scale weak supervision* (Whisper)
- Khan et al., 2025: *MemoCMT*, the cross-modal emotion model we adapted
