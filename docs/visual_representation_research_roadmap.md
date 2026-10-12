This roadmap studies whether compact facial-motion features improve live four-class emotion recognition, and whether they complement learned face-image embeddings. The target is a local Mac/laptop using one webcam and microphone. Robot integration is optional. The first milestone is a two-week pilot, followed by a broader validation study.

Prepared and updated: 2026-10-10. The first IEMOCAP blendshape/image comparison
is now completed. The [async pipeline roadmap](async_multimodal_pipeline_roadmap.md)
contains the current **new ten-working-day schedule**; the original experiment
schedule below is retained as historical planning, not a second active calendar.

**Current evidence and decision.** Nine compact-head runs (three seeds per
representation, 20 epochs each) hold the audio/text checkpoint fixed and use
Sessions 1–3 for training and Session 4 for validation. Reference-text macro F1
is 0.7469 for audio/text, 0.7518 ± 0.0013 with blendshapes, 0.7512 ± 0.0015
with generic ImageNet image embeddings, and 0.7545 ± 0.0026 with both. These
are selection scores and sample standard deviations across seeds, not final
test results. The selected combined head has 74,052 trainable parameters.

The reference-selected 0.25 visual mixture scores 0.7215 with actual Whisper
transcripts, below the audio/text score of 0.7229. A separate validation grid
selects deployment weight 0, preserving the reference experiment unchanged.
The live inspector computes and shows visual probabilities while the default
prediction uses audio/text. No live visual accuracy gain is established.

Controlled missing vision returns the exact baseline on all 1,031 validation
utterances. Time shuffling slightly improves the selected model's reference-text
score (0.7562 versus 0.7553), so temporal order has not earned a contribution
claim. Prioritize a masked-mean pooling control, matched six-second text/audio
inputs, and actual-ASR selection before increasing model complexity. Source
results and verification are in `models/mvp/visual_day5/` and
`data/processed/mvp/day5_7/`.

The runtime has passed paced raw dialogue replay and a short real webcam/mic
capture check. The latter contained no detected speech. Browser appearance review
and longer spoken sessions remain pending. Session 5 is untouched. AU tooling,
MEAD adaptation, DISFA and Aff-Wild2 work described below remain unrun; dataset
access has not been newly verified by this implementation update.

**The research question is:** when added to a fixed audio/text emotion model, how do MediaPipe blendshapes, predicted facial action units, and learned face embeddings compare in accuracy, robustness, and inference cost? Does combining facial-motion features with image embeddings provide additional information? A separate experiment asks whether clean external video training improves transfer to IEMOCAP.

The downstream labels are anger, happiness, sadness, and neutral. This study establishes evidence about emotion classification and facial representations. A claim about human–robot trust would require a later study with trust annotations.

MediaPipe Face Landmarker outputs 52 blendshape coefficients. These must be named and evaluated as blendshapes; they are not automatically FACS action-unit measurements. A separate AU estimator supplies the AU comparison. [MediaPipe documentation](https://developers.google.cn/edge/mediapipe/solutions/vision/face_landmarker)

The proposed data acquisition order is **MEAD first, DISFA access next, and Aff-Wild2 access for the follow-up**. Existing IEMOCAP remains the paired audio/text/video benchmark.

| Priority | Dataset and role | What to obtain | Initial scope |
| --- | --- | --- | --- |
| Required, reuse locally | IEMOCAP: downstream multimodal training and evaluation | Original dialogue videos, utterance audio, transcripts, emotion annotations, timing metadata, and speaker/session identities | All five sessions; retain the existing Session 5 test boundary |
| Required for the external-data experiment | MEAD: clean talking-face training | Original front-view videos, actor IDs, emotion and intensity metadata, and available sentence IDs/audio; preserve the release's naming and metadata | Verify four actors first, then expand to a proposed 24-actor pilot. Keep all provided emotion categories and intensities. Other views can wait |
| Request now; needed for the AU measurement study | DISFA: human-coded facial-action supervision | Original videos, frame-level AU intensity annotations, participant IDs, frame numbering, and dataset documentation | All 27 participants if access permits; one consistent camera view is sufficient initially |
| Request now if convenient; follow-up scope | Aff-Wild2: less controlled expression/AU evaluation | AU and expression annotation subsets, official split lists, and corresponding ordered face crops or video with frame IDs/timing | Begin with annotated train/validation portions. Inventory overlap between annotation subsets before defining joint experiments |

MEAD is the first choice for the clean-video question: its paper describes 1920 × 1080 recordings, eight emotions, multiple intensities, and seven camera views. The 24-actor frontal subset above is our proposed pilot, not an official benchmark split. Use the authors' [dataset page](https://wywu.github.io/projects/MEAD/MEAD.html) and [repository](https://github.com/uniBruce/Mead); the technical specifications are in the [original paper](https://personal.ie.cuhk.edu.hk/~ccloy/files/eccv_2020_mead.pdf). The dataset page was available through search indexing during this review, but direct retrieval failed; archive availability and download sizes remain unverified. Confirm access with the small sample before a bulk download.

DISFA provides recordings of 27 participants at 1024 × 768 and expert-coded AU intensities on a 0–5 scale. It supports studying whether our facial features retain information about annotated facial actions. Request the original DISFA release through its [official page and agreement](https://mohammadmahoor.com/pages/databases/disfa/). DISFA+ is a separate release and is not required for this pilot.

Aff-Wild2 provides expression and AU annotation subsets, including a 12-AU task. These subsets must not be assumed to contain identical annotated frames. Its [official access instructions and dataset descriptions](https://affective-behavior-analysis-in-the-wild.github.io/10th/) provide the contact route; ask for research access and the applicable release rather than treating the completed competition dates as a current deadline.

**A critical evaluation constraint:** OpenFace lists DISFA among its AU training datasets. Evaluating an arbitrary released OpenFace checkpoint on DISFA cannot establish independent AU generalization. DISFA can support participant-held-out training of our own blendshape-to-AU probe, or supervised AU development with a documented split. Independent evaluation of an existing AU estimator requires verified training provenance and an evaluation set it did not train on. [OpenFace AU documentation](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Action-Units)

The minimum acquisition needed to begin is complete IEMOCAP video plus a small MEAD sample. A pretrained AU estimator can be compared as an emotion feature extractor before DISFA arrives; human AU annotations are needed for claims about AU measurement quality. Large additional corpora are deferred until this comparison identifies a specific data gap.

For every download, retain the source URL, release identifier, access terms, archive checksum, original filenames, and subject IDs. Measure archive size and extraction cost on the sample before expanding. Store raw data separately from generated feature caches. New raw roots may be data/MEAD/ and data/DISFA/; preserve the existing IEMOCAP tree. No downloads or access requests are performed by this document.

The experiment matrix holds the audio/text checkpoint fixed. If explicit prosody is included in that checkpoint, keep it identical across every row.

| ID | Added visual input | Question answered | Priority |
| --- | --- | --- | --- |
| B0 | None | What does the existing audio/text model achieve? | Core |
| B1 | MediaPipe blendshape sequence | Are compact facial-motion features useful at low inference cost? | Core |
| B2 | Predicted AU sequence from a documented estimator | How does an explicit AU representation compare? | Required for the AU comparison; subject to an early runtime check |
| B3 | Compact pretrained image embeddings from face crops | What does the learned image representation contribute? | Core; initial implementation uses generic ImageNet MobileNetV3-small |
| B4 | Image embeddings plus blendshapes or AUs | Does the descriptor add information beyond the image representation? | Use the descriptor selected on validation data |

Use the same speaker tracks, frame timestamps, sampling rate, temporal aggregation family, fusion mechanism, and tuning budget. Project inputs to a common feature width and report trainable parameter counts. Record each extractor's pretraining provenance; different pretrained extractors are practical system comparisons, not proof that representation type alone caused a difference.

Start with frozen extractors and small trainable temporal/fusion heads. Cache features once so repeated runs fit the laptop budget. Use one simple fusion design for this study; changing the fusion architecture belongs to a later experiment. Preserve temporal features and missing-frame masks rather than collapsing every recording into one untraceable vector.

The external-data experiment is a matched comparison of B3 with and without MEAD adaptation. Begin from the same image checkpoint. Train a small visual adapter/temporal head with a separate MEAD emotion head, then train the same IEMOCAP fusion head in both conditions. Keep IEMOCAP updates and selection rules identical, and report the added source-data training cost. This tests the benefit of MEAD adaptation; it does not by itself isolate video resolution as the cause.

Retain MEAD's native labels in its source head. For IEMOCAP, preserve all original labels in raw metadata and define the four-class subset only in the experiment configuration. Use the established happiness/excitement merge and checkpoint label order consistently. Never attach unrelated IEMOCAP transcripts or audio to MEAD videos to construct artificial multimodal examples.

The following table is the original proposed schedule. Its local B0/B1/B3/B4
foundation is now implemented; external-data and AU rows remain conditional.
Use the updated main roadmap for the next two weeks. Do not run the Session 5
step here independently of that roadmap's configuration-freeze gate.

| Working days | Work | Reviewable output and exit condition |
| --- | --- | --- |
| 1–2 | Audit IEMOCAP video, verify a small MEAD sample, choose the image/AU extractors, and freeze the protocol | Data inventory, split manifests, label mappings, checkpoint provenance, and a visual inspection sample. Correct speaker and utterance/video alignment must be verified before bulk extraction |
| 3–4 | Extract timestamped blendshapes, image embeddings, and AUs; benchmark a short replay on the Mac | Shared feature schema, extraction/coverage report, and inspected timelines. Decide by day 3 whether the AU estimator is practical; keep B2 explicitly pending if it fails this gate |
| 5–6 | Fit small visual heads and run the matched MEAD adaptation experiment | Source-validation results and B3 checkpoints with/without adaptation. Confirm that extra data helps target validation before selecting it for the demo |
| 7–8 | Train the fusion comparisons and run controlled robustness checks | Validation results for all completed rows, three fixed seeds for the small heads where feasible, per-class errors, and a frozen list of final configurations |
| 9 | Evaluate the frozen configurations on Session 5 and benchmark full replay inference | Final result table, seed variation, coverage, memory, and latency measurements. No further tuning on Session 5 |
| 10 | Connect the validation-selected representation to the live/replay inspector and write the research report | Ten-minute local run, replay fallback, processing visualization, reproducible configuration/commands, and a clear statement of completed and pending comparisons |

If MEAD access is delayed, finish the IEMOCAP representation comparison and mark external-data transfer pending. If AU tooling is delayed, deliver B0/B1/B3 and the feasible complementary comparison, explicitly describing the result as a blendshape/image study. Neither delay justifies renaming blendshapes as AUs or reporting an unrun experiment.

The evaluation protocol is fixed before final testing:

1. Use Sessions 1–3 for initial fitting and Session 4 for validation. After selecting configurations and training duration, optionally refit all Sessions 1–4 under one identical rule for every model. Session 5 remains untouched until final evaluation. Fit normalization and feature selection only on the relevant training partition.
2. Split MEAD by actor. A proposed 24-actor pilot uses 16/4/4 actors for source train/validation/test. Keep every view, intensity, and repetition of an actor together. Record sentence-template overlap and use a template-held-out diagnostic if feasible. Do not claim source benchmark comparability for this custom split.
3. Split any newly trained DISFA probe by participant before fitting or selecting features. Verify inherited model training data separately; a downstream split cannot undo pretrained-model exposure.
4. Report utterance-level macro F1 as the primary emotion metric, with unweighted average recall, per-class recall, and a confusion matrix. Report failed face tracking and feature coverage by split/class. Use the same fixed audio/text fallback for unavailable vision and also report a common-valid-utterance comparison.
5. Measure p50/p95 visual processing latency, full utterance-end-to-prediction latency, throughput, and peak memory on the target Mac. Include decoding, face detection/tracking, and preprocessing in visual cost. Cached-feature speed is not live inference speed.
6. Repeat small-head training with three fixed seeds if the measured budget permits. Report actual runs and variability. Session 5 contains only two actors, so seed variation or frame-level resampling does not establish broad population generalization.

The original standalone engineering target was **10 visual updates per second,
no growing processing backlog, and p95 visual processing below 100 ms**. The
days 3–4 visual-only replay met that target. The trained runtime uses the matching
5 Hz feature contract; its 35-second full dialogue replay produced 11 predictions
with p95 speech-end-to-result 1.474 seconds and no dropped queues. This small
integration check does not establish long-session latency or live accuracy.

Use a few focused diagnostics to explain the results:

- **Video quality:** downsample or blur copies of the same evaluation clips and rerun the extractors. Keep originals and degraded versions in the same split. This paired test separates a quality intervention from the many differences between datasets.
- **Speech-related mouth movement:** compare an upper-face descriptor subset with the full descriptor set. Treat this as a sensitivity analysis; it cannot alone prove the remaining signal is emotion-specific.
- **Temporal information:** compare ordered and time-shuffled features within an utterance on validation data to test whether the temporal component adds value.
- **Missing vision:** evaluate occlusion, tracking loss, and the audio/text fallback without silently excluding difficult utterances.
- **Live compatibility:** use only frames available at the prediction time. Check that appending future video does not change already-emitted features or predictions. Avoid whole-recording AU calibration or smoothing that gives offline evaluation information unavailable live.

The demo inspector should show the tracked face, current blendshape/AU values with correct labels, feature availability, emotion probabilities, and measured latency. A short consented webcam recording can test integration and failure handling. It is a deployment diagnostic, not independent evidence of emotion or trust accuracy.

After the two-week pilot, the next research milestone is AU measurement and cross-domain validation. Train a small participant-held-out blendshape-to-AU probe on DISFA and assess AU intensity error/agreement separately from downstream emotion usefulness. Check each AU's support and select thresholds only on validation data. Add Aff-Wild2 for in-the-wild evaluation using its annotation conventions, missing-label masks, and official split provenance. Evaluate pretrained AU models only on an appropriate unseen corpus or clearly label training-set overlap.

If the pilot supports continuing, broaden target evaluation across held-out IEMOCAP sessions, expand the frontal MEAD actor pool, and add non-frontal views as a controlled extension. Larger expression datasets should be added only after identifying whether the limiting issue is subject diversity, pose, spontaneous behavior, label coverage, or image quality. Multiple views and many neighboring frames increase observations without creating new independent participants.

The final report should distinguish three outcomes: facial features improve emotion recognition; facial features retain similar accuracy with lower runtime cost; or they provide no useful gain under this protocol. A complementary-feature claim requires improvement over the image-only visual branch, not merely over B0. An external-data claim requires the matched with/without-MEAD result. A null result is a valid reason to keep the simpler live system.

The local manifest, coverage/alignment reports, experiment configurations,
validation comparisons, checkpoint provenance and short runtime measurements
are now available. Final-test results, external-data comparisons, independent
AU measurement, longer spoken-session validation and the research report remain
future deliverables under the updated schedule.
