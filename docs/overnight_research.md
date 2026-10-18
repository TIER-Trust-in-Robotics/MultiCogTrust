# Overnight Research — 2026-10-17, about 22:40 to 04:40 EDT

**Completed on 18 October at about 02:00 EDT:** all five phases and the optional happiness-only visual experiment. The timed status notes below are historical. Final results, recommendations and verification are recorded later in this document. Nothing was deployed, committed or pushed.

The aim is to strengthen the evidence behind the demo's models and answer questions that matter for HRI, using only data already on this machine. Results are added under each phase as it finishes; the summary at the end comes last.

## Guardrails

- IEMOCAP Session 5 is never encoded or evaluated. The MSP-Podcast test sets and the MOSEI test split stay untouched.
- Choices are made out-of-fold on Sessions 1-3 (train on two, predict the third). Session 4 is evaluated once per question.
- At most four parallel training processes (18 GB RAM; six were reaped for low memory earlier). New disk use stays under about 5 GB (21 GB free); large temporary caches are deleted at the end.
- No commits, pushes or uploads. The HRI tool was stopped so the camera and microphone are off overnight.
- Negative results are reported. A phase that clearly is not paying off is stopped early.
- Code is in `research/overnight/` (untracked, for review). Outputs are in `data/processed/research/` (gitignored).

## Plan

| # | Question | Why it matters | Budget |
|---|---|---|---|
| 1 | Is B's gain over A real once seed noise is measured? Current-line vs previous-line recipe, 5 seeds each, single weight-averaged checkpoint, out-of-fold and Session 4 (Whisper). | B's F1 gain was within the noise of 2 seeds; its gain on emotion changes looked robust. A paired comparison over 5 seeds settles both. | 1 h |
| 2 | Does what the **other** party just said help? Variants: the other speaker's last line; own + other; own last two lines. Also an HRI variant where the other line is known exactly, as a robot's scripted line would be, while the participant's line comes from ASR. | In HRI the robot's words are known exactly, so this is free context. | 1.5 h |
| 3 | Which signal carries what: valence, arousal and dominance predicted from text vs audio (pooled frozen features, ridge). IEMOCAP Sessions 1-3 → Session 4, plus MSP-Podcast text valence. | Explains the anger/happiness confusion (both high arousal) and informs a trust signal, since trust relates to valence. | 1 h |
| 4 | Do the models know when they don't know? Calibration error, temperature scaling fitted out-of-fold, and accuracy vs coverage when unsure predictions are withheld, for A and B. | An HRI system should act only on confident readings. | 1 h |
| 5 | Write-up, recommendations and next steps; memory update; clean up temporary files. | | 0.5 h |

Stretch, if time remains: out-of-fold face-head runs to learn a face override limited to happiness (the face reads smiles well).

---

## Results

### Status at 22:48: stopped early (usage limit reached)

- **Phase 1** (40 seed-robustness runs, 4 at a time) is still training in the background. It writes to `data/processed/research/runs/*.npz`, and its log is `data/processed/research/logs/phase1.log`. No results have been analysed yet. To analyse them, use `research/overnight/metrics.py` (`oof(recipe, seed, "reference")` for out-of-fold, and `load("full_<recipe>_s<seed>", "4:asr" | "4:prevasr")` for Session 4).
- **Phase 2** encodings finished at 22:55: 4,290 lines per reference source (Sessions 1-4) and 1,027 per Session 4 Whisper or HRI source, 1.2 GB in `data/processed/research/text_*/`. The context strings have not been spot-checked yet, and no Phase 2 training has started.
- **Phases 3-5** have not started. The V/A/D order in metadata `vad` is confirmed as [valence, arousal, dominance] from the EmoEvaluation header.

### Phase 1 results: seed robustness (finished 23:33)

Five seeds (42-46) per recipe, scored with the epoch 3-8 weight average. Out-of-fold (OOF) = each of Sessions 1-3 predicted by a model trained on the other two, reference text. Session 4 was evaluated once. "Change" = accuracy on lines whose label differs from the speaker's previous **scored** line within the dialogue. Unscored lines may supply context but do not define change targets. This is a classification diagnostic, not a measured live switching latency. The CI is a paired t interval over seeds (4 df).

| metric | A recipe mean (sd) | B recipe mean (sd) | B-A paired mean | 95% CI | B wins |
|---|---|---|---|---|---|
| OOF F1 | 0.7123 (0.0047) | 0.7366 (0.0036) | +0.0243 | [+0.0161, +0.0324] | 5/5 |
| OOF change | 0.6074 (0.0053) | 0.6029 (0.0061) | -0.0045 | [-0.0168, +0.0077] | 1/5 |
| S4 Whisper F1 | 0.7032 (0.0066) | 0.7449 (0.0162) | +0.0417 | [+0.0174, +0.0660] | 5/5 |
| S4 Whisper change | 0.6752 (0.0204) | 0.7188 (0.0320) | +0.0436 | [-0.0050, +0.0922] | 5/5 |
| S4 ref F1 | 0.7388 (0.0045) | 0.7875 (0.0071) | +0.0487 | [+0.0381, +0.0594] | 5/5 |
| S4 ref change | 0.7200 (0.0100) | 0.7491 (0.0286) | +0.0291 | [-0.0017, +0.0598] | 4/5 |

- **Macro F1: B's gain is consistent across these seeds.** B beats A on all 5 seeds in every setting: +2.4 points OOF, +4.2 on Session 4 Whisper and +4.9 on Session 4 reference text. Every seed-based confidence interval excludes zero. These intervals measure training-seed variation on fixed data; they do not establish generalization to new people, datasets or HRI interactions.
- **Accuracy on emotion changes: not settled.** On Session 4, B is +4.4 points (Whisper) and +2.9 (reference), but both intervals touch zero. OOF it is flat (−0.5). The change-accuracy gain reported earlier depends on the session and should not be claimed as robust.
- Code: `research/overnight/analyze_phase1.py`.

### Continuation: 18 October, after 01:00 EDT

The continuation contract is saved in `data/processed/research/continuation_contract.json`. The earlier stopped status above is historical. Nothing has been committed, pushed or uploaded.

- **Phase 2 is running.** The original other-speaker cache chose context by utterance start time. Across development metadata, 374 of those previous-other lines end after the target itself ends. That cache is unsuitable for a causal HRI comparison. New context requires the earlier line to finish before the target starts. `encode_causal_context.py` rebuilt eight conditions with per-utterance context IDs, timing and feature hashes; the audit passed, with zero truncated current lines. `run_phase2.py` runs 18 OOF fits and then freezes the OOF ranking before six full training fits. Seeds are 42/43; all fits use the fixed epoch 3-8 weight average. The HRI condition uses the other human's exact reference text as a proxy for known robot words, not a validation of robot deployment.
- **Phase 3 is running.** All 7,869 finite IEMOCAP development V/A/D annotations are retained, including seven values outside the nominal 1-5 scale, confirmed against the original release files. They are neither clipped nor dropped. MSP-Podcast uses every native Train/Development transcript (169,190/34,399), across all emotion codes. Its ridge regularization transfers the IEMOCAP OOF text-valence choice; MSP Development will not tune it. No test transcripts are encoded.
- **Phase 4 is complete.** Temperature and rejection thresholds are fitted separately per recipe/seed on Sessions 1-3 OOF reference predictions, before loading Session 4 predictions. `phase4_summary.json` stores all five seeds, ten-bin ECE, NLL, Brier score and coverage by class. Temperatures are roughly 1.6-1.7. This changes confidence without changing the predicted class.

| Session 4 Whisper, five-seed mean | A: current line | B: previous own line |
|---|---|---|
| ECE before → after temperature scaling | 0.1292 → 0.0357 | 0.1050 → 0.0325 |
| NLL before → after | 0.7846 → 0.6993 | 0.6927 → 0.6201 |
| Accuracy, all available transcripts | 0.7077 | 0.7500 |
| Accuracy when retaining about 81% of all targets | 0.7621 | 0.8101 |
| Accuracy when retaining about 52% of all targets | 0.8543 | 0.9113 |

Four empty ASR transcripts are always abstentions. Calibration metrics and ordinary accuracy use 1,027 available predictions; coverage uses all 1,031 targets. The OOF target coverage levels transfer imperfectly to Session 4. At B's approximately 52% coverage point, class coverage is 78% anger, 44% happiness, 41% sadness and 34% neutral: high retained accuracy comes with unequal rejection. OOF post-fit calibration metrics are optimistic because the same OOF predictions fit the temperature. These are development-set diagnostics, not validated trust probabilities or a selected live rejection policy.

The calibration table concerns the five retrained A/B recipes. Its per-seed temperatures are specific to those experiment heads; they have not been validated as calibration for the shipped checkpoints. The original Phase 1 runs retained predictions but not their head weights or per-run source manifests. Their tables can be recomputed from the 40 arrays, but those original heads cannot now be replayed. All 24 new context heads are saved with source hashes and will be replayed before cache cleanup.

The optional happiness-only face experiment is also queued under `stretch_face.py`. Its protocol freezes a combined visual head, seeds 42/43, twenty epochs with weights averaged over 11-20, and an OOF-only correction grid before any fit. Each OOF head trains on two of Sessions 1-3; normalization uses only that fold's training frames. The gate can only replace a non-happiness decision with happiness when vision is available, the face head supports happiness and the audio/text head is within a fitted confidence limit. A no-op policy is included. The gate is frozen before the Session 4 reports and is not deployed.

`precision_control.py` adds an eight-fit comparator because the earlier previous-own-line cache is float32 while the new context stores are float16. It rounds the existing reference/Whisper text arrays to float16 and uses the same seeds 42/43, eight epochs, and epoch 3-8 average. This protocol was frozen before Phase 2's Session 4 results. It adds no hyperparameter search; its three workers leave one training slot for the optional face run after the primary four-worker controller finishes. The original `encode_text.py` generator is preserved byte-for-byte in `research/overnight/archive/encode_text.py.txt`; the old command now directs callers to the causal replacement.

The final context/dimension tables, recommendations and temporary-cache cleanup remain pending. Research source and these findings should be reviewed separately from the already committed HRI demo.

### Verification checkpoint, about 01:35 EDT

- All eight causal text caches passed reconstruction, timing, count, shape, finite-value and checksum checks in `audit_context.py`; the report is `verified_context_cache.json`.
- The primary 18 OOF fits are complete. The new-variant ranking is frozen as `causalown2` before any primary Session 4 result was inspected. OOF mean macro F1 over seeds 42/43: other-only 0.7169, own+other 0.7342, own-two 0.7469. Their respective change accuracies are 0.5878, 0.5909 and 0.5847. The older float32 previous-own-line recipe scores 0.7373 / 0.6043 on the same two seeds; the float16 control remains pending. These are development selection scores, not evidence that longer context transfers to HRI.
- The primary Session 4 fits are finishing; regression waits for them. MSP encoding finished at 203,589 rows, including 117 empty transcripts; the empty rows remain in the native analysis.
- The 4,290 scored frozen-encoder rows match the existing index, metadata and raw WAV/text hashes. All 7,869 current encoder arrays were checked for valid shapes/values and pinned in `encoder_input_manifest.jsonl`. The older 3,579 unscored arrays lack a historical source manifest: 16 deterministic samples were freshly encoded from raw inputs and matched, but that sample does not establish provenance for every unscored row. This limitation is retained in `encoder_input_audit.json`.
- A separate deterministic sample of 16 reference and 16 Whisper previous-own-line arrays matches fresh pinned text-encoder outputs (maximum absolute difference 0.000036). This is a spot-check, not full re-encoding; see `legacy_prev_cache_audit.json`.
- The suite passes **88 tests**, including future-speech exclusion, exact-other/own-ASR routing, rejection of Session 5, independent scikit-learn ridge comparisons, and restricted face-gate behavior. The command and test hashes are in `tests_report.json`.
- After all fits finish, run `audit_results.py` to replay the 32 new context/control heads and verify calibration/regression artifacts, then `audit_face.py` for the eight fresh face heads. `cleanup.py` refuses to remove temporary arrays until both audits pass and the fitting/audit processes have exited. Final recommendations, the requested memory note and the completion audit follow those checks.

### Phase 2: primary context results

The 24 primary fits and eight matched-comparator fits are complete. Every cell below is a mean over seeds 42/43, not an ensemble. OOF uses reference transcripts and chooses among the three new variants. All variants and the matched previous-own-line comparator use float16 text storage, the same cached frozen audio features and fixed averaged head. This controls storage precision for the comparison; inputs are cast back to float32 for training. It is not a deployed-model quantization or latency result. Differences from the older Phase 1 runs should not be attributed solely to quantization.

| Context before the current line | OOF macro F1 | OOF change accuracy | S4 reference F1 | S4 Whisper F1 | S4 Whisper change accuracy | S4 exact-other HRI proxy F1 |
|---|---|---|---|---|---|---|
| Own last completed line, matched comparator | 0.7405 | **0.6136** | 0.7926 | 0.7390 | **0.7091** | Not applicable |
| Other speaker's last completed line | 0.7169 | 0.5878 | 0.7547 | 0.7309 | 0.6758 | 0.7412 |
| Own + other speaker's last completed lines | 0.7342 | 0.5909 | 0.7799 | 0.7458 | 0.6788 | 0.7500 |
| Own two last completed lines | **0.7469** | 0.5847 | 0.7952 | 0.7523 | 0.7030 | Not applicable |

Own-two was selected among the three new variants using OOF macro F1, before the Session 4 reports. Against the matched comparator it gains **0.64 F1 points OOF but loses 2.89 points on emotion changes**, with that tradeoff present in both seeds. Session 4 Whisper F1 gains 1.33 points while change accuracy loses 0.61 points; reference-text change accuracy loses 4.55 points. The reference F1 gain reverses in seed 43. A higher aggregate F1 does not settle the HRI tradeoff, and these results do not warrant replacing the shorter context in the live tool.

Other-only and own+other are both below the matched previous-own-line comparator on OOF macro F1. Exact-other text improves their Whisper conditions by 1.03 and 0.42 F1 points, respectively. That comparison substitutes human reference transcripts for the other person's ASR; it does not validate transfer to a robot conversation. Context availability is defined by completed speech, without simulating ASR queue latency. Only two seeds were run, and Session 4 remains development data. Stored utterance boundaries/transcripts also do not test live VAD quality or end-to-end latency.

### Phase 3: affect dimensions

Ridge regressors use the mean and population standard deviation of each frozen encoder's unpadded sequence (audio capped at the first six seconds, text at 297 tokens). Text is DistilBERT; audio is DistilHuBERT, not an isolated handcrafted-prosody stream. Text/audio concatenation is the `both` condition. Each dimension's regularization is chosen by Sessions 1-3 OOF MSE. Session 4 is reported once. These probes measure information available in the representations; they do not identify causal reliance of the learned emotion fusion head.

All finite native IEMOCAP dimension labels are retained: 5,766 training and 2,103 Session 4 targets, including emotions outside the four-class classification view. Scores below are **MAE / Pearson r**, on the nominal 1-5 annotation scale.

| Frozen features | Valence | Arousal | Dominance |
|---|---|---|---|
| Text | 0.5819 / 0.5752 | 0.5045 / 0.3994 | 0.5743 / 0.3822 |
| Audio | 0.5863 / 0.5673 | 0.3923 / 0.7092 | 0.4874 / 0.5746 |
| Text + audio | **0.5210 / 0.6792** | 0.3900 / 0.7152 | 0.4882 / 0.5720 |

Audio carries most of the arousal and dominance signal in this probe. Valence benefits from both streams: concatenation reduces MAE by about 10.5% against text alone and 11.2% against audio alone. Adding text barely changes arousal and slightly worsens dominance MAE. This supports complementary information, rather than a universal claim that text or audio dominates. It does not by itself explain individual anger/happiness classification mistakes.

For MSP-Podcast, all native Train/Development transcripts and emotion codes are included (169,190 / 34,399). The ridge coefficients are fitted on MSP Train; only the regularization value transfers from IEMOCAP. This is not zero-shot IEMOCAP-to-MSP transfer. Text-only valence gives **MAE 0.7225, r 0.5758, R² 0.2822** on the nominal 1-7 scale. The training-mean baseline gives MAE 0.9010: the text model reduces MAE by about 19.8%. Its normalized MAE is 0.1204. The 117 empty transcripts remain in the analysis. This is a reference-transcript valence diagnostic, not a four-class emotion, ASR, acoustic or trust evaluation. No MSP audio was available or downloaded.

Every IEMOCAP dimension selected regularization 1.0, the strongest point in the predeclared grid [0.0001, 0.001, 0.01, 0.1, 1.0]; the optimum may lie beyond it. The objective is mean squared loss plus lambda times squared coefficient norm, with an unpenalized intercept and training-only feature standardization (equivalent scikit-learn Ridge alpha is n times lambda). MSP inherits the IEMOCAP OOF text-valence value instead of tuning on MSP Development. No grid expansion or clipping was selected after inspecting development results. The seven original IEMOCAP labels outside the nominal 1-5 range remain unchanged. Ridge pooling discards temporal structure, and the older unscored-cache provenance limitation above applies to these native IEMOCAP probes. Cross-dataset scale/domain differences prevent interpreting normalized errors as a direct ranking of task difficulty. None of these targets is measured trust.

### Stretch: restricted happiness correction

Eight fresh combined visual heads are complete: three OOF folds plus one full Sessions 1-3 fit for each seed 42/43. The OOF gate chooses a happiness label only when the visual head's top class is happiness with probability at least 0.5, the raw audio/text confidence is at most 0.6, visual features are available and the base class is not already happiness. Empty ASR still abstains. A no-op policy was in the selection grid.

| Macro F1, mean of two seeds | B recipe | B + restricted visual correction | Difference |
|---|---|---|---|
| OOF reference | 0.7373 | 0.7409 | +0.0036 |
| S4 reference | 0.7870 | 0.7887 | +0.0017 |
| S4 Whisper | 0.7359 | 0.7392 | +0.0033 |

The gain is not consistent across seeds. On Session 4 Whisper, seed 42 rises from 0.7284 to 0.7376 (15 errors fixed, 5 correct decisions broken), while seed 43 falls from 0.7434 to 0.7407 (7 fixed, 9 broken). This is a small exploratory gain, not evidence for replacing the shadow visual route. Combined-head availability means at least one image/blendshape stream is available; it does not mean every frame has a detected face. The rule changes labels only and does not produce calibrated fused probabilities. It was fitted against raw base confidence, so combining it with temperature scaling would require a new OOF check of the combined policy.

`audit_face.py` replays all eight saved heads exactly, recomputes their fold-specific normalization and gate scores, and verifies the frozen OOF policy. The first audit attempted bitwise equality for normalization buffers; the fixed ten-state float32 average produces about one ulp of roundoff. Its checker now permits that numerical difference while requiring exact prediction replay. The failed check and successful rerun are both retained in the log. No trained weights or result arrays were changed to pass the audit.

### Recommendations for the MVP

1. **Retain the existing A/B demonstration and shorter-context B option.** The five-seed study supports the previous-own-line recipe over a matched current-line recipe, but its emotion-change advantage is not established. Two previous own lines make a small F1 gain while weakening change detection. Select future HRI models against both aggregate F1 and responsiveness to changes, with that tradeoff declared before evaluation.
2. **Keep both audio and text.** The probes show complementary valence information and a strong audio advantage for arousal/dominance. A targeted next experiment is auxiliary valence/arousal supervision or dimension-specific fusion, tested on the same OOF folds and matched budgets. These representation probes are not a replacement for matched unimodal training and fusion interventions when measuring the classifier's actual modality reliance.
3. **Make uncertainty inspectable before using it for robot behavior.** Temperature scaling helps these heads, but deployment needs a matching head/checksum, transcript pipeline and validation of class coverage. Keep raw predictions available for inspection. The 91% retained accuracy point with roughly half the utterances withheld is not 91% accuracy over the whole conversation.
4. **Keep vision in shadow for now.** The happiness gate has a small mean benefit and a negative seed result. Other-speaker context also fails to beat own-line context out-of-fold. Neither result supports automatic promotion into the live system. The exact-other proxy remains a useful hypothesis for a later robot study.
5. **Make comparability the next research milestone.** Freeze labels, speakers/sessions, transcript source, preprocessing, seeds and training/selection budget for a MemoCMT reference and the local distilled model. Report macro F1, per-class confusion, change accuracy and Mac utterance latency together. A wider ridge grid can be considered using training-only validation; the development results here must not become another tuning target. This overnight study is not a SOTA reproduction or a live speed benchmark.
6. **Validate trust separately.** The current targets are emotion and affect dimensions. Valence, confidence and a smile are not validated measures of trust in the robot. The existing consented HRI recorder can support later participant-disjoint evaluation against its separate trust self-reports; no such trust model was trained or validated overnight.

### Reproducibility and retained limits

Code is under `research/overnight/`; machine-readable summaries, predictions, head checkpoints, manifests and logs are under `data/processed/research/`. Phase 1 can be reanalysed with `uv run python research/overnight/analyze_phase1.py`. `plot_dimensions.py` writes PNG/SVG correlation figures; `calibration.py` produced the selective-accuracy figure. The regression and head-replay audits require the temporary feature arrays and therefore run before cleanup. Completed training commands intentionally refuse to overwrite existing runs. A new scientific run should use a separate output location and frozen protocol.

After the experiment, a cache-resume guard was fixed: a completed MSP progress marker could otherwise skip encoding when cleanup had removed the matrix. Missing matrices now restart at row zero; a changed source contract still rejects reuse. Three regression tests cover the state transitions. The complete suite passes **91 tests**. The actually executed dimension source is preserved byte-for-byte in `archive/dimensions_8a568602d1ca.py.txt`; its original hash in the finished experiment is unchanged. `postrun_cache_guard.json` records the later source fix, and the final audit verifies the executed archive rather than rewriting scientific provenance. The earlier successful audit and test report are also preserved.

All scores remain development evidence. The two-seed extensions do not establish population uncertainty, the original 40 Phase 1 heads were not retained, the unscored encoder cache has only sampled raw-input replay verification, and no end-to-end HRI trust or latency result is claimed. Session 5, MSP test transcripts/audio and MOSEI test inference remain excluded by this work.

### Completion and cleanup

- **Phase 1:** all 40 original prediction runs are accounted for; the five-seed paired table and limitations are reproduced in `phase1_summary.json`.
- **Phase 2:** 24 primary context fits and eight matched storage/trainer control fits are complete, with the OOF ranking frozen before Session 4 reports. Every saved fusion head replays exactly across all 44 declared evaluation conditions (maximum probability difference 0).
- **Phase 3:** native IEMOCAP text/audio/combined V/A/D probes and MSP Train-to-Development text valence are complete. The saved regression outputs reproduce from their coefficients and the pre-cleanup arrays. The native train/development MSP speaker identifiers are disjoint. The final audit verifies the actual archived dimension source and retains the sampled-provenance limitation.
- **Phase 4:** ten OOF temperature fits, fixed coverage policies and Session 4 reports are verified, including NLL optimality checks, unchanged class decisions and recomputed coverage/calibration metrics.
- **Stretch:** all eight fresh visual heads replay exactly; training-only normalization and the OOF policy selection were independently checked. Mixed seed results are retained.
- **Phase 5:** the write-up, recommendations and requested memory note are saved. The complete suite passes **91 tests**. The post-run cache guard also resets its marker before allocating a new matrix, so an interrupted model load cannot leave a zero matrix marked complete. A synthetic failure-injection check exercised that failure and its retry without real corpus access, inference or training; both stayed incomplete at row zero. Executed sources, earlier audit/test records and failures are preserved separately.
- Cleanup removed **41,329 temporary arrays / 3.373 GB**. Research output occupied **4.197 GB** immediately before cleanup, within the approximate 5 GB cap; about **0.832 GB** remains. Raw datasets, original encoder/improvement/visual stores, every saved head, all predictions, contracts, manifests and logs are retained. The cleanup manifest records every removed path, byte count and checksum. No temporary text matrices or native dimension feature matrices remain.

The authoritative local verification records are `result_audit.json`, `stretch_face_audit.json`, `tests_report.json`, `postrun_cache_guard.json`, `cache_resume_fault_injection.json`, `cleanup_manifest.json` and `cleanup_complete.json` under `data/processed/research/`. PNG/SVG dimension figures and the selective-accuracy PNG are retained there as inspectable artifacts. The final requirement-by-requirement audit is `completion_audit.json`. Deployed checkpoint checksums remain unchanged, the experiment processes have exited, and the existing HRI capture tool remains stopped.
