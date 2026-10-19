"""Render a two-pane demo video: an IEMOCAP dialogue beside the model's output.

Left: the original split-screen dialogue video, the tracked actor highlighted,
with the IEMOCAP reference transcript. Right: what the deployed audio/text
model produces for the tracked actor, recomputed here through actual Whisper
tiny.en ASR and the emotion checkpoint. Each prediction appears when its
utterance ends plus the processing time measured on this machine.

Two right-pane views: "shift" follows one actor's emotion change; "modalities"
sets the deployed prediction beside text-only and voice-only models on the same
utterance. Both show the face-only model, which deployment runs at weight 0,
and the validated 25% face mixture that was not shipped. Faces are replayed
through the runtime visual extractor on the source video.

--compare renders A (deployed, current line only) beside B (retrained with the
speaker's previous Whisper line as text context, models/mvp/context_baseline.pt)
as compare_<clip>.mp4. A and B run through the live tool's shared encoders (one
audio encoding, then A and B in turn), and both results appear once both have
finished. Compare runs require --device cpu, since on the Apple GPU each new audio
length pays a one-off compile.

Utterance boundaries come from the IEMOCAP segmentation, not live VAD. Only
development sessions are accepted; Session 5 stays held out.

    uv run python src/render_inference_demo.py --clip all --device cpu --rerun-inference
    uv run python src/render_inference_demo.py --clip complaint_call --preview-at 40
    uv run python src/render_inference_demo.py --compare --clip all --device cpu
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import soundfile as sf

from src.prepare_iemocap import DEFAULT_METADATA

CHECKPOINT = ROOT / "models/mvp/audio_text_baseline.pt"
# Deployment uses visual weight 0, so its output is this audio/text ASR diagnostic.
CACHED_PREDICTIONS = ROOT / "data/processed/mvp/asr_validation/predictions.jsonl"
CACHED_TRANSCRIPTS = ROOT / "data/processed/mvp/asr_validation/transcripts.jsonl"
OUTPUT_DIR = ROOT / "data/processed/demo"
# deployment.pt is selected.pt's head at face weight 0 (it runs in shadow);
# selected.pt mixes it in at 0.25.
VISUAL_DIR = ROOT / "models/mvp/visual_day5"
CACHED_FACE = VISUAL_DIR / "combined_seed42.validation.jsonl"
CACHED_WITH_FACE = ROOT / "data/processed/mvp/day5_7/selected_asr_predictions.jsonl"

# Each clip: one actor's correctly predicted emotion shift, Session 4 only.
CLIPS = {
    "complaint_call": {
        "dialog": "Ses04F_impro08",
        "speaker": "M",
        "start": 110.0,
        "end": 185.0,
        "title": "Neutral → Anger",
        "subtitle": "Improvised · a refund call escalates",
    },
    "what_the_hell": {
        "dialog": "Ses04M_script01_1",
        "speaker": "F",
        "start": 246.0,
        "end": 296.0,
        "title": "Neutral → Anger",
        "subtitle": "Scripted · a family argument erupts",
    },
    "kiss_then_confession": {
        "dialog": "Ses04F_script01_3",
        "speaker": "M",
        "start": 126.5,
        "end": 196.0,
        "title": "Happiness → Sadness",
        "subtitle": "Scripted · a reunion turns to grief",
    },
    # Words alone read neutral; the voice carries the annotated emotion.
    "supervisor": {
        "dialog": "Ses04F_impro05",
        "speaker": "F",
        "start": 125.0,
        "end": 195.0,  # recording ends at 195.06 s
        "view": "modalities",
        "title": "Neutral words, angry voice",
        "subtitle": "Improvised · a lost-luggage complaint",
        "note": "Chosen where the words alone read neutral but the voice does not.",
    },
    "condolences": {
        "dialog": "Ses04M_impro06",
        "speaker": "F",
        "start": 43.5,
        "end": 121.5,
        "view": "modalities",
        "title": "Kind words, sad voice",
        "subtitle": "Improvised · consoling a grieving friend",
        "note": "Chosen where the words alone read neutral but the voice does not.",
    },
    "thanks_a_lot": {
        "dialog": "Ses04M_impro01",
        "speaker": "M",
        "start": 116.0,
        "end": 155.0,  # recording ends at 155.09 s
        "view": "modalities",
        "title": "Polite words, angry voice",
        "subtitle": "Improvised · sarcasm in a DMV line",
        "note": "Chosen where the words alone read neutral but the voice does not.",
    },
    # The face reads the happiness that the words and voice miss.
    "dorky": {
        "dialog": "Ses04F_impro07",
        "speaker": "F",
        "start": 290.5,
        "end": 329.0,
        "view": "modalities",
        "title": "Blunt words, smiling face",
        "subtitle": "Improvised · teasing about college plans",
        "note": "Chosen where the face corrects the deployed model.",
    },
    # A/B scenes (--compare only): where B's previous-line context helps most, and where it hurts most.
    "veiled_anger": {
        "dialog": "Ses04M_script03_2",
        "speaker": "M",
        "start": 96.0,
        "end": 169.0,
        "compare_only": True,
        "title": "Veiled anger: context helps",
        "subtitle": "Scripted · a bitter quarrel",
    },
    "quiet_grief": {
        "dialog": "Ses04F_script01_3",
        "speaker": "M",
        "start": 244.0,
        "end": 319.5,
        "compare_only": True,
        "title": "Quiet grief: context helps",
        "subtitle": "Scripted · a war memory",
    },
    "wedding_plans": {
        "dialog": "Ses04F_impro03",
        "speaker": "F",
        "start": 26.0,
        "end": 103.5,
        "compare_only": True,
        "title": "Excitement misread: context hurts",
        "subtitle": "Improvised · planning a wedding",
    },
}
# Demo v2 scenes (--set v2), chosen on Session 4 by stated rules over every tracked line:
# signals that disagree while B is right, emotion shifts B follows, and where the face could help.
CLIPS_V2 = {
    "1_split_anger": {
        "dialog": "Ses04F_impro04", "speaker": "F", "start": 118.0, "end": 169.0, "view": "multimodal",
        "title": "Mixed signals, clear anger",
        "subtitle": "Improvised · three years of job hunting",
        "note": "Chosen where words and voice disagree on most lines but B is right.",
    },
    "2_split_sadness": {
        "dialog": "Ses04F_impro02", "speaker": "F", "start": 82.5, "end": 152.4, "view": "multimodal",
        "title": "Angry words, sad voice",
        "subtitle": "Improvised · leaving the family to serve",
        "note": "Chosen where words and voice disagree on most lines but B is right.",
    },
    "3_split_three_emotions": {
        "dialog": "Ses04F_script01_3", "speaker": "M", "start": 5.3, "end": 87.5, "view": "multimodal",
        "title": "Anger, calm, then love",
        "subtitle": "Scripted · a proposal",
        "note": "Chosen for disagreeing signals across three emotions; B misses the last line.",
    },
    "4_shift_anger_neutral": {
        "dialog": "Ses04M_script01_2", "speaker": "F", "start": 3.7, "end": 68.0, "view": "multimodal",
        "title": "Anger, a pause, anger again",
        "subtitle": "Scripted · an unwelcome guest",
        "note": "Chosen where the emotion shifts and B follows it.",
    },
    "5_shift_grief_anger": {
        "dialog": "Ses04F_script01_1", "speaker": "M", "start": 10.3, "end": 95.5, "view": "multimodal",
        "title": "Calm, grief, then anger",
        "subtitle": "Scripted · a son confronts his mother",
        "note": "Chosen where the emotion shifts and B follows it.",
    },
    "6_shift_sadness_lifts": {
        "dialog": "Ses04F_script02_2", "speaker": "M", "start": 349.0, "end": 419.5, "view": "multimodal",
        "title": "Sadness lifts",
        "subtitle": "Scripted · a couple by the water at night",
        "note": "Chosen where the emotion shifts and B follows it; B misses the first line.",
    },
    "7_face_knows_better": {
        "dialog": "Ses04M_script03_1", "speaker": "F", "start": 160.9, "end": 230.4, "view": "multimodal",
        "title": "When the face knows better",
        "subtitle": "Scripted · laughing about old fights",
        "note": "Chosen where B is wrong but the face alone is right; B does not use the face.",
    },
}
MODALITY_ROWS = [
    ("text", "Text only", "the words, from the Whisper transcript"),
    ("audio", "Voice only", "tone of voice, first 6 s"),
    ("face", "Face only", "expression at 5 frames/s, first 6 s"),
    ("fused", "Deployed", "words + voice · face weight 0"),
    ("with_face", "With face", "deployed + 25% face · not shipped"),
]
FACE_ROWS = [r for r in MODALITY_ROWS if r[0] in ("face", "with_face")]
# Model B: same architecture, retrained on "previous same-speaker line [SEP] current line".
CONTEXT_CHECKPOINT = ROOT / "models/mvp/context_baseline.pt"
CACHED_UNSCORED_TRANSCRIPTS = ROOT / "data/processed/mvp/asr_train/transcripts_s4_unscored.jsonl"
COMPARE_ROWS = [
    ("fused", "A · Deployed", "current line only"),
    ("context", "B · New", "previous line + current line"),
    ("face", "Face only", "used by neither model"),
]
COMPARE_LANES = [("fused", "A · Deployed"), ("context", "B · New"), ("face", "Face"), ("label", "Annot.")]
# A/B benchmark, real predictor path, Whisper text: macro F1 on all 1,031 scored Session 4 lines;
# emotion-step medians over 2,098 lines (Apple GPU) and 300 lines (CPU). ASR is shared by both.
# Both models through the live tool's shared encoders: 64 ms on CPU (108 ms as two separate predictors).
COMPARE_SUMMARY = {"a_f1": 0.7229, "b_f1": 0.7284, "a_mps": 44.3, "b_mps": 50.1, "a_cpu": 49.5, "b_cpu": 50.9,
                   "shared_cpu": 64}
MODALITY_LANES = [("text", "Text"), ("audio", "Voice"), ("face", "Face"), ("fused", "Deployed"),
                  ("with_face", "With face"), ("label", "Annot.")]
SHIFT_LANES = [("fused", "Deployed"), ("face", "Face"), ("with_face", "With face"), ("label", "Annot.")]
# Demo v2 (--set v2): the live demo's default view, each signal alone and then B, the headline model.
V2_ROWS = [
    ("text", "Words only", "what the transcript alone says"),
    ("audio", "Voice only", "tone of voice alone, first 6 s"),
    ("face", "Face only", "expression alone · not used by B"),
    ("context", "B · Words + voice", "also reads the previous line"),
]
V2_LANES = [("text", "Words"), ("audio", "Voice"), ("face", "Face"), ("context", "B"), ("label", "Annot.")]
# Macro F1 on all 1,031 scored Session 4 lines, Whisper tiny.en with the live demo's settings
# (src/final_eval.py dry run for B; the words-only and voice-only controls on the same transcripts).
V2_SUMMARY = {"b": 0.725, "words": 0.605, "voice": 0.619, "face": 0.383}

LABELS = ["anger", "happiness", "sadness", "neutral"]
SHORT = ["ANG", "HAP", "SAD", "NEU"]
LABEL_CODE_TO_INDEX = {"ang": 0, "hap": 1, "exc": 1, "sad": 2, "neu": 3}
LABEL_CODE_NAMES = {
    "ang": "anger", "hap": "happiness", "exc": "excitement", "sad": "sadness",
    "neu": "neutral", "fru": "frustration", "sur": "surprise", "fea": "fear",
    "dis": "disgust", "oth": "other", "xxx": "no majority",
}
# Validated with the dataviz CVD checker against the #141413 surface, all pairs.
SERIES = [(0xD6, 0x3C, 0x45), (0xC9, 0x85, 0x00), (0x39, 0x87, 0xE5), (0x1B, 0xAF, 0x7A)]
BG = (0x14, 0x14, 0x13)
CARD = (0x1C, 0x1C, 0x1B)
RULE = (0x2E, 0x2E, 0x2B)
TRACK = (0x29, 0x29, 0x27)
INK = (0xF0, 0xEF, 0xEC)
INK2 = (0xC3, 0xC2, 0xB7)
INK3 = (0x8A, 0x89, 0x80)
INK4 = (0x55, 0x54, 0x4F)

W, H = 1920, 1080
# The 720x480 source is letterboxed; both actor panels sit inside this crop.
SOURCE_CROP = "704:238:8:122"
VIDEO_X, VIDEO_Y, VIDEO_W, VIDEO_H = 40, 104, 1080, 366
SCRIPT_Y0, SCRIPT_Y1 = 494, 1004
PANE_X, PANE_R = 1160, 1880
EASE_SEC = 0.45

HELVETICA = "/System/Library/Fonts/HelveticaNeue.ttc"
MONO = "/System/Library/Fonts/SFNSMono.ttf"
SF = "/System/Library/Fonts/SFNS.ttf"


def font(size, weight="regular"):
    index = {"regular": 0, "bold": 1, "medium": 10, "light": 7}[weight]
    return ImageFont.truetype(HELVETICA, size, index=index)


def sf_font(size, weight):
    # Helvetica Neue has no arrow glyph; SF does.
    f = ImageFont.truetype(SF, size)
    f.set_variation_by_name(weight)
    return f


FONTS = {
    "title": sf_font(34, "Bold"),
    "subtitle": font(24, "light"),
    "meta": font(19),
    "eyebrow": font(16, "bold"),
    "body": font(24),
    "body_medium": font(24, "medium"),
    "subs": font(26),
    "tag": font(16, "bold"),
    "label": font(22),
    "label_bold": font(22, "bold"),
    "value": ImageFont.truetype(MONO, 21),
    "small": font(17),
    "small_mono": ImageFont.truetype(MONO, 16),
    "block": font(13, "bold"),
    "footer": font(17),
}


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def ease_out(x):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


def mix(color, background, alpha):
    return tuple(int(round(c * alpha + b * (1 - alpha))) for c, b in zip(color, background))


def clip_rows(spec, metadata_path):
    rows = [
        r for r in load_jsonl(metadata_path)
        if r["dialog_id"] == spec["dialog"]
        and r["dialog_end_sec"] > spec["start"]
        and r["dialog_start_sec"] < spec["end"]
    ]
    if not rows:
        raise ValueError(f"No utterances for {spec['dialog']} in the clip window")
    if any(r["session"] == 5 for r in rows):
        raise ValueError("Session 5 is held out until the frozen final evaluation")
    if any(r["session"] != 4 for r in rows):
        raise ValueError("Demo clips come from Session 4, the validation split the header names")
    return sorted(rows, key=lambda r: r["dialog_start_sec"])


class PooledControls:
    """Text-only and voice-only readings (src/modality_controls.py) for the replayed lines.

    The renderer encodes each line itself, so it pools the deployed encoders' outputs here;
    the live tool reuses its shared encodings instead. Both give identical readings.
    """

    def __init__(self, predictor):
        from src.modality_controls import ModalityControls

        self.predictor = predictor
        self.controls = ModalityControls()

    def pooled_features(self, text, audio, sample_rate):
        import torch

        from src.emotion_checkpoint import prepare_audio, tokenize_text

        predictor = self.predictor
        with torch.inference_mode():
            tokens = tokenize_text(predictor.tokenizer, text, predictor.preprocessing)
            text_features = predictor.model.encode_text(
                tokens["input_ids"].to(predictor.device), tokens["attention_mask"].to(predictor.device)
            )[0].mean(0).cpu()
            waveform = prepare_audio(audio, sample_rate, predictor.preprocessing)[None].to(predictor.device)
            audio_features = predictor.model.encode_audio(waveform)[0].mean(0).cpu()
        return {"text": text_features, "audio": audio_features}

    def predict(self, text, audio, sample_rate):
        readings = self.controls.predict(self.pooled_features(text, audio, sample_rate))
        return {mode: [p[label] for label in LABELS] for mode, p in readings.items()}


def whisper(live_asr):
    """tiny.en with the Session 4 evaluation settings, or the live demo's (FAST_ASR, 4 threads)."""
    from faster_whisper import WhisperModel
    from huggingface_hub import snapshot_download

    from src.inference_pipeline import FAST_ASR

    model_path = Path(snapshot_download("Systran/faster-whisper-tiny.en", local_files_only=True))
    transcriber = WhisperModel(str(model_path), device="cpu", compute_type="int8", cpu_threads=4 if live_asr else 2)
    return transcriber, (FAST_ASR if live_asr else {}), model_path


def run_inference(rows, device, live_asr=False):
    """Actual ASR + emotion model on the tracked speaker's finished utterances."""
    import torch

    from src.emotion_checkpoint import prepare_audio
    from src.emotion_fusion_infer import CrossModalEmotionPredictor
    from src.inference_pipeline import transcribe_samples

    torch.set_num_threads(2)
    transcriber, asr_options, model_path = whisper(live_asr)
    predictor = CrossModalEmotionPredictor(CHECKPOINT, device, local_files_only=True)
    controls = PooledControls(predictor)
    preprocessing = predictor.preprocessing

    def infer(row):
        audio, sr = sf.read(row["audio_path"], dtype="float32")
        started = time.perf_counter()
        text = transcribe_samples(transcriber, prepare_audio(audio, sr, preprocessing).numpy(), **asr_options)
        transcribed = time.perf_counter()
        prediction = predictor.predict(text, audio, sr) if text else None
        finished = time.perf_counter()
        return text, prediction, 1000 * (transcribed - started), 1000 * (finished - transcribed)

    infer(rows[0])  # warm-up: lazy initialization is not per-utterance latency
    cached_predictions = {p["utterance_id"]: p for p in load_jsonl(CACHED_PREDICTIONS)}
    cached_transcripts = {t["utterance_id"]: t for t in load_jsonl(CACHED_TRANSCRIPTS)}
    results = []
    for row in rows:
        uid = row["utterance_id"]
        text, prediction, asr_ms, emotion_ms = infer(row)
        probabilities = (
            [prediction.probabilities[name] for name in LABELS] if prediction else None
        )
        record = {
            "utterance_id": uid,
            "start_sec": row["dialog_start_sec"],
            "end_sec": row["dialog_end_sec"],
            "duration_sec": row["duration_sec"],
            "label_code": row["label_code"],
            "label": LABEL_CODE_TO_INDEX.get(row["label_code"]),
            "reference_text": row["text"],
            "asr_text": text,
            "prediction": prediction.label_index if prediction else None,
            "probabilities": probabilities,
            "asr_ms": asr_ms,
            "emotion_ms": emotion_ms,
            # Untimed: the controls are not part of the deployed latency.
            "controls": controls.predict(text, *sf.read(row["audio_path"], dtype="float32")) if text else None,
        }
        if not live_asr and uid in cached_predictions and probabilities is not None:
            cached = cached_predictions[uid]
            record["parity"] = {
                "asr_text_matches": cached_transcripts[uid]["text"] == text,
                "prediction_matches": cached["prediction"] == record["prediction"],
                "max_abs_probability_diff": float(np.max(np.abs(
                    np.array([cached["probabilities"][name] for name in LABELS]) - probabilities
                ))),
            }
        results.append(record)
    return {
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "checkpoint_epoch": predictor.provenance.get("epoch"),
        "asr_model": "Systran/faster-whisper-tiny.en",
        "asr_revision": model_path.name,
        "asr_settings": "live demo (FAST_ASR, 4 threads)" if live_asr else "Session 4 evaluation",
        "device": device,
        "utterances": results,
    }


def run_face_inference(rows, inference, device):
    """Uncached face replay (runtime extractor) and the visual head, per utterance."""
    from contextlib import closing

    import torch

    from src.extract_iemocap_visual import decoded_video
    from src.visual_features import panel_crops, speaker_side
    from src.visual_fusion import VisualFusionPredictor
    from src.visual_stream import VisualExtractor, extract_interval

    torch.set_num_threads(2)
    shadow = VisualFusionPredictor(VISUAL_DIR / "deployment.pt", CHECKPOINT)
    mixture = VisualFusionPredictor(VISUAL_DIR / "selected.pt", CHECKPOINT)
    video = rows[0]["video_path"]
    with closing(decoded_video(video)) as frames:
        crops = panel_crops(next(frames)[2])
    cached_face = {r["utterance_id"]: r["visual_probabilities"] for r in load_jsonl(CACHED_FACE)}
    cached_with_face = {r["utterance_id"]: r["probabilities"] for r in load_jsonl(CACHED_WITH_FACE)}
    by_id = {u["utterance_id"]: u for u in inference["utterances"]}
    for row in rows:
        u = by_id[row["utterance_id"]]
        side = speaker_side(row["dialog_id"], row["speaker"])
        # Frame timestamps restart per interval, so each needs a fresh extractor.
        extractor = VisualExtractor(shadow.payload["feature_config"], device)
        boxes, frame_ms = [], []
        extract = extractor.face.extract

        def recording(rgb, timestamp_ms, extract=extract, boxes=boxes, offset=crops[side][:2]):
            face = extract(rgb, timestamp_ms)
            box = (face.box + np.tile(offset, 2)).tolist() if face.image_valid else []
            boxes.append([timestamp_ms / 1000, *box])
            return face

        extractor.face.extract = recording
        try:
            visual_input, _ = extract_interval(
                extractor, video, side, row["dialog_start_sec"], row["dialog_end_sec"],
                notify=lambda summary, _: frame_ms.append(summary["processing_ms"]),
            )
        finally:
            extractor.close()
        # Uniform stand-in when ASR abstained: the visual output does not depend on it.
        base = u["probabilities"] or [0.25] * 4
        started = time.perf_counter()
        face = shadow.predict(dict(zip(LABELS, base)), visual_input)
        head_ms = 1000 * (time.perf_counter() - started)
        mixed = mixture.predict(dict(zip(LABELS, base)), visual_input)
        u.update(
            face=[face["visual_probabilities"][n] for n in LABELS] if face["visual_available"] else None,
            with_face=[mixed["probabilities"][n] for n in LABELS] if u["probabilities"] else None,
            face_boxes=boxes,
            face_frame_ms=float(np.median(frame_ms)) if frame_ms else None,
            face_head_ms=head_ms,
        )
        if u["face"] and cached_face.get(u["utterance_id"]):
            u["face_parity"] = {
                "face_max_abs_diff": float(np.max(np.abs(np.array(cached_face[u["utterance_id"]]) - u["face"]))),
            }
            if u["with_face"] and u["utterance_id"] in cached_with_face:
                u["face_parity"]["with_face_max_abs_diff"] = float(
                    np.max(np.abs(np.array(cached_with_face[u["utterance_id"]]) - u["with_face"]))
                )
    inference["visual_checkpoints"] = {
        "shadow": str((VISUAL_DIR / "deployment.pt").relative_to(ROOT)),
        "with_face": str((VISUAL_DIR / "selected.pt").relative_to(ROOT)),
        "with_face_weight": mixture.alpha,
    }
    return inference


def run_context_inference(rows, inference, device, metadata_path, live_asr=False):
    """Model B on the tracked actor's lines: previous same-speaker Whisper line [SEP] current line."""
    import torch

    from src.emotion_fusion_infer import CrossModalEmotionPredictor
    from src.live_demo import SharedEncoders

    torch.set_num_threads(2)
    predictor = CrossModalEmotionPredictor(CONTEXT_CHECKPOINT, device, local_files_only=True)
    deployed = CrossModalEmotionPredictor(CHECKPOINT, device, local_files_only=True)
    # The live HRI tool's path: one audio encoding, then A and B one after the other.
    shared = SharedEncoders(deployed, predictor)
    cached = {t["utterance_id"]: t["text"]
              for path in (CACHED_TRANSCRIPTS, CACHED_UNSCORED_TRANSCRIPTS) for t in load_jsonl(path)}
    first = rows[0]
    earlier = sorted(
        (r for r in load_jsonl(metadata_path)
         if r["dialog_id"] == first["dialog_id"] and r["speaker"] == first["speaker"]
         and r["dialog_start_sec"] < first["dialog_start_sec"]),
        key=lambda r: r["dialog_start_sec"],
    )
    if live_asr:  # transcribe back from the clip start with the live settings until a line has words
        from src.emotion_checkpoint import prepare_audio
        from src.inference_pipeline import transcribe_samples

        transcriber, asr_options, _ = whisper(True)
        previous = None
        for r in reversed(earlier):
            audio, sr = sf.read(r["audio_path"], dtype="float32")
            previous = transcribe_samples(transcriber, prepare_audio(audio, sr, predictor.preprocessing).numpy(),
                                          **asr_options) or None
            if previous:
                break
    else:  # lines before the clip were transcribed by the same Whisper settings (cached, deterministic)
        previous = next((cached[r["utterance_id"]] for r in reversed(earlier) if cached.get(r["utterance_id"])), None)
    silence = np.zeros(16000 * 3, np.float32)
    for _ in range(3):  # warm-up is not per-utterance latency
        shared.predict("Ready.", "Ready for the next utterance. [SEP] Ready.", silence, 16000)
    by_id = {u["utterance_id"]: u for u in inference["utterances"]}
    for row in rows:
        u = by_id[row["utterance_id"]]
        text = u["asr_text"]
        if not text:  # deployment abstains; the context carries over unchanged
            u["context_model"] = None
            continue
        audio, sr = sf.read(row["audio_path"], dtype="float32")
        started = time.perf_counter()
        out = shared.predict(text, f"{previous} [SEP] {text}" if previous else text, audio, sr)
        both_ms = 1000 * (time.perf_counter() - started)
        (a, a_ms), (b, b_ms) = out["A"], out["B"]
        u["context_model"] = {
            "previous_text": previous,
            "probabilities": [b.probabilities[name] for name in LABELS],
            "prediction": b.label_index,
            "emotion_ms": b_ms,
            "a_emotion_ms": a_ms,
            "both_emotion_ms": both_ms,  # A then B with shared encoders; both results appear after this
            "a_max_abs_diff": float(np.max(np.abs(
                np.array([a.probabilities[name] for name in LABELS]) - u["probabilities"]))),
        }
        previous = text
    inference["context_checkpoint"] = str(CONTEXT_CHECKPOINT.relative_to(ROOT))
    inference["context_device"] = device
    return inference


def wrap(draw, text, fnt, width):
    lines, current = [], []
    for word in text.split():
        trial = " ".join(current + [word])
        if current and draw.textlength(trial, font=fnt) > width:
            lines.append(current)
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(current)
    return lines


def text_right(draw, xy, text, fnt, fill):
    draw.text((xy[0] - draw.textlength(text, font=fnt), xy[1]), text, font=fnt, fill=fill)


def fits(draw, text, x0, x1):
    return x1 - x0 >= draw.textlength(text, font=FONTS["block"]) + 9


def modality_probabilities(u, key):
    if key in ("text", "audio"):
        values = u["controls"][key]
    elif key == "context":
        values = (u.get("context_model") or {}).get("probabilities")
    else:
        values = u.get({"fused": "probabilities"}.get(key, key))
    return None if values is None else np.array(values)


def compare_readout(u, a, b):
    if a == b:
        return f"Both read {LABELS[a]}."
    if u["label"] is not None and u["label"] in (a, b):
        verb = "fixes" if u["label"] == b else "breaks"
        return f"Context {verb} it: A says {LABELS[a]}, B says {LABELS[b]}."
    return f"A reads {LABELS[a]}; B reads {LABELS[b]}."


def modality_readout(vectors):
    t, a, f = (LABELS[int(np.argmax(vectors[k]))] for k in ("text", "audio", "fused"))
    if vectors["with_face"] is not None and LABELS[int(np.argmax(vectors["with_face"]))] != f:
        return f"Adding the face turns {f} into {LABELS[int(np.argmax(vectors['with_face']))]}."
    if t != f and a == f:
        return f"Words alone read {t}; the voice carries {f}."
    if t != f:
        return f"Words and voice each miss {f}; fused, they get it."
    if a != f:
        return f"The words carry {f}; the voice alone reads {a}."
    return f"Words and voice agree: {f}."


def multimodal_readout(vectors):
    """The live demo's readout: where each signal agrees with B. Correctness is shown separately."""
    b = LABELS[int(np.argmax(vectors["context"]))]
    words, voice, face = (None if vectors.get(k) is None else LABELS[int(np.argmax(vectors[k]))]
                          for k in ("text", "audio", "face"))
    if words is None or voice is None:
        line = f"B reads {b}."
    elif words == voice == b:
        line = f"Words and voice agree: {b}."
    elif words == voice:
        line = f"Words and voice each read {words}; with the previous line, B reads {b}."
    elif voice == b:
        line = f"The words alone read {words}; the voice carries {b}."
    elif words == b:
        line = f"The words carry {b}; the voice alone reads {voice}."
    else:
        line = f"The words read {words} and the voice {voice}; combined, B reads {b}."
    if face is not None:
        line += " The face agrees." if face == b else f" The face alone reads {face}."
    return line


def annotator_name(u):
    if u["label_code"] == "exc":
        return "excitement (= happiness)"
    return LABEL_CODE_NAMES.get(u["label_code"], u["label_code"])


def draw_check(draw, x, y, ok, color):
    if ok:
        draw.line([(x, y + 9), (x + 6, y + 15), (x + 17, y + 2)], fill=color, width=3)
    else:
        draw.line([(x + 2, y + 2), (x + 15, y + 15)], fill=color, width=3)
        draw.line([(x + 15, y + 2), (x + 2, y + 15)], fill=color, width=3)


class Renderer:
    def __init__(self, name, spec, rows, inference, fps):
        self.spec, self.fps = spec, fps
        self.t0, self.t1 = spec["start"], spec["end"]
        self.tracked = spec["speaker"]
        left_speaker = spec["dialog"][5]
        self.tracked_side = "left" if self.tracked == left_speaker else "right"
        self.rows = rows
        self.view = spec.get("view", "shift")
        if self.view == "modalities":
            # Single-signal rows, then a gap, then the two combined rows.
            self.signal_rows = MODALITY_ROWS
            self.row_y = [384 + i * 56 + (14 if i >= 3 else 0) for i in range(len(MODALITY_ROWS))]
            self.lanes, lane_y0 = MODALITY_LANES, 826
        elif self.view == "multimodal":
            # Each signal alone, a gap, then B.
            self.signal_rows = V2_ROWS
            self.row_y = [420 + i * 56 + (14 if i >= 3 else 0) for i in range(len(V2_ROWS))]
            self.lanes, lane_y0 = V2_LANES, 826
        elif self.view == "compare":
            # Two model rows with a latency line each, then the face reference row.
            self.signal_rows, self.row_y = COMPARE_ROWS, [416, 496, 588]
            self.lanes, lane_y0 = COMPARE_LANES, 808
            self.plot_x0, self.lane_h = PANE_X + 122, 24
        else:
            self.signal_rows, self.row_y = FACE_ROWS, [594, 652]
            self.lanes, lane_y0 = SHIFT_LANES, 874
        pitch = 32 if self.view == "compare" else 24
        self.lane_y = {key: lane_y0 + i * pitch for i, (key, _) in enumerate(self.lanes)}
        by_id = {u["utterance_id"]: u for u in inference["utterances"]}
        self.utts = []
        for row in rows:
            if row["speaker"] != self.tracked:
                continue
            u = dict(by_id.get(row["utterance_id"], {}))
            u.update(start_sec=row["dialog_start_sec"], end_sec=row["dialog_end_sec"],
                     label_code=row["label_code"], label=LABEL_CODE_TO_INDEX.get(row["label_code"]))
            # In the comparison both results appear once A and then B have run, as live.
            context = u.get("context_model") or {}
            u["step_ms"] = (context["both_emotion_ms"] if self.view in ("compare", "multimodal") and context
                            else u.get("emotion_ms", 0))
            u["land_sec"] = (
                u["end_sec"] + (u["asr_ms"] + u["step_ms"]) / 1000
                if u.get("probabilities") else None
            )
            self.utts.append(u)
        self.base = self.static_layer(name)

    # ---- layout helpers -------------------------------------------------
    def tx(self, t):
        """Clip time -> x on the trajectory chart and lanes."""
        return self.plot_x0 + (t - self.t0) / (self.t1 - self.t0) * (self.plot_x1 - self.plot_x0)

    plot_x0, plot_x1 = PANE_X + 104, PANE_R - 16
    plot_y0, plot_y1 = 780, 856
    lane_h = 20

    def legend(self, d, y):
        x = PANE_R - 20
        for i in reversed(range(4)):
            x -= d.textlength(LABELS[i].capitalize(), font=FONTS["small"])
            d.text((x, y), LABELS[i].capitalize(), font=FONTS["small"], fill=INK2)
            x -= 18
            d.rounded_rectangle([x, y + 4, x + 12, y + 16], radius=2, fill=SERIES[i])
            x -= 18

    def time_axis(self, d, y):
        for s in range(0, int(self.t1 - self.t0) + 1, 10):
            x = self.tx(self.t0 + s)
            d.line([(x, y), (x, y + 6)], fill=INK4, width=1)
            label = f"{s // 60}:{s % 60:02d}"
            w = d.textlength(label, font=FONTS["small_mono"])
            d.text((min(x - w / 2, PANE_R - 12 - w), y + 10), label, font=FONTS["small_mono"], fill=INK3)

    def static_signal_rows(self, d):
        for (key, title, caption), y in zip(self.signal_rows, self.row_y):
            title_font = FONTS["label_bold"] if key in ("fused", "context") else FONTS["label"]
            d.text((PANE_X + 20, y), title, font=title_font, fill=INK)
            d.text((PANE_X + 32 + d.textlength(title, font=title_font), y + 4), caption,
                   font=FONTS["small"], fill=INK3)

    def static_timeline(self, d, y0):
        """The "over this clip" card header, lane labels and time axis."""
        d.rounded_rectangle([PANE_X, y0, PANE_R, 1004], radius=10, fill=CARD)
        d.text((PANE_X + 20, y0 + 14), "OVER THIS CLIP", font=FONTS["eyebrow"], fill=INK3)
        self.legend(d, y0 + 12)
        for key, label in self.lanes:
            text_right(d, (self.plot_x0 - 10, self.lane_y[key]), label,
                       FONTS["small"], INK if key == "fused" else INK2)
        self.time_axis(d, self.lane_y["label"] + self.lane_h + 6)

    def static_modalities(self, d):
        d.rounded_rectangle([PANE_X, 340, PANE_R, 764], radius=10, fill=CARD)
        d.text((PANE_X + 20, 354), "SAME UTTERANCE  ·  EACH SIGNAL ALONE, THEN COMBINED",
               font=FONTS["eyebrow"], fill=INK3)
        rule_y = (self.row_y[2] + 46 + self.row_y[3]) // 2
        d.line([(PANE_X + 20, rule_y), (PANE_R - 20, rule_y)], fill=RULE, width=1)
        self.static_timeline(d, 780)
        d.text((40, 1026),
               f"{self.spec['note']} Text, voice and face only are separate comparison models (text and voice: "
               "3 seeds on the deployed encoders). Annot. = annotator majority.",
               font=FONTS["footer"], fill=INK3)
        d.text((40, 1050),
               "Session 4 macro F1 with Whisper ASR: deployed 0.723 · text only 0.596 · voice only 0.631 · "
               "face only 0.383 · with face 0.721, so the face was not shipped.",
               font=FONTS["footer"], fill=INK3)

    def static_multimodal(self, d):
        d.rounded_rectangle([PANE_X, 372, PANE_R, 764], radius=10, fill=CARD)
        d.text((PANE_X + 20, 386), "SAME UTTERANCE  ·  EACH SIGNAL ALONE, THEN B", font=FONTS["eyebrow"], fill=INK3)
        rule_y = (self.row_y[2] + 46 + self.row_y[3]) // 2
        d.line([(PANE_X + 20, rule_y), (PANE_R - 20, rule_y)], fill=RULE, width=1)
        self.static_timeline(d, 780)
        d.text((40, 1026),
               f"{self.spec['note']} Words, voice and face only are separate comparison models; B combines words and "
               "voice and also reads the speaker's previous line. Annot. = annotator majority.",
               font=FONTS["footer"], fill=INK3)
        summary = V2_SUMMARY
        d.text((40, 1050),
               f"Session 4 macro F1, Whisper ASR with the live demo's settings: B {summary['b']:.3f} · words only "
               f"{summary['words']:.3f} · voice only {summary['voice']:.3f} · face only {summary['face']:.3f}. "
               "Timings: CPU, as the live demo runs.",
               font=FONTS["footer"], fill=INK3)

    def static_layer(self, name):
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        spec = self.spec
        d.text((40, 34), spec["title"], font=FONTS["title"], fill=INK)
        tw = d.textlength(spec["title"], font=FONTS["title"])
        compare = "A vs B  ·  " if self.view == "compare" else ""
        subtitle = f"{spec['subtitle']}  ·  {compare}{spec['dialog']}"
        d.text((40 + tw + 20, 43), subtitle, font=FONTS["subtitle"], fill=INK2)
        meta = "IEMOCAP Session 4 · validation split · actors not seen in training"
        if 40 + tw + 20 + d.textlength(subtitle, font=FONTS["subtitle"]) + 24 > PANE_R - d.textlength(meta, font=FONTS["meta"]):
            raise ValueError(f"{name}: title and subtitle overlap the header; shorten them")
        text_right(d, (PANE_R, 45), meta, FONTS["meta"], INK3)
        d.line([(40, 86), (PANE_R, 86)], fill=RULE, width=1)

        # Right pane cards.
        for y0, y1 in ((104, 184), (200, 356 if self.view in ("compare", "multimodal") else 324)):
            d.rounded_rectangle([PANE_X, y0, PANE_R, y1], radius=10, fill=CARD)
        d.text((PANE_X + 20, 118), "MODEL OUTPUT", font=FONTS["eyebrow"], fill=INK3)
        actor = "female" if self.tracked == "F" else "male"
        d.text((PANE_X + 20, 140), f"Tracking the {actor} actor ({self.tracked_side} panel)",
               font=FONTS["body_medium"], fill=INK)
        d.text((PANE_X + 20, 214), "HEARD BY THE MODEL  ·  WHISPER TINY.EN", font=FONTS["eyebrow"], fill=INK3)
        if self.view == "modalities":
            self.static_modalities(d)
        elif self.view == "multimodal":
            self.static_multimodal(d)
        elif self.view == "compare":
            self.static_compare(d)
        else:
            self.static_shift(d)
        self.static_signal_rows(d)
        return np.asarray(img).copy()

    def static_compare(self, d):
        d.rounded_rectangle([PANE_X, 372, PANE_R, 740], radius=10, fill=CARD)
        d.text((PANE_X + 20, 386), "SAME UTTERANCE  ·  TWO MODELS", font=FONTS["eyebrow"], fill=INK3)
        d.line([(PANE_X + 20, 574), (PANE_R - 20, 574)], fill=RULE, width=1)
        self.static_timeline(d, 756)
        summary = COMPARE_SUMMARY
        d.text((40, 1026),
               "A = deployed audio_text_baseline.pt (current line only). B = same architecture retrained with the "
               "speaker's previous line as text context (context_baseline.pt). Face = face-only model.",
               font=FONTS["footer"], fill=INK3)
        d.text((40, 1050),
               f"Session 4, Whisper ASR: macro F1 A {summary['a_f1']:.3f} · B {summary['b_f1']:.3f}. "
               f"Emotion step median on CPU: A {summary['a_cpu']:.0f} ms · B {summary['b_cpu']:.0f} ms alone; "
               f"both via the live tool's shared encoders {summary['shared_cpu']:.0f} ms (per-line times shown).",
               font=FONTS["footer"], fill=INK3)

    def static_shift(self, d):
        d.rounded_rectangle([PANE_X, 340, PANE_R, 714], radius=10, fill=CARD)
        d.text((PANE_X + 20, 354), "EMOTION PROBABILITIES  ·  DEPLOYED (WORDS + VOICE)",
               font=FONTS["eyebrow"], fill=INK3)
        d.line([(PANE_X + 20, 578), (PANE_R - 20, 578)], fill=RULE, width=1)
        self.static_timeline(d, 730)
        # Chart frame: recessive gridlines at 0, 0.5, 1.
        for v in (0.0, 0.5, 1.0):
            y = self.plot_y1 - v * (self.plot_y1 - self.plot_y0)
            d.line([(self.plot_x0, y), (self.plot_x1, y)], fill=RULE, width=1)
            text_right(d, (self.plot_x0 - 10, y - 9), f"{v:.1f}", FONTS["small_mono"], INK3)
        d.text((40, 1026),
               "Clip chosen to show a correctly predicted shift. Deployed = prediction after each utterance from "
               "words + voice. Face = face-only model; With face = deployed + 25% face (not shipped).",
               font=FONTS["footer"], fill=INK3)
        d.text((40, 1050),
               "Session 4 with Whisper ASR: deployed 72.4% accurate, macro F1 0.723 · face only 0.383 · "
               "with face 0.721. Models see the first 6 s of each utterance. Annot. = annotator majority.",
               font=FONTS["footer"], fill=INK3)

    # ---- per-frame state ------------------------------------------------
    def landed(self, t):
        return [u for u in self.utts if u["land_sec"] is not None and u["land_sec"] <= t]

    def displayed_probabilities(self, t):
        done = self.landed(t)
        if not done:
            return None, None
        current = np.array(done[-1]["probabilities"])
        previous = np.array(done[-2]["probabilities"]) if len(done) > 1 else np.zeros(4)
        k = ease_out((t - done[-1]["land_sec"]) / EASE_SEC)
        return previous + (current - previous) * k, done[-1]

    def status(self, t):
        for u in self.utts:
            if u["start_sec"] <= t < u["end_sec"]:
                return "listening", u
        for u in self.utts:
            if u["land_sec"] is not None and u["end_sec"] <= t < u["land_sec"]:
                return "processing", u
        return "waiting", None

    # ---- drawing --------------------------------------------------------
    def frame(self, t, video_rgb):
        arr = self.base.copy()
        self.draw_video(arr, video_rgb)
        if self.view == "shift":
            self.draw_trajectory_lines(arr, t)
        img = Image.fromarray(arr)
        d = ImageDraw.Draw(img)
        self.draw_video_overlay(d, t)
        self.draw_script(d, t)
        self.draw_status(d, t)
        self.draw_heard(d, t)
        if self.view == "shift":
            self.draw_bars(d, t)
            self.draw_signal_rows(d, t)
        elif self.view == "compare":
            self.draw_compare(d, t)
        elif self.view == "multimodal":
            self.draw_multimodal(d, t)
        else:
            self.draw_stacks(d, t)
        self.draw_lanes(d, t)
        return img

    def draw_video(self, arr, video_rgb):
        frame = video_rgb.astype(np.float32)
        half = VIDEO_W // 2
        other = slice(0, half) if self.tracked_side == "right" else slice(half, VIDEO_W)
        frame[:, other] *= 0.5
        arr[VIDEO_Y:VIDEO_Y + VIDEO_H, VIDEO_X:VIDEO_X + VIDEO_W] = frame.astype(np.uint8)

    def face_box(self, t):
        """The face the visual model saw in the current 5 Hz slot, in source pixels."""
        for u in self.utts:
            for timestamp, *box in u.get("face_boxes", []):
                if box and timestamp <= t < timestamp + 0.2:
                    return box
        return None

    def draw_video_overlay(self, d, t):
        half = VIDEO_W // 2
        x0 = VIDEO_X + (half if self.tracked_side == "right" else 0)
        d.rectangle([x0 + 1, VIDEO_Y + 1, x0 + half - 2, VIDEO_Y + VIDEO_H - 2], outline=INK, width=3)
        label = f"{self.tracked} · TRACKED"
        w = d.textlength(label, font=FONTS["tag"])
        d.rounded_rectangle([x0 + 14, VIDEO_Y + 14, x0 + 34 + w, VIDEO_Y + 44], radius=6, fill=INK)
        d.text((x0 + 24, VIDEO_Y + 20), label, font=FONTS["tag"], fill=BG)
        box = self.face_box(t)
        if box is None:
            return
        crop_w, crop_h, crop_x, crop_y = map(int, SOURCE_CROP.split(":"))
        bx0, bx1 = (VIDEO_X + (box[i] - crop_x) * VIDEO_W / crop_w for i in (0, 2))
        by0, by1 = (VIDEO_Y + (box[i] - crop_y) * VIDEO_H / crop_h for i in (1, 3))
        d.rectangle([bx0, by0, bx1, by1], outline=INK, width=2)
        w = d.textlength("FACE", font=FONTS["block"])
        d.rounded_rectangle([bx0, by0 - 21, bx0 + w + 12, by0 - 2], radius=4, fill=INK)
        d.text((bx0 + 6, by0 - 18), "FACE", font=FONTS["block"], fill=BG)

    def draw_script(self, d, t):
        """The scene's dialogue as a running script; the current words are lit."""
        width = VIDEO_W - 48
        blocks = []
        for row in self.rows:
            if row["dialog_start_sec"] > t:
                break
            lines = wrap(d, row["text"], FONTS["subs"], width)
            blocks.append((row, lines, len(lines) * 34 + 16))
        while blocks and sum(b[2] for b in blocks) > SCRIPT_Y1 - SCRIPT_Y0:
            blocks.pop(0)
        y = SCRIPT_Y0
        for row, lines, height in blocks:
            tracked = row["speaker"] == self.tracked
            active = t < row["dialog_end_sec"] + 0.4
            d.rounded_rectangle([40, y + 3, 72, y + 31], radius=5, fill=INK if tracked else INK4)
            d.text((49, y + 7), row["speaker"], font=FONTS["tag"], fill=BG if tracked else INK)
            aligned = row.get("words") or []
            begin = aligned[0]["start_sec"] if aligned else 0.0
            finish = aligned[-1]["end_sec"] if aligned else row["duration_sec"]
            progress = (t - row["dialog_start_sec"] - begin) / max(finish - begin, 0.1)
            spoken = int(np.ceil(min(max(progress, 0), 1) * sum(len(line) for line in lines)))
            lit = INK if tracked else INK2
            index = 0
            for li, line in enumerate(lines):
                x = 88
                for word in line:
                    fill = (lit if index < spoken else INK4) if active else INK3
                    d.text((x, y + 2 + li * 34), word, font=FONTS["subs"], fill=fill)
                    x += d.textlength(word + " ", font=FONTS["subs"])
                    index += 1
            y += height

    def draw_status(self, d, t):
        state, u = self.status(t)
        if state == "listening":
            pulse = 0.55 + 0.45 * (0.5 + 0.5 * np.sin(t * 2 * np.pi * 1.2))
            text, color = "LISTENING · WATCHING FACE" if self.face_box(t) else "LISTENING", INK
            dot = mix(INK, CARD, pulse)
        elif state == "processing":
            text, color, dot = f"ASR + EMOTION  {(u['asr_ms'] + u['step_ms']) / 1000:.2f} s", INK2, INK3
        else:
            text, color, dot = "WAITING FOR SPEECH", INK3, INK4
        w = d.textlength(text, font=FONTS["eyebrow"])
        x = PANE_R - 20 - w
        d.text((x, 118), text, font=FONTS["eyebrow"], fill=color)
        d.ellipse([x - 20, 120, x - 8, 132], fill=dot)

    def draw_heard(self, d, t):
        done = self.landed(t)
        if not done:
            d.text((PANE_X + 20, 244), "—", font=FONTS["body"], fill=INK4)
            return
        u = done[-1]
        fresh = ease_out((t - u["land_sec"]) / EASE_SEC)
        timing = (f"ASR {u['asr_ms']:.0f} ms · A then B {u['step_ms']:.0f} ms" if self.view == "compare"
                  else f"ASR {u['asr_ms']:.0f} ms · emotion {u['step_ms']:.0f} ms")
        text_right(d, (PANE_R - 20, 215), timing, FONTS["small_mono"], INK3)
        lines = wrap(d, f"“{u['asr_text']}”", FONTS["body"], PANE_R - PANE_X - 40)
        if len(lines) > 2:
            lines = lines[:2]
            lines[-1] = lines[-1] + ["…"]
        for i, line in enumerate(lines):
            d.text((PANE_X + 20, 244 + i * 32), " ".join(line), font=FONTS["body"], fill=mix(INK, CARD, fresh))

    def draw_bars(self, d, t):
        probs, u = self.displayed_probabilities(t)
        top = int(np.argmax(probs)) if probs is not None else None
        bar_x0, bar_x1 = PANE_X + 170, PANE_R - 96
        for i in range(4):
            y = 388 + i * 36
            is_top = i == top
            d.text((PANE_X + 20, y - 2), LABELS[i].capitalize(),
                   font=FONTS["label_bold"] if is_top else FONTS["label"], fill=INK if is_top else INK2)
            d.rounded_rectangle([bar_x0, y + 4, bar_x1, y + 20], radius=4, fill=TRACK)
            if probs is None:
                text_right(d, (PANE_R - 20, y), "—", FONTS["value"], INK4)
                continue
            p = float(probs[i])
            x = bar_x0 + p * (bar_x1 - bar_x0)
            if x - bar_x0 >= 2:
                d.rounded_rectangle([bar_x0, y + 4, x, y + 20], radius=4,
                                    fill=SERIES[i] if is_top else mix(SERIES[i], CARD, 0.55))
            text_right(d, (PANE_R - 20, y), f"{p:.2f}", FONTS["value"], INK if is_top else INK3)
        if u is None:
            d.text((PANE_X + 20, 540), "Prediction appears when the tracked actor finishes an utterance",
                   font=FONTS["small"], fill=INK3)
            return
        self.draw_verdict(d, u, 540, "Prediction")

    def draw_verdict(self, d, u, y, lead):
        pred = u["prediction"]
        d.text((PANE_X + 20, y), lead, font=FONTS["small"], fill=INK3)
        x = PANE_X + 20 + d.textlength(lead + "  ", font=FONTS["small"])
        d.text((x, y - 3), LABELS[pred].capitalize(), font=FONTS["label_bold"], fill=INK)
        x += d.textlength(LABELS[pred].capitalize() + "      ", font=FONTS["label_bold"])
        d.text((x, y), "Annotators", font=FONTS["small"], fill=INK3)
        x += d.textlength("Annotators  ", font=FONTS["small"])
        name = annotator_name(u)
        d.text((x, y - 3), name, font=FONTS["label"], fill=INK)
        x += d.textlength(name + "  ", font=FONTS["label"])
        if u["label"] is None:
            d.text((x, y), "\u00b7  unscored", font=FONTS["small"], fill=INK3)
        else:
            draw_check(d, x, y + 1, u["label"] == pred, INK)

    def draw_signal_rows(self, d, t):
        """One stacked bar per signal for the latest utterance; returns its vectors."""
        done = self.landed(t)
        x0, x1 = PANE_X + 20, PANE_R - 20
        u = done[-1] if done else None
        k = ease_out((t - u["land_sec"]) / EASE_SEC) if u else 0
        vectors = {}
        for (key, _, _), y in zip(self.signal_rows, self.row_y):
            bar_y = y + 28
            d.rounded_rectangle([x0, bar_y, x1, bar_y + 18], radius=4, fill=TRACK)
            current = vectors[key] = modality_probabilities(u, key) if u else None
            if current is None:
                text_right(d, (x1, y), "no face found" if u else "—", FONTS["small"] if u else FONTS["value"], INK3)
                continue
            prior = modality_probabilities(done[-2], key) if len(done) > 1 else None
            previous = np.zeros(4) if prior is None else prior
            p = previous + (current - previous) * k
            top = int(np.argmax(current))
            # Agreement with the annotators, per row; unscored utterances get no mark.
            if u["label"] is not None:
                draw_check(d, x1 - 17, y + 3, u["label"] == top, INK)
            value = f"{p[top]:.2f}"
            text_right(d, (x1 - 30, y), value, FONTS["value"], INK)
            text_right(d, (x1 - 40 - d.textlength(value, font=FONTS["value"]), y - 1),
                       LABELS[top].capitalize(), FONTS["label_bold"], INK)
            # 100% stacked bar in fixed class order, 2px surface gaps between segments.
            shown = [c for c in range(4) if p[c] * (x1 - x0) >= 1]
            width = (x1 - x0) - 2 * max(len(shown) - 1, 0)
            x = x0
            for c in shown:
                w = p[c] * width
                fill = SERIES[c] if c == top else mix(SERIES[c], CARD, 0.45)
                if w >= 8:
                    d.rounded_rectangle([x, bar_y, x + w, bar_y + 18], radius=4, fill=fill)
                else:
                    d.rectangle([x, bar_y, x + w, bar_y + 18], fill=fill)
                x += w + 2
        return u, k, vectors

    def draw_compare(self, d, t):
        """A and B on the latest utterance, each with its measured emotion-step time."""
        u, k, vectors = self.draw_signal_rows(d, t)
        x0 = PANE_X + 20
        if u is None:
            d.text((x0, 660), "Predictions appear when the tracked actor finishes an utterance",
                   font=FONTS["small"], fill=INK3)
            return
        context = u.get("context_model") or {}
        for (_, ms), y in zip((("fused", context.get("a_emotion_ms")), ("context", context.get("emotion_ms"))),
                                self.row_y):
            if ms is not None:
                d.text((x0, y + 52), f"emotion step {ms:.0f} ms", font=FONTS["small_mono"], fill=INK3)
        self.draw_previous_line(d, context, k)
        if vectors.get("fused") is not None and vectors.get("context") is not None:
            readout = compare_readout(u, int(np.argmax(vectors["fused"])), int(np.argmax(vectors["context"])))
            d.text((x0, 654), readout, font=FONTS["body_medium"], fill=mix(INK, CARD, k))
        d.text((x0, 696), "Annotators", font=FONTS["small"], fill=INK3)
        x = x0 + d.textlength("Annotators  ", font=FONTS["small"])
        d.text((x, 693), annotator_name(u), font=FONTS["label"], fill=INK)
        if u["label"] is None:
            x += d.textlength(annotator_name(u) + "  ", font=FONTS["label"])
            d.text((x, 696), "·  unscored", font=FONTS["small"], fill=INK3)

    def draw_previous_line(self, d, context, k):
        """The speaker's previous transcript, which B reads before the current line."""
        x0 = PANE_X + 20
        if not context.get("previous_text"):
            return
        lead = "B also reads the previous line: "
        d.text((x0, 316), lead, font=FONTS["small"], fill=INK3)
        room = PANE_R - 20 - x0 - d.textlength(lead, font=FONTS["small"])
        body = context["previous_text"]
        quote = f"“{body}”"
        while d.textlength(quote, font=FONTS["small"]) > room and body:
            body = body[:-1]
            quote = f"“{body.rstrip()}…”"
        d.text((x0 + d.textlength(lead, font=FONTS["small"]), 316), quote, font=FONTS["small"],
               fill=mix(INK2, CARD, k))

    def draw_multimodal(self, d, t):
        """Each signal alone, then B, with the live demo's readout and the annotators' label."""
        u, k, vectors = self.draw_signal_rows(d, t)
        x0 = PANE_X + 20
        if u is None:
            d.text((x0, 690), "Predictions appear when the tracked actor finishes an utterance",
                   font=FONTS["small"], fill=INK3)
            return
        self.draw_previous_line(d, u.get("context_model") or {}, k)
        if vectors.get("context") is not None:
            for i, line in enumerate(wrap(d, multimodal_readout(vectors), FONTS["body_medium"], PANE_R - PANE_X - 40)[:2]):
                d.text((x0, 662 + i * 28), " ".join(line), font=FONTS["body_medium"], fill=mix(INK, CARD, k))
        d.text((x0, 728), "Annotators", font=FONTS["small"], fill=INK3)
        x = x0 + d.textlength("Annotators  ", font=FONTS["small"])
        d.text((x, 725), annotator_name(u), font=FONTS["label"], fill=INK)
        if u["label"] is None:
            x += d.textlength(annotator_name(u) + "  ", font=FONTS["label"])
            d.text((x, 728), "·  unscored", font=FONTS["small"], fill=INK3)

    def draw_stacks(self, d, t):
        """Each signal alone, the deployed fusion and the face mixture, with a readout."""
        u, k, vectors = self.draw_signal_rows(d, t)
        x0 = PANE_X + 20
        if u is None:
            d.text((x0, 690), "Predictions appear when the tracked actor finishes an utterance",
                   font=FONTS["small"], fill=INK3)
            return
        d.text((x0, 684), modality_readout(vectors), font=FONTS["body_medium"], fill=mix(INK, CARD, k))
        d.text((x0, 724), "Annotators", font=FONTS["small"], fill=INK3)
        x = x0 + d.textlength("Annotators  ", font=FONTS["small"])
        d.text((x, 721), annotator_name(u), font=FONTS["label"], fill=INK)
        if u["label"] is None:
            x += d.textlength(annotator_name(u) + "  ", font=FONTS["label"])
            d.text((x, 724), "·  unscored", font=FONTS["small"], fill=INK3)

    def draw_lanes(self, d, t):
        """One block per utterance and signal; annotator labels are revealed with speech."""
        h = self.lane_h
        for u in self.utts:
            if u["start_sec"] > t:
                continue
            x0 = max(self.tx(u["start_sec"]), self.plot_x0)
            x1 = self.tx(min(u["end_sec"], t))
            if x1 - x0 < 2:
                continue
            landed = u["land_sec"] is not None and u["land_sec"] <= t
            for key, _ in self.lanes[:-1]:
                y = self.lane_y[key]
                if not landed:
                    d.rounded_rectangle([x0, y, x1 - 2, y + h], radius=4, outline=INK3, width=1)
                    continue
                end = self.tx(u["end_sec"]) - 2
                values = modality_probabilities(u, key)
                if values is None:
                    d.rounded_rectangle([x0, y, end, y + h], radius=4, fill=TRACK)
                    continue
                c = int(np.argmax(values))
                d.rounded_rectangle([x0, y, end, y + h], radius=4, fill=SERIES[c])
                if fits(d, SHORT[c], x0, end):
                    d.text((x0 + 5, y + (h - 14) // 2), SHORT[c], font=FONTS["block"], fill=BG)
                # The shift view flags deployed misses above its first lane.
                if self.view == "shift" and key == "fused" and u["label"] is not None and u["label"] != c:
                    draw_check(d, end - 20, y - 20, False, INK)
            self.draw_label_block(d, u, x0, x1 - 2, self.lane_y["label"])

    def draw_label_block(self, d, u, x0, end, y):
        """Annotator label, revealed alongside speech rather than ahead of it."""
        h = self.lane_h
        inset = (h - 14) // 2
        if u["label"] is None:
            d.rounded_rectangle([x0, y, end, y + h], radius=4, fill=TRACK)
            if fits(d, u["label_code"].upper(), x0, end):
                d.text((x0 + 5, y + inset), u["label_code"].upper(), font=FONTS["block"], fill=INK3)
        else:
            d.rounded_rectangle([x0, y, end, y + h], radius=4, fill=mix(SERIES[u["label"]], CARD, 0.75))
            if fits(d, SHORT[u["label"]], x0, end):
                d.text((x0 + 5, y + inset), SHORT[u["label"]], font=FONTS["block"], fill=BG)

    def draw_trajectory_lines(self, arr, t):
        done = self.landed(t)
        if not done:
            return
        span_y = self.plot_y1 - self.plot_y0
        for c in range(4):
            points, previous = [], None
            for u in done:
                p = u["probabilities"][c]
                if previous is None:
                    points.append((self.tx(u["land_sec"]), p))
                else:
                    points.append((self.tx(u["land_sec"]), previous))
                    points.append((self.tx(min(u["land_sec"] + EASE_SEC, t)),
                                   previous + (p - previous) * ease_out((t - u["land_sec"]) / EASE_SEC)))
                previous = p
            points.append((self.tx(t), points[-1][1]))
            pts = np.array([[x, self.plot_y1 - v * span_y] for x, v in points], np.float64)
            cv2.polylines(arr, [np.round(pts * 4).astype(np.int32)], False, SERIES[c], 3, cv2.LINE_AA, shift=2)
        lanes_end = self.lane_y["label"] + self.lane_h
        cv2.line(arr, (int(self.tx(t)), self.plot_y0 - 6), (int(self.tx(t)), lanes_end), INK3, 1, cv2.LINE_AA)


def video_frames(spec, video_path, fps):
    command = [
        "ffmpeg", "-v", "error", "-ss", f"{spec['start']:.3f}", "-i", str(video_path),
        "-t", f"{spec['end'] - spec['start']:.3f}",
        "-vf", f"fps={fps},crop={SOURCE_CROP},scale={VIDEO_W}:{VIDEO_H}:flags=lanczos",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    size = VIDEO_W * VIDEO_H * 3
    try:
        while chunk := process.stdout.read(size):
            if len(chunk) < size:
                break
            yield np.frombuffer(chunk, np.uint8).reshape(VIDEO_H, VIDEO_W, 3)
    finally:
        process.kill()
        process.wait()


def render(name, spec, rows, inference, out_path, fps, preview_at=None):
    renderer = Renderer(name, spec, rows, inference, fps)
    video_path = rows[0]["video_path"]
    if preview_at is not None:
        frame_spec = dict(spec, start=spec["start"] + preview_at, end=spec["start"] + preview_at + 1)
        video = next(video_frames(frame_spec, video_path, fps))
        renderer.frame(spec["start"] + preview_at, video).save(out_path)
        return
    duration = spec["end"] - spec["start"]
    encoder = subprocess.Popen([
        "ffmpeg", "-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
        "-ss", f"{spec['start']:.3f}", "-t", f"{duration:.3f}", "-i", str(video_path),
        "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-shortest",
        "-movflags", "+faststart", str(out_path),
    ], stdin=subprocess.PIPE)
    for i, video in enumerate(video_frames(spec, video_path, fps)):
        encoder.stdin.write(renderer.frame(spec["start"] + i / fps, video).tobytes())
        if (i + 1) % (fps * 10) == 0:
            print(f"  {name}: {(i + 1) / fps:.0f}/{duration:.0f} s", flush=True)
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", choices=("v1", "v2"), default="v1",
                        help="v2: the live demo's view (each signal alone, then B), written to demo/demov2/")
    parser.add_argument("--clip", choices=[*CLIPS, *CLIPS_V2, "all"], default="all")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, help="Default: data/processed/demo (v1) or its demov2/ folder (v2)")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")  # mps adds one-off compiles to latencies
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--rerun-inference", action="store_true")
    parser.add_argument("--preview-at", type=float, help="Write one PNG frame at this clip offset (s)")
    parser.add_argument("--compare", action="store_true",
                        help="Render A (deployed) beside B (previous-line context model) as compare_<clip>.mp4")
    args = parser.parse_args()
    v2 = args.set == "v2"
    if (args.compare or v2) and args.device != "cpu":
        parser.error("--compare and --set v2 time A and B as deployed on CPU; pass --device cpu")
    if args.compare and v2:
        parser.error("--compare renders the v1 A/B scenes")
    clips = CLIPS_V2 if v2 else CLIPS
    if args.clip != "all" and args.clip not in clips:
        parser.error(f"{args.clip} is not a {args.set} clip")
    args.output_dir = args.output_dir or (OUTPUT_DIR / "demov2" if v2 else OUTPUT_DIR)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    names = [n for n in clips if args.compare or not clips[n].get("compare_only")]
    for name in names if args.clip == "all" else [args.clip]:
        if clips[name].get("compare_only") and not args.compare:
            parser.error(f"{name} is an A/B scene; render it with --compare")
        spec = dict(clips[name], view="compare") if args.compare else clips[name]
        rows = clip_rows(spec, args.metadata)
        inference_path = args.output_dir / f"{name}.inference.json"
        tracked = [r for r in rows if r["speaker"] == spec["speaker"] and r["dialog_end_sec"] <= spec["end"]]
        if inference_path.exists() and not args.rerun_inference:
            inference = json.loads(inference_path.read_text())
        else:
            print(f"{name}: inference on {len(tracked)} utterances", flush=True)
            inference = {"clip": name, **spec, **run_inference(tracked, args.device, live_asr=v2)}
            inference_path.write_text(json.dumps(inference, indent=2))
        if any("face" not in u for u in inference["utterances"]):
            print(f"{name}: face replay on {len(tracked)} utterances", flush=True)
            inference = run_face_inference(tracked, inference, args.device)
            inference_path.write_text(json.dumps(inference, indent=2))
        if (args.compare or v2) and any("context_model" not in u or (u["context_model"] and "both_emotion_ms" not in u["context_model"])
                                        for u in inference["utterances"]):
            print(f"{name}: context model (B) on {len(tracked)} utterances", flush=True)
            inference = run_context_inference(tracked, inference, args.device, args.metadata, live_asr=v2)
            inference_path.write_text(json.dumps(inference, indent=2))
        for u in inference["utterances"]:
            parity = u.get("parity")
            face_parity = u.get("face_parity") or {}
            scored = "unscored" if u["label"] is None else ("correct" if u["label"] == u["prediction"] else "WRONG")
            print(f"  {u['utterance_id']} {u['label_code']} -> {SHORT[u['prediction']] if u['prediction'] is not None else '-'}"
                  f" {scored:8} {u['asr_ms']:.0f}+{u['emotion_ms']:.0f} ms"
                  + (f"  parity text={parity['asr_text_matches']} max|dp|={parity['max_abs_probability_diff']:.1e}" if parity else "")
                  + (f"  text-only {SHORT[int(np.argmax(u['controls']['text']))]} voice-only {SHORT[int(np.argmax(u['controls']['audio']))]}"
                     if u.get("controls") else "")
                  + (f" face-only {SHORT[int(np.argmax(u['face']))] if u['face'] else '-'}"
                     f" with-face {SHORT[int(np.argmax(u['with_face']))] if u['with_face'] else '-'}"
                     f" {u['face_frame_ms'] or 0:.0f} ms/frame" if "face" in u else "")
                  + "".join(f"  {k}={v:.1e}" for k, v in face_parity.items())
                  + (f"  A {u['context_model']['a_emotion_ms']:.0f} ms (|dp| {u['context_model']['a_max_abs_diff']:.0e})  B {SHORT[u['context_model']['prediction']]} {u['context_model']['emotion_ms']:.0f} ms"
                     if u.get("context_model") else ""))
        prefix = "compare_" if args.compare else ""
        if args.preview_at is not None:
            out = args.output_dir / f"{prefix}{name}_preview_{args.preview_at:g}s.png"
            render(name, spec, rows, inference, out, args.fps, args.preview_at)
        else:
            out = args.output_dir / f"{prefix}{name}.mp4"
            render(name, spec, rows, inference, out, args.fps)
        print(f"Wrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}", flush=True)


if __name__ == "__main__":
    main()
