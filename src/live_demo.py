"""Live HRI session tool: a scripted robot speaks, the participant answers naturally, and
two emotion models score each answer on a localhost page.

A is the deployed audio_text_baseline.pt (current line only). B is context_baseline.pt
(the speaker's previous transcript [SEP] current transcript). Whisper runs once per
utterance and both models score it; the face-only model runs in shadow (weight 0, as
deployed). Every context-smoothing preset is computed so the page switches instantly.

Scenarios (hri_scenarios.json) give the participant a short setting. The robot's scripted
lines are spoken with macOS text-to-speech, and speech overlapping the robot is ignored so
the models never score the robot. Once a participant ID is entered and consent confirmed,
each trial records the robot's lines, the participant's utterance audio (16 kHz WAV),
transcripts, model outputs and a post-trial self-report under data/processed/hri_sessions/
(local, gitignored). Without a session nothing is written.

    uv run python src/live_demo.py --offline
    uv run python src/live_demo.py --offline --replay-dialog Ses04F_impro05 --speaker F --silent-robot

macOS asks the terminal for microphone and camera access on first use.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import soundfile as sf

from src.emotion_checkpoint import EMOTION_LABELS, prepare_audio, tokenize_text
from src.emotion_context import PRESETS, SpeakerContext
from src.inference_pipeline import FAST_ASR, SpeechProcessor

SCENARIOS = Path(__file__).with_name("hri_scenarios.json")
SESSIONS = ROOT / "data/processed/hri_sessions"
A_CHECKPOINT = ROOT / "models/mvp/audio_text_baseline.pt"
B_CHECKPOINT = ROOT / "models/mvp/context_baseline.pt"
VISUAL_CHECKPOINT = ROOT / "models/mvp/visual_day5/deployment.pt"
# Self-report options map onto the models' four classes, plus "other".
FEELINGS = {"anger": "Angry or frustrated", "happiness": "Happy or amused", "sadness": "Sad or disappointed",
            "neutral": "Neutral or calm", "other": "Something else"}


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class Robot:
    """Speaks scripted lines with macOS `say` and remembers when, so its echo is ignored."""

    def __init__(self, voice="Samantha", rate=170, silent=False):
        self.voice, self.rate, self.silent = voice, rate, silent
        self.lock = threading.Lock()
        self.windows = []  # (start, end) in monotonic seconds, the clock of live audio
        self.speaking_since = None
        self.line = None

    @property
    def speaking(self):
        return self.speaking_since is not None

    def speak(self, text, on_done=None):
        with self.lock:
            if self.speaking_since is not None:
                raise ValueError("The robot is still speaking")
            self.speaking_since, self.line = time.monotonic(), text

        def run():
            command = ["say", "-v", self.voice, "-r", str(self.rate)]
            if self.silent:  # synthesize without playback, then take as long as playback would
                with tempfile.TemporaryDirectory() as folder:
                    path = Path(folder) / "line.aiff"
                    subprocess.run([*command, "-o", str(path), text], check=True)
                    time.sleep(sf.info(str(path)).duration)
            else:
                subprocess.run([*command, text], check=True)
            with self.lock:
                started, ended = self.speaking_since, time.monotonic()
                self.windows.append((started, ended))
                self.speaking_since = None
            if on_done:
                on_done(started, ended)

        threading.Thread(target=run, daemon=True).start()

    def overlaps(self, start, end, tail=0.5):
        """True when a speech segment overlaps the robot's voice (plus room echo)."""
        with self.lock:
            if self.speaking_since is not None and end >= self.speaking_since:
                return True
            return any(start < stop + tail and end > begin for begin, stop in self.windows)


class StudyRecorder:
    """One participant's consented session: an event log and per-utterance WAV files."""

    def __init__(self, root, participant, metadata):
        safe = re.sub(r"[^A-Za-z0-9_-]", "", participant)[:40]
        if not safe:
            raise ValueError("Participant ID needs letters or digits")
        self.folder = Path(root) / f"{time.strftime('%Y%m%d-%H%M%S')}_{safe}"
        (self.folder / "audio").mkdir(parents=True)
        self.lock = threading.Lock()
        self.utterances = 0
        (self.folder / "session.json").write_text(json.dumps(
            {"participant": safe, "consent_confirmed_at": now_iso(), **metadata}, indent=2) + "\n")
        self.events = (self.folder / "events.jsonl").open("a")

    def log(self, event, **fields):
        with self.lock:
            self.events.write(json.dumps({"event": event, "wall_time": now_iso(),
                                          "monotonic": time.monotonic(), **fields}, allow_nan=False) + "\n")
            self.events.flush()

    def save_audio(self, samples, sample_rate, trial):
        with self.lock:
            self.utterances += 1
            name = f"audio/trial{trial:02d}_utt{self.utterances:03d}.wav"
        sf.write(self.folder / name, samples, sample_rate)
        return name

    def close(self):
        with self.lock:
            self.events.close()


class SharedEncoders:
    """A and B share the same frozen encoders: encode the audio once and each distinct text once,
    then run both fusion heads. Outputs equal each predictor's own predict()."""

    KEYS = ("text_model_name", "text_revision", "audio_model_name", "audio_revision")

    def __init__(self, model_a, model_b):
        config_a, config_b = model_a.model.model_config, model_b.model.model_config
        if any(config_a[k] != config_b[k] for k in self.KEYS) or model_a.preprocessing != model_b.preprocessing:
            raise ValueError("A and B must use identical encoders and preprocessing to share them")
        self.a, self.b = model_a, model_b
        # B's own encoder copies are never used again; dropping them frees roughly 300 MB.
        model_b.model.text_encoder = model_b.model.audio_encoder = None

    def predict(self, text_a, text_b, audio, sample_rate):
        """Returns {"A": (prediction, ms), "B": (prediction, ms)}; each ms includes the shared audio encoding."""
        import torch

        from src.emotion_fusion_infer import EmotionPrediction

        a = self.a
        with torch.inference_mode():
            started = time.perf_counter()
            waveform = prepare_audio(audio, sample_rate, a.preprocessing).unsqueeze(0).to(a.device)
            audio_embeddings = a.model.encode_audio(waveform)
            audio_ms = 1000 * (time.perf_counter() - started)
            texts, out = {}, {}
            for name, predictor, text in (("A", a, text_a), ("B", self.b, text_b)):
                started = time.perf_counter()
                if text not in texts:
                    tokens = tokenize_text(a.tokenizer, text, a.preprocessing)
                    mask = tokens["attention_mask"].to(a.device)
                    texts[text] = (a.model.encode_text(tokens["input_ids"].to(a.device), mask), mask)
                embeddings, mask = texts[text]
                probabilities = predictor.model.forward_from_embeddings(embeddings, audio_embeddings, mask) \
                    .logits.softmax(-1)[0].cpu()
                index = int(probabilities.argmax())
                prediction = EmotionPrediction(predictor.label_names[index], index,
                                               {n: float(probabilities[i]) for i, n in enumerate(predictor.label_names)})
                out[name] = (prediction, audio_ms + 1000 * (time.perf_counter() - started))
        return out


class ComparisonProcessor:
    """One Whisper pass, then A on the current line and B on previous + current line."""

    def __init__(self, transcriber, model_a, model_b, robot=None, audio_sink=None, shared=None, asr_options=None):
        self.a = SpeechProcessor(transcriber, None if shared else model_a, asr_options)
        self.a.preprocessing = model_a.preprocessing
        self.model_a, self.b, self.shared = model_a, model_b, shared
        self.robot = robot
        self.audio_sink = audio_sink  # called with each scored segment; returns a saved path or None
        self.lock = threading.Lock()
        self.reset()

    def reset(self, payload=None):
        """Forget the previous line and every smoothing state (new conversation)."""
        with self.lock:
            self.previous_text = None
            self.contexts = {(name, model): SpeakerContext.preset(name) for name in PRESETS for model in "AB"}

    def process(self, segment):
        if self.robot is not None and self.robot.overlaps(segment.start_time, segment.end_time):
            return {"start_time": segment.start_time, "end_time": segment.end_time, "text": "",
                    "status": "ignored_robot_speech", "emotion": None, "timings_ms": {}}
        result = self.a.process(segment)
        if self.audio_sink is not None:
            result["recording"] = self.audio_sink(segment)
        text = result["text"]
        if not text or (result["emotion"] is None and not self.shared):
            return result
        sample_rate = self.a.preprocessing.sample_rate
        audio = prepare_audio(segment.samples, segment.sample_rate, self.a.preprocessing).numpy()
        with self.lock:
            previous, self.previous_text = self.previous_text, text
        prompt = f"{previous} [SEP] {text}" if previous else text
        if self.shared:
            started = time.perf_counter()
            both = self.shared.predict(text, prompt, audio, sample_rate)
            result["timings_ms"]["emotion_both_models"] = 1000 * (time.perf_counter() - started)
            (a_prediction, a_ms), (prediction, b_ms) = both["A"], both["B"]
            result["emotion"] = asdict(a_prediction)
            result["timings_ms"]["emotion"] = a_ms
            result["model_provenance"] = {k: self.model_a.provenance[k] for k in ("kind", "epoch", "test_session")
                                          if k in self.model_a.provenance}
        else:
            started = time.perf_counter()
            prediction = self.b.predict(prompt, audio, sample_rate)
            b_ms = 1000 * (time.perf_counter() - started)
        raw = {"A": result["emotion"]["probabilities"], "B": prediction.probabilities}
        with self.lock:
            smoothed = {name: {m: self.contexts[(name, m)].update(raw[m]) for m in "AB"} for name in PRESETS}
        result["comparison"] = {
            "previous_text": previous,
            "A": {"probabilities": raw["A"], "emotion_ms": result["timings_ms"]["emotion"]},
            "B": {"probabilities": raw["B"], "emotion_ms": b_ms},
            "smoothed": smoothed,
        }
        result["timings_ms"]["context_model"] = b_ms
        return result


class Study:
    """Session, trial, robot-turn and self-report actions behind the page's controls."""

    def __init__(self, dashboard, processor, robot, scenarios, metadata, root=SESSIONS):
        self.dashboard, self.processor, self.robot = dashboard, processor, robot
        self.scenarios = {s["id"]: s for s in scenarios}
        self.metadata, self.root = metadata, root
        self.recorder = None
        self.lock = threading.RLock()
        self.state = {"session": None, "trial": None, "awaiting_report": None, "trials_done": 0}
        processor.audio_sink = self.save_audio
        self.publish()

    def actions(self):
        return {"/session/start": self.start_session, "/session/end": self.end_session,
                "/trial/start": self.start_trial, "/trial/robot": self.robot_next,
                "/trial/end": self.end_trial, "/trial/report": self.report,
                "/reset": self.processor.reset}

    def publish(self):
        with self.lock:
            self.dashboard.update(hri={**self.state, "robot": {"speaking": self.robot.speaking,
                                                                "line": self.robot.line},
                                       "feelings": FEELINGS})

    def log(self, event, **fields):
        if self.recorder:
            self.recorder.log(event, **fields)

    def start_session(self, payload):
        with self.lock:
            if self.recorder:
                raise ValueError("A session is already recording")
            if payload.get("consent") is not True:
                raise ValueError("Confirm the participant's consent before recording")
            self.recorder = StudyRecorder(self.root, str(payload.get("participant", "")), self.metadata)
            folder = self.recorder.folder
            self.state.update(session={"participant": folder.name.split("_", 1)[1],
                                       "folder": str(folder.relative_to(ROOT) if folder.is_relative_to(ROOT) else folder)},
                              trials_done=0, trial=None, awaiting_report=None)
            self.log("session_start")
        self.publish()
        return self.state["session"]

    def end_session(self, payload):
        with self.lock:
            if self.state["trial"]:
                self.end_trial({})
            if self.recorder:
                self.log("session_end", trials=self.state["trials_done"])
                self.recorder.close()
            self.recorder = None
            self.state.update(session=None, awaiting_report=None)
        self.publish()

    def start_trial(self, payload):
        scenario = self.scenarios[payload["scenario"]]
        with self.lock:
            if self.state["trial"]:
                self.end_trial({})
            self.processor.reset()
            index = self.state["trials_done"] + 1
            self.state.update(awaiting_report=None, trial={
                "index": index, "scenario": scenario["id"], "title": scenario["title"],
                "next_turn": 0, "turns": len(scenario["turns"]), "started_monotonic": time.monotonic()})
            self.log("trial_start", trial=index, scenario=scenario)
        self.publish()

    def robot_next(self, payload):
        with self.lock:
            trial = self.state["trial"]
            if not trial:
                raise ValueError("Start a trial first")
            if trial["next_turn"] >= trial["turns"]:
                raise ValueError("The robot has no more lines in this scenario")
            turn = trial["next_turn"]
            line = self.scenarios[trial["scenario"]]["turns"][turn]
            index = trial["index"]

            def done(started, ended):
                self.log("robot_turn", trial=index, turn=turn + 1, text=line,
                         start_monotonic=started, end_monotonic=ended, voice=self.robot.voice)
                self.publish()

            self.robot.speak(line, on_done=done)
            trial["next_turn"] = turn + 1
        self.publish()

    def end_trial(self, payload):
        with self.lock:
            trial = self.state["trial"]
            if not trial:
                raise ValueError("No trial is running")
            self.log("trial_end", trial=trial["index"], scenario=trial["scenario"],
                     robot_turns_spoken=trial["next_turn"])
            self.state.update(trial=None, trials_done=trial["index"],
                              awaiting_report={"index": trial["index"], "scenario": trial["scenario"],
                                               "title": trial["title"]} if self.recorder else None)
        self.publish()

    def report(self, payload):
        with self.lock:
            pending = self.state["awaiting_report"]
            if not pending:
                raise ValueError("No trial is waiting for a self-report")
            feeling, trust = payload.get("feeling"), payload.get("trust")
            if feeling not in FEELINGS or not isinstance(trust, int) or not 1 <= trust <= 7:
                raise ValueError("Choose a feeling and a trust rating from 1 to 7")
            self.log("self_report", trial=pending["index"], scenario=pending["scenario"], feeling=feeling,
                     trust=trust, note=str(payload.get("note", ""))[:500])
            self.state["awaiting_report"] = None
        self.publish()

    def save_audio(self, segment):
        with self.lock:
            trial = self.state["trial"]
            if not (self.recorder and trial):
                return None
            return self.recorder.save_audio(segment.samples, segment.sample_rate, trial["index"])

    def on_result(self, result):
        """Log every scored or ignored segment that falls inside a recorded trial."""
        with self.lock:
            trial = self.state["trial"]
            if not (self.recorder and trial) or result.get("event") != "prediction":
                return
            if result.get("status") == "ignored_robot_speech":
                self.log("ignored_robot_echo", trial=trial["index"], start_monotonic=result.get("start_time"),
                         end_monotonic=result.get("end_time"))
                return
            emotion = result.get("emotion") or {}
            self.log("utterance", trial=trial["index"], scenario=trial["scenario"],
                     start_monotonic=result.get("start_time"), end_monotonic=result.get("end_time"),
                     text=result.get("text"), status=result.get("status"), audio=result.get("recording"),
                     comparison=result.get("comparison"), face=emotion.get("visual_probabilities"),
                     timings_ms=result.get("timings_ms"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu",
                        help="Emotion models; CPU keeps per-utterance latency steady")
    parser.add_argument("--offline", action="store_true", help="Require locally cached model assets")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--no-camera", action="store_true", help="Audio only; no face model")
    parser.add_argument("--microphone-index", type=int)
    parser.add_argument("--threshold", type=float, default=0.4, help="VAD threshold")
    parser.add_argument("--robot-voice", default="Samantha", help="macOS `say` voice for the robot")
    parser.add_argument("--robot-rate", type=int, default=170, help="Robot speaking rate, words per minute")
    parser.add_argument("--silent-robot", action="store_true", help="Time robot lines without playing them")
    parser.add_argument("--deployed-asr", action="store_true",
                        help="Whisper settings of the Session 4 evaluation (slower; default uses FAST_ASR, 4 threads)")
    parser.add_argument("--replay-dialog", help="Stream a development IEMOCAP dialogue instead of devices")
    parser.add_argument("--speaker", choices=("F", "M"), default="F")
    parser.add_argument("--output", type=Path, help="Also write each raw result as JSONL")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    if not 0 < args.threshold < 1:
        parser.error("--threshold must be between 0 and 1")
    return args


def main(argv=None):
    args = parse_args(argv)
    import numpy as np
    import torch
    from faster_whisper import WhisperModel

    from src.emotion_fusion_infer import CrossModalEmotionPredictor
    from src.iemocap_embedding_cache import sha256
    from src.inspector_dashboard import InspectorDashboard
    from src.pipeline_demo import replay_source
    from src.stream_runtime import run_stream
    from src.visual_fusion import VisualFusionPredictor
    from src.visual_stream import VisualExtractor

    voices = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout
    if not re.search(rf"^{re.escape(args.robot_voice)}\s", voices, re.MULTILINE):
        raise SystemExit(f"Voice {args.robot_voice!r} is not installed; see `say -v '?'`")
    torch.set_num_threads(2)
    model_a = CrossModalEmotionPredictor(A_CHECKPOINT, args.device, args.offline)
    model_b = CrossModalEmotionPredictor(B_CHECKPOINT, args.device, args.offline)
    if model_b.provenance.get("text_context") != "previous_same_speaker_line":
        raise ValueError(f"{B_CHECKPOINT} is not a previous-line context checkpoint")
    transcriber = WhisperModel("tiny.en", device="cpu", compute_type="int8",
                               cpu_threads=2 if args.deployed_asr else 4, local_files_only=args.offline)
    robot = Robot(args.robot_voice, args.robot_rate, args.silent_robot)
    shared = SharedEncoders(model_a, model_b)
    processor = ComparisonProcessor(transcriber, model_a, model_b, robot, shared=shared,
                                    asr_options=None if args.deployed_asr else FAST_ASR)
    source = None
    if args.replay_dialog:
        source, _ = replay_source(SimpleNamespace(
            audio_file=None, replay_utterance=None, replay_dialog=args.replay_dialog,
            speaker=args.speaker, stream_replay=False))
    use_vision = not args.no_camera and (source is None or source.video_path is not None)
    visual_predictor = VisualFusionPredictor(VISUAL_CHECKPOINT, A_CHECKPOINT) if use_vision else None
    extractor = VisualExtractor(visual_predictor.payload["feature_config"], args.device) if use_vision else None
    scenarios = json.loads(SCENARIOS.read_text())
    dashboard = InspectorDashboard(args.port, page="live_demo.html")
    study = Study(dashboard, processor, robot, scenarios, metadata={
        "scenarios_file": SCENARIOS.name, "scenarios_sha256": sha256(SCENARIOS),
        "models": {name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
                   for name, path in (("A", A_CHECKPOINT), ("B", B_CHECKPOINT), ("face", VISUAL_CHECKPOINT))},
        "robot_voice": args.robot_voice, "robot_rate": args.robot_rate,
        "source": f"replay {args.replay_dialog} ({args.speaker})" if source else "microphone + camera",
        "audio": "16 kHz mono WAV per detected utterance during trials; robot-overlapping speech is not saved",
    })
    dashboard.actions.update(study.actions())
    dashboard.update(demo={
        "labels": EMOTION_LABELS,
        "presets": list(PRESETS),
        "models": {"A": A_CHECKPOINT.name, "B": B_CHECKPOINT.name},
        "face": VISUAL_CHECKPOINT.name if use_vision else None,
        "source": f"{args.replay_dialog} ({args.speaker})" if source else "microphone + camera",
        "scenarios": scenarios,
    })
    silence = np.zeros(16000 * 6, np.float32)
    for seconds in (1, 6):  # warm-up is not utterance latency
        shared.predict("Ready.", "Ready for the next utterance. [SEP] Ready.", silence[: 16000 * seconds], 16000)
    output = args.output.open("w") if args.output else None

    def emit(result):
        study.on_result(result)
        if output:
            output.write(json.dumps(result, allow_nan=False) + "\n")
            output.flush()
        comparison = result.get("comparison")
        if comparison:
            top = {m: max(comparison[m]["probabilities"], key=comparison[m]["probabilities"].get) for m in "AB"}
            print(f"A {top['A']:9}  B {top['B']:9}  “{result['text']}”", flush=True)

    print(f"HRI session tool: {dashboard.url}  (Ctrl+C or Stop to end)", file=sys.stderr, flush=True)
    if not args.no_browser:
        webbrowser.open(dashboard.url)
    try:
        asyncio.run(run_stream(
            processor, emit,
            visual_predictor=visual_predictor, extractor=extractor, replay=source,
            camera_index=args.camera_index if source is None and extractor else None,
            microphone_index=args.microphone_index, threshold=args.threshold, dashboard=dashboard,
        ))
        if source is not None:  # keep the finished replay on screen until Stop
            dashboard.hold(3600)
    except KeyboardInterrupt:
        pass
    finally:
        study.end_session({})
        if extractor:
            extractor.close()
        if output:
            output.close()
        dashboard.close()


if __name__ == "__main__":
    main()
