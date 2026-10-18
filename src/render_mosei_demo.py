"""Render a CMU-MOSEI demo: single-speaker validation clips with text-only vs audio+text sentiment.

Each clip plays, then three diverging bars appear on a -3..+3 scale: the annotators' score,
a text-only model (frozen DistilBERT features, ridge regression on MOSEI train) and the
legacy audio+text model (models/mosei/mosei_cross_modal.pt, DistilBERT + DistilHuBERT,
replayed validation predictions). Validation split only; the test split is never touched.

    uv run python src/render_mosei_demo.py
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw

from src.render_inference_demo import (
    BG, CARD, FONTS, H, INK, INK2, INK3, INK4, PANE_R, PANE_X, RULE, TRACK, W, ease_out, text_right, wrap,
)

PREDICTIONS = ROOT / "data/processed/improve/mosei_modalities/demo_predictions.json"
RAW = ROOT / "data/MOSEI/Raw"
OUTPUT = ROOT / "data/processed/demo/mosei_voice_intensity.mp4"
FPS, HOLD_SEC, REVEAL_DELAY = 30, 3.8, 0.35
# Diverging pair for sentiment (not the emotion palette): negative red, positive blue, gray zero.
NEGATIVE, POSITIVE = (0xD6, 0x3C, 0x45), (0x39, 0x87, 0xE5)
VIDEO_X, VIDEO_Y, VIDEO_W, VIDEO_H = 40, 104, 1080, 810
CLIPS = [
    ("266791/18", "Clear words: the text model and the audio+text model both read it."),
    ("80855/1", "The words alone read nearly neutral; the voice carries the sentiment."),
    ("31474/12", "The words alone point the wrong way; adding the voice corrects it."),
    ("33170/3", "The words read nearly neutral; the voice adds the strength."),
    ("241629/3", "The words read nearly neutral; the voice adds the strength."),
    ("rojyUp4FPSg/11", "Counterexample: a calm, flat delivery misleads the audio+text model."),
]
ROWS = [("label", "Annotators", "majority of human raters"), ("text_only", "Text only", "the words"),
        ("audio_text", "Audio + text", "the words and the voice")]


def frames(path):
    command = ["ffmpeg", "-v", "error", "-i", str(path), "-vf",
               f"fps={FPS},scale={VIDEO_W}:{VIDEO_H}:force_original_aspect_ratio=decrease:flags=lanczos,"
               f"pad={VIDEO_W}:{VIDEO_H}:(ow-iw)/2:(oh-ih)/2:color=black",  # letterbox 16:9 sources, never stretch
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    size = VIDEO_W * VIDEO_H * 3
    try:
        while (chunk := process.stdout.read(size)) and len(chunk) == size:
            yield np.frombuffer(chunk, np.uint8).reshape(VIDEO_H, VIDEO_W, 3)
    finally:
        process.kill()
        process.wait()


class Page:
    def __init__(self, index, clip_id, note, values):
        self.index, self.clip_id, self.note, self.values = index, clip_id, note, values
        self.scale_x0, self.scale_x1 = PANE_X + 40, PANE_R - 40
        self.base = self.static()

    def tx(self, v):
        return self.scale_x0 + (v + 3) / 6 * (self.scale_x1 - self.scale_x0)

    def static(self):
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        d.text((40, 34), "Words give the direction, the voice adds the strength", font=FONTS["title"], fill=INK)
        text_right(d, (PANE_R, 45), f"CMU-MOSEI validation · clip {self.index + 1} of {len(CLIPS)}", FONTS["meta"], INK3)
        d.line([(40, 86), (PANE_R, 86)], fill=RULE, width=1)
        d.rounded_rectangle([PANE_X, 104, PANE_R, 640], radius=10, fill=CARD)
        d.text((PANE_X + 20, 118), "SENTIMENT  ·  −3 NEGATIVE TO +3 POSITIVE", font=FONTS["eyebrow"], fill=INK3)
        for i, (_, title, caption) in enumerate(ROWS):
            y = 172 + i * 118
            d.text((PANE_X + 20, y), title, font=FONTS["label_bold"], fill=INK)
            d.text((PANE_X + 32 + d.textlength(title, font=FONTS["label_bold"]), y + 4), caption, font=FONTS["small"], fill=INK3)
            d.rounded_rectangle([self.scale_x0, y + 40, self.scale_x1, y + 64], radius=4, fill=TRACK)
            d.line([(self.tx(0), y + 34), (self.tx(0), y + 70)], fill=INK3, width=2)
        for v in range(-3, 4):
            x = self.tx(v)
            d.line([(x, 520), (x, 528)], fill=INK4, width=1)
            label = f"{v:+d}" if v else "0"
            d.text((x - d.textlength(label, font=FONTS["small_mono"]) / 2, 534), label, font=FONTS["small_mono"], fill=INK3)
        d.text((self.scale_x0, 554), "Negative", font=FONTS["small"], fill=INK2)
        text_right(d, (self.scale_x1, 554), "Positive", FONTS["small"], INK2)
        d.rounded_rectangle([PANE_X, 656, PANE_R, 1004], radius=10, fill=CARD)
        d.text((PANE_X + 20, 670), "ALL 1,871 VALIDATION CLIPS", font=FONTS["eyebrow"], fill=INK3)
        lines = [("Direction right (non-zero clips)", "0.837", "0.845"), ("Mean absolute error", "0.594", "0.548"),
                 ("Average strength on strong clips (true 2.27)", "0.96", "1.23")]
        d.text((PANE_X + 430, 700), "Text", font=FONTS["small"], fill=INK3)
        d.text((PANE_X + 540, 700), "Audio + text", font=FONTS["small"], fill=INK3)
        for i, (name, a, b) in enumerate(lines):
            y = 730 + i * 36
            d.text((PANE_X + 20, y), name, font=FONTS["small"], fill=INK2)
            d.text((PANE_X + 430, y), a, font=FONTS["value"], fill=INK)
            d.text((PANE_X + 540, y), b, font=FONTS["value"], fill=INK)
        for i, line in enumerate(wrap(d, "The gain is mostly strength, not direction, and it needs learned fusion: "
                                         "the same frozen features in a linear model gain nothing from audio.",
                                      FONTS["small"], PANE_R - PANE_X - 40)):
            d.text((PANE_X + 20, 852 + i * 24), " ".join(line), font=FONTS["small"], fill=INK3)
        for i, line in enumerate(wrap(d, "Clips chosen to illustrate, including one where the voice misleads. Text only: "
                                         "frozen DistilBERT + ridge. Audio + text: models/mosei/mosei_cross_modal.pt.",
                                      FONTS["small"], PANE_R - PANE_X - 40)):
            d.text((PANE_X + 20, 930 + i * 24), " ".join(line), font=FONTS["small"], fill=INK3)
        transcript = wrap(d, f"“{self.values['text']}”", FONTS["subs"], VIDEO_W)
        for i, line in enumerate(transcript[:3]):
            d.text((VIDEO_X, 928 + i * 34), " ".join(line), font=FONTS["subs"], fill=INK2)
        d.text((40, 1046), f"MOSEI clip {self.clip_id} · single speaker, YouTube review/opinion · shown locally only",
               font=FONTS["footer"], fill=INK3)
        return np.asarray(img).copy()

    def frame(self, t, video, duration):
        arr = self.base.copy()
        arr[VIDEO_Y:VIDEO_Y + VIDEO_H, VIDEO_X:VIDEO_X + VIDEO_W] = video
        img = Image.fromarray(arr)
        d = ImageDraw.Draw(img)
        k = ease_out((t - duration - REVEAL_DELAY) / 0.6)
        for i, (key, _, _) in enumerate(ROWS):
            y = 172 + i * 118
            v = self.values[key]
            if k <= 0:
                text_right(d, (PANE_R - 20, y), "—", FONTS["value"], INK4)
                continue
            end = self.tx(v * k)
            color = NEGATIVE if v < 0 else POSITIVE
            x0, x1 = sorted((self.tx(0), end))
            if x1 - x0 >= 2:
                d.rounded_rectangle([x0, y + 40, x1, y + 64], radius=4, fill=color)
            text_right(d, (PANE_R - 20, y), f"{v * k:+.2f}", FONTS["value"], INK)
        if k > 0:
            for i, line in enumerate(wrap(d, self.note, FONTS["body_medium"], PANE_R - PANE_X - 40)[:2]):
                d.text((PANE_X + 20, 580 + i * 28), " ".join(line), font=FONTS["body_medium"], fill=INK if k >= 1 else INK2)
        else:
            d.text((PANE_X + 20, 580), "Listening…", font=FONTS["body_medium"], fill=INK3)
        return img


def render_clip(index, clip_id, note, values, folder):
    path = RAW / f"{clip_id}.mp4"
    duration = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                     str(path)], capture_output=True, text=True, check=True).stdout)
    page = Page(index, clip_id, note, values)
    out = Path(folder) / f"segment{index:02d}.mp4"
    total = duration + HOLD_SEC
    encoder = subprocess.Popen([
        "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-i", str(path), "-map", "0:v", "-map", "1:a", "-af", f"apad=whole_dur={total:.3f},aresample=48000",
        "-ac", "2", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k", "-t", f"{total:.3f}", str(out)], stdin=subprocess.PIPE)
    last = None
    count = int(round(total * FPS))
    video = frames(path)
    for i in range(count):
        t = i / FPS
        if t < duration:
            last = next(video, last)
        held = last if t < duration else (last * 0.55).astype(np.uint8)  # dim the frozen frame while results show
        encoder.stdin.write(page.frame(t, held, duration).tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {clip_id}")
    return out


def main():
    predictions = json.loads(PREDICTIONS.read_text())
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as folder:
        segments = [render_clip(i, clip, note, predictions[clip], folder) for i, (clip, note) in enumerate(CLIPS)]
        listing = Path(folder) / "segments.txt"
        listing.write_text("".join(f"file '{s}'\n" for s in segments))
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
                        "-c", "copy", "-movflags", "+faststart", str(OUTPUT)], check=True)
    print(f"Wrote {OUTPUT.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
