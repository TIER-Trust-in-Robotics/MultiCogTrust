"""Render the onboarding inference diagram as a vector PDF and a PNG preview.

From the repository root (does not change project dependencies):
uv run --no-project --python 3.11 --with reportlab --with pymupdf \
    python docs/assets/render_inference_diagram.py
"""

from pathlib import Path
import math

import pymupdf
from reportlab.lib.colors import HexColor
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output/pdf"
WIDTH, HEIGHT = 1200, 850
INK = "#19324A"
MUTED = "#53677D"
BLUE = "#3567A8"
PURPLE = "#7153A3"
TEAL = "#187B78"
PDF = OUTPUT / "multicogtrust_inference.pdf"
PNG = OUTPUT / "multicogtrust_inference.png"


def render():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(PDF), pagesize=(WIDTH, HEIGHT), pageCompression=1)
    c.setTitle("MultiCogTrust: when to predict and how to combine clues")
    c.setAuthor("MultiCogTrust")
    c.setSubject("Speech buffering, multimodal fusion, latency and design choices")

    def text(x, y, value, size=12, color=INK, bold=False, max_width=None):
        font = "Helvetica-Bold" if bold else "Helvetica"
        if max_width is not None:
            assert stringWidth(value, font, size) <= max_width, value
        c.setFillColor(HexColor(color))
        c.setFont(font, size)
        c.drawString(x, HEIGHT - y, value)

    def box(x, y, width, height, fill, stroke=None, radius=12, dashed=False):
        c.saveState()
        c.setFillColor(HexColor(fill))
        c.setStrokeColor(HexColor(stroke or fill))
        c.setLineWidth(1.3)
        if dashed:
            c.setDash(6, 4)
        c.roundRect(x, HEIGHT - y - height, width, height, radius, fill=1, stroke=1)
        c.restoreState()

    def arrow(points, color=INK, dashed=False):
        c.saveState()
        c.setStrokeColor(HexColor(color))
        c.setFillColor(HexColor(color))
        c.setLineWidth(1.65)
        if dashed:
            c.setDash(5, 4)
        path = c.beginPath()
        path.moveTo(points[0][0], HEIGHT - points[0][1])
        for x, y in points[1:]:
            path.lineTo(x, HEIGHT - y)
        c.drawPath(path)
        c.setDash()
        x, y = points[-1]
        px, py = points[-2]
        angle = math.atan2(y - py, x - px)
        head = c.beginPath()
        head.moveTo(x, HEIGHT - y)
        for offset in (-0.48, 0.48):
            head.lineTo(
                x - 8 * math.cos(angle + offset),
                HEIGHT - (y - 8 * math.sin(angle + offset)),
            )
        head.close()
        c.drawPath(head, fill=1, stroke=0)
        c.restoreState()

    def card(x, y, width, height, title, lines, *, fill="#FFFFFF", color=INK,
             stroke="#CBD7E3", dashed=False, title_size=14, body_size=11.7):
        box(x, y, width, height, fill, stroke, dashed=dashed)
        text(x + 14, y + 25, title, title_size, color, True, width - 28)
        for i, line in enumerate(lines):
            text(x + 14, y + 47 + i * 18, line, body_size, MUTED,
                 max_width=width - 28)

    box(0, 0, WIDTH, HEIGHT, "#FFFFFF", radius=0)
    text(36, 40, "MULTICOGTRUST / THE BIG PICTURE", 11, TEAL, True)
    text(36, 82, "Listen, gather context, combine clues", 31, INK, True)
    text(36, 107, "The challenge: useful emotion estimates, soon enough for a conversation.", 13, MUTED)
    box(999, 36, 165, 30, "#EAF4F3", radius=15)
    text(1018, 56, "FOUR-CLASS MVP", 12, TEAL, True)

    box(36, 129, 1128, 368, "#F4F7FB", "#E3EAF2", radius=18)
    text(56, 156, "01  WHAT THE LIVE BASELINE DOES NOW", 12, BLUE, True)
    text(884, 156, "Solid = current | dashed = next", 10.5, MUTED)
    text(56, 180, "Capture is continuous. Emotion prediction waits for a completed speech segment.", 11.5, MUTED)

    arrow([(211, 274), (249, 274)])
    arrow([(426, 255), (454, 255), (454, 230), (482, 230)], BLUE)
    arrow([(426, 294), (454, 294), (454, 351), (482, 351)], PURPLE)
    arrow([(664, 230), (688, 230), (688, 265), (718, 265)], BLUE)
    arrow([(664, 351), (688, 351), (688, 312), (718, 312)], PURPLE)
    arrow([(922, 288), (977, 288)], TEAL)
    arrow([(816, 418), (816, 356)], TEAL, dashed=True)

    card(56, 224, 155, 100, "Listen",
         ["Speech arrives", "in short pieces.", "VAD checks for voice."], body_size=11.2)
    card(249, 224, 177, 100, "Wait for a pause",
         ["Keep a speech buffer.", "Release a segment.", "Process it in order."], title_size=13.7)
    card(482, 188, 182, 81, "How it sounds",
         ["Describe voice patterns.", "E.g. emphasis and rhythm."],
         fill="#E7F0FD", color=BLUE, stroke="#C5D8F0")
    card(482, 310, 182, 83, "What was said",
         ["Transcribe the speech.", "Represent its content."],
         fill="#F0EAF9", color=PURPLE, stroke="#DACDEC")
    card(718, 226, 204, 130, "Combine the clues",
         ["Use learned relationships", "between voice and words.", "Estimate an emotion."],
         fill="#E7F4F1", color=TEAL, stroke="#B9DCD5")

    box(977, 203, 171, 174, INK)
    text(992, 230, "Four outputs", 15, "#FFFFFF", True)
    text(992, 249, "Emotion probabilities", 10.8, "#C5D5E5")
    for i, name in enumerate(("Anger", "Happiness", "Sadness", "Neutral")):
        y = 261 + i * 26
        box(992, y, 141, 21, "#2C4864", radius=6)
        text(1003, y + 15, name, 11.5, "#FFFFFF")

    text(56, 420, "A pause can occur inside a sentence.", 12, INK, True)
    text(56, 443, "Current limit: only the first six seconds of a segment are analyzed.", 11, MUTED)
    text(56, 463, "Long speech needs a deliberate handling policy.", 11, MUTED)
    card(746, 418, 402, 60, "Next: add matching face cues",
         ["Features exist; live fusion still needs evaluation."],
         fill="#E7F4F1", color=TEAL, stroke=TEAL, dashed=True, body_size=11.5)

    box(36, 521, 1128, 124, "#F3F9F8", "#D9EAE7", radius=18)
    text(56, 548, "02  THE DELAY AFTER THE PERSON STOPS SPEAKING", 12, TEAL, True)
    labels = ("Confirm the pause", "Wait behind other work", "Run the models", "Display the result")
    for i, label in enumerate(labels):
        x = 56 + 280 * i
        box(x, 570, 246, 43, "#FFFFFF", "#BBD8D2", radius=9)
        text(x + 14, 597, label, 13, INK, True, 218)
        if i < 3:
            arrow([(x + 246, 591), (x + 280, 591)], TEAL)
    text(56, 633, "Measure the whole wait. Faster models alone cannot remove pause detection or a backlog.", 11, MUTED)

    box(36, 666, 1128, 139, "#FAF8F3", "#E9E3D8", radius=18)
    text(56, 694, "03  WHEN TO UPDATE IS A DESIGN CHOICE", 12, "#8A6B32", True)
    card(56, 711, 344, 79, "Rolling windows",
         ["Earlier hints; less context.", "Overlapping work may repeat."],
         stroke="#E0D7C6", color="#8A6B32")
    card(424, 711, 344, 79, "After a pause (current)",
         ["More context; fewer model calls.", "Long turns mean more waiting."],
         stroke="#E0D7C6", color="#8A6B32")
    card(792, 711, 352, 79, "Hybrid (proposal)",
         ["Tentative updates, then a final result.", "More work and coordination."],
         stroke="#8A6B32", color="#8A6B32", dashed=True)

    text(36, 830, "Better speed must preserve useful emotion predictions. Trust-specific learning is a later task.", 11, MUTED)

    c.showPage()
    c.save()

    # Render the delivered PDF itself so the embedded preview matches it exactly.
    with pymupdf.open(PDF) as document:
        assert len(document) == 1
        document[0].get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False).save(PNG)
    print(PDF)
    print(PNG)


if __name__ == "__main__":
    render()
