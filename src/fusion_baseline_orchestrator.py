from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.events import ProsodyFeatureEvent, TranscriptEvent  # noqa: E402
from src.iemocap_aligned_dataset import load_iemocap_aligned  # noqa: E402
from src.models.simple_fusion import SimpleTextProsodyFusionClassifier  # noqa: E402
from src.text.nlp import NLPConfig, run_nlp_worker  # noqa: E402
from src.workers.fusion import (  # noqa: E402
    SimpleFusionWorkerConfig,
    run_simple_fusion_worker,
)


def load_model(
    checkpoint_path: Path | None,
    num_labels: int,
    model_name: str,
    device: str,
) -> SimpleTextProsodyFusionClassifier:
    model = SimpleTextProsodyFusionClassifier(
        num_labels=num_labels,
        text_model_name=model_name,
        freeze_text_encoder=True,
    )
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
    model.to(torch.device(device))
    model.eval()
    return model


def record_to_events(
    record: dict,
    start_time: float,
) -> tuple[TranscriptEvent, ProsodyFeatureEvent]:
    duration_sec = float(record["duration_sec"])
    words = [word["word"] for word in record["words"]]
    transcript = TranscriptEvent(
        start_time=start_time,
        end_time=start_time + duration_sec,
        text=record["text"],
    )
    prosody = ProsodyFeatureEvent(
        start_time=start_time,
        end_time=start_time + duration_sec,
        utterance_id=record["utterance_id"],
        words=words,
        word_features=np.asarray(record["word_audio_features"], dtype=np.float32),
        word_feature_mask=np.asarray(
            record["word_audio_feature_mask"],
            dtype=np.float32,
        ),
    )
    return transcript, prosody


async def run_replay(args: argparse.Namespace) -> None:
    train_ds, test_ds, num_labels, label_names = load_iemocap_aligned(args.aligned_dir)
    dataset = test_ds if args.split == "test" else train_ds
    records = [dataset[idx] for idx in range(min(args.limit, len(dataset)))]

    stop_event = asyncio.Event()
    transcript_queue: asyncio.Queue[TranscriptEvent] = asyncio.Queue(maxsize=16)
    text_queue = asyncio.Queue(maxsize=16)
    prosody_queue: asyncio.Queue[ProsodyFeatureEvent] = asyncio.Queue(maxsize=16)
    prediction_queue = asyncio.Queue(maxsize=16)

    model = load_model(args.checkpoint, num_labels, args.model_name, args.device)
    tasks = [
        asyncio.create_task(
            run_nlp_worker(
                transcript_queue,
                text_queue,
                stop_event,
                NLPConfig(model_name=args.model_name, device=args.device),
                encoder=None,
            )
        ),
        asyncio.create_task(
            run_simple_fusion_worker(
                text_queue,
                prosody_queue,
                prediction_queue,
                stop_event,
                model,
                SimpleFusionWorkerConfig(device=args.device),
            )
        ),
    ]

    try:
        current_time = 0.0
        for record in records:
            transcript, prosody = record_to_events(record, current_time)
            await prosody_queue.put(prosody)
            await transcript_queue.put(transcript)
            current_time = transcript.end_time + 0.25

        for _ in records:
            prediction = await asyncio.wait_for(prediction_queue.get(), timeout=60.0)
            label_name = label_names[prediction.predicted_label]
            confidence = float(prediction.probabilities[prediction.predicted_label])
            print(
                f"{prediction.utterance_id}: pred={label_name} "
                f"confidence={confidence:.3f} text={prediction.text!r}"
            )
    finally:
        stop_event.set()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay aligned IEMOCAP examples through NLP and simple fusion queues."
        )
    )
    parser.add_argument(
        "--aligned-dir",
        type=Path,
        default=ROOT / "data" / "iemocap_aligned_features",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    asyncio.run(run_replay(parse_args()))


if __name__ == "__main__":
    main()
