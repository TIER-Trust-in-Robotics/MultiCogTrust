import asyncio
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import TinyTokenizer
from src.core.events import AudioChunk, SpeechSegment
from src.segmentAudioTorch import SileroVAD
from src.train_cross_modal_finetune_iemocap import make_collate
from src.prepare_iemocap import alignment_end
from src.iemocap_aligned_dataset import parse_wdseg


def test_documented_entry_points_start_without_devices():
    for path in [
        "src/pipeline_demo.py",
        "src/workers_demo.py",
        "src/train_cross_modal_finetune_iemocap.py",
        "src/prepare_iemocap.py",
    ]:
        result = subprocess.run(
            [sys.executable, path, "--help"], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout


def test_alignment_end_includes_silence(tmp_path):
    path = tmp_path / "sample.wdseg"
    path.write_text("0 9 123 <s>\n10 29 123 HELLO(2)\n30 99 123 </s>\n")
    assert alignment_end(path) == 1.0
    assert parse_wdseg(path)[0].word == "hello"
    assert parse_wdseg(path)[0].end_sec == 0.3


def test_collator_retains_audio_lengths():
    import torch

    batch = make_collate(TinyTokenizer())(
        [("hi", torch.ones(1024), 0), ("hello there", torch.ones(1600), 1)]
    )
    assert batch[3].sum(1).tolist() == [1024, 1600]
    assert not batch[3][0, 1024:].any()


def test_collator_runs_in_spawned_mac_style_worker():
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(
        [("hello", torch.ones(1024), 0), ("hello there", torch.ones(1600), 1)],
        batch_size=2,
        collate_fn=make_collate(TinyTokenizer()),
        num_workers=1,
        multiprocessing_context="spawn",
        timeout=30,
    )
    batches = list(loader)
    assert len(batches) == 1
    assert batches[0][3].sum(1).tolist() == [1024, 1600]
    assert batches[0][4].tolist() == [0, 1]


def test_vad_worker_segment_boundaries_and_pcm_conversion(monkeypatch):
    monkeypatch.setattr("src.segmentAudioTorch.load_silero_vad", lambda: object())
    vad = SileroVAD(min_speech_ms=32, min_silence_ms=64, pre_buffer_size=3)
    probabilities = iter([0.0, 1.0, 1.0, 0.0, 0.0])
    monkeypatch.setattr(vad, "_speach_prob", lambda samples: next(probabilities))
    segment = None
    for i in range(5):
        _, segment = vad.process_chunk(
            AudioChunk(i * 0.032, 16000, np.full(512, 1000, dtype=np.int16))
        )
    assert segment is not None
    assert segment.start_time == 0 and segment.end_time == pytest.approx(0.160)
    assert len(segment.samples) == 2560
    np.testing.assert_allclose(segment.samples, 1000 / 32768)


def test_async_asr_does_not_block_event_loop(monkeypatch):
    from src.text.transcribe import WhisperConfig, run_transcribe_worker

    class FakeWhisper:
        def __init__(self, model_name, *, device, compute_type):
            pass  # beam_size must not be passed to the constructor

        def transcribe(self, samples, **kwargs):
            assert kwargs["beam_size"] == 1

            def generate():
                time.sleep(0.08)
                yield SimpleNamespace(text="hello")

            return generate(), None

    monkeypatch.setattr("src.text.transcribe.WhisperModel", FakeWhisper)

    async def run():
        incoming, outgoing, stop = asyncio.Queue(), asyncio.Queue(), asyncio.Event()
        await incoming.put(
            SpeechSegment(1.0, 2.0, 16000, np.zeros(16000, dtype=np.float32))
        )
        worker = asyncio.create_task(
            run_transcribe_worker(incoming, outgoing, stop, WhisperConfig())
        )
        ticks = 0

        async def ticker():
            nonlocal ticks
            while not stop.is_set():
                ticks += 1
                await asyncio.sleep(0.005)

        clock = asyncio.create_task(ticker())
        try:
            result = await asyncio.wait_for(outgoing.get(), timeout=2)
            assert (result.start_time, result.end_time, result.text) == (
                1.0,
                2.0,
                "hello",
            )
            assert ticks >= 4
        finally:
            stop.set()
            await asyncio.gather(worker, clock)

    asyncio.run(run())


def test_final_evaluation_reloads_best_validation_checkpoint(
    tiny_model, tmp_path, monkeypatch
):
    import torch
    from src import train_cross_modal_finetune_iemocap as training

    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text("{}\n")
    samples = [
        training.Sample("hello", "unused.wav", 0, session, f"Ses0{session}_example")
        for session in (1, 4, 5)
    ]
    monkeypatch.setattr(training, "load_samples", lambda *a: samples)
    monkeypatch.setattr(
        training.AutoTokenizer, "from_pretrained", lambda *a, **k: TinyTokenizer()
    )
    monkeypatch.setattr(
        training, "CrossModalTransformerFusionClassifier", lambda **k: tiny_model()
    )

    class Data:
        def __init__(self, samples, max_seconds):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            return "hello", torch.ones(64), 0

    monkeypatch.setattr(training, "IemocapDataset", Data)
    calls = []

    def evaluate(model, loader, device):
        calls.append(len(calls) + 1)
        with torch.no_grad():
            if len(calls) <= 2:
                model.classifier.bias.fill_(len(calls))
                return (
                    (1.0, 1.0, np.eye(4)) if len(calls) == 1 else (0.0, 0.0, np.eye(4))
                )
            torch.testing.assert_close(
                model.classifier.bias, torch.ones_like(model.classifier.bias)
            )
            return 1.0, 1.0, np.eye(4, dtype=int)

    monkeypatch.setattr(training, "evaluate", evaluate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--metadata",
            str(metadata),
            "--epochs",
            "2",
            "--freeze-audio",
            "--device",
            "cpu",
            "--num-workers",
            "0",
            "--out",
            str(tmp_path / "best.pt"),
        ],
    )
    training.main()
    assert len(calls) == 3


def test_training_refuses_metadata_with_wrong_session_split(tmp_path):
    import json
    from src.train_cross_modal_finetune_iemocap import load_samples

    path = tmp_path / "metadata.jsonl"
    path.write_text(
        json.dumps(
            {
                "utterance_id": "Ses05F_example",
                "session": 5,
                "split": "train",
                "label_code": "neu",
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="Session split mismatch"):
        load_samples(str(path), str(tmp_path))


def test_training_refuses_metadata_changed_after_audit(tmp_path):
    import json
    from src.train_cross_modal_finetune_iemocap import load_samples

    path = tmp_path / "metadata.jsonl"
    path.write_text("{}\n")
    path.with_name("validation_report.json").write_text(
        json.dumps({"passed": True, "metadata_sha256": "not-the-current-file"})
    )
    with pytest.raises(ValueError, match="passing data audit"):
        load_samples(str(path), str(tmp_path))
