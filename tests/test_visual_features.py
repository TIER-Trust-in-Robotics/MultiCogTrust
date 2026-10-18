from types import SimpleNamespace
import numpy as np
import pytest

from src.audit_iemocap_video_alignment import match_audio
from src.extract_iemocap_visual import sample_times, extract_dialog
from src.visual_features import speaker_side, panel_crops, choose_face


def test_speaker_mapping_changes_with_dialog_marker():
    assert speaker_side("Ses01F_impro01", "F") == "left"
    assert speaker_side("Ses01F_impro01", "M") == "right"
    assert speaker_side("Ses02M_impro01", "M") == "left"
    assert speaker_side("Ses02M_impro01", "F") == "right"
    with pytest.raises(ValueError):
        speaker_side("unknown", "F")


def test_panel_and_face_selection_do_not_cross_participants():
    frame = np.zeros((60, 100, 3), np.uint8)
    frame[10:50, :48] = 80
    frame[10:50, 52:] = 160
    crops = panel_crops(frame)
    assert crops["left"][2] <= 50 <= crops["right"][0]
    assert choose_face([[1, 1, 21, 21], [60, 2, 65, 7]], (60, 100, 3)) == (0, 0)
    assert choose_face([[1, 1, 21, 21], [60, 2, 80, 22]], (60, 100, 3)) == (None, 3)
    assert choose_face([], frame.shape) == (None, 1)


def test_audio_video_alignment_recovers_offset_and_channel():
    rng = np.random.default_rng(12)
    reference = rng.normal(size=3200)
    source = np.zeros((32000, 2))
    source[8400:11600, 1] = reference
    result = match_audio(reference, source, 0.5)
    assert result["channel"] == 1
    assert result["correlation"] == pytest.approx(1)
    assert result["offset_sec"] == pytest.approx(0.025)


def test_sampling_never_reaches_end_or_audio_truncation_boundary():
    times = sample_times(2, 12, 5, 6)
    assert len(times) == 30 and times[0] == 2 and times[-1] < 8
    assert len(sample_times(1, 1.05, 5, 6)) == 1


def test_extraction_keeps_masks_timestamps_and_is_prefix_causal(tmp_path, monkeypatch):
    from src import extract_iemocap_visual as extraction

    video = tmp_path / "video.avi"
    video.write_bytes(b"source bytes")
    frames = []
    for i in range(10):
        frame = np.full((30, 60, 3), 40, np.uint8)
        frame[:, 30:] = 120
        frames.append((i, i / 10, frame))
    monkeypatch.setattr(
        extraction, "decoded_video", lambda p, stats=None: (f for f in frames)
    )

    class Face:
        def __init__(self, path):
            pass

        def close(self):
            pass

        def extract(self, frame, timestamp):
            missing = timestamp == 300
            image_only = timestamp == 500
            value = float(frame.mean())
            return SimpleNamespace(
                status=1 if missing else (6 if image_only else 0),
                face_count=0 if missing else 1,
                milliseconds=0,
                blendshapes=np.full(
                    52, 0 if missing or image_only else value / 255, np.float32
                ),
                image_valid=not missing,
                blendshape_valid=not (missing or image_only),
                detection_score=0.0 if missing else 0.99,
                box=np.array([2, 2, 12, 12]),
                crop=None if missing else frame,
            )

    class Images:
        def encode(self, crops):
            return np.stack([np.full(576, c.mean(), np.float32) for c in crops])

    monkeypatch.setattr(extraction, "FaceFeatureExtractor", Face)
    rows = []
    for speaker in ("F", "M"):
        rows.append(
            {
                "utterance_id": f"Ses01M_demo_{speaker}000",
                "dialog_id": "Ses01M_demo",
                "session": 1,
                "video_path": str(video),
                "dialog_start_sec": 0.1,
                "dialog_end_sec": 0.8,
                "speaker": speaker,
                "emotion4_label": 0,
                "video_span_valid": True,
            }
        )
    contract = {"sample_hz": 5, "max_seconds": 0.6}
    result = extract_dialog(
        rows, tmp_path / "first", contract, Images(), tmp_path / "weights"
    )
    frames.extend([(10, 1.0, np.full((30, 60, 3), 255, np.uint8))])
    repeated = extract_dialog(
        rows, tmp_path / "second", contract, Images(), tmp_path / "weights"
    )
    for first, second in zip(result["utterances"], repeated["utterances"]):
        with (
            np.load(tmp_path / "first" / first["feature_path"]) as a,
            np.load(tmp_path / "second" / second["feature_path"]) as b,
        ):
            assert a["valid"].tolist() == [True, False, False]
            assert a["image_valid"].tolist() == [True, False, True]
            assert not a["blendshapes"][~a["valid"]].any()
            assert not a["image_embeddings"][~a["image_valid"]].any()
            assert np.all(a["timestamps_sec"] < 0.7)
            for key in a.files:
                np.testing.assert_equal(a[key], b[key])
            assert a["image_embeddings"][0, 0] == (
                40 if first["speaker"] == "M" else 120
            )


def test_visual_batch_keeps_missing_face_separate_from_padding():
    import torch
    from src.visual_feature_dataset import collate_visual

    batch = []
    for i, length in enumerate((2, 4)):
        batch.append(
            {
                "utterance_id": str(i),
                "label": i,
                "blendshapes": torch.zeros(length, 52),
                "image_embeddings": torch.zeros(length, 576),
                "times_sec": torch.arange(length).float(),
                "visual_mask": torch.tensor([True] + [False] * (length - 1)),
                "image_mask": torch.ones(length, dtype=torch.bool),
                "blendshape_mask": torch.tensor([True] + [False] * (length - 1)),
                "decoded_mask": torch.ones(length, dtype=torch.bool),
            }
        )
    result = collate_visual(batch)
    assert result["sequence_mask"][0].tolist() == [True, True, False, False]
    assert result["decoded_mask"][0].tolist() == [True, True, False, False]
    assert result["visual_mask"][0].tolist() == [True, False, False, False]


def test_decoder_masks_untimed_tail_without_shifting_source_indices(monkeypatch):
    from src import extract_iemocap_visual as extraction

    class Capture:
        def __init__(self, times):
            self.times = times
            self.index = -1

        def isOpened(self):
            return True

        def read(self):
            self.index += 1
            return (
                (True, np.zeros((4, 4, 3), np.uint8))
                if self.index < len(self.times)
                else (False, None)
            )

        def get(self, key):
            return self.times[self.index]

        def release(self):
            pass

    monkeypatch.setattr(
        extraction.cv2, "VideoCapture", lambda path: Capture([66.7, 100.0, 0.0])
    )
    stats = {}
    frames = list(extraction.decoded_video("source", stats))
    assert [f[0] for f in frames] == [0, 1]
    assert stats["untimed_tail_frame_indices"] == [2]
    monkeypatch.setattr(
        extraction.cv2, "VideoCapture", lambda path: Capture([66.7, 0.0, 100.0])
    )
    with pytest.raises(ValueError, match="inside video"):
        list(extraction.decoded_video("source"))
