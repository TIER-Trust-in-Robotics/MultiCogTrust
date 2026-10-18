import numpy as np

from src.core.events import SpeechSegment
from src.emotion_fusion_infer import EmotionPrediction
from src.inference_pipeline import SpeechProcessor


class RecordingPredictor:
    def __init__(self, text_context=None):
        from src.emotion_checkpoint import PreprocessingConfig

        self.preprocessing = PreprocessingConfig()
        self.provenance = {"kind": "test"} | ({"text_context": text_context} if text_context else {})
        self.seen = []

    def predict(self, text, audio, sample_rate):
        self.seen.append(text)
        return EmotionPrediction("neutral", 3, {"anger": 0.1, "happiness": 0.1, "sadness": 0.1, "neutral": 0.7})


def segment():
    return SpeechSegment(0.0, 1.0, 16000, np.zeros(16000, np.float32))


def test_context_checkpoint_reads_the_previous_transcript():
    predictor = RecordingPredictor("previous_same_speaker_line")
    processor = SpeechProcessor(predictor=predictor)
    first = processor.process(segment(), "I did what they told me.")
    processor.process(segment(), "")  # empty transcript: abstain, context unchanged
    second = processor.process(segment(), "Then why is it lost?")
    assert predictor.seen == ["I did what they told me.", "I did what they told me. [SEP] Then why is it lost?"]
    assert first["context_text"] is None and second["context_text"] == "I did what they told me."
    assert second["model_provenance"]["text_context"] == "previous_same_speaker_line"


def test_training_context_uses_the_same_speakers_previous_line(tmp_path):
    import json

    from src.train_cross_modal_finetune_iemocap import Sample
    from src.train_mvp_context import with_previous_line

    rows = [
        {"utterance_id": "d_F000", "dialog_id": "d", "speaker": "F", "session": 1, "dialog_start_sec": 0.0, "text": "Hi."},
        {"utterance_id": "d_M000", "dialog_id": "d", "speaker": "M", "session": 1, "dialog_start_sec": 1.0, "text": "Hey."},
        {"utterance_id": "d_F001", "dialog_id": "d", "speaker": "F", "session": 1, "dialog_start_sec": 2.0, "text": "Ugh."},  # unscored
        {"utterance_id": "d_F002", "dialog_id": "d", "speaker": "F", "session": 1, "dialog_start_sec": 3.0, "text": "Fine."},
    ]
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text("".join(json.dumps(r) + "\n" for r in rows))
    samples = [Sample(text=r["text"], audio_path="", label=0, session=1, utterance_id=r["utterance_id"])
               for r in rows if r["utterance_id"] != "d_F001"]
    texts = {s.utterance_id: s.text for s in with_previous_line(samples, metadata)}
    assert texts == {"d_F000": "Hi.", "d_M000": "Hey.", "d_F002": "Ugh. [SEP] Fine."}


def test_other_checkpoints_are_unchanged():
    predictor = RecordingPredictor()
    processor = SpeechProcessor(predictor=predictor)
    processor.process(segment(), "One.")
    result = processor.process(segment(), "Two.")
    assert predictor.seen == ["One.", "Two."]
    assert "context_text" not in result


class WordTokenizer:
    """Whitespace stand-in for DistilBERT: one id per word, [CLS] = 1, [SEP] = 2."""

    cls_token_id, sep_token_id = 1, 2

    def __init__(self):
        self.vocab = {"[SEP]": 2}

    def __call__(self, text, add_special_tokens=True, return_tensors=None, truncation=False, max_length=None):
        import torch

        ids = [self.vocab.setdefault(w, len(self.vocab) + 3) for w in text.split()]
        if add_special_tokens:
            ids = [1] + ids[: max_length - 2 if truncation else None] + [2]
        if return_tensors == "pt":
            ids = torch.tensor([ids])
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        return {"input_ids": ids}


def ids_of(tokens):
    return tokens["input_ids"][0].tolist()


def test_context_is_cut_from_its_start_and_the_current_line_is_kept():
    from src.emotion_checkpoint import PreprocessingConfig, tokenize_text

    tok, config = WordTokenizer(), PreprocessingConfig(max_tokens=8)
    assert ids_of(tokenize_text(tok, "a b [SEP] c d", config)) == ids_of(tok("a b [SEP] c d", return_tensors="pt"))
    long = tokenize_text(tok, "p1 p2 p3 p4 p5 p6 [SEP] c1 c2 c3", config)
    assert ids_of(long) == ids_of(tok("p5 p6 [SEP] c1 c2 c3", return_tensors="pt"))
    assert long["attention_mask"].shape == long["input_ids"].shape == (1, 8)
    # A current line that cannot fit with any context is truncated alone.
    alone = tokenize_text(tok, "p1 [SEP] c1 c2 c3 c4 c5 c6 c7", config)
    assert ids_of(alone) == ids_of(tok("c1 c2 c3 c4 c5 c6", return_tensors="pt"))


def test_real_tokenizer_ids_match_the_joined_string():
    import pytest

    from src.emotion_checkpoint import PreprocessingConfig, tokenize_text

    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("distilbert-base-uncased", local_files_only=True)
    except OSError:
        pytest.skip("DistilBERT tokenizer is not cached locally")
    text = "I did exactly what they told me to do. [SEP] Then why is it lost?"
    assert ids_of(tokenize_text(tok, text, PreprocessingConfig())) == ids_of(tok(text, return_tensors="pt"))
    cut = ids_of(tokenize_text(tok, "word " * 400 + "[SEP] Then why is it lost?", PreprocessingConfig()))
    current = ids_of(tok("Then why is it lost?", return_tensors="pt"))  # [CLS] then why is it lost ? [SEP]
    assert len(cut) == 297 and cut[0] == current[0] and cut[-len(current):] == [tok.sep_token_id] + current[1:]
