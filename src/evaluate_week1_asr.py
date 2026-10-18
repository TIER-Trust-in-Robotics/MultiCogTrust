"""Matched Session 4 ASR diagnostics for baseline/reference/improvement heads.

Reuses the verified six-second audio embedding cache and the existing actual
tiny.en transcripts. Text encodings are pinned and cached per encoder family.
All 1,031 samples, including empty-ASR abstentions, remain in each comparison.
This is a development diagnostic; it performs no test inference or selection.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.emotion_checkpoint import PreprocessingConfig, read_checkpoint, load_checkpoint_weights
from src.evaluate_asr_baseline import diagnostic_metrics
from src.iemocap_embedding_cache import feature_path, sha256, write_json, collate_embeddings, CachedDataset
from src.models import CrossModalTransformerFusionClassifier
from src.prepare_iemocap import DEFAULT_METADATA
from src.train_cross_modal_finetune_iemocap import load_samples
from src.train_mvp_baseline import evaluate as evaluate_reference


class ASRDataset(Dataset):
    def __init__(self, samples, audio_cache, text_cache):
        self.samples, self.audio_cache, self.text_cache = samples, audio_cache, text_cache

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        sample = self.samples[i]
        with np.load(feature_path(self.audio_cache, sample.utterance_id), allow_pickle=False) as entry:
            audio = torch.from_numpy(entry["audio"])
        text = torch.from_numpy(np.load(self.text_cache / f"{sample.utterance_id}.npy", allow_pickle=False))
        return text, audio, sample.label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/benchmarks/week1_v1/iemocap_asr")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()
    if (args.output / "summary.json").exists():
        raise FileExistsError("Preserving complete diagnostic")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    metadata_hash = sha256(DEFAULT_METADATA)
    samples = [s for s in load_samples(str(DEFAULT_METADATA), str(ROOT)) if s.session == 4]
    if len(samples) != 1031:
        raise ValueError("Unexpected validation coverage")
    transcripts_path = ROOT / "data/processed/mvp/asr_validation/transcripts.jsonl"
    transcripts = {r["utterance_id"]: r for r in map(json.loads, transcripts_path.read_text().splitlines())}
    asr_config = json.loads(transcripts_path.with_name("config.json").read_text())
    preprocessing = PreprocessingConfig(**asr_config["preprocessing"])
    if asr_config["metadata_sha256"] != metadata_hash or set(transcripts) != {s.utterance_id for s in samples}:
        raise ValueError("ASR source or coverage changed")
    for sample in samples:
        if transcripts[sample.utterance_id]["audio_sha256"] != sha256(sample.audio_path):
            raise ValueError("ASR/raw waveform changed")
    nonempty = [s for s in samples if transcripts[s.utterance_id]["text"]]
    if len(nonempty) != 1027:
        raise ValueError("Unexpected ASR abstention count")
    groups = {
        "distilled": {
            "cache": ROOT / "data/processed/iemocap_encoder_cache",
            "runs": [("baseline_seed42", ROOT / "models/mvp/audio_text_baseline.pt"), ("baseline_seed43", ROOT / "data/processed/benchmarks/iemocap_cmt_v1/seed43.pt"), ("baseline_seed44", ROOT / "data/processed/benchmarks/iemocap_cmt_v1/seed44.pt")]
                + [(f"smoothing_seed{s}", ROOT / f"data/processed/benchmarks/week1_v1/iemocap_smoothing/seed{s}.pt") for s in (42, 43, 44)],
        },
        "base": {
            "cache": ROOT / "data/processed/benchmarks/week1_v1/iemocap_base_encoder_cache",
            "runs": [(f"base_seed{s}", ROOT / f"data/processed/benchmarks/week1_v1/iemocap_base/seed{s}.pt") for s in (42, 43, 44)],
        },
    }
    reports = {}
    for family, spec in groups.items():
        first = read_checkpoint(spec["runs"][0][1], "iemocap_emotion_4")
        config = first["model_config"]
        index = json.loads((spec["cache"] / "index.json").read_text())
        encoders = {k: config[k] for k in ("text_model_name", "text_revision", "audio_model_name", "audio_revision")}
        if not index["complete"] or index["contract"]["metadata_sha256"] != metadata_hash or index["contract"]["encoders"] != encoders or index["contract"]["preprocessing"] != asdict(preprocessing):
            raise ValueError("Reference audio cache/ASR contract mismatch")
        entries = {e["utterance_id"]: e for e in index["entries"]}
        for sample in samples:
            if entries[sample.utterance_id]["session"] != 4 or sha256(feature_path(spec["cache"], sample.utterance_id)) != entries[sample.utterance_id]["feature_sha256"]:
                raise ValueError("Audio embedding provenance changed")
        model = CrossModalTransformerFusionClassifier(
            **config, freeze_audio_encoder=True, freeze_text_encoder=True,
            local_files_only=True,
        ).to(args.device).eval()
        tokenizer = AutoTokenizer.from_pretrained(config["text_model_name"], revision=config["text_revision"], local_files_only=True)
        text_cache = args.output / f"{family}_text_cache"
        text_cache.mkdir(exist_ok=True)
        contract = {"encoders": encoders, "metadata_sha256": metadata_hash, "transcripts_sha256": sha256(transcripts_path), "audio_index_sha256": sha256(spec["cache"] / "index.json"), "preprocessing": asdict(preprocessing), "test_model_inference": False, "sample_ids": [s.utterance_id for s in nonempty]}
        contract_path = text_cache / "contract.json"
        if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
            raise ValueError("ASR text cache contract changed")
        write_json(contract_path, contract)
        hashes_path = text_cache / "hashes.json"
        if hashes_path.exists():
            hashes = json.loads(hashes_path.read_text())
            if set(hashes) != {s.utterance_id for s in nonempty} or any(
                sha256(text_cache / f"{uid}.npy") != digest
                for uid, digest in hashes.items()
            ):
                raise ValueError("Cached ASR text bytes changed")
        with torch.inference_mode():
            for i, sample in enumerate(nonempty):
                path = text_cache / f"{sample.utterance_id}.npy"
                if path.exists():
                    continue
                tokens = tokenizer(transcripts[sample.utterance_id]["text"], return_tensors="pt", truncation=True, max_length=preprocessing.max_tokens)
                text = model.encode_text(tokens["input_ids"].to(args.device), tokens["attention_mask"].to(args.device))[0].cpu().numpy()
                if text.dtype != np.float32 or not np.isfinite(text).all():
                    raise ValueError("Nonfinite ASR text embedding")
                temporary = path.with_suffix(".tmp")
                with temporary.open("wb") as handle:
                    np.save(handle, text, allow_pickle=False)
                temporary.replace(path)
                if (i + 1) % 200 == 0:
                    print(f"{family} ASR text encoded {i + 1}/{len(nonempty)}", flush=True)
        write_json(text_cache / "hashes.json", {s.utterance_id: sha256(text_cache / f"{s.utterance_id}.npy") for s in nonempty})
        loader = DataLoader(ASRDataset(nonempty, spec["cache"], text_cache), batch_size=16, collate_fn=collate_embeddings)
        reference_loader = DataLoader(CachedDataset(samples, spec["cache"]), batch_size=16, collate_fn=collate_embeddings)
        for name, checkpoint_path in spec["runs"]:
            checkpoint = read_checkpoint(checkpoint_path, "iemocap_emotion_4")
            if checkpoint["model_config"] != config or checkpoint["provenance"]["metadata_sha256"] != metadata_hash or checkpoint["preprocessing"] != asdict(preprocessing):
                raise ValueError("Comparator input contract changed")
            load_checkpoint_weights(model, checkpoint)
            reference_metrics, reference_probabilities = evaluate_reference(model, reference_loader, args.device)
            saved_reference = [json.loads(line) for line in checkpoint_path.with_suffix(".validation.jsonl").read_text().splitlines()]
            if [r["utterance_id"] for r in saved_reference] != [s.utterance_id for s in samples]:
                raise ValueError("Reference comparator coverage/order differs")
            saved_probabilities = np.array([r["probabilities"] for r in saved_reference])
            if not np.allclose(reference_probabilities, saved_probabilities, rtol=1e-5, atol=1e-5):
                raise AssertionError("Independent reference replay failed")
            probabilities = []
            with torch.inference_mode():
                for text, tm, audio, am, _ in loader:
                    logits = model.forward_from_embeddings(text.to(args.device), audio.to(args.device), tm.to(args.device), am.to(args.device)).logits
                    probabilities.extend(logits.softmax(-1).cpu().tolist())
            available = {s.utterance_id: p for s, p in zip(nonempty, probabilities, strict=True)}
            predictions = [{"utterance_id": s.utterance_id, "label": s.label, "prediction": int(np.argmax(available[s.utterance_id])) if s.utterance_id in available else -1, "probabilities": available.get(s.utterance_id)} for s in samples]
            if name == "baseline_seed42":
                previous = {r["utterance_id"]: r for r in map(json.loads, transcripts_path.with_name("predictions.jsonl").read_text().splitlines())}
                if any(previous[r["utterance_id"]]["prediction"] != r["prediction"] for r in predictions):
                    raise AssertionError("Cached ASR path fails to reproduce public-predictor labels")
            metrics = diagnostic_metrics(predictions)
            reports[name] = {"checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path), "metrics": metrics, "selected_epoch": checkpoint["provenance"]["epoch"], "selection_used_asr": False, "family": family, "test_model_inference": False, "text_cache_contract_sha256": sha256(contract_path), "reference_replay": reference_metrics, "reference_replay_max_abs_difference": float(np.max(np.abs(np.asarray(reference_probabilities) - saved_probabilities)))}
            write_json(args.output / f"{name}.metrics.json", reports[name])
            (args.output / f"{name}.predictions.jsonl").write_text("".join(json.dumps(p) + "\n" for p in predictions))
            print(f"{name}: ASR macro F1 {metrics['macro_f1']:.4f}, {metrics['count']} examples, {metrics['abstentions']} abstentions", flush=True)
        del model, tokenizer
    write_json(args.output / "summary.json", {"test_model_inference": False, "source_sha256": sha256(Path(__file__)), "transcripts_sha256": sha256(transcripts_path), "asr_config": asr_config, "diagnostic_no_selection": True, "runs": reports})


if __name__ == "__main__":
    main()
