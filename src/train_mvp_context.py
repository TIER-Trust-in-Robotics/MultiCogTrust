"""Frozen-encoder fusion head that reads the speaker's previous line as text context.

Same architecture and encoders as train_mvp_baseline.py. The text input is
"{previous same-speaker line} [SEP] {current line}" (no prefix for a speaker's
first line), built from reference transcripts in dialogue order, including
unscored lines. Live, the previous line is the previous Whisper transcript.

The recipe is fixed (chosen out-of-fold on Sessions 1-3): train a set number of
epochs, then average the head's weights over an epoch window. Session 4 is
reported, never used for selection. Session 5 is never encoded or evaluated.

    uv run python src/train_mvp_context.py --offline
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.emotion_checkpoint import PreprocessingConfig, save_checkpoint
from src.iemocap_embedding_cache import CachedDataset, build_cache, collate_embeddings, sha256, write_json
from src.models import CrossModalTransformerFusionClassifier
from src.prepare_iemocap import DEFAULT_METADATA
from src.train_cross_modal_finetune_iemocap import choose_device, load_samples
from src.train_mvp_baseline import evaluate, forward


def with_previous_line(samples, metadata_path):
    """Prefix each sample's text with its speaker's previous line in the dialogue."""
    rows = [json.loads(line) for line in Path(metadata_path).read_text().splitlines()]
    chains = defaultdict(list)
    for row in rows:
        if row["session"] != 5:
            chains[(row["dialog_id"], row["speaker"])].append(row)
    previous = {}
    for chain in chains.values():
        last = None
        for row in sorted(chain, key=lambda r: r["dialog_start_sec"]):
            previous[row["utterance_id"]] = last
            if row["text"].strip():
                last = row["text"].strip()
    return [
        replace(s, text=f"{previous[s.utterance_id]} [SEP] {s.text}" if previous[s.utterance_id] else s.text)
        for s in samples
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/mvp_context.json")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--cache", type=Path, default=ROOT / "data/processed/iemocap_context_encoder_cache")
    parser.add_argument("--output", type=Path, default=ROOT / "models/mvp/context_baseline.pt")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if (
        config["train_sessions"] != [1, 2, 3]
        or config["validation_session"] != 4
        or config["test_session"] != 5
        or config["test_evaluated"]
        or config["text_context"] != "previous_same_speaker_line"
        or not config["freeze_audio_encoder"]
        or not config["freeze_text_encoder"]
    ):
        raise ValueError("Context baseline requires frozen encoders, fixed sessions and previous-line context")
    if args.output.exists():
        raise FileExistsError(f"Preserving existing run: {args.output}; use another --output")
    first, last = config["weight_average_epochs"]
    if not 1 <= first <= last == config["epochs"]:
        raise ValueError("Weight-average window must end at the final epoch")
    started = time.perf_counter()
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(4)
    device = choose_device(args.device)
    development = [s for s in load_samples(str(args.metadata), str(ROOT)) if s.session != config["test_session"]]
    samples = with_previous_line(development, args.metadata)
    training = [s for s in samples if s.session in config["train_sessions"]]
    validation = [s for s in samples if s.session == config["validation_session"]]
    preprocessing = PreprocessingConfig(max_audio_sec=config["max_audio_sec"], max_tokens=config["max_tokens"])
    model = CrossModalTransformerFusionClassifier(
        num_labels=4,
        fusion_head_output_type=config["fusion_head"],
        freeze_text_encoder=True,
        freeze_audio_encoder=True,
        local_files_only=args.offline,
        **{k: config[k] for k in ("text_model_name", "text_revision", "audio_model_name", "audio_revision")},
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        config["text_model_name"], revision=config["text_revision"], local_files_only=args.offline
    )
    print(f"Device {device}; training {len(training)}; validation {len(validation)}; Session 5 excluded", flush=True)
    build_cache(training + validation, model, tokenizer, preprocessing, args.cache, sha256(args.metadata), device)
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(CachedDataset(training, args.cache), batch_size=config["batch_size"], shuffle=True,
                              collate_fn=collate_embeddings, generator=generator)
    val_loader = DataLoader(CachedDataset(validation, args.cache), batch_size=config["batch_size"],
                            collate_fn=collate_embeddings)
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.Adam([p for _, p in trainable], lr=config["learning_rate"])
    criterion = torch.nn.CrossEntropyLoss()
    snapshots, history = [], []
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        epoch_start, total_loss, seen = time.perf_counter(), 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits, labels = forward(model, batch, device)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(labels)
            seen += len(labels)
        if epoch >= first:
            snapshots.append({n: p.detach().cpu().clone() for n, p in trainable})
        history.append({"epoch": epoch, "train_loss": total_loss / seen, "seconds": time.perf_counter() - epoch_start})
        print(f"Epoch {epoch:02d}: loss={history[-1]['train_loss']:.4f} ({history[-1]['seconds']:.1f}s)", flush=True)
    model.load_state_dict({n: torch.stack([s[n] for s in snapshots]).mean(0) for n, _ in trainable}, strict=False)
    metrics, _ = evaluate(model, val_loader, device)
    provenance = {
        "kind": "context_development_baseline",
        "text_context": config["text_context"],
        "text_format": "{previous same-speaker transcript} [SEP] {current transcript}; first line of a speaker has no prefix",
        "seed": config["seed"],
        "train_count": len(training),
        "validation_count": len(validation),
        "validation_session": 4,
        "test_session": 5,
        "test_evaluated": False,
        "weight_average_epochs": [first, last],
        "selection": config["selection"],
        "metadata_sha256": sha256(args.metadata),
        "config": config,
        "config_sha256": sha256(args.config),
        "cache_index_sha256": sha256(args.cache / "index.json"),
        "transcript_source": "iemocap_reference",
        "device": device,
        "trainer_sha256": sha256(Path(__file__)),
        "session4_reference_text": metrics,
    }
    save_checkpoint(args.output, model, preprocessing=preprocessing, include_encoders=False, provenance=provenance)
    write_json(args.output.with_suffix(".metrics.json"), {
        "finished": True, "test_evaluated": False, "validation_reference_text": metrics, "history": history,
        "checkpoint_sha256": sha256(args.output), "elapsed_sec": time.perf_counter() - started,
    })
    print(json.dumps({"validation_reference_text": metrics, "test_evaluated": False}, indent=2), flush=True)


if __name__ == "__main__":
    main()
