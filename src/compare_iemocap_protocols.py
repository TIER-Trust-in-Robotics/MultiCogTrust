"""Legacy exploratory split comparison on IEMOCAP (4-class).

This is NOT a MemoCMT paper reproduction. Its random 80/10/10 partition differs
from the released two-stage sklearn split (approximately 81/9/10), and the
encoders and audio crop also differ. A split changes example composition and
training-set size as well as speaker overlap, so the score gap does not isolate
speaker leakage. The session protocol is a single held-out session, not full
leave-one-session-out cross-validation.

WARNING: this legacy command encodes all five sessions and evaluates test data.
Use benchmark_controls.py for development with Session 5 reserved.

Example:
    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python src/compare_iemocap_protocols.py \
        --encode-device mps --epochs 40
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch import nn
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.iemocap_aligned_dataset import resolve_data_path  # noqa: E402
from src.models import CrossModalTransformerFusionClassifier  # noqa: E402

LABEL_MAP = {"ang": 0, "hap": 1, "exc": 1, "sad": 2, "neu": 3}
LABEL_NAMES = ["anger", "happiness", "sadness", "neutral"]
SAMPLE_RATE = 16000


@dataclass
class Sample:
    text: str
    audio_path: str
    label: int
    session: int


@dataclass
class Cached:
    text_embedding: torch.Tensor  # [1, T_text, dim]
    audio_embedding: torch.Tensor  # [1, T_audio, dim]
    label: int


def load_samples(metadata_path: str, data_root: str) -> list[Sample]:
    samples: list[Sample] = []
    missing = 0
    with open(metadata_path) as handle:
        for line in handle:
            record = json.loads(line)
            code = record["label_code"]
            if code not in LABEL_MAP:
                continue
            local_path = resolve_data_path(
                record["audio_path"], Path(data_root) / "data"
            )
            if not local_path.exists():
                missing += 1
                continue
            samples.append(
                Sample(record["text"], str(local_path), LABEL_MAP[code], int(record["session"]))
            )
    if not samples:
        raise FileNotFoundError(
            f"No audio from {metadata_path} resolved under {data_root} "
            f"({missing} paths missing); restore data/IEMOCAP_full_release/ "
            "or pass --data-root"
        )
    if missing:
        print(f"Warning: skipped {missing} utterances with missing audio")
    return samples


def cap_per_class(samples: list[Sample], max_samples: int, rng: random.Random) -> list[Sample]:
    if max_samples <= 0 or max_samples >= len(samples):
        return samples
    by_label: dict[int, list[Sample]] = {}
    for sample in samples:
        by_label.setdefault(sample.label, []).append(sample)
    per_label = max(1, max_samples // len(by_label))
    chosen: list[Sample] = []
    for label_samples in by_label.values():
        rng.shuffle(label_samples)
        chosen.extend(label_samples[:per_label])
    return chosen


def load_audio(path: str, max_seconds: float) -> torch.Tensor:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    waveform = torch.from_numpy(audio)
    if sr != SAMPLE_RATE:
        import torchaudio

        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    max_len = int(max_seconds * SAMPLE_RATE)
    if waveform.numel() > max_len:
        waveform = waveform[:max_len]
    return waveform


def precompute(samples, model, tokenizer, max_seconds, device) -> list[Cached]:
    cache: list[Cached] = []
    model.eval()
    start = time.time()
    with torch.no_grad():
        for i, sample in enumerate(samples):
            encoded = tokenizer(sample.text, return_tensors="pt", truncation=True, max_length=297)
            text_embedding = model.encode_text(
                encoded["input_ids"].to(device), encoded["attention_mask"].to(device)
            )
            waveform = load_audio(sample.audio_path, max_seconds).unsqueeze(0).to(device)
            audio_embedding = model.encode_audio(waveform)
            cache.append(Cached(text_embedding.cpu(), audio_embedding.cpu(), sample.label))
            if (i + 1) % 250 == 0 or (i + 1) == len(samples):
                rate = (i + 1) / (time.time() - start)
                print(f"  encoded {i + 1}/{len(samples)}  ({rate:.1f} utt/s)")
    return cache


def split_random(n, seed, val_frac=0.1, test_frac=0.1):
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    return indices[n_test + n_val :], indices[n_test : n_test + n_val], indices[:n_test]


def split_session(samples, test_session, val_frac, seed):
    test_idx = [i for i, s in enumerate(samples) if s.session == test_session]
    train_pool = [i for i, s in enumerate(samples) if s.session != test_session]
    random.Random(seed).shuffle(train_pool)
    n_val = int(len(train_pool) * val_frac)
    return train_pool[n_val:], train_pool[:n_val], test_idx


def evaluate(model, cache, indices, device):
    model.eval()
    confusion = np.zeros((len(LABEL_NAMES), len(LABEL_NAMES)), dtype=int)
    with torch.no_grad():
        for i in indices:
            item = cache[i]
            output = model.forward_from_embeddings(
                item.text_embedding.to(device), item.audio_embedding.to(device)
            )
            confusion[item.label, int(output.logits.argmax(-1).item())] += 1
    total = confusion.sum()
    weighted_acc = confusion.trace() / max(1, total)
    recalls = confusion.diagonal() / confusion.sum(axis=1).clip(min=1)
    return weighted_acc, float(recalls.mean()), confusion


def train_protocol(name, cache, train_idx, val_idx, test_idx, args, device):
    print(f"\n===== {name} =====")
    print(f"train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")
    model = CrossModalTransformerFusionClassifier(
        num_labels=len(LABEL_NAMES),
        fusion_head_output_type=args.fusion_head,
        text_model_name=args.text_model,
        audio_model_name=args.audio_model,
        freeze_text_encoder=True,
        freeze_audio_encoder=True,
    ).to(device)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    order = list(train_idx)
    best_val_ua = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.Random(epoch).shuffle(order)
        total_loss = 0.0
        for i in order:
            item = cache[i]
            optimizer.zero_grad()
            output = model.forward_from_embeddings(
                item.text_embedding.to(device), item.audio_embedding.to(device)
            )
            loss = criterion(output.logits, torch.tensor([item.label], device=device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        val_wa, val_ua, _ = evaluate(model, cache, val_idx, device)
        if val_ua > best_val_ua:
            best_val_ua = val_ua
            best_state = copy.deepcopy(model.state_dict())
        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            print(
                f"  epoch {epoch:3d}  loss={total_loss / len(order):.4f}  "
                f"val_WA={val_wa:.3f}  val_UA={val_ua:.3f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    test_wa, test_ua, confusion = evaluate(model, cache, test_idx, device)
    print(f"  TEST  WA={test_wa:.3f}  UA={test_ua:.3f}  (best val_UA={best_val_ua:.3f})")
    # Save only the trained fusion + head; the frozen encoders are reloadable by
    # name (saving them would be ~850MB of redundant pretrained weights).
    head_state = {
        k: v for k, v in model.state_dict().items()
        if not k.startswith(("text_encoder.", "audio_encoder."))
    }
    out = os.path.join(args.out_dir, f"cross_modal_{args.tag}_{name}.pt")
    torch.save(
        {"fusion_head_state_dict": head_state, "label_names": LABEL_NAMES,
         "fusion_head": args.fusion_head, "protocol": name,
         "text_model": args.text_model, "audio_model": args.audio_model,
         "test_wa": test_wa, "test_ua": test_ua},
        out,
    )
    return {"WA": test_wa, "UA": test_ua, "confusion": confusion, "checkpoint": out}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="data/iemocap_aligned_features/metadata.jsonl")
    parser.add_argument("--data-root", default=os.path.abspath("."))
    parser.add_argument("--max-samples", type=int, default=0, help="0 = full dataset")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-audio-sec", type=float, default=6.0)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-session", type=int, default=5)
    parser.add_argument("--fusion-head", default="min", choices=["cls", "mean", "max", "min"])
    parser.add_argument("--text-model", default="distilbert-base-uncased")
    parser.add_argument("--audio-model", default="ntu-spml/distilhubert")
    parser.add_argument("--tag", default="distilled", help="label for output checkpoints/table")
    parser.add_argument("--split-seed", type=int, default=0, help="paper used seed 0")
    parser.add_argument("--encode-device", default="auto")
    parser.add_argument("--train-device", default="cpu")
    parser.add_argument("--out-dir", default="models")
    args = parser.parse_args()

    if args.encode_device == "auto":
        args.encode_device = (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.split_seed)

    samples = load_samples(args.metadata, args.data_root)
    samples = cap_per_class(samples, args.max_samples, random.Random(args.split_seed))
    counts = {name: 0 for name in LABEL_NAMES}
    for sample in samples:
        counts[LABEL_NAMES[sample.label]] += 1
    print(f"Loaded {len(samples)} IEMOCAP 4-class utterances: {counts}")

    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    encoder = CrossModalTransformerFusionClassifier(
        num_labels=len(LABEL_NAMES), fusion_head_output_type=args.fusion_head,
        text_model_name=args.text_model, audio_model_name=args.audio_model,
        freeze_text_encoder=True, freeze_audio_encoder=True,
    ).to(args.encode_device)
    print(f"Encoders [{args.tag}]: text={args.text_model}  audio={args.audio_model}")
    print(f"Pre-computing frozen embeddings on {args.encode_device} (shared across both protocols)...")
    cache = precompute(samples, encoder, tokenizer, args.max_audio_sec, args.encode_device)
    del encoder

    tr, va, te = split_random(len(samples), args.split_seed, args.val_frac, 0.1)
    paper = train_protocol("random-exploratory", cache, tr, va, te, args, args.train_device)

    tr, va, te = split_session(samples, args.test_session, args.val_frac, args.split_seed)
    session = train_protocol("session-independent", cache, tr, va, te, args, args.train_device)

    print("\n" + "=" * 58)
    print(f"COMPARISON [{args.tag}] (4-class IEMOCAP, frozen encoders, {args.fusion_head} agg)")
    print("=" * 58)
    print(f"{'protocol':<26}{'WA':>8}{'UA':>8}")
    print("MemoCMT's reported scores use a different protocol and are not ranked here.")
    print(f"{'random 80/10/10':<26}{paper['WA']:>8.3f}{paper['UA']:>8.3f}")
    print(f"{'session-independent':<26}{session['WA']:>8.3f}{session['UA']:>8.3f}")
    print(
        f"\nObserved split difference (random - session), not a causal leakage estimate: "
        f"WA {paper['WA'] - session['WA']:+.3f}  UA {paper['UA'] - session['UA']:+.3f}"
    )


if __name__ == "__main__":
    main()
