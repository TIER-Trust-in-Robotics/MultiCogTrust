"""Overnight research harness: fusion heads on cached frozen-encoder sequences (Sessions 1-4 only).

    run --tag T --train 1 2 3 --train-text SOURCE --eval 4:asr 1:prevref ... --seed S

Text sources: "reference" (encoder cache), "asr" (Whisper), "prevref"/"prevasr" (previous own line
[SEP] current), and research sources under data/processed/research/text_<source>/ built by
encode_text.py. Every run saves each epoch's probabilities plus those of the weight average over
epochs 3-8 (one deployable checkpoint), per eval set.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.emotion_checkpoint import EMOTION_LABEL_MAP, PreprocessingConfig, save_checkpoint
from src.iemocap_embedding_cache import collate_embeddings, feature_path, sha256, write_json
from src.models import CrossModalTransformerFusionClassifier
from src.prepare_iemocap import DEFAULT_METADATA
from src.train_cross_modal_finetune_iemocap import load_samples

CONFIG = json.loads((ROOT / "configs/mvp_day3_baseline.json").read_text())
CACHE = ROOT / "data/processed/iemocap_encoder_cache"          # scored lines: reference text + audio
CONTEXT = ROOT / "data/processed/improve/context_cache"          # unscored lines: reference text + audio
IMPROVE = ROOT / "data/processed/improve"
OUT = ROOT / "data/processed/research"
ENCODERS = ("text_model_name", "text_revision", "audio_model_name", "audio_revision")


def text_dir(source):
    if source == "asr":
        return IMPROVE / "asr_text"
    if source in ("prevref", "prevasr"):
        return IMPROVE / f"text_{source}"
    return OUT / f"text_{source}"


class Item:
    """Unscored utterance: context only, never a metric target."""

    def __init__(self, row):
        self.utterance_id, self.session, self.label = row["utterance_id"], row["session"], -1


def scored_samples():
    samples = [s for s in load_samples(str(DEFAULT_METADATA), str(ROOT)) if s.session in (1, 2, 3, 4)]
    return {k: [s for s in samples if s.session == k] for k in (1, 2, 3, 4)}


def unscored(session):
    assert session in (1, 2, 3, 4), "Session 5 is held out"
    return [Item(r) for r in map(json.loads, Path(DEFAULT_METADATA).read_text().splitlines())
            if r["session"] == session and r["label_code"] not in EMOTION_LABEL_MAP]


class Embeddings(Dataset):
    def __init__(self, samples, text="reference"):
        if any(s.session == 5 for s in samples):
            raise ValueError("Session 5 is held out")
        self.text = text
        if text != "reference":  # missing file = empty transcript (deployment abstains)
            samples = [s for s in samples if (text_dir(text) / f"{s.utterance_id}.npy").exists()]
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        source = CONTEXT / f"{s.utterance_id}.npz" if s.label < 0 else feature_path(CACHE, s.utterance_id)
        with np.load(source, allow_pickle=False) as item:
            audio, text = torch.from_numpy(item["audio"]), torch.from_numpy(item["text"])
        if self.text != "reference":  # research sources are stored as float16
            text = torch.from_numpy(np.load(text_dir(self.text) / f"{s.utterance_id}.npy").astype(np.float32))
        return text, audio, max(s.label, 0)


def make_model(device):
    return CrossModalTransformerFusionClassifier(
        num_labels=4, fusion_head_output_type=CONFIG["fusion_head"], freeze_text_encoder=True,
        freeze_audio_encoder=True, local_files_only=True, **{k: CONFIG[k] for k in ENCODERS}).to(device)


@torch.inference_mode()
def predict(model, dataset, device):
    model.eval()
    out = []
    for text, tm, audio, am, _ in DataLoader(dataset, batch_size=64, collate_fn=collate_embeddings):
        out.append(model.forward_from_embeddings(text.to(device), audio.to(device), tm.to(device), am.to(device))
                   .logits.softmax(-1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 4), np.float32)


def run(args):
    result_path = OUT / "runs" / f"{args.tag}.npz"
    if result_path.exists():
        raise FileExistsError(f"Preserving completed run: {result_path}")
    if not args.train or len(set(args.train)) != len(args.train) or any(k not in (1, 2, 3) for k in args.train):
        raise ValueError("Train on distinct Sessions 1-3 only")
    if args.epochs != 8:
        raise ValueError("Research contract fixes eight epochs and averaging over 3-8")
    device = args.device
    groups = scored_samples()
    train = [s for k in args.train for s in groups[k]]
    train_set = Embeddings(train, args.train_text)
    evals = {}
    for spec in args.eval:
        session, text = spec.split(":")
        if int(session) not in (1, 2, 3, 4):
            raise ValueError("Session 5 is held out")
        if text.startswith("ctx-"):
            evals[spec] = Embeddings(unscored(int(session)), text[4:])
        else:
            evals[spec] = Embeddings(groups[int(session)], text)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = make_model(device)
    # Cached-head training never calls these encoders. Keep their weights off
    # the GPU to bound four-worker MPS memory without changing head arithmetic.
    model.text_encoder.to("cpu")
    model.audio_encoder.to("cpu")
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    loader = DataLoader(train_set, batch_size=16, shuffle=True, collate_fn=collate_embeddings,
                        generator=torch.Generator().manual_seed(args.seed))
    history, snapshots = {k: [] for k in evals}, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.perf_counter()
        for text, tm, audio, am, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model.forward_from_embeddings(text.to(device), audio.to(device), tm.to(device), am.to(device)).logits
            criterion(logits, labels.to(device)).backward()
            optimizer.step()
        if not args.final_only_eval:
            for k, ds in evals.items():
                history[k].append(predict(model, ds, device))
        state = model.state_dict()
        snapshots.append({n: state[n].detach().cpu().clone() for n in trainable})
        print(f"{args.tag} epoch {epoch} ({time.perf_counter() - started:.0f}s)", flush=True)
    window = snapshots[2:]  # Exactly the preregistered epochs 3-8.
    model.load_state_dict({n: torch.stack([s[n] for s in window]).mean(0) for n in trainable}, strict=False)
    swa = {f"swa3::{k}": predict(model, ds, device) for k, ds in evals.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "runs").mkdir(exist_ok=True)
    temporary = result_path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle,
             **{f"probs::{k}": np.stack(v) for k, v in history.items() if v},
             **{f"ids::{k}": np.array([s.utterance_id for s in evals[k].samples]) for k in evals}, **swa)
    temporary.replace(result_path)
    provenance = {"tag": args.tag, "seed": args.seed, "train_sessions": args.train,
                  "train_text": args.train_text, "train_count": len(train_set),
                  "train_utterance_ids": [s.utterance_id for s in train_set.samples],
                  "eval_counts": {key: len(dataset) for key, dataset in evals.items()},
                  "epochs": args.epochs, "weight_average_epochs": [3, 8],
                  "metadata_sha256": sha256(DEFAULT_METADATA), "config_sha256": sha256(ROOT / "configs/mvp_day3_baseline.json"),
                  "trainer_sha256": sha256(Path(__file__)), "device": device,
                  "test_evaluated": False, "text_cache_contract_sha256": sha256(text_dir(args.train_text) / "contract.json")
                  if (text_dir(args.train_text) / "contract.json").exists() else None,
                  "final_only_eval": args.final_only_eval, "predictions_sha256": sha256(result_path)}
    if args.save_weights:
        checkpoint = OUT / "checkpoints" / f"{args.tag}.pt"
        if checkpoint.exists():
            raise FileExistsError(checkpoint)
        save_checkpoint(checkpoint, model, preprocessing=PreprocessingConfig(), provenance=provenance, include_encoders=False)
        provenance["checkpoint_sha256"] = sha256(checkpoint)
    write_json(result_path.with_suffix(".json"), provenance)
    print(f"{args.tag} done", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run",))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--train", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--train-text", default="reference")
    parser.add_argument("--eval", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--final-only-eval", action="store_true", help="Only score the frozen averaged head")
    parser.add_argument("--save-weights", action="store_true")
    a = parser.parse_args()
    assert all(k in (1, 2, 3) for k in a.train), "train on Sessions 1-3 only"
    torch.set_num_threads(2)
    run(a)
