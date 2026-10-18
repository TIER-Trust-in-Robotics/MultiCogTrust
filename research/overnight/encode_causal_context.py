"""Audited replacement for phase-two caches with complete-before-start context."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from context_sources import context_records, fit_context
from harness import CONFIG, OUT
from src.emotion_checkpoint import EMOTION_LABEL_MAP
from src.iemocap_embedding_cache import sha256, write_json
from src.prepare_iemocap import DEFAULT_METADATA


def main():
    rows = [row for row in map(json.loads, Path(DEFAULT_METADATA).read_text().splitlines()) if row["session"] in (1, 2, 3, 4)]
    transcript_paths = [ROOT / "data/processed/mvp/asr_validation/transcripts.jsonl", ROOT / "data/processed/mvp/asr_train/transcripts_s4_unscored.jsonl"]
    asr = {row["utterance_id"]: row["text"].strip() for path in transcript_paths for row in map(json.loads, path.read_text().splitlines())}
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["text_model_name"], revision=CONFIG["text_revision"], local_files_only=True)
    torch.set_num_threads(2)
    model = AutoModel.from_pretrained(CONFIG["text_model_name"], revision=CONFIG["text_revision"], local_files_only=True).to("mps").eval()
    reports = []
    for variant in ("other", "ownother", "own2"):
        for mode in ("ref", "asr", "hri"):
            if variant == "own2" and mode == "hri":
                continue
            source = f"causal{variant}{mode}"
            directory = OUT / f"text_{source}"
            directory.mkdir(exist_ok=True)
            records = [fit_context(record, tokenizer, CONFIG["max_tokens"]) for record in context_records(rows, variant, mode, asr, EMOTION_LABEL_MAP)]
            contract = {"source": source, "metadata_sha256": sha256(DEFAULT_METADATA), "asr_hashes": {str(path): sha256(path) for path in transcript_paths},
                        "text_model": CONFIG["text_model_name"], "revision": CONFIG["text_revision"], "max_tokens": CONFIG["max_tokens"],
                        "dtype": "float16", "timing": "all context utterances end no later than target start", "record_count": len(records)}
            contract_path = directory / "contract.json"
            if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
                raise ValueError("Context cache contract changed")
            write_json(contract_path, contract)
            for start in range(0, len(records), 16):
                batch = [record for record in records[start:start + 16] if not (directory / f"{record['utterance_id']}.npy").exists()]
                if batch:
                    tokens = tokenizer([record["encoded_text"] for record in batch], return_tensors="pt", padding=True, truncation=True, max_length=CONFIG["max_tokens"])
                    with torch.inference_mode():
                        embeddings = model(**{name: value.to("mps") for name, value in tokens.items()}).last_hidden_state.cpu().numpy()
                    for i, record in enumerate(batch):
                        value = embeddings[i, :int(tokens["attention_mask"][i].sum())].astype(np.float16)
                        if not np.isfinite(value).all():
                            raise ValueError("Nonfinite context encoding")
                        path = directory / f"{record['utterance_id']}.npy"
                        temporary = path.with_suffix(".tmp")
                        with temporary.open("wb") as handle:
                            np.save(handle, value, allow_pickle=False)
                        temporary.replace(path)
                if start % 800 == 0:
                    print(f"{source}: {start}/{len(records)}", flush=True)
            with (directory / "manifest.jsonl").open("w") as handle:
                for record in records:
                    record["feature_sha256"] = sha256(directory / f"{record['utterance_id']}.npy")
                    handle.write(json.dumps(record) + "\n")
            reports.append({"source": source, "count": len(records), "manifest_sha256": sha256(directory / "manifest.jsonl"),
                            "dropped_context_lines": sum(record["dropped_context_lines"] for record in records),
                            "current_truncated": sum(record["current_truncated"] for record in records),
                            "samples": [{name: record[name] for name in ("utterance_id", "context_ids", "encoded_text")} for record in records[:3]]})
            print(f"{source}: complete ({len(records)})", flush=True)
    write_json(OUT / "context_source_audit.json", {"passed": True, "sources": reports, "test_model_inference": False,
               "generator_sha256": sha256(Path(__file__)), "builder_sha256": sha256(Path(__file__).with_name("context_sources.py")),
               "reason": "Earlier cache used preceding start time; 374 other-speaker utterances across the development metadata end after the target ends. New caches require completed context before target starts."})


if __name__ == "__main__":
    main()
