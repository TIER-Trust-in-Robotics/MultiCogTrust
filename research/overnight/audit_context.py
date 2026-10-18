"""Verify actual cached context against metadata, transcript sources and hashes.

Run before deleting the temporary feature arrays. Retain manifests and this
report afterward so cleanup cannot be mistaken for missing experiment evidence.
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from transformers import AutoTokenizer

from context_sources import context_records, fit_context
from harness import CONFIG, OUT, ROOT
from src.emotion_checkpoint import EMOTION_LABEL_MAP
from src.iemocap_embedding_cache import sha256, write_json
from src.prepare_iemocap import DEFAULT_METADATA


def main():
    rows = [row for row in map(json.loads, Path(DEFAULT_METADATA).read_text().splitlines()) if row["session"] in (1, 2, 3, 4)]
    transcript_paths = [ROOT / "data/processed/mvp/asr_validation/transcripts.jsonl", ROOT / "data/processed/mvp/asr_train/transcripts_s4_unscored.jsonl"]
    asr = {row["utterance_id"]: row["text"].strip() for path in transcript_paths for row in map(json.loads, path.read_text().splitlines())}
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["text_model_name"], revision=CONFIG["text_revision"], local_files_only=True)
    reports = []
    for variant in ("other", "ownother", "own2"):
        for mode in ("ref", "asr", "hri"):
            if variant == "own2" and mode == "hri":
                continue
            source = f"causal{variant}{mode}"
            directory = OUT / f"text_{source}"
            expected = [fit_context(row, tokenizer, CONFIG["max_tokens"]) for row in context_records(rows, variant, mode, asr, EMOTION_LABEL_MAP)]
            actual = [json.loads(line) for line in (directory / "manifest.jsonl").read_text().splitlines()]
            assert len(actual) == len(expected) == (4290 if mode == "ref" else 1027)
            assert len({row["utterance_id"] for row in actual}) == len(actual)
            files = set(directory.glob("*.npy"))
            assert len(files) == len(actual)
            for reference, saved in zip(expected, actual):
                assert reference == {key: saved[key] for key in reference}
                assert all(end <= saved["start_sec"] for end in saved["context_end_sec"])
                assert not saved["current_truncated"]
                path = directory / f"{saved['utterance_id']}.npy"
                assert sha256(path) == saved["feature_sha256"]
                value = np.load(path, allow_pickle=False)
                assert value.dtype == np.float16 and value.shape == (saved["token_count_before_truncation"], 768)
                assert np.isfinite(value).all()
            reports.append({"source": source, "count": len(actual), "contract_sha256": sha256(directory / "contract.json"),
                            "manifest_sha256": sha256(directory / "manifest.jsonl"), "bytes": sum(path.stat().st_size for path in files)})
            print(f"Audited {source}: {len(actual)} causal finite feature arrays", flush=True)
    write_json(OUT / "verified_context_cache.json", {"passed": True, "sources": reports, "auditor_sha256": sha256(Path(__file__)),
               "metadata_sha256": sha256(DEFAULT_METADATA), "source_audit_sha256": sha256(OUT / "context_source_audit.json"),
               "checks": ["Exact reconstruction from development metadata and declared ref/ASR/HRI sources", "All context ends before target starts", "Expected coverage and no duplicate or extra arrays", "Every feature hash, finite value, dtype and token dimension checked"],
               "test_model_inference": False})


if __name__ == "__main__":
    main()
