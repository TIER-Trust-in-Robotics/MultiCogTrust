"""Delete only this plan's temporary arrays, after successful result audits.

Raw data, earlier improvement caches, visual stores, predictions, source
manifests, logs and all saved model heads are preserved. The removal manifest
is written before deletion and records every path, checksum and byte count.
"""

import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).parent))
from harness import OUT, ROOT
from src.iemocap_embedding_cache import sha256, write_json


def main():
    if (OUT / "cleanup_complete.json").exists():
        raise FileExistsError("Cleanup was already completed; preserving its manifest")
    result = json.loads((OUT / "result_audit.json").read_text())
    face = json.loads((OUT / "stretch_face_audit.json").read_text())
    assert result["passed"] and face["passed"]
    assert len({row["tag"] for row in result["checkpoint_replays"]}) == 32
    assert len(face["heads"]) == 8
    for name in ("phase2_job.json", "precision_control_job.json"):
        job = json.loads((OUT / name).read_text())
        assert job["status"] == "finished"
    processes = subprocess.run(["ps", "-axo", "command="], capture_output=True, text=True, check=True).stdout.splitlines()
    for command in processes:
        assert not any(f"research/overnight/{name}.py" in command for name in ("harness", "dimensions", "run_phase2", "precision_control", "stretch_face", "audit_results", "audit_face")), command
    sources = [f"{variant}{mode}" for variant in ("other", "ownother", "own2", "causalother", "causalownother", "causalown2")
               for mode in ("ref", "asr", "hri") if not (variant.endswith("own2") and mode == "hri")]
    sources += ["prev16ref", "prev16asr"]
    paths = [path for source in sources for path in (OUT / f"text_{source}").glob("*.npy")]
    paths += [OUT / "msp_dimension_features.npy", OUT / "iemocap_dimension_features.npz"]
    assert all(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(OUT.resolve()) for path in paths)
    before = sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file())
    assert before <= 5_000_000_000, "Research output exceeds the declared approximate 5GB cap"
    removed = [{"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in paths]
    manifest = {"files": removed, "bytes": sum(row["bytes"] for row in removed), "total_research_bytes_before": before,
                "result_audit_sha256": sha256(OUT / "result_audit.json"), "face_audit_sha256": sha256(OUT / "stretch_face_audit.json"),
                "rebuild": {"causal_text": "uv run python research/overnight/encode_causal_context.py",
                            "precision_control_text": "Deterministic float16 conversion of original improve/text_prevref and text_prevasr; source/array hashes retained in each text_prev16 manifest.",
                            "dimension_features": "prepare_iemocap() and prepare_msp() in research/overnight/dimensions.py; native data and encoder assets remain local.",
                            "obsolete_context": "Historical noncausal inputs are intentionally not regenerated; generator is retained as archive/encode_text.py.txt."},
                "preserved": ["Raw datasets", "Original improvement and encoder caches", "Original visual feature stores", "All prediction arrays and saved heads", "Manifests, contracts, logs, summaries and verification reports"]}
    write_json(OUT / "cleanup_manifest.json", manifest)
    for path in paths:
        path.unlink()
    assert all(not (ROOT / row["path"]).exists() for row in removed)
    after = sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file())
    write_json(OUT / "cleanup_complete.json", {"passed": True, "removed_files": len(removed), "removed_bytes": manifest["bytes"],
               "remaining_research_bytes": after, "manifest_sha256": sha256(OUT / "cleanup_manifest.json"), "source_sha256": sha256(Path(__file__))})
    print(f"Removed {len(removed)} temporary arrays, {manifest['bytes'] / 1e9:.3f} GB; retained {after / 1e9:.3f} GB", flush=True)


if __name__ == "__main__":
    main()
