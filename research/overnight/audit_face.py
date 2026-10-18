"""Verify optional face artifacts, fold boundaries and OOF policy selection."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import torch
from harness import OUT
from metrics import LABEL, load, oof, scored
from stretch_face import DIRECTORY, DEST, score
from src.iemocap_embedding_cache import sha256, write_json
from src.models.visual_temporal import fit_normalization
from src.train_mvp_visual import visual_predictions
from src.visual_feature_dataset import VisualFeatureDataset
from src.visual_fusion import load_visual_checkpoint


def main():
    summary = json.loads((OUT / "stretch_face_summary.json").read_text())
    assert summary["finished"] and summary["training_runs"] == 8
    assert sha256(OUT / "stretch_face_protocol.json") == summary["protocol_sha256"]
    assert sha256(OUT / "stretch_face_selection.json") == summary["selection_sha256"]
    assert len([name for name in summary["saved_artifact_hashes"] if name.endswith(".pt")]) == 8
    assert all(sha256(DEST / name) == fingerprint for name, fingerprint in summary["saved_artifact_hashes"].items())
    torch.set_num_threads(2)
    datasets = [VisualFeatureDataset(DIRECTORY, partition) for partition in ("train", "validation")]
    rows = {dataset.records[i]["utterance_id"]: dataset[i] for dataset in datasets for i in range(len(dataset))}
    assert all(row["label"] == LABEL[uid] for uid, row in rows.items())
    oof_face, full_face, checks = {}, {}, []
    for seed in (42, 43):
        oof_face[seed] = {}
        for held in (1, 2, 3, None):
            tag = f"loso{held}_s{seed}" if held else f"full_s{seed}"
            path = DEST / f"{tag}.pt"
            model, payload = load_visual_checkpoint(path)
            provenance = payload["provenance"]
            train_ids = scored(tuple(session for session in (1, 2, 3) if session != held))
            eval_ids = scored((held,)) if held else scored((4,))
            assert set(provenance["train_ids"]) == set(train_ids)
            assert set(provenance["eval_ids"]) == set(eval_ids)
            assert not set(train_ids) & set(eval_ids)
            assert provenance["weight_average_epochs"] == [11, 20]
            assert provenance["source_sha256"] == sha256(Path(__file__).with_name("stretch_face.py"))
            assert all(sha256(DIRECTORY / name) == fingerprint for name, fingerprint in provenance["sources"].items())
            training_rows = [rows[uid] for uid in provenance["train_ids"]]
            stats = fit_normalization(training_rows, .001)
            normalization_difference = 0.
            for field in ("blendshapes", "image_embeddings"):
                for kind in ("mean", "std"):
                    actual = getattr(model, field + "_" + kind)
                    # The fixed weight average also averages ten identical
                    # float32 normalization buffers, introducing ulp roundoff.
                    torch.testing.assert_close(actual, stats[field][kind], rtol=1e-6, atol=1e-7)
                    normalization_difference = max(normalization_difference, float((actual - stats[field][kind]).abs().max()))
            saved = np.load(path.with_suffix(".npz"), allow_pickle=False)
            ordered = [rows[uid] for uid in saved["ids"]]
            p, available = visual_predictions(model, ordered)
            np.testing.assert_array_equal(p.numpy(), saved["probabilities"])
            np.testing.assert_array_equal(available.numpy(), saved["available"])
            assert set(saved["ids"]) == set(eval_ids)
            values = {uid: (probability, bool(flag)) for uid, probability, flag in zip(saved["ids"], saved["probabilities"], saved["available"])}
            if held:
                oof_face[seed].update(values)
            else:
                full_face[seed] = values
            checks.append({"tag": tag, "n": len(values), "normalization_train_only": True,
                           "normalization_max_absolute_difference": normalization_difference, "exact_replay": True})
            print(f"Audited face {tag}", flush=True)
    selection = json.loads((OUT / "stretch_face_selection.json").read_text())
    assert selection["fit_sessions"] == [1, 2, 3]
    for candidate in selection["candidates"]:
        actual = [score(scored((1, 2, 3)), oof("prev", seed, "prevref"), oof_face[seed], candidate["policy"]) for seed in (42, 43)]
        assert candidate["runs"] == actual
        assert abs(candidate["f1_mean"] - np.mean([row["gated_f1"] for row in actual])) < 1e-12
        assert candidate["changed"] == sum(row["changed"] for row in actual)
    assert selection["selected"] == max(selection["candidates"], key=lambda row: (row["f1_mean"], -row["changed"]))
    for result in summary["session4"]:
        seed = result["seed"]
        for condition, source in (("reference", "prevref"), ("asr", "prevasr")):
            assert result[condition] == score(scored((4,)), load(f"full_prev_s{seed}", f"4:{source}"), full_face[seed], selection["selected"]["policy"])
    write_json(OUT / "stretch_face_audit.json", {"passed": True, "heads": checks, "selection_oof_only": True,
               "summary_sha256": sha256(OUT / "stretch_face_summary.json"), "auditor_sha256": sha256(Path(__file__)),
               "note": "All eight serialized heads replay exactly; normalization was recomputed from each declared training fold."})


if __name__ == "__main__":
    main()
