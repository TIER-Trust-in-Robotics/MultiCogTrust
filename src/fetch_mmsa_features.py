"""Fetch a pinned public mirror and verify the official MMSA MOSEI bytes."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.iemocap_embedding_cache import sha256
from src.mmsa_features import RELEASE_SHA256

REPO = "Josie1204/mosei"
REVISION = "638362ecac5d7858e0bcd1e4129f2a8f56a857f3"
BYTES = 13_652_131_313


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "data/MOSEI/Processed")
    args = parser.parse_args()
    path = args.directory / "unaligned_50.pkl"
    if path.exists():
        if path.stat().st_size != BYTES or sha256(path) != RELEASE_SHA256:
            raise FileExistsError("Preserving a different existing feature file; choose another directory")
        print(f"Verified existing official feature bytes: {path}")
        return
    args.directory.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.directory).free < BYTES + 8_000_000_000:
        raise OSError("Need space for the release plus the development feature export")
    downloaded = Path(hf_hub_download(REPO, "unaligned_50.pkl", repo_type="dataset", revision=REVISION, local_dir=args.directory))
    if downloaded.stat().st_size != BYTES or sha256(downloaded) != RELEASE_SHA256:
        raise ValueError("Downloaded bytes differ from the official MMSA release")
    print(f"Verified official feature bytes: {downloaded}")


if __name__ == "__main__":
    main()
