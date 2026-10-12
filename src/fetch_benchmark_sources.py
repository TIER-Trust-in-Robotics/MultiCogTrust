"""Download checksum-pinned protocol references as data; never execute them."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "configs/benchmark_sources.json").read_text())
    for source in manifest["sources"]:
        target = ROOT / source["path"]
        if target.exists():
            data = target.read_bytes()
        else:
            request = urllib.request.Request(source["url"], headers={"User-Agent": "MultiCogTrust-benchmark-audit"})
            with urllib.request.urlopen(request, timeout=30) as response:
                data = response.read()
        if hashlib.sha256(data).hexdigest() != source["sha256"]:
            raise ValueError(f"Source checksum mismatch: {source['name']}")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        print(f"Verified {source['name']} at {source['commit']}")


if __name__ == "__main__":
    main()
