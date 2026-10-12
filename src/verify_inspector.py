"""Check the running localhost inspector API; browser appearance needs separate review."""

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def verify(port=8765, stop=False):
    url = f"http://127.0.0.1:{port}"
    html = urllib.request.urlopen(url, timeout=5).read().decode()
    state = json.load(urllib.request.urlopen(url + "/state", timeout=5))
    if not state.get("history") or not state.get("prediction"):
        raise ValueError(
            "Wait for a completed replay prediction before checking the inspector"
        )
    preview = urllib.request.urlopen(url + "/frame.jpg", timeout=5)
    frame = preview.read()
    if preview.status != 200 or not frame.startswith(b"\xff\xd8"):
        raise ValueError("Inspector has no JPEG preview")
    if "Visual branch" not in html or "Trust prediction requires" not in html:
        raise ValueError("Missing probability/interpretation labels")
    try:
        urllib.request.urlopen(
            urllib.request.Request(url + "/stop", method="POST"), timeout=5
        )
        raise ValueError("Stop endpoint accepted a request without the session token")
    except urllib.error.HTTPError as error:
        if error.code != 403:
            raise
    syntax = "not checked: node unavailable"
    if shutil.which("node"):
        with tempfile.TemporaryDirectory(
            prefix="multicogtrust-inspector-"
        ) as directory:
            script = Path(directory) / "inspector.js"
            script.write_text(html.split("<script>")[1].split("</script>")[0])
            subprocess.run(
                ["node", "--check", str(script)], check=True, capture_output=True
            )
        syntax = "passed"
    stopped = None
    if stop:
        token = re.search("'X-Session':'([^']+)'", html)[1]
        request = urllib.request.Request(
            url + "/stop", method="POST", headers={"X-Session": token}
        )
        stopped = urllib.request.urlopen(request, timeout=5).status
    return {
        "http_contract_passed": True,
        "state_history_count": len(state["history"]),
        "frame_responded": True,
        "unauthorized_stop": 403,
        "authorized_stop": stopped,
        "javascript_syntax": syntax,
        "visual_appearance_verified": False,
        "browser_review": "Separate browser appearance review is required; API checks do not establish visual quality.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop this inspector session after checking it",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/processed/mvp/day5_7/inspector_verification.json",
    )
    args = parser.parse_args()
    result = verify(args.port, args.stop)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
