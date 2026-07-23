from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.iemocap_baseline_training import (  # noqa: E402
    load_metrics,
    write_findings_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run text-only, audio-only, and simple fusion IEMOCAP baselines."
    )
    parser.add_argument(
        "--aligned-dir",
        type=Path,
        default=ROOT / "data" / "iemocap_aligned_features",
    )
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--text-epochs", type=int, default=5)
    parser.add_argument("--audio-epochs", type=int, default=20)
    parser.add_argument("--fusion-epochs", type=int, default=5)
    parser.add_argument("--sliding-fusion-epochs", type=int, default=5)
    parser.add_argument("--text-batch-size", type=int, default=16)
    parser.add_argument("--audio-batch-size", type=int, default=64)
    parser.add_argument("--fusion-batch-size", type=int, default=16)
    parser.add_argument("--sliding-fusion-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "models" / "baselines",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=ROOT / "models" / "baselines" / "iemocap_baseline_findings.md",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Only regenerate the report from existing metrics JSON files.",
    )
    return parser.parse_args()


def add_common_args(command: list[str], args: argparse.Namespace) -> list[str]:
    command.extend(["--aligned-dir", str(args.aligned_dir), "--lr", str(args.lr)])
    if args.device:
        command.extend(["--device", args.device])
    if args.limit_train is not None:
        command.extend(["--limit-train", str(args.limit_train)])
    if args.limit_test is not None:
        command.extend(["--limit-test", str(args.limit_test)])
    return command


def run_training(args: argparse.Namespace) -> None:
    commands = [
        add_common_args(
            [
                sys.executable,
                str(ROOT / "src" / "train_text_baseline_iemocap.py"),
                "--model-name",
                args.model_name,
                "--epochs",
                str(args.text_epochs),
                "--batch-size",
                str(args.text_batch_size),
                "--output",
                str(args.output_dir / "text_only_distilbert.pt"),
                "--metrics-output",
                str(args.output_dir / "text_only_metrics.json"),
            ],
            args,
        ),
        add_common_args(
            [
                sys.executable,
                str(ROOT / "src" / "train_audio_baseline_iemocap.py"),
                "--epochs",
                str(args.audio_epochs),
                "--batch-size",
                str(args.audio_batch_size),
                "--output",
                str(args.output_dir / "audio_only_gru.pt"),
                "--metrics-output",
                str(args.output_dir / "audio_only_metrics.json"),
            ],
            args,
        ),
        add_common_args(
            [
                sys.executable,
                str(ROOT / "src" / "train_simple_fusion_iemocap.py"),
                "--model-name",
                args.model_name,
                "--epochs",
                str(args.fusion_epochs),
                "--batch-size",
                str(args.fusion_batch_size),
                "--output",
                str(args.output_dir / "simple_fusion.pt"),
                "--metrics-output",
                str(args.output_dir / "simple_fusion_metrics.json"),
            ],
            args,
        ),
        add_common_args(
            [
                sys.executable,
                str(ROOT / "src" / "train_sliding_frame_fusion_iemocap.py"),
                "--model-name",
                args.model_name,
                "--epochs",
                str(args.sliding_fusion_epochs),
                "--batch-size",
                str(args.sliding_fusion_batch_size),
                "--output",
                str(args.output_dir / "sliding_frame_fusion.pt"),
                "--metrics-output",
                str(args.output_dir / "sliding_frame_fusion_metrics.json"),
                "--equivalence-output",
                str(args.output_dir / "sliding_embedding_equivalence.json"),
            ],
            args,
        ),
    ]
    for command in commands:
        print("Running:", " ".join(command))
        subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    if not args.skip_training:
        run_training(args)

    metrics_paths = {
        "Text-only": args.output_dir / "text_only_metrics.json",
        "Audio-only": args.output_dir / "audio_only_metrics.json",
        "Simple fusion": args.output_dir / "simple_fusion_metrics.json",
        "Sliding-frame fusion": args.output_dir / "sliding_frame_fusion_metrics.json",
    }
    metrics_by_model = {
        name: load_metrics(path)
        for name, path in metrics_paths.items()
        if path.exists()
    }
    if not metrics_by_model:
        raise FileNotFoundError(f"No metrics files found under {args.output_dir}")
    write_findings_report(
        output_path=args.report_output,
        metrics_paths=metrics_paths,
        metrics_by_model=metrics_by_model,
    )


if __name__ == "__main__":
    main()
