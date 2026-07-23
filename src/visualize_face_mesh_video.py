"""Visualize MediaPipe FaceLandmarker mesh tracking on a video file.

Examples:
    uv run python src/visualize_face_mesh_video.py \
        data/processed/iemocap_speaker_videos/Session1/Ses01F_impro01/Ses01F_impro01_F.mp4

    uv run python src/visualize_face_mesh_video.py \
        data/processed/iemocap_speaker_videos/Session1/Ses01F_impro01/Ses01F_impro01_F.mp4 \
        --output data/processed/face_mesh_debug/Ses01F_impro01_F_mesh.mp4 \
        --no-display --max-frames 300

Press ``q`` or ``Esc`` to quit the display window.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions


DEFAULT_OUTPUT_ROOT = Path("data/processed/face_mesh_debug")
DEFAULT_MODEL_PATH = Path("src/weights/face_landmarker.task")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw MediaPipe Face Mesh landmarks over a video for tracking QA."
    )
    parser.add_argument("video", type=Path, help="Input video path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for an annotated output video.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not open a GUI window; useful when only writing --output.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after this many frames.",
    )
    parser.add_argument(
        "--start-sec",
        type=float,
        default=0.0,
        help="Seek to this time before processing.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Display/output scale factor. Use e.g. 2.0 for small speaker crops.",
    )
    parser.add_argument(
        "--max-num-faces",
        type=int,
        default=1,
        help="Maximum number of faces to track.",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.5,
        help="MediaPipe face detection confidence threshold.",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.5,
        help="MediaPipe landmark tracking confidence threshold.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to MediaPipe face_landmarker.task.",
    )
    parser.add_argument(
        "--draw",
        choices=["all", "contours", "tesselation"],
        default="all",
        help="Landmark overlay density.",
    )
    parser.add_argument(
        "--hide-text",
        action="store_true",
        help="Do not draw frame/FPS/detection text.",
    )
    parser.add_argument(
        "--fourcc",
        default="mp4v",
        help="OpenCV fourcc for --output. Defaults to mp4v.",
    )
    return parser.parse_args()


def scaled_frame(frame, scale: float):
    if scale == 1.0:
        return frame
    if scale <= 0:
        raise ValueError("--scale must be positive")
    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)


def make_writer(output_path: Path, fps: float, frame_size: tuple[int, int], fourcc: str):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    codec = cv2.VideoWriter_fourcc(*fourcc)
    writer = cv2.VideoWriter(str(output_path), codec, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video writer: {output_path}")
    return writer


def draw_status(frame, frame_index: int, fps_now: float, face_count: int) -> None:
    color = (90, 220, 90) if face_count else (0, 0, 255)
    label = f"frame={frame_index} fps={fps_now:.1f} faces={face_count}"
    cv2.putText(
        frame,
        label,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    args = parse_args()
    video_path = args.video.expanduser()
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    model_path = args.model.expanduser()
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if args.no_display and args.output is None:
        raise ValueError("--no-display is only useful with --output")

    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    FaceLandmarksConnections = mp.tasks.vision.FaceLandmarksConnections
    VisionRunningMode = mp.tasks.vision.RunningMode
    mp_drawing = mp.tasks.vision.drawing_utils
    mp_styles = mp.tasks.vision.drawing_styles

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = source_fps if source_fps and source_fps > 0 else 30.0
    if args.start_sec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start_sec * 1000.0)

    writer = None
    window_name = f"Face Mesh: {video_path.name}"
    processed = 0
    start_ms = int(args.start_sec * 1000.0)
    previous_time = time.perf_counter()

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=args.max_num_faces,
        min_face_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    with FaceLandmarker.create_from_options(options) as face_landmarker:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            processed += 1
            frame = scaled_frame(frame, args.scale)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = start_ms + int((processed - 1) * 1000.0 / fps)
            results = face_landmarker.detect_for_video(mp_frame, timestamp_ms)

            face_count = len(results.face_landmarks or [])
            if results.face_landmarks:
                for face_landmarks in results.face_landmarks:
                    if args.draw in {"all", "tesselation"}:
                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style(),
                        )
                    if args.draw in {"all", "contours"}:
                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style(),
                        )
                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS
                            + FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_styles.get_default_face_mesh_iris_connections_style(),
                        )

            now = time.perf_counter()
            fps_now = 1.0 / max(now - previous_time, 1e-6)
            previous_time = now
            if not args.hide_text:
                draw_status(frame, processed, fps_now, face_count)

            if args.output is not None:
                if writer is None:
                    height, width = frame.shape[:2]
                    writer = make_writer(args.output, fps, (width, height), args.fourcc)
                writer.write(frame)

            if not args.no_display:
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in {ord("q"), 27}:
                    break

            if args.max_frames is not None and processed >= args.max_frames:
                break

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Wrote overlay video: {args.output}")
    if not args.no_display:
        cv2.destroyAllWindows()
    print(f"Processed {processed} frames from {video_path}")


if __name__ == "__main__":
    main()
