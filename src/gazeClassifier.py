import cv2
import time
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions

# MediaPipe Landmark Constants

# Iris
RIGHT_IRIS = [474, 475, 476, 477]
LEFT_IRIS = [469, 470, 471, 472]

# Eyes
RIGHT_EYE = [
    362,
    382,
    381,
    380,
    374,
    373,
    390,
    249,
    263,
    466,
    388,
    387,
    386,
    385,
    384,
    398,
    463,
    341,
    256,
    252,
    253,
    254,
    339,
    255,
    359,
    467,
    260,
    259,
    257,
    258,
    286,
    414,
]
LEFT_EYE = [
    33,
    7,
    163,
    144,
    145,
    153,
    154,
    155,
    133,
    173,
    157,
    158,
    159,
    160,
    161,
    246,
    130,
    247,
    30,
    29,
    27,
    28,
    56,
    190,
    243,
    112,
    26,
    22,
    23,
    24,
    110,
    25,
]

# Eye corners
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263
LEFT_EYE_INNER = 133
LEFT_EYE_OUTER = 33

# Face bounding landmarks
FACE_TOP = 10
FACE_BOTTOM = 152
FACE_LEFT = 127
FACE_RIGHT = 356
NOSE_TIP = 4


#
# GAZE MODEL
#


class GazeClassifier:
    """
    Determines whether the person in a frame is:
      - facing the camera  (head pose: nose aligned with face center)
      - looking at camera  (eye gaze: iris centered within the eye socket)

    Both checks are normalized by face/eye size so they are robust to
    different distances from the camera.

    Output is a (2,) shape boolean vector
    [Facing, Looking]
    """

    def __init__(
        self,
        camera_w: int = 1920,
        camera_h: int = 1080,
        model_path: str = "face_landmarker.task",
        facing_threshold: float = 0.12,  # max nose-to-face-center offset / face_width
        gaze_threshold: float = 0.10,  # max iris-to-eye-center offset / eye_width
    ):
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.face_mesh = FaceLandmarker.create_from_options(options)
        self.w = camera_w
        self.h = camera_h
        self.facing_threshold = facing_threshold
        self.gaze_threshold = gaze_threshold

    # Landmark Extraction Function

    def landmarks_to_np(self, landmarks) -> np.ndarray:
        return np.array(
            [[l.x * self.w, l.y * self.h] for l in landmarks], dtype=np.float32
        )

    def extract_face_landmarks(self, frame: np.ndarray) -> np.ndarray | None:
        mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        results = self.face_mesh.detect_for_video(mp_frame, int(time.time() * 1000))
        if results.face_landmarks:
            # FIX: if there are multiple face, what to do?
            return self.landmarks_to_np(results.face_landmarks[0])
        return None

    # Geom Function

    def _face_width(self, lm: np.ndarray) -> float:
        """
        Euclidean distance between face boundry points
        """
        return max(float(np.linalg.norm(lm[FACE_LEFT] - lm[FACE_RIGHT])), 1.0)

    def _eye_width(self, lm: np.ndarray, inner: int, outer: int) -> float:
        """
        Euclidean distance between eye boundry points
        """
        return max(float(np.linalg.norm(lm[inner] - lm[outer])), 1.0)

    # Head Gaze

    def head_offset(self, lm: np.ndarray) -> float:
        """
        Normalized distance of nose tip from face center.
        0 = perfectly facing camera
        """
        face_center = (
            lm[FACE_LEFT] + lm[FACE_RIGHT] + lm[FACE_TOP] + lm[FACE_BOTTOM]
        ) / 4
        nose = lm[NOSE_TIP]
        return float(np.linalg.norm(nose - face_center)) / self._face_width(lm)

    def is_facing_camera(self, lm: np.ndarray) -> bool:
        return self.head_offset(lm) < self.facing_threshold

    # Eye Gaze

    def gaze_offset(self, lm: np.ndarray) -> float:
        """
        Average normalized iris offset from eye center across both eyes.
        0 = both irises centered (looking straight at camera).
        """
        right_eye_center = np.mean(lm[RIGHT_EYE], axis=0)
        left_eye_center = np.mean(lm[LEFT_EYE], axis=0)
        right_iris_center = np.mean(lm[RIGHT_IRIS], axis=0)
        left_iris_center = np.mean(lm[LEFT_IRIS], axis=0)

        right_offset = np.linalg.norm(
            right_iris_center - right_eye_center
        ) / self._eye_width(lm, RIGHT_EYE_INNER, RIGHT_EYE_OUTER)
        left_offset = np.linalg.norm(
            left_iris_center - left_eye_center
        ) / self._eye_width(lm, LEFT_EYE_INNER, LEFT_EYE_OUTER)

        return float((right_offset + left_offset) / 2)

    def is_looking_at_camera(self, lm: np.ndarray) -> bool:
        return self.gaze_offset(lm) < self.gaze_threshold

    # API

    def classify(self, frame: np.ndarray) -> dict:
        """
        Process a single frame and return a result dict:
          {
            "face_detected": bool,
            "is_facing":     bool,   # head pose toward camera
            "is_looking":    bool,   # eye gaze toward camera
            "head_offset":   float,  # normalized (lower = more facing)
            "gaze_offset":   float,  # normalized (lower = more looking)
          }
        """
        result = {
            "face_detected": False,
            "is_facing": False,
            "is_looking": False,
            "head_offset": None,
            "gaze_offset": None,
        }

        lm = self.extract_face_landmarks(frame)
        if lm is None:
            return result

        result["face_detected"] = True
        result["head_offset"] = self.head_offset(lm)
        result["gaze_offset"] = self.gaze_offset(lm)
        result["is_facing"] = result["head_offset"] < self.facing_threshold
        result["is_looking"] = result["gaze_offset"] < self.gaze_threshold
        return result

    # Visualization Function

    def draw_overlay(self, frame: np.ndarray, lm: np.ndarray, result: dict):
        """Draw gaze arrows and status text onto the frame."""
        GREEN = (127, 255, 0)
        WHITE = (255, 255, 255)
        RED = (0, 0, 255)
        BLUE = (255, 0, 0)
        YELLOW = (0, 200, 255)

        # Eye gaze arrows (both eyes)
        for eye_ids, iris_ids, inner, outer in [
            (RIGHT_EYE, RIGHT_IRIS, RIGHT_EYE_INNER, RIGHT_EYE_OUTER),
            (LEFT_EYE, LEFT_IRIS, LEFT_EYE_INNER, LEFT_EYE_OUTER),
        ]:
            eye_center = np.mean(lm[eye_ids], axis=0).astype(int)
            iris_center = np.mean(lm[iris_ids], axis=0).astype(int)
            gaze_vec = iris_center - eye_center
            endpoint = (eye_center + gaze_vec * 10).astype(int)

            cv2.circle(frame, tuple(eye_center), radius=3, color=BLUE, thickness=2)
            cv2.arrowedLine(
                frame, tuple(eye_center), tuple(endpoint), color=WHITE, thickness=2
            )

        # Head pose line (face center and nose center)
        face_center = (
            (lm[FACE_LEFT] + lm[FACE_RIGHT] + lm[FACE_TOP] + lm[FACE_BOTTOM]) / 4
        ).astype(int)
        nose = lm[NOSE_TIP].astype(int)
        cv2.circle(frame, tuple(face_center), radius=4, color=BLUE, thickness=2)
        cv2.circle(frame, tuple(nose), radius=4, color=RED, thickness=2)
        cv2.line(frame, tuple(face_center), tuple(nose), color=WHITE, thickness=1)

        # Labels
        facing_color = GREEN if result["is_facing"] else RED
        gaze_color = GREEN if result["is_looking"] else YELLOW

        cv2.putText(
            frame,
            f"Facing: {'YES' if result['is_facing'] else 'NO'}  ({result['head_offset']:.3f})",
            (10, 30),
            cv2.FONT_HERSHEY_PLAIN,
            1.6,
            facing_color,
            2,
        )
        cv2.putText(
            frame,
            f"Looking: {'YES' if result['is_looking'] else 'NO'}  ({result['gaze_offset']:.3f})",
            (10, 55),
            cv2.FONT_HERSHEY_PLAIN,
            1.6,
            gaze_color,
            2,
        )


# Demo

if __name__ == "__main__":
    model = GazeClassifier()
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        _, frame = cap.read()
        if not _:
            continue

        frame = cv2.flip(frame, 1)
        result = model.classify(frame)

        if result["face_detected"]:
            lm = model.extract_face_landmarks(frame)
            model.draw_overlay(frame, lm, result)
        else:
            cv2.putText(
                frame,
                "No face detected",
                (10, 30),
                cv2.FONT_HERSHEY_PLAIN,
                1.6,
                (0, 0, 255),
                2,
            )

        cv2.imshow("GazeClassifier", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
