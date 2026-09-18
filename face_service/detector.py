import threading
from pathlib import Path

import cv2
import numpy as np


class FaceDetector:

    # Detection is performed on a smaller working image.
    # Bounding boxes are mapped back to the original image.
    DETECTION_SIZE = 640

    def __init__(
        self,
        score_threshold: float = 0.7,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ):
        model_path = (
            Path(__file__).resolve().parent.parent
            / "models"
            / "face_detection_yunet_2023mar.onnx"
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"YuNet model not found: {model_path}"
            )

        self._lock = threading.Lock()

        self.detector = cv2.FaceDetectorYN.create(
            str(model_path),
            "",
            (self.DETECTION_SIZE, self.DETECTION_SIZE),
            score_threshold,
            nms_threshold,
            top_k,
        )

    def detect(self, frame: np.ndarray) -> np.ndarray | list:

        if frame is None or frame.size == 0:
            return []

        original_height, original_width = frame.shape[:2]

        if original_height <= 0 or original_width <= 0:
            return []

        # Keep the original aspect ratio.
        scale = min(
            self.DETECTION_SIZE / original_width,
            self.DETECTION_SIZE / original_height,
        )

        new_width = max(1, int(round(original_width * scale)))
        new_height = max(1, int(round(original_height * scale)))

        # Resize only for detection.
        resized = cv2.resize(
            frame,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA,
        )

        with self._lock:
            self.detector.setInputSize((new_width, new_height))
            _, faces = self.detector.detect(resized)

        if faces is None:
            return []

        faces = np.asarray(faces, dtype=np.float32).copy()

        if faces.size == 0:
            return []

        # YuNet face format:
        #
        # [x, y, width, height, ...]
        #
        # Convert coordinates back to the original image.
        inverse_scale = 1.0 / scale

        faces[:, 0] *= inverse_scale
        faces[:, 1] *= inverse_scale
        faces[:, 2] *= inverse_scale
        faces[:, 3] *= inverse_scale

        # YuNet also returns five landmark pairs:
        #
        # right eye
        # left eye
        # nose
        # right mouth
        # left mouth
        #
        # Their coordinates also need to be mapped back.
        faces[:, 5:15] *= inverse_scale

        return faces