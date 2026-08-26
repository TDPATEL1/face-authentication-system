import threading
from pathlib import Path

import cv2
import numpy as np


class FaceDetector:

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
            (320, 320),
            score_threshold,
            nms_threshold,
            top_k,
        )

    def detect(self, frame: np.ndarray) -> np.ndarray | list:

        if frame is None or frame.size == 0:
            return []

        height, width = frame.shape[:2]

        with self._lock:
            self.detector.setInputSize((width, height))
            _, faces = self.detector.detect(frame)

        if faces is None:
            return []

        return faces