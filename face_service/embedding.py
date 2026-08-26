import threading
from pathlib import Path

import cv2
import numpy as np


class FaceEmbedder:

    def __init__(self):

        model_path = (
            Path(__file__).resolve().parent.parent
            / "models"
            / "face_recognition_sface_2021dec.onnx"
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"SFace model not found: {model_path}"
            )

        self._lock = threading.Lock()
        self.model = cv2.FaceRecognizerSF.create(
            str(model_path),
            ""
        )

    def align_crop(self, image: np.ndarray, face: np.ndarray) -> np.ndarray:
        with self._lock:
            return self.model.alignCrop(image, face)

    def feature(self, aligned_face: np.ndarray) -> np.ndarray:
        with self._lock:
            return self.model.feature(aligned_face)

    def extract(self, image: np.ndarray, face: np.ndarray) -> np.ndarray:
        with self._lock:
            aligned_face = self.model.alignCrop(image, face)
            return self.model.feature(aligned_face)