import logging
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


class AntiSpoofingService:
    """
    MiniFASNetV2 ONNX anti-spoofing service.

    Input:
        Face image

    Output:
        is_real
        confidence
        class_id
        probabilities
    """

    MODEL_PATH = (
        Path(__file__).resolve().parents[2]
        / "models"
        / "liveness"
        / "MiniFASNetV2.onnx"
    )

    INPUT_SIZE = (80, 80)

    # MiniFASNetV2 from the selected model has 3 output classes.
    #
    # We intentionally do NOT guess the class mapping here.
    # The first standalone test will expose the raw probabilities
    # so we can verify the mapping before connecting this service
    # to authentication.
    REAL_CLASS_ID = 1

    def __init__(self):
        if not self.MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Anti-spoofing model not found: {self.MODEL_PATH}"
            )

        logger.info(
            "Loading anti-spoofing model: %s",
            self.MODEL_PATH,
        )

        self.session = ort.InferenceSession(
            str(self.MODEL_PATH),
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        input_shape = self.session.get_inputs()[0].shape
        output_shape = self.session.get_outputs()[0].shape

        logger.info(
            "Anti-spoofing model loaded successfully"
        )
        logger.info(
            "Anti-spoofing input: %s",
            input_shape,
        )
        logger.info(
            "Anti-spoofing output: %s",
            output_shape,
        )

    @staticmethod
    def _crop_face(
        frame: np.ndarray,
        face,
        scale: float = 2.7,
    ) -> np.ndarray | None:
        """
        Crop and enlarge the detected face before
        sending it to MiniFASNetV2.

        The selected MiniFASNetV2 reference uses a
        crop scale of approximately 2.7.
        """

        if frame is None or frame.size == 0:
            return None

        try:
            x = float(face[0])
            y = float(face[1])
            w = float(face[2])
            h = float(face[3])
        except (TypeError, ValueError, IndexError):
            return None

        if w <= 0 or h <= 0:
            return None

        frame_height, frame_width = frame.shape[:2]

        center_x = x + (w / 2.0)
        center_y = y + (h / 2.0)

        crop_size = max(w, h) * scale

        x1 = int(center_x - crop_size / 2.0)
        y1 = int(center_y - crop_size / 2.0)
        x2 = int(center_x + crop_size / 2.0)
        y2 = int(center_y + crop_size / 2.0)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_width, x2)
        y2 = min(frame_height, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size == 0:
            return None

        return face_crop

    @staticmethod
    def _preprocess(face_crop: np.ndarray) -> np.ndarray:
        """
        Prepare BGR OpenCV image for MiniFASNetV2.

        The model expects:
            float32
            RGB
            80 x 80
            CHW
            batch dimension
        """

        image = cv2.resize(
            face_crop,
            AntiSpoofingService.INPUT_SIZE,
            interpolation=cv2.INTER_LINEAR,
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        image = image.astype(
            np.float32
        ) / 255.0

        image = np.transpose(
            image,
            (2, 0, 1),
        )

        image = np.expand_dims(
            image,
            axis=0,
        )

        return np.ascontiguousarray(
            image,
            dtype=np.float32,
        )

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        """
        Numerically stable softmax.
        """

        values = values.astype(
            np.float32
        )

        values = values - np.max(values)

        exp_values = np.exp(values)

        return exp_values / np.sum(exp_values)

    def predict(
        self,
        frame: np.ndarray,
        face,
    ) -> dict:
        """
        Run anti-spoofing inference on one detected face.
        """

        face_crop = self._crop_face(
            frame,
            face,
        )

        if face_crop is None:
            raise ValueError(
                "Could not create face crop for anti-spoofing"
            )

        input_tensor = self._preprocess(
            face_crop
        )

        outputs = self.session.run(
            [self.output_name],
            {
                self.input_name: input_tensor,
            },
        )

        raw_output = outputs[0][0]

        probabilities = self._softmax(
            raw_output
        )

        predicted_class = int(
            np.argmax(probabilities)
        )

        confidence = float(
            probabilities[predicted_class]
        )

        is_real = (
            predicted_class
            == self.REAL_CLASS_ID
        )

        return {
            "is_real": is_real,
            "confidence": confidence,
            "class_id": predicted_class,
            "probabilities": [
                float(value)
                for value in probabilities
            ],
        }


anti_spoofing_service = AntiSpoofingService()