import cv2
import numpy as np
from cryptography.fernet import Fernet

from app.core.config import settings
from face_service.detector import FaceDetector
from face_service.embedding import FaceEmbedder


class FaceMatchingService:

    # Configurable Quality Thresholds
    MIN_FACE_WIDTH: int = 60
    MIN_FACE_HEIGHT: int = 60
    MIN_BLUR_SCORE: float = 50.0       # Laplacian variance on aligned face
    MIN_BRIGHTNESS: float = 35.0       # Mean luminance [0-255]
    MAX_BRIGHTNESS: float = 230.0      # Overexposure upper bound [0-255]

    def __init__(
        self,
        detector: FaceDetector | None = None,
        embedder: FaceEmbedder | None = None,
    ):
        self.detector = detector or FaceDetector()
        self.embedder = embedder or FaceEmbedder()

        key = settings.FACE_TEMPLATE_ENCRYPTION_KEY
        if not key:
            raise RuntimeError(
                "FACE_TEMPLATE_ENCRYPTION_KEY is not configured"
            )

        try:
            self.fernet = Fernet(key.encode())
        except Exception as exc:
            raise RuntimeError(
                "FACE_TEMPLATE_ENCRYPTION_KEY is invalid"
            ) from exc

    def validate_and_extract_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Detect face, validate image quality (size, blur, lighting), align,
        and generate L2-normalized 128-D SFace embedding.
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid or empty image")

        # 1. Face detection
        faces = self.detector.detect(image)

        if len(faces) == 0:
            raise ValueError("No face detected. Please look directly at the camera.")

        if len(faces) > 1:
            raise ValueError(
                "Multiple faces detected. Please make sure only one person is visible."
            )

        face = faces[0]
        w, h = float(face[2]), float(face[3])

        # 2. Minimum face size check
        if w < self.MIN_FACE_WIDTH or h < self.MIN_FACE_HEIGHT:
            raise ValueError(
                "Face is too small. Please move closer to the camera."
            )

        # 3. Align & crop face to 112x112
        aligned_face = self.embedder.align_crop(image, face)

        if aligned_face is None or aligned_face.size == 0:
            raise ValueError("Face alignment failed. Please face the camera.")

        # 4. Blur & lighting checks on the cropped aligned face
        gray = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)

        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if blur_score < self.MIN_BLUR_SCORE:
            raise ValueError("Image is too blurry. Please hold steady.")

        brightness = float(np.mean(gray))
        if brightness < self.MIN_BRIGHTNESS:
            raise ValueError("Image is too dark. Please improve lighting.")

        if brightness > self.MAX_BRIGHTNESS:
            raise ValueError("Image is overexposed. Please adjust lighting.")

        # 5. Extract SFace feature & L2-normalize
        feature = self.embedder.feature(aligned_face).astype(np.float32)

        norm = np.linalg.norm(feature)
        if norm == 0:
            raise ValueError("Invalid face embedding generated")

        normalized_feature = (feature / norm).reshape(1, -1)
        return normalized_feature

    def extract_embedding(self, image: np.ndarray) -> np.ndarray:
        """Alias for validate_and_extract_embedding."""
        return self.validate_and_extract_embedding(image)

    def average_normalized_embeddings(
        self,
        embeddings: list[np.ndarray],
    ) -> np.ndarray:
        """
        Average multiple valid embeddings and L2-normalize the resulting vector.
        """
        if not embeddings:
            raise ValueError("No valid embeddings provided for averaging")

        stacked = np.vstack(embeddings).astype(np.float32)
        avg = np.mean(stacked, axis=0, keepdims=True)

        norm = np.linalg.norm(avg)
        if norm == 0:
            raise ValueError("Invalid averaged embedding")

        return (avg / norm).reshape(1, -1)

    def encrypt_embedding(self, embedding: np.ndarray) -> bytes:
        """Encrypt an embedding array into Fernet ciphertext bytes."""
        embedding_bytes = embedding.astype(np.float32).tobytes()
        return self.fernet.encrypt(embedding_bytes)

    def decrypt_embedding(
        self,
        encrypted_embedding: bytes,
    ) -> np.ndarray:
        """
        Decrypt a stored face embedding and verify L2 normalization.
        """
        try:
            embedding_bytes = self.fernet.decrypt(
                encrypted_embedding
            )
        except Exception as exc:
            raise ValueError(
                "Could not decrypt stored face template"
            ) from exc

        embedding = np.frombuffer(
            embedding_bytes,
            dtype=np.float32
        )

        if embedding.size != 128:
            raise ValueError(
                f"Stored face template has invalid dimension: {embedding.size}"
            )

        embedding = embedding.reshape(1, -1)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def compare(
        self,
        captured_embedding: np.ndarray,
        stored_embedding: np.ndarray,
    ) -> float:
        """
        Compare two SFace embeddings using cosine similarity.
        Returns a similarity score in [-1.0, 1.0].
        """
        captured = captured_embedding.reshape(1, -1).astype(np.float32)
        stored = stored_embedding.reshape(1, -1).astype(np.float32)

        if captured.shape != stored.shape:
            raise ValueError(
                "Face embedding dimensions do not match"
            )

        captured_norm = np.linalg.norm(captured)
        stored_norm = np.linalg.norm(stored)

        if captured_norm == 0 or stored_norm == 0:
            raise ValueError(
                "Invalid face embedding norm"
            )

        similarity = float(
            np.dot(captured, stored.T)[0][0]
            / (captured_norm * stored_norm)
        )

        return similarity