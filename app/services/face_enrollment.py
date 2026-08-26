import numpy as np

from app.services.face_matching import FaceMatchingService


class FaceEnrollmentService:

    def __init__(self, matcher: FaceMatchingService | None = None):
        self.matcher = matcher or FaceMatchingService()

    def enroll(self, image: np.ndarray) -> bytes:
        """
        Enroll a single face image after strict quality validation.
        Returns Fernet-encrypted embedding bytes.
        """
        embedding = self.matcher.validate_and_extract_embedding(image)
        return self.matcher.encrypt_embedding(embedding)

    def enroll_multiple_frames(self, images: list[np.ndarray]) -> bytes:
        """
        Enroll multiple frames (e.g. 3 frames).
        Validates quality for each frame, averages embeddings, L2-normalizes,
        and returns Fernet-encrypted embedding bytes.
        """
        if not images:
            raise ValueError("No images provided for enrollment")

        valid_embeddings = []
        errors = []

        for idx, img in enumerate(images):
            try:
                emb = self.matcher.validate_and_extract_embedding(img)
                valid_embeddings.append(emb)
            except ValueError as e:
                errors.append(f"Frame {idx + 1}: {str(e)}")

        if not valid_embeddings:
            error_summary = "; ".join(errors) if errors else "All frames failed quality checks"
            raise ValueError(f"Face enrollment failed. {error_summary}")

        averaged_embedding = self.matcher.average_normalized_embeddings(valid_embeddings)
        return self.matcher.encrypt_embedding(averaged_embedding)