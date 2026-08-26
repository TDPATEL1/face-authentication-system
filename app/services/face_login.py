import logging
import numpy as np
from sqlalchemy.orm import Session

from app.models.face_template import FaceTemplate
from app.services.face_matching import FaceMatchingService

logger = logging.getLogger(__name__)


class FaceLoginService:

    # Calibrated OpenCV SFace baseline threshold
    SIMILARITY_THRESHOLD: float = 0.363

    def __init__(self, matcher: FaceMatchingService | None = None):
        self.matcher = matcher or FaceMatchingService()

    def authenticate(
        self,
        image: np.ndarray,
        db: Session,
    ) -> tuple[int | None, float]:
        """
        Identify the user from a captured face after quality validation.
        Returns (best_user_id, best_similarity).
        """
        # Validate quality & generate embedding from captured image
        captured_embedding = self.matcher.validate_and_extract_embedding(
            image
        )

        # Get all enrolled face templates
        templates = (
            db.query(FaceTemplate)
            .all()
        )

        if not templates:
            return None, -1.0

        best_user_id = None
        best_similarity = -1.0

        for template in templates:
            try:
                stored_embedding = (
                    self.matcher.decrypt_embedding(
                        template.embedding
                    )
                )

                similarity = self.matcher.compare(
                    captured_embedding,
                    stored_embedding
                )

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_user_id = template.user_id

            except ValueError as exc:
                logger.warning(
                    "Skipping invalid face template id=%s: %s",
                    template.id,
                    exc,
                )

        if (
            best_user_id is None
            or best_similarity < self.SIMILARITY_THRESHOLD
        ):
            return None, best_similarity

        return best_user_id, best_similarity