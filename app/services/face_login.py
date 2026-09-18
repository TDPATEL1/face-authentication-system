import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from sqlalchemy.orm import Session

from app.models.face_template import FaceTemplate
from app.services.face_matching import FaceMatchingService

logger = logging.getLogger(__name__)


@dataclass
class _CachedEmbedding:
    embedding: np.ndarray
    expires_at: float


class FaceLoginService:
    """
    Face authentication service.

    Stored face embeddings remain encrypted in the database.
    Decrypted embeddings are cached in process memory for a short,
    bounded period to avoid repeated Fernet decryption during login.
    """

    SIMILARITY_THRESHOLD = 0.363

    # Cache safety/performance limits.
    CACHE_MAX_ENTRIES = 1000
    CACHE_TTL_SECONDS = 300.0

    def __init__(self, matcher=None):
        self.matcher = matcher or FaceMatchingService()

        # OrderedDict provides a simple bounded LRU-style cache.
        self._embedding_cache: OrderedDict[int, _CachedEmbedding] = (
            OrderedDict()
        )

        # Protect cache access when multiple requests are processed
        # concurrently by the same application worker.
        self._cache_lock = threading.RLock()

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _get_cached_embedding(self, user_id: int):
        """
        Return a cached decrypted embedding if it exists and has not expired.
        Returns None on cache miss/expiry.
        """
        now = time.monotonic()

        with self._cache_lock:
            cached = self._embedding_cache.get(user_id)

            if cached is None:
                return None

            if cached.expires_at <= now:
                self._embedding_cache.pop(user_id, None)
                return None

            # Refresh LRU position.
            self._embedding_cache.move_to_end(user_id)

            return cached.embedding

    def _cache_embedding(self, user_id: int, embedding: np.ndarray) -> None:
        """
        Store a decrypted embedding in the bounded cache.
        """
        if not isinstance(user_id, int) or user_id <= 0:
            return

        embedding = np.asarray(embedding, dtype=np.float32)

        # Store a private copy so callers cannot accidentally mutate
        # the cached array.
        embedding = embedding.copy()

        cached = _CachedEmbedding(
            embedding=embedding,
            expires_at=time.monotonic() + self.CACHE_TTL_SECONDS,
        )

        with self._cache_lock:
            self._embedding_cache[user_id] = cached
            self._embedding_cache.move_to_end(user_id)

            while len(self._embedding_cache) > self.CACHE_MAX_ENTRIES:
                self._embedding_cache.popitem(last=False)

    def invalidate_embedding(self, user_id: int) -> None:
        """
        Remove one user's decrypted embedding from the cache.

        Call this after enrollment/re-enrollment or any operation that
        changes the user's stored face template.
        """
        if not isinstance(user_id, int) or user_id <= 0:
            return

        with self._cache_lock:
            self._embedding_cache.pop(user_id, None)

    def clear_embedding_cache(self) -> None:
        """
        Clear all cached decrypted embeddings.

        Useful for security events, key rotation, shutdown handling,
        or tests.
        """
        with self._cache_lock:
            self._embedding_cache.clear()

    def _get_embedding(self, template: FaceTemplate):
        """
        Get a decrypted embedding from cache or decrypt it from the DB.

        Returns:
            np.ndarray on success
            None if the template is invalid
        """
        user_id = template.user_id

        cached_embedding = self._get_cached_embedding(user_id)

        if cached_embedding is not None:
            return cached_embedding

        try:
            embedding = self.matcher.decrypt_embedding(template.embedding)

            embedding = np.asarray(embedding, dtype=np.float32)

            if embedding.size != 128:
                raise ValueError("Invalid face embedding dimension")

            embedding = embedding.reshape(1, -1)

            self._cache_embedding(user_id, embedding)

            return embedding

        except (ValueError, TypeError) as exc:
            logger.warning(
                "Skipping invalid face template id=%s: %s",
                template.id,
                exc,
            )
            return None

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def authenticate(self, image, db: Session):
        """
        Authenticate a face against enrolled face templates.

        Returns:
            (user_id, similarity)

        If authentication fails:
            (None, best_similarity)

        If there are no templates:
            (None, -1.0)
        """
        captured_embedding = self.matcher.validate_and_extract_embedding(image)

        templates = db.query(FaceTemplate).all()

        if not templates:
            return None, -1.0

        best_user_id = None
        best_similarity = -1.0

        for template in templates:
            stored_embedding = self._get_embedding(template)

            if stored_embedding is None:
                continue

            try:
                similarity = self.matcher.compare(
                    captured_embedding,
                    stored_embedding,
                )

                # Ignore invalid/non-finite similarity values.
                if not np.isfinite(similarity):
                    logger.warning(
                        "Skipping non-finite similarity for face template id=%s",
                        template.id,
                    )
                    continue

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_user_id = template.user_id

            except (ValueError, TypeError) as exc:
                logger.warning(
                    "Skipping face comparison for template id=%s: %s",
                    template.id,
                    exc,
                )

        if (
            best_user_id is None
            or best_similarity < self.SIMILARITY_THRESHOLD
        ):
            return None, best_similarity

        return best_user_id, best_similarity