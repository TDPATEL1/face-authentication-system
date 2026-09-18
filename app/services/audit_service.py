import logging
import math
import re
from typing import Optional

from sqlalchemy.orm import Session

from app.models.audit_log import AuditLog


logger = logging.getLogger(__name__)


# --------------------------------------------------
# Audit Security Configuration
# --------------------------------------------------

MAX_EVENT_TYPE_LENGTH = 50
MAX_DETAILS_LENGTH = 2000

ALLOWED_EVENT_TYPES = {
    "DOOR_UNLOCK",
    "DOOR_LOCK",
    "FACE_ENROLLMENT_FAILED",
    "FACE_ENROLLMENT_SUCCESS",
    "FACE_LOGIN_FAILED",
    "FACE_LOGIN_SUCCESS",
    "LIVENESS_FAILED",
    "ANTI_SPOOF_FAILED",
    "AUTH_SESSION_COMPLETION_FAILED",
}


# Sensitive values that must never be persisted
# inside audit details.
SENSITIVE_PATTERNS = [
    # Authorization headers, including Bearer tokens
    re.compile(
        r"authorization\s*[:=]\s*(?:bearer\s+)?\S+",
        re.IGNORECASE,
    ),

    # Standalone Bearer tokens
    re.compile(
        r"\bbearer\s+\S+",
        re.IGNORECASE,
    ),

    # Passwords
    re.compile(
        r"password\s*[:=]\s*\S+",
        re.IGNORECASE,
    ),

    # Secret keys
    re.compile(
        r"secret[_-]?key\s*[:=]\s*\S+",
        re.IGNORECASE,
    ),

    # Access tokens
    re.compile(
        r"access[_-]?token\s*[:=]\s*\S+",
        re.IGNORECASE,
    ),

    # Refresh tokens
    re.compile(
        r"refresh[_-]?token\s*[:=]\s*\S+",
        re.IGNORECASE,
    ),

    # JWT values
    re.compile(
        r"jwt\s*[:=]\s*\S+",
        re.IGNORECASE,
    ),
]


class AuditService:

    @staticmethod
    def _sanitize_details(
        details: Optional[str],
    ) -> Optional[str]:
        """
        Sanitize audit details before database storage.

        Prevents:
        - Excessively large audit entries
        - Authentication token leakage
        - Password leakage
        - Secret-key leakage
        """

        if details is None:
            return None

        if not isinstance(details, str):
            details = str(details)

        details = details.strip()

        # ------------------------------------------
        # Remove sensitive authentication values
        # ------------------------------------------

        for pattern in SENSITIVE_PATTERNS:
            details = pattern.sub(
                "[REDACTED]",
                details,
            )

        # ------------------------------------------
        # Limit audit record size
        # ------------------------------------------

        if len(details) > MAX_DETAILS_LENGTH:
            details = (
                details[:MAX_DETAILS_LENGTH]
                + "...[truncated]"
            )

        return details

    @staticmethod
    def log(
        db: Session,
        event_type: str,
        success: bool,
        user_id: Optional[int] = None,
        similarity: Optional[float] = None,
        details: Optional[str] = None,
    ) -> AuditLog:
        """
        Create a security audit log.

        Security protections:
        - Validates event type
        - Validates user ID
        - Validates similarity
        - Sanitizes sensitive details
        - Limits details size
        """

        # ------------------------------------------
        # Event type validation
        # ------------------------------------------

        if not isinstance(event_type, str):
            raise ValueError(
                "Invalid audit event type"
            )

        event_type = event_type.strip().upper()

        if not event_type:
            raise ValueError(
                "Audit event type cannot be empty"
            )

        if len(event_type) > MAX_EVENT_TYPE_LENGTH:
            raise ValueError(
                "Audit event type is too long"
            )

        if event_type not in ALLOWED_EVENT_TYPES:
            raise ValueError(
                "Unsupported audit event type"
            )

        # ------------------------------------------
        # Success validation
        # ------------------------------------------

        if not isinstance(success, bool):
            raise ValueError(
                "Audit success value must be boolean"
            )

        # ------------------------------------------
        # User ID validation
        # ------------------------------------------

        if user_id is not None:
            if not isinstance(user_id, int):
                raise ValueError(
                    "Invalid audit user ID"
                )

            if user_id <= 0:
                raise ValueError(
                    "Invalid audit user ID"
                )

        # ------------------------------------------
        # Similarity validation
        # ------------------------------------------

        normalized_similarity: Optional[str] = None

        if similarity is not None:

            if not isinstance(
                similarity,
                (int, float),
            ):
                raise ValueError(
                    "Invalid similarity value"
                )

            similarity = float(similarity)

            if not math.isfinite(similarity):
                raise ValueError(
                    "Similarity must be finite"
                )

            if not 0.0 <= similarity <= 1.0:
                raise ValueError(
                    "Similarity must be between 0 and 1"
                )

            normalized_similarity = str(
                round(similarity, 6)
            )

        # ------------------------------------------
        # Details sanitization
        # ------------------------------------------

        sanitized_details = (
            AuditService._sanitize_details(
                details
            )
        )

        # ------------------------------------------
        # Create audit record
        # ------------------------------------------

        audit_log = AuditLog(
            user_id=user_id,
            event_type=event_type,
            success=success,
            similarity=normalized_similarity,
            details=sanitized_details,
        )

        db.add(audit_log)
        db.commit()
        db.refresh(audit_log)

        logger.info(
            "Audit event: event=%s user_id=%s success=%s",
            event_type,
            user_id,
            success,
        )

        return audit_log