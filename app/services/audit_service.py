import logging
from typing import Optional

from sqlalchemy.orm import Session

from app.models.audit_log import AuditLog

logger = logging.getLogger(__name__)


class AuditService:
    @staticmethod
    def log(
        db: Session,
        event_type: str,
        success: bool,
        user_id: Optional[int] = None,
        similarity: Optional[float] = None,
        details: Optional[str] = None,
    ) -> AuditLog:
        audit_log = AuditLog(
            user_id=user_id,
            event_type=event_type,
            success=success,
            similarity=(
                str(round(similarity, 6))
                if similarity is not None
                else None
            ),
            details=details,
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
