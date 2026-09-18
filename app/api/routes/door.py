import logging

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user
from app.core.database import get_db
from app.services.audit_service import AuditService
from app.services.door_service import DoorService
from app.services.unified_auth_session import (
    unified_auth_session_manager,
)


logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/api/v1/door",
    tags=["Door Control"],
)


door_service = DoorService()


@router.post("/unlock")
async def unlock_door(
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
    door_authorization: str | None = Header(
        default=None,
        alias="X-Door-Authorization",
    ),
):
    """
    Unlock the door using a valid one-time biometric
    authentication grant.

    A JWT alone is NOT sufficient for door access.
    """

    # ==================================================
    # STEP 1: REQUIRE DOOR AUTHORIZATION GRANT
    # ==================================================

    if not door_authorization:
        AuditService.log(
            db=db,
            event_type="DOOR_UNLOCK",
            success=False,
            user_id=current_user.id,
            details=(
                "Door unlock rejected: "
                "missing door authorization grant"
            ),
        )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Door authorization is required",
        )

    # ==================================================
    # STEP 2: ATOMICALLY CONSUME GRANT
    # ==================================================

    authorized, reason = (
        unified_auth_session_manager.consume_door_grant(
            grant=door_authorization,
            user_id=current_user.id,
        )
    )

    if not authorized:
        AuditService.log(
            db=db,
            event_type="DOOR_UNLOCK",
            success=False,
            user_id=current_user.id,
            details=(
                "Door unlock rejected: "
                f"{reason}"
            ),
        )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=reason,
        )

    # ==================================================
    # STEP 3: UNLOCK DOOR
    # ==================================================

    success = door_service.unlock()

    # ==================================================
    # STEP 4: AUDIT RESULT
    # ==================================================

    AuditService.log(
        db=db,
        event_type="DOOR_UNLOCK",
        success=success,
        user_id=current_user.id,
        details=(
            "Door unlocked successfully"
            if success
            else "Door controller failed to unlock"
        ),
    )

    # ==================================================
    # STEP 5: RESPONSE
    # ==================================================

    if not success:
        logger.error(
            "Door authorization was valid but "
            "door unlock failed. user_id=%s",
            current_user.id,
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Door unlock failed",
        )

    logger.info(
        "Door unlocked using one-time biometric "
        "authorization. user_id=%s",
        current_user.id,
    )

    return {
        "success": True,
        "door": "unlocked",
        "user_id": current_user.id,
    }


@router.post("/lock")
async def lock_door(
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    success = door_service.lock()

    AuditService.log(
        db=db,
        event_type="DOOR_LOCK",
        success=success,
        user_id=current_user.id,
        details=(
            "Door locked successfully"
            if success
            else "Door lock failed"
        ),
    )

    return {
        "success": success,
        "door": "locked" if success else "lock_failed",
        "user_id": current_user.id,
    }


@router.get("/status")
async def door_status(
    current_user=Depends(get_current_user),
):
    return {
        "unlocked": door_service.is_unlocked(),
        "user_id": current_user.id,
    }