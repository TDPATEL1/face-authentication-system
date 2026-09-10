from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user
from app.core.database import get_db
from app.services.audit_service import AuditService
from app.services.door_service import DoorService

router = APIRouter(
    prefix="/api/v1/door",
    tags=["Door Control"],
)

door_service = DoorService()


@router.post("/unlock")
async def unlock_door(
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    success = door_service.unlock()

    AuditService.log(
        db=db,
        event_type="DOOR_UNLOCK",
        success=success,
        user_id=current_user.id,
        details=(
            "Door unlocked successfully"
            if success
            else "Door unlock failed"
        ),
    )

    return {
        "success": success,
        "door": "unlocked" if success else "unlock_failed",
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