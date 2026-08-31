from fastapi import APIRouter

from app.services.door_service import DoorService


router = APIRouter(
    prefix="/api/v1/door",
    tags=["Door Control"],
)


# Single shared door service instance
door_service = DoorService()


@router.post("/unlock")
async def unlock_door():
    success = door_service.unlock()

    return {
        "success": success,
        "door": "unlocked" if success else "unlock_failed",
    }


@router.post("/lock")
async def lock_door():
    success = door_service.lock()

    return {
        "success": success,
        "door": "locked" if success else "lock_failed",
    }


@router.get("/status")
async def door_status():
    return {
        "unlocked": door_service.is_unlocked(),
    }