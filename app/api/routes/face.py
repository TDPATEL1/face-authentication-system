import logging

import cv2
import numpy as np
from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user
from app.api.routes.door import door_service
from app.core.database import get_db
from app.core.security import create_access_token
from app.models.face_template import FaceTemplate
from app.models.user import User
from app.services.face_enrollment import FaceEnrollmentService
from app.services.face_login import FaceLoginService
from app.services.face_matching import FaceMatchingService


logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/api/v1/face",
    tags=["Face Recognition"],
)


# ==================================================
# Singleton Face Services
# ==================================================

# Models are loaded once when the application starts.
# This avoids reloading YuNet/SFace for every request.

face_matcher = FaceMatchingService()

face_enrollment_service = FaceEnrollmentService(
    matcher=face_matcher
)

face_login_service = FaceLoginService(
    matcher=face_matcher
)


# ==================================================
# Helper: Decode Uploaded Image
# ==================================================

async def decode_upload_image(
    upload: UploadFile,
) -> np.ndarray:
    """
    Safely read and decode an uploaded image.

    Returns:
        BGR NumPy image.
    """

    # Check MIME type
    if (
        not upload.content_type
        or not upload.content_type.startswith("image/")
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be an image (JPEG/PNG)",
        )

    # Read uploaded bytes
    image_bytes = await upload.read()

    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded image file is empty",
        )

    # Convert bytes to NumPy array
    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8,
    )

    # Decode image using OpenCV
    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR,
    )

    if frame is None or frame.size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not decode image",
        )

    return frame


# ==================================================
# FACE ENROLLMENT
# ==================================================

@router.post(
    "/enroll",
    status_code=status.HTTP_201_CREATED,
)
async def enroll_face(
    image: UploadFile | None = File(None),
    file: UploadFile | None = File(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Enroll the authenticated user's face.

    The endpoint accepts either:
        image
        file

    This keeps compatibility with different frontend
    form-data field names.
    """

    # ----------------------------------------------
    # Select uploaded image
    # ----------------------------------------------

    upload = image or file

    if upload is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image file provided in form data",
        )

    # ----------------------------------------------
    # Decode image
    # ----------------------------------------------

    frame = await decode_upload_image(upload)

    # ----------------------------------------------
    # Generate encrypted face embedding
    # ----------------------------------------------

    try:
        encrypted_embedding = await run_in_threadpool(
            face_enrollment_service.enroll,
            frame,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    except Exception:
        logger.error(
            "Face enrollment error",
            exc_info=True,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Face enrollment failed",
        )

    # ----------------------------------------------
    # Save face template
    # ----------------------------------------------

    try:
        existing_template = (
            db.query(FaceTemplate)
            .filter(
                FaceTemplate.user_id == current_user.id
            )
            .first()
        )

        # Update existing template
        if existing_template:
            existing_template.embedding = encrypted_embedding

        # Create new template
        else:
            new_template = FaceTemplate(
                user_id=current_user.id,
                embedding=encrypted_embedding,
            )

            db.add(new_template)

        db.commit()

    except SQLAlchemyError:
        db.rollback()

        logger.error(
            "Face template database error",
            exc_info=True,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not save face template to database",
        )

    # ----------------------------------------------
    # Successful enrollment response
    # ----------------------------------------------

    return {
        "message": "Face enrolled successfully",
        "user_id": current_user.id,
    }


# ==================================================
# FACE LOGIN
# ==================================================

@router.post(
    "/login",
    status_code=status.HTTP_200_OK,
)
async def face_login(
    image: UploadFile | None = File(None),
    file: UploadFile | None = File(None),
    db: Session = Depends(get_db),
):
    """
    Authenticate a user using face recognition.

    Successful authentication automatically unlocks
    the simulated door for the configured duration
    (currently 5 seconds).
    """

    # ----------------------------------------------
    # Select uploaded image
    # ----------------------------------------------

    upload = image or file

    if upload is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image file provided in form data",
        )

    # ----------------------------------------------
    # Decode uploaded image
    # ----------------------------------------------

    frame = await decode_upload_image(upload)

    # ----------------------------------------------
    # Authenticate face
    # ----------------------------------------------

    try:
        user_id, similarity = await run_in_threadpool(
            face_login_service.authenticate,
            frame,
            db,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    except Exception:
        logger.error(
            "Face login error",
            exc_info=True,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Face authentication failed",
        )

    # ----------------------------------------------
    # Face was not recognized
    # ----------------------------------------------

    if user_id is None:
        logger.warning(
            "Face authentication failed. Similarity=%.4f",
            similarity,
        )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Face not recognized",
        )

    # ----------------------------------------------
    # Find authenticated user
    # ----------------------------------------------

    user = (
        db.query(User)
        .filter(User.id == user_id)
        .first()
    )

    if not user:
        logger.error(
            "Face template belongs to missing user. "
            "user_id=%s",
            user_id,
        )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    # ==================================================
    # SUCCESSFUL FACE AUTHENTICATION
    # ==================================================

    # At this point:
    #
    # 1. Face was detected
    # 2. Face passed quality checks
    # 3. Face matched a stored template
    # 4. User was found in the database
    #
    # Only NOW do we unlock the door.

    door_unlocked = door_service.unlock()

    if door_unlocked:
        logger.info(
            "Face authentication successful. "
            "Door unlocked for user_id=%s",
            user.id,
        )

    else:
        logger.error(
            "Face authentication successful but "
            "door failed to unlock. user_id=%s",
            user.id,
        )

    # ----------------------------------------------
    # Create JWT access token
    # ----------------------------------------------

    access_token = create_access_token(
        data={
            "sub": str(user.id),
            "email": user.email,
        }
    )

    # ----------------------------------------------
    # Successful response
    # ----------------------------------------------

    return {
        "message": "Face login successful",
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "name": user.name,
            "email": user.email,
        },
        "similarity": round(
            similarity,
            4,
        ),
        "door": (
            "unlocked"
            if door_unlocked
            else "unlock_failed"
        ),
    }