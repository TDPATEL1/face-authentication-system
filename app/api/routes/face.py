import logging
from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)

import cv2
import numpy as np
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user
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


# --------------------------------------------------
# Singleton Service Instances (Loaded once per process)
# --------------------------------------------------

face_matcher = FaceMatchingService()
face_enrollment_service = FaceEnrollmentService(matcher=face_matcher)
face_login_service = FaceLoginService(matcher=face_matcher)


# --------------------------------------------------
# Helper: Decode Uploaded Image Safely
# --------------------------------------------------

async def decode_upload_image(upload: UploadFile) -> np.ndarray:
    if not upload.content_type or not upload.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be an image (JPEG/PNG)",
        )

    image_bytes = await upload.read()
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded image file is empty",
        )

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None or frame.size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not decode image",
        )

    return frame


# --------------------------------------------------
# Face Enrollment
# --------------------------------------------------

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
    upload = image or file
    if upload is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image file provided in form data",
        )

    frame = await decode_upload_image(upload)

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
    except Exception as exc:
        logger.error("Face enrollment error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Face enrollment failed",
        )

    try:
        existing_template = (
            db.query(FaceTemplate)
            .filter(FaceTemplate.user_id == current_user.id)
            .first()
        )

        if existing_template:
            existing_template.embedding = encrypted_embedding
        else:
            new_template = FaceTemplate(
                user_id=current_user.id,
                embedding=encrypted_embedding,
            )
            db.add(new_template)

        db.commit()

    except SQLAlchemyError as exc:
        db.rollback()
        logger.error("Face template database error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not save face template to database",
        )

    return {
        "message": "Face enrolled successfully",
        "user_id": current_user.id,
    }


# --------------------------------------------------
# Face Login
# --------------------------------------------------

@router.post(
    "/login",
    status_code=status.HTTP_200_OK,
)
async def face_login(
    image: UploadFile | None = File(None),
    file: UploadFile | None = File(None),
    db: Session = Depends(get_db),
):
    upload = image or file
    if upload is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image file provided in form data",
        )

    frame = await decode_upload_image(upload)

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
    except Exception as exc:
        logger.error("Face login error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Face authentication failed",
        )

    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Face not recognized",
        )

    user = (
        db.query(User)
        .filter(User.id == user_id)
        .first()
    )

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    access_token = create_access_token(
        data={
            "sub": str(user.id),
            "email": user.email,
        }
    )

    return {
        "message": "Face login successful",
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "name": user.name,
            "email": user.email,
        },
        "similarity": round(similarity, 4),
    }