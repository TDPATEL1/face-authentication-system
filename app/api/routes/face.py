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

from app.core.config import settings
from app.api.dependencies import get_current_user
from app.api.routes.door import door_service
from app.core.database import get_db
from app.core.security import create_access_token
from app.models.face_template import FaceTemplate
from app.models.user import User
from app.services.audit_service import AuditService
from app.services.anti_spoofing import anti_spoofing_service
from app.services.anti_spoofing_session import (
    anti_spoofing_session_manager,
)
from app.services.face_enrollment import FaceEnrollmentService
from app.services.face_login import FaceLoginService
from app.services.face_matching import FaceMatchingService
from app.services.unified_auth_session import (
    AuthState,
    unified_auth_session_manager,
)
from app.services.liveness_session import liveness_session_manager


logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/api/v1/face",
    tags=["Face Recognition"],
)


# ==================================================
# Singleton Face Services
# ==================================================

# Models are loaded once when the application starts.
# This avoids reloading YuNet/SFace/anti-spoofing
# models for every request.

face_matcher = FaceMatchingService()

face_enrollment_service = FaceEnrollmentService(
    matcher=face_matcher
)

face_login_service = FaceLoginService(
    matcher=face_matcher
)


# ==================================================
# Anti-Spoofing Configuration
# ==================================================

# MiniFASNetV2 verified REAL class.
#
# Class 2 was observed for a real live face during
# the standalone anti-spoofing test.

ANTI_SPOOF_REAL_CLASS_ID = 2


# Minimum confidence required for a REAL prediction.

ANTI_SPOOF_MIN_CONFIDENCE = 0.80


# ==================================================
# Helper: Safe Audit Logging
# ==================================================

def safe_audit_log(
    db: Session,
    event_type: str,
    success: bool,
    user_id: int | None = None,
    similarity: float | None = None,
    details: str | None = None,
) -> None:
    """
    Write an audit event without allowing an audit
    failure to break the main authentication operation.
    """

    try:
        AuditService.log(
            db=db,
            event_type=event_type,
            success=success,
            user_id=user_id,
            similarity=similarity,
            details=details,
        )

    except Exception:
        # An audit failure must never break authentication,
        # face recognition, liveness, or door control.
        logger.exception(
            "Audit logging failed: event=%s user_id=%s",
            event_type,
            user_id,
        )

        # Roll back any failed SQLAlchemy transaction so the
        # database session remains usable for the main operation.
        try:
            db.rollback()
        except Exception:
            logger.exception(
                "Audit database rollback failed"
            )


# ==================================================
# Helper: Decode Uploaded Image
# ==================================================

async def decode_upload_image(upload: UploadFile) -> np.ndarray:
    """
    Safely decode an uploaded face image.

    Security protections:
    - Validates MIME type
    - Limits compressed file size
    - Rejects empty files
    - Validates image decoding
    - Limits width and height
    - Limits total decoded pixels
    """

    # ----------------------------------------------
    # Content-Type validation
    # ----------------------------------------------

    if not upload.content_type or not upload.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be an image",
        )

    # ----------------------------------------------
    # File-size protection
    # ----------------------------------------------

    max_size = settings.MAX_FACE_IMAGE_SIZE_BYTES

    # Read one extra byte so we can detect files larger
    # than the configured limit.
    image_bytes = await upload.read(max_size + 1)

    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded image is empty",
        )

    if len(image_bytes) > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"Image file is too large. "
                f"Maximum allowed size is {max_size // (1024 * 1024)} MB"
            ),
        )

    # ----------------------------------------------
    # OpenCV decoding
    # ----------------------------------------------

    try:
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR,
        )

    except Exception as exc:
        logging.exception("Failed to decode uploaded image")

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image file",
        ) from exc

    if image is None or image.size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image file",
        )

    # ----------------------------------------------
    # Dimension protection
    # ----------------------------------------------

    height, width = image.shape[:2]

    if width > settings.MAX_FACE_IMAGE_WIDTH:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"Image width exceeds the maximum allowed "
                f"width of {settings.MAX_FACE_IMAGE_WIDTH} pixels"
            ),
        )

    if height > settings.MAX_FACE_IMAGE_HEIGHT:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"Image height exceeds the maximum allowed "
                f"height of {settings.MAX_FACE_IMAGE_HEIGHT} pixels"
            ),
        )

    # ----------------------------------------------
    # Pixel-count protection
    # ----------------------------------------------

    total_pixels = width * height

    if total_pixels > settings.MAX_FACE_IMAGE_PIXELS:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"Image resolution is too large. "
                f"Maximum allowed resolution is "
                f"{settings.MAX_FACE_IMAGE_PIXELS} pixels"
            ),
        )

    return image


def commit_frame_nonce_reservation(
    session_id: str,
    nonce: str,
) -> str:
    """Commit a processed frame's nonce reservation."""

    committed, reason, next_nonce = (
        unified_auth_session_manager
        .commit_nonce_reservation(
            session_id,
            nonce,
        )
    )

    if not committed or next_nonce is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=reason,
        )

    return next_nonce


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
    """

    upload = image or file

    if upload is None:
        safe_audit_log(
            db=db,
            event_type="FACE_ENROLLMENT_FAILED",
            success=False,
            user_id=current_user.id,
            details="No image file provided",
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image file provided in form data",
        )

    try:
        frame = await decode_upload_image(upload)

    except HTTPException as exc:
        safe_audit_log(
            db=db,
            event_type="FACE_ENROLLMENT_FAILED",
            success=False,
            user_id=current_user.id,
            details=f"Image validation failed: {exc.detail}",
        )
        raise

    try:
        encrypted_embedding = await run_in_threadpool(
            face_enrollment_service.enroll,
            frame,
        )

    except ValueError as exc:
        safe_audit_log(
            db=db,
            event_type="FACE_ENROLLMENT_FAILED",
            success=False,
            user_id=current_user.id,
            details=str(exc),
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    except Exception:
        logger.error(
            "Face enrollment error",
            exc_info=True,
        )

        safe_audit_log(
            db=db,
            event_type="FACE_ENROLLMENT_FAILED",
            success=False,
            user_id=current_user.id,
            details="Face embedding generation failed",
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Face enrollment failed",
        )

    try:
        existing_template = (
            db.query(FaceTemplate)
            .filter(
                FaceTemplate.user_id == current_user.id
            )
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
        
        face_login_service.invalidate_embedding(current_user.id)

    except SQLAlchemyError:
        db.rollback()

        logger.error(
            "Face template database error",
            exc_info=True,
        )

        safe_audit_log(
            db=db,
            event_type="FACE_ENROLLMENT_FAILED",
            success=False,
            user_id=current_user.id,
            details="Could not save face template to database",
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not save face template to database",
        )

    safe_audit_log(
        db=db,
        event_type="FACE_ENROLLMENT_SUCCESS",
        success=True,
        user_id=current_user.id,
        details="Face template enrolled successfully",
    )

    return {
        "message": "Face enrolled successfully",
        "user_id": current_user.id,
    }


# ==================================================
# UNIFIED AUTH SESSION START
# ==================================================

@router.post(
    "/session/start",
    status_code=status.HTTP_200_OK,
)
async def start_unified_auth_session():
    """
    Start a unified authentication session enforcing
    the state order:

    CREATED
        ->
    LIVENESS_PASSED
        ->
    ANTI_SPOOF_PASSED
        ->
    FACE_MATCHED
        ->
    COMPLETED
    """

    session_id = unified_auth_session_manager.create_session()

    session = unified_auth_session_manager.get_session(
        session_id
    )

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not create unified authentication session",
        )

    # Backward compatibility:
    # mirror the same detector into the legacy liveness manager.

    from app.services.liveness_session import LivenessSession

    liveness_session_manager.sessions[session_id] = LivenessSession(
        detector=session.liveness_detector,
        created_at=session.created_at,
    )

    return {
        "message": "Authentication session started",
        "session_id": session_id,
        "nonce": session.current_nonce,
        "expires_in": (
            unified_auth_session_manager.SESSION_TIMEOUT_SECONDS
        ),
        "challenge": session.liveness_detector.challenge,
        "challenge_message": (
            session.liveness_detector.challenge_message()
        ),
        "state": session.state.value,
    }


# ==================================================
# LIVENESS SESSION START
# ==================================================

@router.post(
    "/liveness/start",
    status_code=status.HTTP_200_OK,
)
async def start_liveness():
    """
    Start a new short-lived liveness verification session.

    A random directional challenge is generated for
    this session.
    """

    session_id = (
        unified_auth_session_manager.create_session()
    )

    session = (
        unified_auth_session_manager.get_session(
            session_id
        )
    )

    if session is None:
        logger.error(
            "Could not retrieve newly created "
            "unified liveness session"
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not create liveness session",
        )

    # Mirror to legacy liveness_session_manager
    # for backward compatibility.

    from app.services.liveness_session import LivenessSession

    liveness_session_manager.sessions[session_id] = LivenessSession(
        detector=session.liveness_detector,
        created_at=session.created_at,
    )

    challenge = session.liveness_detector.challenge

    challenge_message = (
        session.liveness_detector.challenge_message()
    )

    logger.info(
        "Liveness session created. "
        "session_id=%s challenge=%s",
        session_id,
        challenge,
    )

    return {
        "message": "Liveness session started",
        "session_id": session_id,
        "nonce": session.current_nonce,
        "expires_in": (
            unified_auth_session_manager
            .SESSION_TIMEOUT_SECONDS
        ),
        "challenge": challenge,
        "challenge_message": challenge_message,
        "state": session.state.value,
    }


# ==================================================
# LIVENESS FRAME
# ==================================================

@router.post(
    "/liveness/frame",
    status_code=status.HTTP_200_OK,
)
async def process_liveness_frame(
    session_id: str,
    nonce: str,
    image: UploadFile | None = File(None),
    file: UploadFile | None = File(None),
):
    """
    Process one camera frame for challenge-based
    liveness detection.

    Every frame must contain the current server-generated
    one-time nonce.

    The nonce is reserved before processing and consumed only after a
    response-producing frame operation succeeds.
    """

    # ==================================================
    # REPLAY PROTECTION
    # ==================================================

    nonce_valid, nonce_reason = (
        unified_auth_session_manager.reserve_nonce(
            session_id,
            nonce,
        )
    )

    if not nonce_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=nonce_reason,
        )

    session = (
        unified_auth_session_manager.get_session(
            session_id
        )
    )

    if session is None:
        unified_auth_session_manager.rollback_nonce_reservation(
            session_id,
            nonce,
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Liveness session is invalid or expired",
        )

    if session.state not in (
        AuthState.CREATED,
        AuthState.LIVENESS_PASSED,
    ):
        unified_auth_session_manager.rollback_nonce_reservation(
            session_id,
            nonce,
        )
        unified_auth_session_manager.invalidate_session(
            session_id
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Liveness verification cannot be run "
                f"in state {session.state.value}"
            ),
        )

    upload = image or file

    if upload is None:
        unified_auth_session_manager.rollback_nonce_reservation(
            session_id,
            nonce,
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image file provided in form data",
        )

    try:
        frame = await decode_upload_image(upload)

    except HTTPException:
        unified_auth_session_manager.rollback_nonce_reservation(
            session_id,
            nonce,
        )
        raise

    try:
        faces = await run_in_threadpool(
            face_matcher.detector.detect,
            frame,
        )

    except Exception:
        unified_auth_session_manager.rollback_nonce_reservation(
            session_id,
            nonce,
        )
        logger.error(
            "Liveness face detection error",
            exc_info=True,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not process liveness frame",
        )

    if len(faces) == 0:
        next_nonce = commit_frame_nonce_reservation(
            session_id,
            nonce,
        )

        return {
            "liveness": False,
            "status": "no_face",
            "challenge": (
                session.liveness_detector.challenge
            ),
            "challenge_message": (
                session.liveness_detector.challenge_message()
            ),
            "message": (
                "No face detected. "
                "Please look at the camera."
            ),
            "nonce": next_nonce,
        }

    if len(faces) > 1:
        next_nonce = commit_frame_nonce_reservation(
            session_id,
            nonce,
        )

        return {
            "liveness": False,
            "status": "multiple_faces",
            "challenge": (
                session.liveness_detector.challenge
            ),
            "challenge_message": (
                session.liveness_detector.challenge_message()
            ),
            "message": (
                "Multiple faces detected. "
                "Only one person should be visible."
            ),
            "nonce": next_nonce,
        }

    face = faces[0]

    try:
        result = await run_in_threadpool(
            unified_auth_session_manager.process_liveness_frame,
            session_id,
            face,
        )

    except Exception:
        unified_auth_session_manager.rollback_nonce_reservation(
            session_id,
            nonce,
        )
        logger.error(
            "Liveness state processing error",
            exc_info=True,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not process liveness frame",
        )

    if not result.get("valid"):
        unified_auth_session_manager.rollback_nonce_reservation(
            session_id,
            nonce,
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get(
                "reason",
                "Liveness check failed",
            ),
        )

    result["nonce"] = commit_frame_nonce_reservation(
        session_id,
        nonce,
    )

    return result

# ==================================================
# ANTI-SPOOFING SESSION START
# ==================================================

@router.post(
    "/anti-spoof/start",
    status_code=status.HTTP_200_OK,
)
async def start_anti_spoofing():
    """
    Legacy anti-spoofing session endpoint.

    The main frontend authentication flow now uses
    the unified authentication session.

    This endpoint is retained temporarily for backward
    compatibility.
    """

    session_id = (
        anti_spoofing_session_manager.create_session()
    )

    logger.info(
        "Legacy anti-spoofing session created. "
        "session_id=%s",
        session_id,
    )

    return {
        "message": "Anti-spoofing session started",
        "session_id": session_id,
        "expires_in": (
            anti_spoofing_session_manager
            .SESSION_TIMEOUT_SECONDS
        ),
        "required_frames": (
            anti_spoofing_session_manager
            .REQUIRED_FRAMES
        ),
    }


# ==================================================
# ANTI-SPOOFING FRAME
# ==================================================

@router.post(
    "/anti-spoof/frame",
    status_code=status.HTTP_200_OK,
)
async def process_anti_spoofing_frame(
    session_id: str,
    nonce: str,
    image: UploadFile | None = File(None),
    file: UploadFile | None = File(None),
):
    """
    Process one frame through MiniFASNetV2.

    Multiple frames are collected under the same
    unified authentication session.

    Every frame must contain the current server-generated
    one-time nonce.
    """

    # ==================================================
    # REPLAY PROTECTION
    # ==================================================

    nonce_valid, nonce_reason = (
        unified_auth_session_manager.reserve_nonce(
            session_id,
            nonce,
        )
    )

    if not nonce_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=nonce_reason,
        )

    session = (
        unified_auth_session_manager.get_session(
            session_id
        )
    )

    if session is None:
        unified_auth_session_manager.rollback_nonce_reservation(
            session_id,
            nonce,
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Anti-spoofing session "
                "is invalid or expired"
            ),
        )

    if session.state != AuthState.LIVENESS_PASSED:
        unified_auth_session_manager.rollback_nonce_reservation(
            session_id,
            nonce,
        )
        unified_auth_session_manager.invalidate_session(
            session_id
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Liveness verification must pass "
                "before anti-spoofing. "
                f"Current state: {session.state.value}"
            ),
        )

    upload = image or file

    if upload is None:
        unified_auth_session_manager.rollback_nonce_reservation(
            session_id,
            nonce,
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "No image file provided "
                "in form data"
            ),
        )

    try:
        frame = await decode_upload_image(upload)

    except HTTPException:
        unified_auth_session_manager.rollback_nonce_reservation(
            session_id,
            nonce,
        )
        raise

    # ----------------------------------------------
    # Detect face
    # ----------------------------------------------

    try:
        faces = await run_in_threadpool(
            face_matcher.detector.detect,
            frame,
        )

    except Exception:
        unified_auth_session_manager.rollback_nonce_reservation(
            session_id,
            nonce,
        )
        logger.error(
            "Anti-spoofing face detection error",
            exc_info=True,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Could not process "
                "anti-spoofing frame"
            ),
        )

    if len(faces) == 0:
        next_nonce = commit_frame_nonce_reservation(
            session_id,
            nonce,
        )

        return {
            "status": "no_face",
            "message": "No face detected.",
            "nonce": next_nonce,
        }

    if len(faces) > 1:
        next_nonce = commit_frame_nonce_reservation(
            session_id,
            nonce,
        )

        return {
            "status": "multiple_faces",
            "message": (
                "Multiple faces detected. "
                "Only one person should be visible."
            ),
            "nonce": next_nonce,
        }

    face = faces[0]

    # ----------------------------------------------
    # MiniFASNetV2 prediction
    # ----------------------------------------------

    try:
        prediction = await run_in_threadpool(
            anti_spoofing_service.predict,
            frame,
            face,
        )

    except Exception:
        unified_auth_session_manager.rollback_nonce_reservation(
            session_id,
            nonce,
        )
        logger.error(
            "Anti-spoofing prediction error",
            exc_info=True,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Anti-spoofing prediction failed"
            ),
        )

    # ----------------------------------------------
    # Store prediction in unified session
    # ----------------------------------------------

    try:
        res = await run_in_threadpool(
            unified_auth_session_manager.add_anti_spoof_prediction,
            session_id,
            prediction,
        )

    except Exception:
        unified_auth_session_manager.rollback_nonce_reservation(
            session_id,
            nonce,
        )
        logger.error(
            "Anti-spoofing state processing error",
            exc_info=True,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Anti-spoofing prediction failed",
        )

    if not res.get("valid"):
        unified_auth_session_manager.rollback_nonce_reservation(
            session_id,
            nonce,
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=res.get(
                "reason",
                "Anti-spoofing check failed",
            ),
        )

    if res.get("status") == "checking":
        next_nonce = commit_frame_nonce_reservation(
            session_id,
            nonce,
        )

        return {
            "status": "checking",
            "frame_number": res["frame_number"],
            "required_frames": res["required_frames"],
            "class_id": prediction["class_id"],
            "confidence": round(
                prediction["confidence"],
                4,
            ),
            "is_real": (
                prediction["class_id"]
                == ANTI_SPOOF_REAL_CLASS_ID
                and prediction["confidence"]
                >= ANTI_SPOOF_MIN_CONFIDENCE
            ),
            "nonce": next_nonce,
        }

    if res.get("status") == "passed":
        next_nonce = commit_frame_nonce_reservation(
            session_id,
            nonce,
        )

        return {
            "status": "passed",
            "message": (
                "Anti-spoofing verification successful"
            ),
            "nonce": next_nonce,
            **res,
        }

    next_nonce = commit_frame_nonce_reservation(
        session_id,
        nonce,
    )

    return {
        "status": "failed",
        "message": (
            "Anti-spoofing verification failed"
        ),
        "nonce": next_nonce,
        **res,
    }


# ==================================================
# FACE LOGIN
# ==================================================

@router.post(
    "/login",
    status_code=status.HTTP_200_OK,
)
async def face_login(
    session_id: str,
    nonce: str | None = None,
    image: UploadFile | None = File(None),
    file: UploadFile | None = File(None),
    db: Session = Depends(get_db),
):
    """
    Authenticate a user using face recognition.

    Required security pipeline:

        1. Validate unified authentication session
        2. Validate one-time request nonce
        3. Completed liveness challenge
        4. Completed multi-frame anti-spoofing
        5. Atomically reserve session for face matching
        6. SFace face recognition
        7. User lookup
        8. Complete and consume unified session
        9. Door unlock
        10. JWT creation

    Important:
        session validation is performed before nonce/image
        validation so invalid sessions are rejected with
        401 instead of FastAPI returning 422 for missing
        request fields.
    """

    # ==================================================
    # STEP 1: VALIDATE / RESERVE SESSION FIRST
    # ==================================================

    # This MUST happen before requiring the nonce or image.
    #
    # Otherwise FastAPI can return 422 for a missing nonce
    # before the security layer gets a chance to reject an
    # invalid or expired authentication session.

    session = unified_auth_session_manager.get_session(
        session_id
    )

    if session is None:
        safe_audit_log(
            db=db,
            event_type="FACE_LOGIN_FAILED",
            success=False,
            details="Unified authentication session is invalid or expired",
        )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication session is invalid or expired",
        )

    # ==================================================
    # STEP 2: ATOMICALLY RESERVE SESSION FOR FACE MATCH
    # ==================================================

    # This verifies that the authentication pipeline has
    # reached the required state:
    #
    # LIVENESS_PASSED
    #        ↓
    # ANTI_SPOOF_PASSED
    #        ↓
    # FACE_MATCHED / reserved for login
    #
    # The transition is atomic inside the unified
    # authentication session manager.

    valid, reason = (
        unified_auth_session_manager
        .validate_and_consume_for_login(
            session_id
        )
    )

    if not valid:
        safe_audit_log(
            db=db,
            event_type="LIVENESS_FAILED",
            success=False,
            details=(
                "Unified auth session validation failed: "
                f"{reason}"
            ),
        )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=reason,
        )

    # ==================================================
    # STEP 3: REQUIRE NONCE
    # ==================================================

    if not nonce:
        unified_auth_session_manager.invalidate_session(
            session_id
        )

        liveness_session_manager.remove_session(
            session_id
        )

        safe_audit_log(
            db=db,
            event_type="FACE_LOGIN_FAILED",
            success=False,
            details="Login request missing nonce",
        )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication nonce is required",
        )

    # ==================================================
    # STEP 4: RESERVE ONE-TIME REQUEST NONCE
    # ==================================================

    nonce_valid, nonce_reason = (
        unified_auth_session_manager.reserve_nonce(
            session_id,
            nonce,
        )
    )

    if not nonce_valid:
        unified_auth_session_manager.invalidate_session(
            session_id
        )

        liveness_session_manager.remove_session(
            session_id
        )

        safe_audit_log(
            db=db,
            event_type="FACE_LOGIN_FAILED",
            success=False,
            details=(
                "Replay protection rejected login request: "
                f"{nonce_reason}"
            ),
        )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=nonce_reason,
        )

    # ==================================================
    # STEP 5: SELECT UPLOADED IMAGE
    # ==================================================

    upload = image or file

    if upload is None:
        unified_auth_session_manager.invalidate_session(
            session_id
        )

        liveness_session_manager.remove_session(
            session_id
        )

        safe_audit_log(
            db=db,
            event_type="FACE_LOGIN_FAILED",
            success=False,
            details="No image file provided",
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "No image file provided "
                "in form data"
            ),
        )

    # ==================================================
    # STEP 6: DECODE IMAGE
    # ==================================================

    try:
        frame = await decode_upload_image(
            upload
        )

    except HTTPException as exc:
        unified_auth_session_manager.invalidate_session(
            session_id
        )

        liveness_session_manager.remove_session(
            session_id
        )

        safe_audit_log(
            db=db,
            event_type="FACE_LOGIN_FAILED",
            success=False,
            details=(
                f"Image validation failed: "
                f"{exc.detail}"
            ),
        )

        raise

    # ==================================================
    # STEP 7: DETECT FACE FOR ANTI-SPOOFING
    # ==================================================

    try:
        faces = await run_in_threadpool(
            face_matcher.detector.detect,
            frame,
        )

    except Exception:
        logger.error(
            "Anti-spoofing face detection error",
            exc_info=True,
        )

        unified_auth_session_manager.invalidate_session(
            session_id
        )

        liveness_session_manager.remove_session(
            session_id
        )

        safe_audit_log(
            db=db,
            event_type="ANTI_SPOOF_FAILED",
            success=False,
            details=(
                "Could not detect face "
                "for anti-spoofing"
            ),
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Could not process "
                "anti-spoofing verification"
            ),
        )

    # ==================================================
    # STEP 8: REQUIRE EXACTLY ONE FACE
    # ==================================================

    if len(faces) == 0:
        unified_auth_session_manager.invalidate_session(
            session_id
        )

        liveness_session_manager.remove_session(
            session_id
        )

        safe_audit_log(
            db=db,
            event_type="ANTI_SPOOF_FAILED",
            success=False,
            details="No face detected",
        )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No face detected",
        )

    if len(faces) > 1:
        unified_auth_session_manager.invalidate_session(
            session_id
        )

        liveness_session_manager.remove_session(
            session_id
        )

        safe_audit_log(
            db=db,
            event_type="ANTI_SPOOF_FAILED",
            success=False,
            details="Multiple faces detected",
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Multiple faces detected. "
                "Only one person should be visible."
            ),
        )

    face = faces[0]

    # ==================================================
    # STEP 9: MINI FASNET V2 ANTI-SPOOFING
    # ==================================================

    try:
        anti_spoof_result = (
            await run_in_threadpool(
                anti_spoofing_service.predict,
                frame,
                face,
            )
        )

    except ValueError as exc:
        logger.error(
            "Anti-spoofing validation error: %s",
            exc,
        )

        unified_auth_session_manager.invalidate_session(
            session_id
        )

        liveness_session_manager.remove_session(
            session_id
        )

        safe_audit_log(
            db=db,
            event_type="ANTI_SPOOF_FAILED",
            success=False,
            details=str(exc),
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Anti-spoofing verification "
                "could not process the face"
            ),
        )

    except Exception:
        logger.error(
            "Anti-spoofing inference error",
            exc_info=True,
        )

        unified_auth_session_manager.invalidate_session(
            session_id
        )

        liveness_session_manager.remove_session(
            session_id
        )

        safe_audit_log(
            db=db,
            event_type="ANTI_SPOOF_FAILED",
            success=False,
            details=(
                "Anti-spoofing model "
                "inference failed"
            ),
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Anti-spoofing verification failed"
            ),
        )

    # --------------------------------------------------
    # Read anti-spoofing result
    # --------------------------------------------------

    spoof_class_id = (
        anti_spoof_result["class_id"]
    )

    spoof_confidence = (
        anti_spoof_result["confidence"]
    )

    is_real = (
        spoof_class_id
        == ANTI_SPOOF_REAL_CLASS_ID
        and spoof_confidence
        >= ANTI_SPOOF_MIN_CONFIDENCE
    )

    logger.info(
        "Anti-spoofing result: "
        "class_id=%s confidence=%.4f "
        "is_real=%s probabilities=%s",
        spoof_class_id,
        spoof_confidence,
        is_real,
        anti_spoof_result["probabilities"],
    )

    # ==================================================
    # STEP 10: REJECT SPOOF
    # ==================================================

    if not is_real:
        unified_auth_session_manager.invalidate_session(
            session_id
        )

        liveness_session_manager.remove_session(
            session_id
        )

        safe_audit_log(
            db=db,
            event_type="ANTI_SPOOF_FAILED",
            success=False,
            details=(
                "Potential spoof detected. "
                f"class_id={spoof_class_id}, "
                f"confidence={spoof_confidence:.4f}, "
                f"probabilities="
                f"{anti_spoof_result['probabilities']}"
            ),
        )

        logger.warning(
            "Potential spoof detected. "
            "class_id=%s confidence=%.4f",
            spoof_class_id,
            spoof_confidence,
        )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Anti-spoofing verification failed"
            ),
        )

    # ==================================================
    # STEP 11: SFACE FACE AUTHENTICATION
    # ==================================================

    try:
        user_id, similarity = (
            await run_in_threadpool(
                face_login_service.authenticate,
                frame,
                db,
            )
        )

    except ValueError as exc:
        unified_auth_session_manager.invalidate_session(
            session_id
        )

        liveness_session_manager.remove_session(
            session_id
        )

        safe_audit_log(
            db=db,
            event_type="FACE_LOGIN_FAILED",
            success=False,
            details=str(exc),
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    except Exception:
        logger.error(
            "Face login error",
            exc_info=True,
        )

        unified_auth_session_manager.invalidate_session(
            session_id
        )

        liveness_session_manager.remove_session(
            session_id
        )

        safe_audit_log(
            db=db,
            event_type="FACE_LOGIN_FAILED",
            success=False,
            details=(
                "Face authentication "
                "processing failed"
            ),
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Face authentication failed",
        )

    # ==================================================
    # STEP 12: FACE NOT RECOGNIZED
    # ==================================================

    if user_id is None:
        logger.warning(
            "Face authentication failed. "
            "Similarity=%.4f. "
            "Invalidating authentication session.",
            similarity,
        )

        unified_auth_session_manager.invalidate_session(
            session_id
        )

        liveness_session_manager.remove_session(
            session_id
        )

        safe_audit_log(
            db=db,
            event_type="FACE_LOGIN_FAILED",
            success=False,
            similarity=similarity,
            details=(
                "Face was not recognized "
                "after anti-spoofing passed"
            ),
        )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Face not recognized",
        )

    # ==================================================
    # STEP 13: FIND AUTHENTICATED USER
    # ==================================================

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

        unified_auth_session_manager.invalidate_session(
            session_id
        )

        liveness_session_manager.remove_session(
            session_id
        )

        safe_audit_log(
            db=db,
            event_type="FACE_LOGIN_FAILED",
            success=False,
            user_id=user_id,
            similarity=similarity,
            details=(
                "Face template belongs "
                "to missing user"
            ),
        )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    # ==================================================
    # STEP 14: COMPLETE AND CONSUME SESSION
    # ==================================================

    session_completed = (
        unified_auth_session_manager.complete_session(
            session_id=session_id,
            user_id=user.id,
            similarity=similarity,
        )
    )

    if not session_completed:
        logger.warning(
            "Authentication session completion failed. "
            "Door unlock blocked. "
            "session_id=%s user_id=%s",
            session_id,
            user.id,
        )

        unified_auth_session_manager.invalidate_session(
            session_id
        )

        liveness_session_manager.remove_session(
            session_id
        )

        safe_audit_log(
            db=db,
            event_type="AUTH_SESSION_COMPLETION_FAILED",
            success=False,
            user_id=user.id,
            similarity=similarity,
            details=(
                "Unified authentication session "
                "could not be completed or was "
                "already consumed"
            ),
        )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication session is no longer valid.",
        )

    # ==================================================
    # STEP 15: CLEAN UP LEGACY LIVENESS SESSION
    # ==================================================

    liveness_session_manager.remove_session(
        session_id
    )

    # ==================================================
    # STEP 16: UNLOCK DOOR
    # ==================================================

    door_grant = unified_auth_session_manager.create_door_grant(
        user_id=user.id,
    )

    logger.info(
        "Face authentication successful. "
        "One-time door authorization grant issued "
        "for user_id=%s",
        user.id,
    )

    # ==================================================
    # STEP 17: AUDIT SUCCESSFUL FACE AUTHENTICATION
    # ==================================================

    safe_audit_log(
        db=db,
        event_type="FACE_LOGIN_SUCCESS",
        success=True,
        user_id=user.id,
        similarity=similarity,
        details=(
            "Face authentication successful; "
            "liveness passed; "
            "anti-spoofing passed; "
            f"anti_spoof_confidence={spoof_confidence:.4f}; "
            "one-time door authorization issued"
        ),
    )

    # ==================================================
    # STEP 18: CREATE JWT ACCESS TOKEN
    # ==================================================

    access_token = create_access_token(
        data={
            "sub": str(user.id),
            "email": user.email,
        }
    )

    # ==================================================
    # STEP 19: SUCCESSFUL RESPONSE
    # ==================================================

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
        "anti_spoofing": {
            "is_real": True,
            "class_id": spoof_class_id,
            "confidence": round(
                spoof_confidence,
                4,
            ),
        },
        "door": "authorization_required",
        "door_grant": door_grant,
    }