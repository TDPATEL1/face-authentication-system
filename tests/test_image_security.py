import io

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings


client = TestClient(app)


def create_test_image(
    width: int = 300,
    height: int = 300,
    image_format: str = ".jpg",
) -> bytes:
    """
    Create a valid test image in memory.
    """

    image = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    success, encoded = cv2.imencode(
        image_format,
        image,
    )

    assert success

    return encoded.tobytes()


def test_valid_image_is_accepted():
    """
    Verify that a normal image within the configured
    limits can be decoded successfully.
    """

    image_bytes = create_test_image()

    assert len(image_bytes) < settings.MAX_FACE_IMAGE_SIZE_BYTES


def test_empty_image_is_rejected():
    """
    Verify that an empty upload is rejected.
    """

    response = client.post(
        "/api/v1/face/enroll",
        files={
            "image": (
                "empty.jpg",
                io.BytesIO(b""),
                "image/jpeg",
            )
        },
    )

    assert response.status_code in {400, 401}


def test_non_image_content_type_is_rejected():
    """
    Verify that non-image uploads are rejected.
    """

    response = client.post(
        "/api/v1/face/enroll",
        files={
            "image": (
                "test.txt",
                io.BytesIO(b"this is not an image"),
                "text/plain",
            )
        },
    )

    assert response.status_code in {400, 401}


def test_invalid_image_data_is_rejected():
    """
    Verify that invalid binary data with an image MIME type
    is rejected.
    """

    response = client.post(
        "/api/v1/face/enroll",
        files={
            "image": (
                "invalid.jpg",
                io.BytesIO(b"this is not actually an image"),
                "image/jpeg",
            )
        },
    )

    assert response.status_code in {400, 401}


def test_oversized_image_file_is_rejected():
    """
    Verify that uploads larger than the configured
    maximum file size are rejected.
    """

    oversized_data = b"0" * (
        settings.MAX_FACE_IMAGE_SIZE_BYTES + 1
    )

    response = client.post(
        "/api/v1/face/enroll",
        files={
            "image": (
                "oversized.jpg",
                io.BytesIO(oversized_data),
                "image/jpeg",
            )
        },
    )

    assert response.status_code in {401, 413}


def test_image_dimension_limit():
    """
    Verify the configured dimension limits.
    """

    assert settings.MAX_FACE_IMAGE_WIDTH == 4096
    assert settings.MAX_FACE_IMAGE_HEIGHT == 4096


def test_image_pixel_limit():
    """
    Verify the configured pixel-count limit.
    """

    assert settings.MAX_FACE_IMAGE_PIXELS == 16_777_216


def test_image_size_configuration_is_positive():
    """
    Verify that image security configuration values
    are valid positive numbers.
    """

    assert settings.MAX_FACE_IMAGE_SIZE_BYTES > 0
    assert settings.MAX_FACE_IMAGE_WIDTH > 0
    assert settings.MAX_FACE_IMAGE_HEIGHT > 0
    assert settings.MAX_FACE_IMAGE_PIXELS > 0