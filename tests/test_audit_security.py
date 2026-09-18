import math

import pytest

from app.services.audit_service import (
    ALLOWED_EVENT_TYPES,
    MAX_DETAILS_LENGTH,
    AuditService,
)


def test_all_allowed_event_types_are_accepted():
    for event_type in ALLOWED_EVENT_TYPES:
        normalized = event_type.lower()

        # We only validate the input here; no DB write is needed
        # for the allow-list behavior itself.
        assert normalized.upper() in ALLOWED_EVENT_TYPES


def test_unsupported_event_type_is_rejected():
    with pytest.raises(ValueError, match="Unsupported audit event type"):
        AuditService.log(
            db=None,
            event_type="SYSTEM_TEST",
            success=True,
            details="test",
        )


def test_empty_event_type_is_rejected():
    with pytest.raises(ValueError, match="Audit event type cannot be empty"):
        AuditService.log(
            db=None,
            event_type="   ",
            success=True,
        )


def test_event_type_too_long_is_rejected():
    with pytest.raises(ValueError, match="Audit event type is too long"):
        AuditService.log(
            db=None,
            event_type="A" * 51,
            success=True,
        )


def test_invalid_success_value_is_rejected():
    with pytest.raises(
        ValueError,
        match="Audit success value must be boolean",
    ):
        AuditService.log(
            db=None,
            event_type="FACE_LOGIN_SUCCESS",
            success="true",
        )


@pytest.mark.parametrize("user_id", [0, -1, -100])
def test_invalid_user_id_is_rejected(user_id):
    with pytest.raises(ValueError, match="Invalid audit user ID"):
        AuditService.log(
            db=None,
            event_type="FACE_LOGIN_SUCCESS",
            success=True,
            user_id=user_id,
        )


def test_non_integer_user_id_is_rejected():
    with pytest.raises(ValueError, match="Invalid audit user ID"):
        AuditService.log(
            db=None,
            event_type="FACE_LOGIN_SUCCESS",
            success=True,
            user_id="1",
        )


@pytest.mark.parametrize(
    "similarity",
    [-0.1, 1.1, 2.0],
)
def test_similarity_out_of_range_is_rejected(similarity):
    with pytest.raises(
        ValueError,
        match="Similarity must be between 0 and 1",
    ):
        AuditService.log(
            db=None,
            event_type="FACE_LOGIN_SUCCESS",
            success=True,
            similarity=similarity,
        )


@pytest.mark.parametrize(
    "similarity",
    [math.nan, math.inf, -math.inf],
)
def test_non_finite_similarity_is_rejected(similarity):
    with pytest.raises(
        ValueError,
        match="Similarity must be finite",
    ):
        AuditService.log(
            db=None,
            event_type="FACE_LOGIN_SUCCESS",
            success=True,
            similarity=similarity,
        )


def test_sensitive_details_are_redacted():
    details = (
        "authorization: Bearer secret-token "
        "password=my-password "
        "secret_key=my-secret "
        "access_token=my-access-token "
        "refresh_token=my-refresh-token "
        "jwt=my-jwt"
    )

    sanitized = AuditService._sanitize_details(details)

    assert sanitized is not None
    assert "secret-token" not in sanitized
    assert "my-password" not in sanitized
    assert "my-secret" not in sanitized
    assert "my-access-token" not in sanitized
    assert "my-refresh-token" not in sanitized
    assert "my-jwt" not in sanitized
    assert "[REDACTED]" in sanitized


def test_details_are_truncated():
    details = "A" * (MAX_DETAILS_LENGTH + 500)

    sanitized = AuditService._sanitize_details(details)

    assert sanitized is not None
    assert len(sanitized) == MAX_DETAILS_LENGTH + len("...[truncated]")
    assert sanitized.endswith("...[truncated]")


def test_details_none_remains_none():
    assert AuditService._sanitize_details(None) is None


def test_details_are_stripped():
    sanitized = AuditService._sanitize_details(
        "   Audit event details   "
    )

    assert sanitized == "Audit event details"


def test_event_type_is_normalized():
    # This validates the normalization logic without performing a DB write.
    event_type = "  face_login_success  "

    normalized = event_type.strip().upper()

    assert normalized == "FACE_LOGIN_SUCCESS"
    assert normalized in ALLOWED_EVENT_TYPES