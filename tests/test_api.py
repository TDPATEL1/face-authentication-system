import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.database import SessionLocal
from app.models.user import User
from app.models.face_template import FaceTemplate
from app.services.face_matching import FaceMatchingService

client = TestClient(app)


from app.models.audit_log import AuditLog

@pytest.fixture(autouse=True)
def clean_test_users():
    db = SessionLocal()
    users = db.query(User).filter(User.email.like('%@testsuite.com')).all()
    for u in users:
        db.query(AuditLog).filter(AuditLog.user_id == u.id).delete()
        db.delete(u)
    db.commit()
    db.close()
    yield
    db = SessionLocal()
    users = db.query(User).filter(User.email.like('%@testsuite.com')).all()
    for u in users:
        db.query(AuditLog).filter(AuditLog.user_id == u.id).delete()
        db.delete(u)
    db.commit()
    db.close()


def test_register_and_duplicate():
    # 1. Successful registration
    res = client.post('/api/v1/auth/register', json={
        'name': 'Test User',
        'email': 'user1@testsuite.com',
        'password': 'StrongPassword123!'
    })
    assert res.status_code == 201
    data = res.json()
    assert data['email'] == 'user1@testsuite.com'
    assert 'password' not in data
    assert 'password_hash' not in data

    # 2. Duplicate registration
    res_dup = client.post('/api/v1/auth/register', json={
        'name': 'Test User Duplicate',
        'email': 'user1@testsuite.com',
        'password': 'StrongPassword123!'
    })
    assert res_dup.status_code == 409
    assert 'already registered' in res_dup.json()['detail'].lower()


def test_password_login_flow():
    # Register user
    client.post('/api/v1/auth/register', json={
        'name': 'Login Tester',
        'email': 'login_user@testsuite.com',
        'password': 'MyPassword123!'
    })

    # Valid login
    res_ok = client.post('/api/v1/auth/login', json={
        'email': 'login_user@testsuite.com',
        'password': 'MyPassword123!'
    })
    assert res_ok.status_code == 200
    token = res_ok.json()['access_token']
    assert token

    # Wrong password
    res_wrong = client.post('/api/v1/auth/login', json={
        'email': 'login_user@testsuite.com',
        'password': 'WrongPassword999'
    })
    assert res_wrong.status_code == 401

    # Unknown user (constant-time flow)
    res_unknown = client.post('/api/v1/auth/login', json={
        'email': 'non_existent@testsuite.com',
        'password': 'AnyPassword123!'
    })
    assert res_unknown.status_code == 401

    # Protected /me
    res_me = client.get('/api/v1/auth/me', headers={'Authorization': f'Bearer {token}'})
    assert res_me.status_code == 200
    assert res_me.json()['email'] == 'login_user@testsuite.com'


def test_protected_routes_without_token():
    res = client.get('/api/v1/auth/me')
    assert res.status_code == 403 or res.status_code == 401


def test_face_enrollment_validation():
    # Register & login
    client.post('/api/v1/auth/register', json={
        'name': 'Face Tester',
        'email': 'facetest@testsuite.com',
        'password': 'MyPassword123!'
    })
    res_login = client.post('/api/v1/auth/login', json={
        'email': 'facetest@testsuite.com',
        'password': 'MyPassword123!'
    })
    token = res_login.json()['access_token']

    # 1. Empty image
    res_empty = client.post(
        '/api/v1/face/enroll',
        headers={'Authorization': f'Bearer {token}'},
        files={'image': ('empty.jpg', b'', 'image/jpeg')}
    )
    assert res_empty.status_code == 400

    # 2. Black frame (no face)
    no_face = np.zeros((300, 300, 3), dtype=np.uint8)
    _, encoded_no_face = cv2.imencode('.jpg', no_face)
    res_no_face = client.post(
        '/api/v1/face/enroll',
        headers={'Authorization': f'Bearer {token}'},
        files={'image': ('noface.jpg', encoded_no_face.tobytes(), 'image/jpeg')}
    )
    assert res_no_face.status_code == 400
    assert 'No face detected' in res_no_face.json()['detail']


def test_face_login_invalid_session_rejected():
    no_face = np.zeros((300, 300, 3), dtype=np.uint8)

    _, encoded_no_face = cv2.imencode(
        ".jpg",
        no_face
    )

    res = client.post(
        "/api/v1/face/login?session_id=dummy_session_123",
        files={
            "image": (
                "noface.jpg",
                encoded_no_face.tobytes(),
                "image/jpeg",
            )
        },
    )

    assert res.status_code == 401

    detail = res.json()["detail"].lower()

    assert (
        "invalid" in detail
        or "expired" in detail
        or "authentication requires" in detail
    )

    # The request must be rejected before face processing.
    assert "no face detected" not in detail
