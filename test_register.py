from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_register_user():
    payload = {
        "name": "Test User",
        "email": "test123@example.com",
        "password": "Password@123",
    }

    response = client.post(
        "/api/v1/auth/register",
        json=payload,
    )

    print("STATUS CODE:", response.status_code)
    print("RESPONSE:")
    print(response.text)

    assert response.status_code in (200, 201, 400, 409)