import threading

from fastapi.testclient import TestClient

from app.main import app
from app.services.unified_auth_session import unified_auth_session_manager


client = TestClient(app)


def test_concurrent_session_start_requests():
    results = []
    errors = []
    lock = threading.Lock()

    def start_session():
        try:
            response = client.post("/api/v1/face/session/start")

            with lock:
                results.append(response)

        except Exception as exc:
            with lock:
                errors.append(exc)

    threads = [
        threading.Thread(target=start_session)
        for _ in range(10)
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    assert errors == []
    assert len(results) == 10

    # Every request should receive a valid HTTP response.
    assert all(response.status_code < 500 for response in results)

    # Each successful session-start response should have a unique session ID.
    successful = [
        response
        for response in results
        if response.status_code == 200
    ]

    session_ids = []

    for response in successful:
        data = response.json()

        if "session_id" in data:
            session_ids.append(data["session_id"])

    assert len(session_ids) == len(set(session_ids))