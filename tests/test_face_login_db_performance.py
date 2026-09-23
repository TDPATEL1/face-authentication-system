import time

from app.core.database import SessionLocal
from app.models.face_template import FaceTemplate


def test_face_template_query_performance():
    db = SessionLocal()

    try:
        # Warm-up query
        db.query(FaceTemplate).all()

        measurements = []

        for _ in range(20):
            start = time.perf_counter()

            templates = db.query(FaceTemplate).all()

            elapsed = (time.perf_counter() - start) * 1000
            measurements.append(elapsed)

        average_ms = sum(measurements) / len(measurements)

        print(f"\nFaceTemplate DB query:")
        print(f"Templates loaded: {len(templates)}")
        print(f"Average query time: {average_ms:.3f} ms")

        assert templates is not None

    finally:
        db.close()