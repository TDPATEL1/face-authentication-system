import statistics
import time

import cv2
import numpy as np

from app.services.face_login import FaceLoginService
from app.services.face_matching import FaceMatchingService


IMAGE_PATH = "tests/fixtures/face.jpg"


def benchmark(name, func, iterations=10):
    times = []

    # Warm-up
    for _ in range(2):
        func()

    for _ in range(iterations):
        start = time.perf_counter()
        func()
        times.append((time.perf_counter() - start) * 1000)

    average = statistics.mean(times)
    median = statistics.median(times)

    sorted_times = sorted(times)
    p95_index = min(
        len(sorted_times) - 1,
        int(len(sorted_times) * 0.95),
    )

    p95 = sorted_times[p95_index]

    print(f"\n{name}")
    print(f"  Average : {average:.2f} ms")
    print(f"  Median  : {median:.2f} ms")
    print(f"  P95     : {p95:.2f} ms")
    print(f"  Min     : {min(times):.2f} ms")
    print(f"  Max     : {max(times):.2f} ms")

    return times


class FakeQuery:
    def __init__(self, templates):
        self.templates = templates

    def all(self):
        return self.templates


class FakeDB:
    def __init__(self, templates):
        self.templates = templates

    def query(self, model):
        return FakeQuery(self.templates)


class FakeTemplate:
    def __init__(self, template_id, user_id, embedding):
        self.id = template_id
        self.user_id = user_id
        self.embedding = embedding


def test_face_login_scaling_performance():
    image = cv2.imread(IMAGE_PATH)

    assert image is not None, "Could not load benchmark image"

    matcher = FaceMatchingService()

    # Generate one valid embedding.
    reference_embedding = matcher.validate_and_extract_embedding(image)

    # Encrypt the same valid embedding so that the benchmark
    # measures the real Fernet decryption + comparison path.
    encrypted_embedding = matcher.encrypt_embedding(
        reference_embedding
    )

    login_service = FaceLoginService(matcher=matcher)

    print("\n" + "=" * 70)
    print("FACE LOGIN SCALING BENCHMARK")
    print("=" * 70)

    print(f"Image resolution: {image.shape[1]}x{image.shape[0]}")

    for template_count in [1, 10, 100, 500, 1000]:
        templates = [
            FakeTemplate(
                template_id=index + 1,
                user_id=index + 1,
                embedding=encrypted_embedding,
            )
            for index in range(template_count)
        ]

        db = FakeDB(templates)

        times = benchmark(
            f"{template_count} enrolled templates",
            lambda: login_service.authenticate(
                image,
                db,
            ),
            iterations=5,
        )

        average = statistics.mean(times)

        print(
            f"  Throughput: "
            f"{1000 / average:.2f} logins/sec"
        )

    print("\n" + "=" * 70)
    print("SCALING BENCHMARK COMPLETE")
    print("=" * 70)