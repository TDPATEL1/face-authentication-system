import time

import numpy as np

from app.services.face_matching import FaceMatchingService


def test_face_comparison_performance():
    matcher = FaceMatchingService()

    captured_embedding = np.random.rand(1, 128).astype(np.float32)
    stored_embeddings = [
        np.random.rand(1, 128).astype(np.float32)
        for _ in range(1000)
    ]

    # Warm-up
    matcher.compare(captured_embedding, stored_embeddings[0])

    measurements = []

    for _ in range(20):
        start = time.perf_counter()

        best_similarity = -1.0

        for stored_embedding in stored_embeddings:
            similarity = matcher.compare(
                captured_embedding,
                stored_embedding,
            )

            if similarity > best_similarity:
                best_similarity = similarity

        elapsed = (time.perf_counter() - start) * 1000
        measurements.append(elapsed)

    average_ms = sum(measurements) / len(measurements)

    print("\nFace comparison benchmark:")
    print(f"Embeddings compared: {len(stored_embeddings)}")
    print(f"Average comparison loop: {average_ms:.3f} ms")
    print(f"Comparisons per second: {1000 / average_ms:.2f}")

    assert best_similarity >= -1.0