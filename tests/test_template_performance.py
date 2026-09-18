import statistics
import time

import numpy as np

from app.services.face_matching import FaceMatchingService


ITERATIONS = 1000


def benchmark(name, func, iterations=ITERATIONS):
    times = []

    for _ in range(20):
        func()

    for _ in range(iterations):
        start = time.perf_counter()
        func()
        times.append((time.perf_counter() - start) * 1000)

    average = statistics.mean(times)
    median = statistics.median(times)

    print(f"\n{name}")
    print(f"  Average : {average:.4f} ms")
    print(f"  Median  : {median:.4f} ms")
    print(f"  Min     : {min(times):.4f} ms")
    print(f"  Max     : {max(times):.4f} ms")

    return times


def test_template_operations_performance():
    matcher = FaceMatchingService()

    embedding = np.random.default_rng(42).random(
        (1, 128),
        dtype=np.float32,
    )

    embedding /= np.linalg.norm(embedding)

    encrypted = matcher.encrypt_embedding(embedding)

    print("\n" + "=" * 60)
    print("FACE TEMPLATE OPERATION BENCHMARK")
    print("=" * 60)

    decrypt_times = benchmark(
        "1. Fernet Decryption",
        lambda: matcher.decrypt_embedding(encrypted),
    )

    decrypted = matcher.decrypt_embedding(encrypted)

    compare_times = benchmark(
        "2. Embedding Comparison",
        lambda: matcher.compare(
            embedding,
            decrypted,
        ),
    )

    print("\n" + "=" * 60)
    print("PER-TEMPLATE COST")
    print("=" * 60)

    decrypt_avg = statistics.mean(decrypt_times)
    compare_avg = statistics.mean(compare_times)

    print(f"Decryption : {decrypt_avg:.4f} ms")
    print(f"Comparison : {compare_avg:.4f} ms")
    print(
        f"Combined   : "
        f"{decrypt_avg + compare_avg:.4f} ms/template"
    )

    print("=" * 60)

    assert decrypt_avg > 0
    assert compare_avg > 0