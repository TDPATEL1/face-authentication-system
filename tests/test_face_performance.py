import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from app.services.face_matching import FaceMatchingService


IMAGE_PATH = Path(__file__).parent / "fixtures" / "face.jpg"
ITERATIONS = 20


def benchmark(name, func, iterations=ITERATIONS):
    times = []

    # Warm-up
    for _ in range(3):
        func()

    for _ in range(iterations):
        start = time.perf_counter()
        func()
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)

    times.sort()

    p50 = statistics.median(times)
    p95_index = min(len(times) - 1, int(len(times) * 0.95))
    p95 = times[p95_index]

    print(f"\n{name}")
    print(f"  Average : {statistics.mean(times):.2f} ms")
    print(f"  Median  : {p50:.2f} ms")
    print(f"  P95     : {p95:.2f} ms")
    print(f"  Min     : {min(times):.2f} ms")
    print(f"  Max     : {max(times):.2f} ms")

    return times


def test_face_recognition_performance():
    assert IMAGE_PATH.exists(), f"Benchmark image not found: {IMAGE_PATH}"

    image = cv2.imread(str(IMAGE_PATH))

    assert image is not None, "Could not load benchmark image"
    assert image.ndim == 3
    assert image.shape[2] == 3

    matcher = FaceMatchingService()

    print("\n" + "=" * 60)
    print("FACE RECOGNITION PERFORMANCE BENCHMARK")
    print("=" * 60)

    print(f"Image: {IMAGE_PATH}")
    print(f"Resolution: {image.shape[1]}x{image.shape[0]}")
    print(f"Iterations: {ITERATIONS}")

    # ---------------------------------------------------------
    # 1. Face detection
    # ---------------------------------------------------------

    faces = benchmark(
        "1. YuNet Face Detection",
        lambda: matcher.detector.detect(image),
    )

    detected_faces = matcher.detector.detect(image)

    assert len(detected_faces) == 1, (
        f"Expected exactly one face for benchmark, "
        f"but detected {len(detected_faces)}"
    )

    face = detected_faces[0]

    # ---------------------------------------------------------
    # 2. Face alignment
    # ---------------------------------------------------------

    aligned_face = benchmark(
        "2. Face Alignment",
        lambda: matcher.embedder.align_crop(image, face),
    )

    aligned = matcher.embedder.align_crop(image, face)

    assert aligned is not None
    assert aligned.size > 0

    # ---------------------------------------------------------
    # 3. SFace feature extraction
    # ---------------------------------------------------------

    feature = benchmark(
        "3. SFace Feature Extraction",
        lambda: matcher.embedder.feature(aligned),
    )

    feature_array = matcher.embedder.feature(aligned).astype(np.float32)

    assert feature_array.size == 128

    norm = np.linalg.norm(feature_array)

    assert norm > 0

    embedding = (feature_array / norm).reshape(1, -1)

    # ---------------------------------------------------------
    # 4. Full pipeline
    # ---------------------------------------------------------

    full_pipeline = benchmark(
        "4. Full Face Pipeline",
        lambda: matcher.validate_and_extract_embedding(image),
    )

    # ---------------------------------------------------------
    # 5. Face comparison
    # ---------------------------------------------------------

    comparison = benchmark(
        "5. Face Embedding Comparison",
        lambda: matcher.compare(embedding, embedding),
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)

    print(f"Detection average : {statistics.mean(faces):.2f} ms")
    print(f"Alignment average : {statistics.mean(benchmark("alignment", lambda: matcher.embedder.align_crop(image, face))):.2f} ms")
    print(f"Feature average   : {statistics.mean(feature):.2f} ms")
    print(f"Pipeline average  : {statistics.mean(full_pipeline):.2f} ms")
    print(f"Compare average   : {statistics.mean(comparison):.2f} ms")

    assert statistics.mean(full_pipeline) > 0
    assert statistics.mean(comparison) > 0