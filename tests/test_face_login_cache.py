import numpy as np

from app.services.face_login import FaceLoginService


def test_embedding_is_cached():
    service = FaceLoginService()

    user_id = 1
    embedding = np.random.rand(1, 128).astype(np.float32)

    service._cache_embedding(user_id, embedding)

    cached = service._get_cached_embedding(user_id)

    assert cached is not None
    np.testing.assert_array_equal(cached, embedding)


def test_cached_embedding_is_reused():
    service = FaceLoginService()

    user_id = 1
    embedding = np.random.rand(1, 128).astype(np.float32)

    service._cache_embedding(user_id, embedding)

    first = service._get_cached_embedding(user_id)
    second = service._get_cached_embedding(user_id)

    assert first is not None
    assert second is not None
    np.testing.assert_array_equal(first, second)


def test_embedding_cache_invalidation():
    service = FaceLoginService()

    user_id = 1
    embedding = np.random.rand(1, 128).astype(np.float32)

    service._cache_embedding(user_id, embedding)

    # Verify it is cached before invalidation.
    assert service._get_cached_embedding(user_id) is not None

    # Simulate successful face re-enrollment.
    service.invalidate_embedding(user_id)

    # Old embedding must no longer be available.
    assert service._get_cached_embedding(user_id) is None


def test_invalidate_only_removes_requested_user():
    service = FaceLoginService()

    embedding_1 = np.random.rand(1, 128).astype(np.float32)
    embedding_2 = np.random.rand(1, 128).astype(np.float32)

    service._cache_embedding(1, embedding_1)
    service._cache_embedding(2, embedding_2)

    service.invalidate_embedding(1)

    assert service._get_cached_embedding(1) is None

    cached_user_2 = service._get_cached_embedding(2)

    assert cached_user_2 is not None
    np.testing.assert_array_equal(cached_user_2, embedding_2)


def test_clear_embedding_cache():
    service = FaceLoginService()

    service._cache_embedding(
        1,
        np.random.rand(1, 128).astype(np.float32),
    )
    service._cache_embedding(
        2,
        np.random.rand(1, 128).astype(np.float32),
    )

    assert service._get_cached_embedding(1) is not None
    assert service._get_cached_embedding(2) is not None

    service.clear_embedding_cache()

    assert service._get_cached_embedding(1) is None
    assert service._get_cached_embedding(2) is None