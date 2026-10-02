"""Tests for the thread-safe round-robin API key pool."""

import pytest

from free_claude_code.core.api_key_pool import ApiKeyPool


def test_api_key_pool_single_key():
    pool = ApiKeyPool("key1")
    assert pool.size == 1
    assert pool.current_key() == "key1"
    # Rotate should return None because there's only one key, which has been tried
    assert pool.rotate() is None
    # After reset, it can be tried again
    pool.reset()
    assert pool.current_key() == "key1"


def test_api_key_pool_multi_key():
    pool = ApiKeyPool("key1, key2, key3")
    assert pool.size == 3
    assert pool.current_key() == "key1"

    # First rotation
    assert pool.rotate() == "key2"
    assert pool.current_key() == "key2"

    # Second rotation
    assert pool.rotate() == "key3"
    assert pool.current_key() == "key3"

    # Third rotation (all keys exhausted)
    assert pool.rotate() is None
    # The active key remains key3
    assert pool.current_key() == "key3"


def test_api_key_pool_reset():
    pool = ApiKeyPool("key1, key2")
    assert pool.current_key() == "key1"
    assert pool.rotate() == "key2"

    # Reset starts with the currently active key (key2) as tried
    pool.reset()
    assert pool.current_key() == "key2"

    # Rotate should go to key1
    assert pool.rotate() == "key1"
    assert pool.current_key() == "key1"

    # Next rotation returns None
    assert pool.rotate() is None


def test_api_key_pool_whitespace_and_empty():
    pool = ApiKeyPool("  key1  , , key2  ")
    assert pool.size == 2
    assert pool._keys == ["key1", "key2"]

    with pytest.raises(ValueError):
        ApiKeyPool("")

    with pytest.raises(ValueError):
        ApiKeyPool(" , , ")
