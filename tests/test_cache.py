"""Tests for NVMe LRU cache tier."""

from __future__ import annotations

from pathlib import Path

import pytest

from flowstate.storage.cache import CacheConfig, LRUCache


@pytest.fixture
def cache(tmp_path: Path) -> LRUCache:
    config = CacheConfig(cache_dir=tmp_path / "cache", max_size_bytes=10000)
    return LRUCache(config)


def _create_file(path: Path, size: int = 100) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


class TestLRUCache:
    def test_put_and_get(self, cache: LRUCache, tmp_path: Path):
        src = _create_file(tmp_path / "src" / "test.parquet")
        cache.put("data/test.parquet", src)
        result = cache.get("data/test.parquet")
        assert result is not None
        assert result.exists()

    def test_miss_returns_none(self, cache: LRUCache):
        result = cache.get("nonexistent")
        assert result is None
        assert cache.stats.misses == 1

    def test_hit_tracking(self, cache: LRUCache, tmp_path: Path):
        src = _create_file(tmp_path / "src" / "test.parquet")
        cache.put("test.parquet", src)
        cache.get("test.parquet")
        cache.get("test.parquet")
        assert cache.stats.hits == 2
        assert cache.stats.hit_rate > 0

    def test_eviction(self, tmp_path: Path):
        config = CacheConfig(cache_dir=tmp_path / "cache", max_size_bytes=250)
        cache = LRUCache(config)

        for i in range(5):
            src = _create_file(tmp_path / "src" / f"file{i}.parquet", size=100)
            cache.put(f"file{i}.parquet", src)

        # Cache can hold ~2 files at 100 bytes each (250 max)
        assert cache.stats.evictions > 0
        assert cache.stats.current_size_bytes <= 250

    def test_lru_eviction_order(self, tmp_path: Path):
        config = CacheConfig(cache_dir=tmp_path / "cache", max_size_bytes=250)
        cache = LRUCache(config)

        for i in range(3):
            src = _create_file(tmp_path / "src" / f"file{i}.parquet", size=100)
            cache.put(f"file{i}.parquet", src)

        # file0 should have been evicted (LRU)
        assert not cache.contains("file0.parquet")
        # Most recent should still be there
        assert cache.contains("file2.parquet")

    def test_contains(self, cache: LRUCache, tmp_path: Path):
        src = _create_file(tmp_path / "src" / "test.parquet")
        assert not cache.contains("test.parquet")
        cache.put("test.parquet", src)
        assert cache.contains("test.parquet")

    def test_remove(self, cache: LRUCache, tmp_path: Path):
        src = _create_file(tmp_path / "src" / "test.parquet")
        cache.put("test.parquet", src)
        assert cache.remove("test.parquet")
        assert not cache.contains("test.parquet")

    def test_remove_nonexistent(self, cache: LRUCache):
        assert not cache.remove("nonexistent")

    def test_clear(self, cache: LRUCache, tmp_path: Path):
        for i in range(3):
            src = _create_file(tmp_path / "src" / f"file{i}.parquet", size=50)
            cache.put(f"file{i}.parquet", src)

        cache.clear()
        assert cache.stats.current_size_bytes == 0
        assert cache.stats.file_count == 0

    def test_stats(self, cache: LRUCache, tmp_path: Path):
        src = _create_file(tmp_path / "src" / "test.parquet", size=200)
        cache.put("test.parquet", src)
        assert cache.stats.current_size_bytes == 200
        assert cache.stats.file_count == 1
