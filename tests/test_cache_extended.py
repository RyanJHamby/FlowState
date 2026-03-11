"""Extended tests for NVMe LRU cache tier — thread safety, edge cases, persistence."""

from __future__ import annotations

import threading
from pathlib import Path

from flowstate.storage.cache import CacheConfig, CacheStats, LRUCache


def _create_file(path: Path, size: int = 100) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


class TestCacheStats:
    def test_hit_rate_no_ops(self):
        stats = CacheStats()
        assert stats.hit_rate == 0.0

    def test_hit_rate_all_hits(self):
        stats = CacheStats(hits=10, misses=0)
        assert stats.hit_rate == 1.0

    def test_hit_rate_mixed(self):
        stats = CacheStats(hits=3, misses=7)
        assert abs(stats.hit_rate - 0.3) < 1e-9


class TestLRUAccess:
    def test_get_refreshes_lru_order(self, tmp_path: Path):
        """Accessing an entry moves it to most-recently-used."""
        config = CacheConfig(cache_dir=tmp_path / "cache", max_size_bytes=250)
        cache = LRUCache(config)

        # Put f0 and f1 (both fit: 200 <= 250)
        for i in range(2):
            src = _create_file(tmp_path / "src" / f"f{i}.parquet", size=100)
            cache.put(f"f{i}.parquet", src)

        assert cache.contains("f0.parquet")
        assert cache.contains("f1.parquet")

        # Access f0 to refresh it (now f1 is LRU)
        cache.get("f0.parquet")

        # Adding f2 should evict f1 (the LRU), not f0
        src = _create_file(tmp_path / "src" / "f2.parquet", size=100)
        cache.put("f2.parquet", src)
        assert cache.contains("f0.parquet")  # Was refreshed
        assert not cache.contains("f1.parquet")  # LRU evicted

    def test_put_update_existing(self, tmp_path: Path):
        """Updating an existing entry replaces the size tracking."""
        config = CacheConfig(cache_dir=tmp_path / "cache", max_size_bytes=10000)
        cache = LRUCache(config)

        src1 = _create_file(tmp_path / "src" / "f.parquet", size=100)
        cache.put("f.parquet", src1)
        assert cache.stats.current_size_bytes == 100

        # Update with larger file
        src2 = _create_file(tmp_path / "src" / "f_big.parquet", size=300)
        cache.put("f.parquet", src2)
        assert cache.stats.current_size_bytes == 300
        assert cache.stats.file_count == 1


class TestCachePersistence:
    def test_scan_existing_on_init(self, tmp_path: Path):
        """Cache picks up existing files from disk on init."""
        cache_dir = tmp_path / "cache"
        # Pre-populate the cache directory
        f1 = cache_dir / "a.parquet"
        f1.parent.mkdir(parents=True)
        f1.write_bytes(b"x" * 200)
        f2 = cache_dir / "sub" / "b.parquet"
        f2.parent.mkdir(parents=True)
        f2.write_bytes(b"y" * 300)

        config = CacheConfig(cache_dir=cache_dir, max_size_bytes=10000)
        cache = LRUCache(config)
        assert cache.stats.file_count == 2
        assert cache.stats.current_size_bytes == 500


class TestCacheEviction:
    def test_eviction_cleans_up_files(self, tmp_path: Path):
        config = CacheConfig(cache_dir=tmp_path / "cache", max_size_bytes=150)
        cache = LRUCache(config)

        src1 = _create_file(tmp_path / "src" / "f1.parquet", size=100)
        cache.put("f1.parquet", src1)
        cached_path = cache.get("f1.parquet")
        assert cached_path is not None
        assert cached_path.exists()

        # This should evict f1
        src2 = _create_file(tmp_path / "src" / "f2.parquet", size=100)
        cache.put("f2.parquet", src2)
        assert not cached_path.exists()

    def test_eviction_counter(self, tmp_path: Path):
        config = CacheConfig(cache_dir=tmp_path / "cache", max_size_bytes=150)
        cache = LRUCache(config)

        for i in range(5):
            src = _create_file(tmp_path / "src" / f"f{i}.parquet", size=100)
            cache.put(f"f{i}.parquet", src)

        assert cache.stats.evictions >= 3

    def test_file_larger_than_cache(self, tmp_path: Path):
        """A file that requires evicting everything still gets cached."""
        config = CacheConfig(cache_dir=tmp_path / "cache", max_size_bytes=150)
        cache = LRUCache(config)

        src1 = _create_file(tmp_path / "src" / "small.parquet", size=50)
        cache.put("small.parquet", src1)

        # 200 > 150, but after eviction there's nothing left, so it fits
        src2 = _create_file(tmp_path / "src" / "big.parquet", size=200)
        cache.put("big.parquet", src2)
        # small was evicted, big may exceed max but was stored
        assert not cache.contains("small.parquet")


class TestThreadSafety:
    def test_concurrent_puts(self, tmp_path: Path):
        config = CacheConfig(cache_dir=tmp_path / "cache", max_size_bytes=100000)
        cache = LRUCache(config)

        errors: list[Exception] = []

        def worker(worker_id: int) -> None:
            try:
                for i in range(10):
                    key = f"w{worker_id}_f{i}.parquet"
                    src = _create_file(
                        tmp_path / "src" / f"w{worker_id}" / f"f{i}.parquet",
                        size=50,
                    )
                    cache.put(key, src)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(w,)) for w in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors in concurrent puts: {errors}"
        assert cache.stats.file_count == 40

    def test_concurrent_get_and_put(self, tmp_path: Path):
        config = CacheConfig(cache_dir=tmp_path / "cache", max_size_bytes=100000)
        cache = LRUCache(config)

        # Pre-populate
        for i in range(10):
            src = _create_file(tmp_path / "src" / f"f{i}.parquet", size=50)
            cache.put(f"f{i}.parquet", src)

        errors: list[Exception] = []

        def reader() -> None:
            try:
                for _ in range(100):
                    for i in range(10):
                        cache.get(f"f{i}.parquet")
            except Exception as e:
                errors.append(e)

        def writer() -> None:
            try:
                for i in range(10, 20):
                    src = _create_file(
                        tmp_path / "src2" / f"f{i}.parquet", size=50,
                    )
                    cache.put(f"f{i}.parquet", src)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=reader),
            threading.Thread(target=reader),
            threading.Thread(target=writer),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors


class TestClearAndRemove:
    def test_remove_updates_size(self, tmp_path: Path):
        config = CacheConfig(cache_dir=tmp_path / "cache", max_size_bytes=10000)
        cache = LRUCache(config)

        src = _create_file(tmp_path / "src" / "f.parquet", size=200)
        cache.put("f.parquet", src)
        assert cache.stats.current_size_bytes == 200

        cache.remove("f.parquet")
        assert cache.stats.current_size_bytes == 0

    def test_clear_resets_all(self, tmp_path: Path):
        config = CacheConfig(cache_dir=tmp_path / "cache", max_size_bytes=10000)
        cache = LRUCache(config)

        for i in range(5):
            src = _create_file(tmp_path / "src" / f"f{i}.parquet", size=50)
            cache.put(f"f{i}.parquet", src)

        cache.clear()
        assert cache.stats.file_count == 0
        assert cache.stats.current_size_bytes == 0
        # Directory still exists
        assert Path(config.cache_dir).exists()

    def test_get_after_clear_misses(self, tmp_path: Path):
        config = CacheConfig(cache_dir=tmp_path / "cache", max_size_bytes=10000)
        cache = LRUCache(config)

        src = _create_file(tmp_path / "src" / "f.parquet", size=50)
        cache.put("f.parquet", src)
        cache.clear()

        assert cache.get("f.parquet") is None
        assert cache.stats.misses == 1
