"""NVMe LRU cache tier for hot data."""

from __future__ import annotations

import contextlib
import logging
import shutil
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for the NVMe cache tier."""

    cache_dir: str | Path
    max_size_bytes: int = 100 * 1024**3  # 100 GB default
    eviction_ratio: float = 0.1  # Evict 10% when full


@dataclass
class CacheStats:
    """Cache statistics."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    current_size_bytes: int = 0
    file_count: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class LRUCache:
    """LRU cache tier backed by local (NVMe) storage.

    Caches Parquet files on fast local storage to avoid repeated reads
    from object storage. Uses LRU eviction when cache is full.
    """

    def __init__(self, config: CacheConfig) -> None:
        self._config = config
        self._cache_dir = Path(config.cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, int] = OrderedDict()  # key -> size_bytes
        self._stats = CacheStats()
        self._scan_existing()

    @property
    def config(self) -> CacheConfig:
        return self._config

    @property
    def stats(self) -> CacheStats:
        return self._stats

    def _scan_existing(self) -> None:
        """Scan cache directory for existing files."""
        for path in self._cache_dir.rglob("*.parquet"):
            key = str(path.relative_to(self._cache_dir))
            size = path.stat().st_size
            self._entries[key] = size
            self._stats.current_size_bytes += size
        self._stats.file_count = len(self._entries)

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / key

    def get(self, key: str) -> Path | None:
        """Get a cached file path, or None if not cached.

        Moves the key to most-recently-used position.
        """
        with self._lock:
            if key not in self._entries:
                self._stats.misses += 1
                return None

            self._entries.move_to_end(key)
            self._stats.hits += 1
            return self._cache_path(key)

    def put(self, key: str, source_path: Path) -> Path:
        """Add a file to the cache.

        If the file already exists, it's treated as an update.
        Evicts LRU entries if the cache would exceed max size.

        Args:
            key: Cache key (typically the partition path).
            source_path: Path to the file to cache.

        Returns:
            The path to the cached file.
        """
        file_size = source_path.stat().st_size

        with self._lock:
            # Evict if necessary
            while (
                self._stats.current_size_bytes + file_size > self._config.max_size_bytes
                and self._entries
            ):
                self._evict_one()

            dest = self._cache_path(key)
            dest.parent.mkdir(parents=True, exist_ok=True)

            # Remove old entry if updating
            if key in self._entries:
                self._stats.current_size_bytes -= self._entries[key]

            shutil.copy2(source_path, dest)
            self._entries[key] = file_size
            self._entries.move_to_end(key)
            self._stats.current_size_bytes += file_size
            self._stats.file_count = len(self._entries)

            return dest

    def _evict_one(self) -> None:
        """Evict the least recently used entry."""
        if not self._entries:
            return
        key, size = self._entries.popitem(last=False)
        path = self._cache_path(key)
        if path.exists():
            path.unlink()
            # Clean up empty parent directories
            with contextlib.suppress(OSError):
                path.parent.rmdir()
        self._stats.current_size_bytes -= size
        self._stats.evictions += 1
        self._stats.file_count = len(self._entries)
        logger.debug(f"Evicted {key} ({size} bytes)")

    def contains(self, key: str) -> bool:
        with self._lock:
            return key in self._entries

    def remove(self, key: str) -> bool:
        """Remove a specific entry from the cache."""
        with self._lock:
            if key not in self._entries:
                return False
            size = self._entries.pop(key)
            path = self._cache_path(key)
            if path.exists():
                path.unlink()
            self._stats.current_size_bytes -= size
            self._stats.file_count = len(self._entries)
            return True

    def clear(self) -> None:
        """Clear the entire cache."""
        with self._lock:
            shutil.rmtree(self._cache_dir)
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._entries.clear()
            self._stats.current_size_bytes = 0
            self._stats.file_count = 0
