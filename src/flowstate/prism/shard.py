"""File-level sharding strategies for multi-GPU distributed replay.

Assigns Parquet files to ranks (GPUs) so each rank replays a disjoint
subset of the data. Three strategies are provided:

- **Round-robin**: Even file distribution regardless of content.
- **Symbol-affinity**: All files for a given symbol go to the same rank,
  preserving per-symbol temporal ordering without cross-rank coordination.
- **Time-range**: Each rank owns a contiguous time slice, useful when
  downstream processing is time-ordered.

All strategies are deterministic: given the same file list and rank count,
they produce the same assignment every time.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

logger = logging.getLogger(__name__)


class ShardStrategy(StrEnum):
    """File-to-rank assignment strategy."""

    ROUND_ROBIN = "round_robin"
    SYMBOL_AFFINITY = "symbol_affinity"
    TIME_RANGE = "time_range"


@dataclass(frozen=True)
class ShardAssignment:
    """Result of sharding: which files belong to which rank."""

    rank: int
    world_size: int
    files: list[Path]
    strategy: ShardStrategy

    @property
    def file_count(self) -> int:
        return len(self.files)


def shard_files(
    files: list[Path],
    rank: int,
    world_size: int,
    strategy: ShardStrategy = ShardStrategy.ROUND_ROBIN,
    symbol_extractor: str = "symbol",
) -> ShardAssignment:
    """Assign files to a rank using the given sharding strategy.

    Args:
        files: All discovered Parquet files (must be the same list on every rank).
        rank: This rank's index (0-based).
        world_size: Total number of ranks.
        strategy: Sharding strategy to use.
        symbol_extractor: Hive partition key for symbol affinity (e.g., "symbol" or "bucket").

    Returns:
        ShardAssignment with this rank's file subset.
    """
    if world_size <= 0:
        raise ValueError(f"world_size must be > 0, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank {rank} out of range [0, {world_size})")
    if not files:
        return ShardAssignment(rank=rank, world_size=world_size, files=[], strategy=strategy)

    # Sort files for deterministic assignment across all ranks
    sorted_files = sorted(files, key=lambda p: str(p))

    if strategy == ShardStrategy.ROUND_ROBIN:
        assigned = _shard_round_robin(sorted_files, rank, world_size)
    elif strategy == ShardStrategy.SYMBOL_AFFINITY:
        assigned = _shard_symbol_affinity(sorted_files, rank, world_size, symbol_extractor)
    elif strategy == ShardStrategy.TIME_RANGE:
        assigned = _shard_time_range(sorted_files, rank, world_size)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    logger.debug(
        f"Rank {rank}/{world_size}: assigned {len(assigned)}/{len(files)} files "
        f"via {strategy.value}"
    )
    return ShardAssignment(rank=rank, world_size=world_size, files=assigned, strategy=strategy)


def _shard_round_robin(files: list[Path], rank: int, world_size: int) -> list[Path]:
    """Assign every N-th file to this rank."""
    return files[rank::world_size]


def _shard_symbol_affinity(
    files: list[Path],
    rank: int,
    world_size: int,
    partition_key: str,
) -> list[Path]:
    """Assign files by hashing the symbol/bucket partition value.

    All files with the same symbol hash go to the same rank, ensuring
    per-symbol temporal ordering without cross-rank merging.
    """
    assigned: list[Path] = []
    for path in files:
        key = _extract_partition_value(path, partition_key)
        owner = _deterministic_hash(key) % world_size
        if owner == rank:
            assigned.append(path)
    return assigned


def _shard_time_range(files: list[Path], rank: int, world_size: int) -> list[Path]:
    """Assign contiguous file ranges to each rank.

    Files are sorted lexicographically (which aligns with date partitions
    in Hive format), then split into world_size contiguous chunks.
    """
    n = len(files)
    chunk_size = n // world_size
    remainder = n % world_size

    # Distribute remainder across first `remainder` ranks
    start = rank * chunk_size + min(rank, remainder)
    end = start + chunk_size + (1 if rank < remainder else 0)
    return files[start:end]


def _extract_partition_value(path: Path, key: str) -> str:
    """Extract a Hive partition value from a file path.

    Looks for patterns like `key=value` in the path components.
    Falls back to the full path string if the key isn't found.
    """
    prefix = f"{key}="
    for part in path.parts:
        if part.startswith(prefix):
            return part[len(prefix):]
    # Fallback: use full path for deterministic hashing
    return str(path)


def _deterministic_hash(value: str) -> int:
    """Deterministic hash for cross-process consistency.

    Uses MD5 (not for security, just for uniform distribution).
    Must produce the same result on every rank.
    """
    return int(hashlib.md5(value.encode(), usedforsecurity=False).hexdigest(), 16)
