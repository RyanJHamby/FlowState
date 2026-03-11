"""Tests for file-level sharding strategies."""

from __future__ import annotations

from pathlib import Path

import pytest

from flowstate.prism.shard import (
    ShardStrategy,
    shard_files,
)


def _make_hive_paths(n: int, base: str = "/data") -> list[Path]:
    """Create fake Hive-partitioned file paths."""
    paths = []
    symbols = ["AAPL", "MSFT", "GOOG", "TSLA", "AMZN", "META", "NVDA", "JPM"]
    for i in range(n):
        sym = symbols[i % len(symbols)]
        date = f"2024-01-{(i % 28) + 1:02d}"
        paths.append(Path(f"{base}/type=trade/date={date}/bucket={sym}/part-{i:04d}.parquet"))
    return paths


class TestRoundRobin:
    def test_even_distribution(self):
        files = _make_hive_paths(12)
        shards = [shard_files(files, r, 4, ShardStrategy.ROUND_ROBIN) for r in range(4)]
        counts = [s.file_count for s in shards]
        assert counts == [3, 3, 3, 3]

    def test_uneven_distribution(self):
        files = _make_hive_paths(10)
        shards = [shard_files(files, r, 3, ShardStrategy.ROUND_ROBIN) for r in range(3)]
        total = sum(s.file_count for s in shards)
        assert total == 10

    def test_all_files_assigned(self):
        files = _make_hive_paths(7)
        all_assigned = set()
        for r in range(3):
            shard = shard_files(files, r, 3, ShardStrategy.ROUND_ROBIN)
            all_assigned.update(str(f) for f in shard.files)
        assert len(all_assigned) == 7

    def test_no_overlap(self):
        files = _make_hive_paths(20)
        all_files: list[list[str]] = []
        for r in range(4):
            shard = shard_files(files, r, 4, ShardStrategy.ROUND_ROBIN)
            all_files.append([str(f) for f in shard.files])
        for i in range(4):
            for j in range(i + 1, 4):
                overlap = set(all_files[i]) & set(all_files[j])
                assert len(overlap) == 0, f"Ranks {i} and {j} share files"

    def test_deterministic(self):
        files = _make_hive_paths(15)
        a = shard_files(files, 1, 4, ShardStrategy.ROUND_ROBIN)
        b = shard_files(files, 1, 4, ShardStrategy.ROUND_ROBIN)
        assert [str(f) for f in a.files] == [str(f) for f in b.files]

    def test_single_rank(self):
        files = _make_hive_paths(10)
        shard = shard_files(files, 0, 1, ShardStrategy.ROUND_ROBIN)
        assert shard.file_count == 10


class TestSymbolAffinity:
    def test_same_symbol_same_rank(self):
        files = _make_hive_paths(24)
        shards = {
            r: shard_files(files, r, 4, ShardStrategy.SYMBOL_AFFINITY, symbol_extractor="bucket")
            for r in range(4)
        }
        # All AAPL files should be on the same rank
        aapl_ranks = set()
        for r, shard in shards.items():
            for f in shard.files:
                if "bucket=AAPL" in str(f):
                    aapl_ranks.add(r)
        assert len(aapl_ranks) == 1, f"AAPL spread across ranks: {aapl_ranks}"

    def test_all_files_assigned(self):
        files = _make_hive_paths(16)
        all_assigned = set()
        for r in range(4):
            shard = shard_files(files, r, 4, ShardStrategy.SYMBOL_AFFINITY)
            all_assigned.update(str(f) for f in shard.files)
        assert len(all_assigned) == 16

    def test_no_overlap(self):
        files = _make_hive_paths(20)
        rank_files: list[set[str]] = []
        for r in range(4):
            shard = shard_files(files, r, 4, ShardStrategy.SYMBOL_AFFINITY)
            rank_files.append({str(f) for f in shard.files})
        for i in range(4):
            for j in range(i + 1, 4):
                assert len(rank_files[i] & rank_files[j]) == 0

    def test_deterministic(self):
        files = _make_hive_paths(20)
        a = shard_files(files, 2, 4, ShardStrategy.SYMBOL_AFFINITY)
        b = shard_files(files, 2, 4, ShardStrategy.SYMBOL_AFFINITY)
        assert [str(f) for f in a.files] == [str(f) for f in b.files]


class TestTimeRange:
    def test_contiguous_chunks(self):
        files = _make_hive_paths(12)
        shards = [shard_files(files, r, 3, ShardStrategy.TIME_RANGE) for r in range(3)]
        # Each rank gets a contiguous block
        all_files = []
        for s in shards:
            all_files.extend(str(f) for f in s.files)
        expected = [str(f) for f in sorted(files, key=lambda p: str(p))]
        assert all_files == expected

    def test_balanced(self):
        files = _make_hive_paths(10)
        shards = [shard_files(files, r, 3, ShardStrategy.TIME_RANGE) for r in range(3)]
        counts = [s.file_count for s in shards]
        # 10 files / 3 ranks → [4, 3, 3] or similar
        assert sum(counts) == 10
        assert max(counts) - min(counts) <= 1

    def test_single_rank(self):
        files = _make_hive_paths(10)
        shard = shard_files(files, 0, 1, ShardStrategy.TIME_RANGE)
        assert shard.file_count == 10


class TestEdgeCases:
    def test_empty_file_list(self):
        shard = shard_files([], 0, 4, ShardStrategy.ROUND_ROBIN)
        assert shard.file_count == 0
        assert shard.files == []

    def test_invalid_rank(self):
        with pytest.raises(ValueError, match="rank 5 out of range"):
            shard_files(_make_hive_paths(10), 5, 4)

    def test_negative_rank(self):
        with pytest.raises(ValueError, match="rank -1 out of range"):
            shard_files(_make_hive_paths(10), -1, 4)

    def test_zero_world_size(self):
        with pytest.raises(ValueError, match="world_size must be > 0"):
            shard_files(_make_hive_paths(10), 0, 0)

    def test_more_ranks_than_files(self):
        files = _make_hive_paths(2)
        shards = [shard_files(files, r, 5, ShardStrategy.ROUND_ROBIN) for r in range(5)]
        total = sum(s.file_count for s in shards)
        assert total == 2
        # Some ranks get 0 files
        assert any(s.file_count == 0 for s in shards)

    def test_assignment_metadata(self):
        files = _make_hive_paths(10)
        shard = shard_files(files, 2, 4, ShardStrategy.ROUND_ROBIN)
        assert shard.rank == 2
        assert shard.world_size == 4
        assert shard.strategy == ShardStrategy.ROUND_ROBIN
