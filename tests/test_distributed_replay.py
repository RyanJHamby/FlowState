"""Tests for distributed replay coordinator."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from flowstate.prism.distributed import DistributedReplay, DistributedReplayConfig
from flowstate.prism.replay import ReplayFilter
from flowstate.prism.shard import ShardStrategy


def _write_hive_data(base_dir: Path, data_type: str, date: str, bucket: int, n_rows: int):
    """Write test Parquet data in Hive partition layout."""
    part_dir = base_dir / f"type={data_type}" / f"date={date}" / f"bucket={bucket:04d}"
    part_dir.mkdir(parents=True, exist_ok=True)

    table = pa.table({
        "timestamp": pa.array(range(n_rows), type=pa.int64()),
        "price": [100.0 + i * 0.1 for i in range(n_rows)],
        "symbol": ["SYM"] * n_rows,
    })
    pq.write_table(table, part_dir / "data.parquet")


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Create a test dataset with multiple partitions."""
    for bucket in range(4):
        _write_hive_data(tmp_path, "trade", "2024-01-15", bucket, 100)
        _write_hive_data(tmp_path, "trade", "2024-01-16", bucket, 100)
    return tmp_path


class TestSingleRank:
    def test_single_rank_gets_all_files(self, data_dir: Path):
        config = DistributedReplayConfig(
            data_dir=str(data_dir),
            rank=0,
            world_size=1,
        )
        dr = DistributedReplay(config)
        assignment = dr.discover_and_shard()
        assert assignment.file_count == 8  # 4 buckets × 2 dates
        assert not dr.is_distributed

    def test_single_rank_replays_all_rows(self, data_dir: Path):
        config = DistributedReplayConfig(
            data_dir=str(data_dir),
            rank=0,
            world_size=1,
        )
        dr = DistributedReplay(config)
        batches = list(dr.replay())
        total_rows = sum(b.num_rows for b in batches)
        assert total_rows == 800  # 8 files × 100 rows


class TestMultiRank:
    def test_all_ranks_cover_all_files(self, data_dir: Path):
        world_size = 4
        all_files: set[str] = set()
        for rank in range(world_size):
            config = DistributedReplayConfig(
                data_dir=str(data_dir),
                rank=rank,
                world_size=world_size,
            )
            dr = DistributedReplay(config)
            assignment = dr.discover_and_shard()
            all_files.update(str(f) for f in assignment.files)
        assert len(all_files) == 8

    def test_no_overlap_between_ranks(self, data_dir: Path):
        world_size = 3
        rank_files: list[set[str]] = []
        for rank in range(world_size):
            config = DistributedReplayConfig(
                data_dir=str(data_dir),
                rank=rank,
                world_size=world_size,
            )
            dr = DistributedReplay(config)
            assignment = dr.discover_and_shard()
            rank_files.append({str(f) for f in assignment.files})
        for i in range(world_size):
            for j in range(i + 1, world_size):
                assert len(rank_files[i] & rank_files[j]) == 0

    def test_all_rows_covered(self, data_dir: Path):
        world_size = 2
        total_rows = 0
        for rank in range(world_size):
            config = DistributedReplayConfig(
                data_dir=str(data_dir),
                rank=rank,
                world_size=world_size,
            )
            dr = DistributedReplay(config)
            batches = list(dr.replay())
            total_rows += sum(b.num_rows for b in batches)
        assert total_rows == 800

    def test_symbol_affinity_strategy(self, data_dir: Path):
        config = DistributedReplayConfig(
            data_dir=str(data_dir),
            rank=0,
            world_size=2,
            strategy=ShardStrategy.SYMBOL_AFFINITY,
        )
        dr = DistributedReplay(config)
        assignment = dr.discover_and_shard()
        assert assignment.strategy == ShardStrategy.SYMBOL_AFFINITY
        assert assignment.file_count > 0

    def test_time_range_strategy(self, data_dir: Path):
        config = DistributedReplayConfig(
            data_dir=str(data_dir),
            rank=0,
            world_size=2,
            strategy=ShardStrategy.TIME_RANGE,
        )
        dr = DistributedReplay(config)
        assignment = dr.discover_and_shard()
        assert assignment.strategy == ShardStrategy.TIME_RANGE
        assert assignment.file_count > 0


class TestMultiEpoch:
    def test_two_epochs_double_rows(self, data_dir: Path):
        config = DistributedReplayConfig(
            data_dir=str(data_dir),
            rank=0,
            world_size=1,
        )
        dr = DistributedReplay(config)
        batches = list(dr.replay(num_epochs=2))
        total_rows = sum(b.num_rows for b in batches)
        assert total_rows == 1600  # 800 × 2 epochs
        assert dr.stats.epochs_completed == 2

    def test_epoch_stats(self, data_dir: Path):
        config = DistributedReplayConfig(
            data_dir=str(data_dir),
            rank=0,
            world_size=1,
        )
        dr = DistributedReplay(config)
        list(dr.replay(num_epochs=3))
        assert dr.stats.epochs_completed == 3
        assert dr.stats.rows_yielded == 2400


class TestWithFilter:
    def test_filter_by_data_type(self, data_dir: Path):
        config = DistributedReplayConfig(
            data_dir=str(data_dir),
            rank=0,
            world_size=1,
        )
        dr = DistributedReplay(config)
        replay_filter = ReplayFilter(data_types=["trade"])
        assignment = dr.discover_and_shard(replay_filter)
        assert assignment.file_count == 8

    def test_filter_nonexistent_type(self, data_dir: Path):
        config = DistributedReplayConfig(
            data_dir=str(data_dir),
            rank=0,
            world_size=1,
        )
        dr = DistributedReplay(config)
        replay_filter = ReplayFilter(data_types=["quote"])
        assignment = dr.discover_and_shard(replay_filter)
        assert assignment.file_count == 0


class TestStats:
    def test_stats_populated(self, data_dir: Path):
        config = DistributedReplayConfig(
            data_dir=str(data_dir),
            rank=0,
            world_size=2,
        )
        dr = DistributedReplay(config)
        list(dr.replay())
        s = dr.stats
        assert s.rank == 0
        assert s.world_size == 2
        assert s.total_files == 8
        assert s.assigned_files > 0
        assert s.batches_yielded > 0
        assert s.rows_yielded > 0

    def test_reset_stats(self, data_dir: Path):
        config = DistributedReplayConfig(
            data_dir=str(data_dir),
            rank=0,
            world_size=1,
        )
        dr = DistributedReplay(config)
        list(dr.replay())
        assert dr.stats.rows_yielded > 0

        dr.reset_stats()
        assert dr.stats.rows_yielded == 0
        assert dr.stats.assigned_files > 0  # Preserved from discovery
