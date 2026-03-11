"""Distributed replay coordinator for multi-GPU training.

Orchestrates per-rank replay across multiple GPUs:
1. Each rank discovers the same global file list.
2. Files are sharded across ranks using a deterministic strategy.
3. Each rank replays only its assigned files in time order.
4. An optional NCCL barrier synchronizes ranks at epoch boundaries.

Single-rank mode (world_size=1) is a no-op shim that runs the standard
ReplayEngine — no distributed overhead.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field

import pyarrow as pa

from flowstate.prism.nccl import MultiGPUComm, NCCLConfig
from flowstate.prism.replay import ReplayConfig, ReplayEngine, ReplayFilter
from flowstate.prism.shard import ShardAssignment, ShardStrategy, shard_files

logger = logging.getLogger(__name__)


@dataclass
class DistributedReplayConfig:
    """Configuration for distributed replay."""

    data_dir: str = ""
    rank: int = 0
    world_size: int = 1
    strategy: ShardStrategy = ShardStrategy.ROUND_ROBIN
    symbol_partition_key: str = "bucket"
    replay_config: ReplayConfig = field(default_factory=ReplayConfig)
    sync_on_epoch: bool = True


@dataclass
class DistributedReplayStats:
    """Per-rank statistics from distributed replay."""

    rank: int = 0
    world_size: int = 1
    total_files: int = 0
    assigned_files: int = 0
    epochs_completed: int = 0
    batches_yielded: int = 0
    rows_yielded: int = 0


class DistributedReplay:
    """Multi-rank replay coordinator.

    Each rank replays a disjoint subset of files, determined by the
    sharding strategy. At epoch boundaries, an optional NCCL barrier
    ensures all ranks have finished before the next epoch begins.

    Example::

        config = DistributedReplayConfig(
            data_dir="/data/market",
            rank=int(os.environ["RANK"]),
            world_size=int(os.environ["WORLD_SIZE"]),
            strategy=ShardStrategy.SYMBOL_AFFINITY,
        )
        dr = DistributedReplay(config)

        for batch in dr.replay(replay_filter, num_epochs=3):
            # Each rank sees only its shard of data
            process(batch)
    """

    def __init__(self, config: DistributedReplayConfig) -> None:
        self._config = config
        self._engine = ReplayEngine(config.data_dir, config=config.replay_config)
        self._comm = MultiGPUComm(NCCLConfig(
            world_size=config.world_size,
            rank=config.rank,
        ))
        self._stats = DistributedReplayStats(
            rank=config.rank,
            world_size=config.world_size,
        )
        self._assignment: ShardAssignment | None = None

    @property
    def stats(self) -> DistributedReplayStats:
        return self._stats

    @property
    def assignment(self) -> ShardAssignment | None:
        return self._assignment

    @property
    def is_distributed(self) -> bool:
        return self._config.world_size > 1

    def discover_and_shard(
        self, replay_filter: ReplayFilter | None = None,
    ) -> ShardAssignment:
        """Discover files and assign this rank's shard.

        Every rank must call this with the same replay_filter to get
        consistent global file lists.
        """
        all_files = self._engine.discover_files(replay_filter)
        self._stats.total_files = len(all_files)

        assignment = shard_files(
            files=all_files,
            rank=self._config.rank,
            world_size=self._config.world_size,
            strategy=self._config.strategy,
            symbol_extractor=self._config.symbol_partition_key,
        )
        self._assignment = assignment
        self._stats.assigned_files = assignment.file_count

        logger.info(
            f"Rank {self._config.rank}: {assignment.file_count}/{len(all_files)} files "
            f"({self._config.strategy.value})"
        )
        return assignment

    def replay(
        self,
        replay_filter: ReplayFilter | None = None,
        num_epochs: int = 1,
    ) -> Iterator[pa.RecordBatch]:
        """Replay this rank's shard of data for the given number of epochs.

        Args:
            replay_filter: Filter for file discovery and row-level filtering.
            num_epochs: Number of full passes over the data.

        Yields:
            RecordBatches in time order within this rank's shard.
        """
        if self._assignment is None:
            self.discover_and_shard(replay_filter)

        for epoch in range(num_epochs):
            yield from self._replay_epoch(replay_filter, epoch)

            self._stats.epochs_completed += 1

            if self._config.sync_on_epoch and self.is_distributed:
                self._comm.barrier()
                logger.debug(f"Rank {self._config.rank}: epoch {epoch} barrier passed")

    def _replay_epoch(
        self,
        replay_filter: ReplayFilter | None,
        epoch: int,
    ) -> Iterator[pa.RecordBatch]:
        """Single epoch: replay assigned files in order."""
        assert self._assignment is not None

        for file_path in self._assignment.files:
            try:
                for batch in self._engine.read_file_batches(file_path, replay_filter):
                    if batch.num_rows > 0:
                        self._stats.batches_yielded += 1
                        self._stats.rows_yielded += batch.num_rows
                        yield batch
            except Exception:
                logger.exception(
                    f"Rank {self._config.rank}: error reading {file_path} "
                    f"(epoch {epoch}), skipping"
                )

    def reset_stats(self) -> None:
        """Reset per-run statistics (keeps assignment)."""
        self._stats = DistributedReplayStats(
            rank=self._config.rank,
            world_size=self._config.world_size,
            total_files=self._stats.total_files,
            assigned_files=self._stats.assigned_files,
        )
