"""End-to-end GPU data feeding pipeline.

Connects replay, temporal alignment, and prefetch into a single pipeline
that delivers aligned, GPU-ready tensors to a training loop.

    replay → align → pin → prefetch → consume

All GPU features degrade gracefully to CPU when CUDA is not available.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa

from flowstate.prism.alignment import (
    AlignmentStats,
    TemporalAligner,
)
from flowstate.prism.pinned_buffer import PinnedBufferConfig, PinnedBufferPool
from flowstate.prism.prefetcher import (
    PrefetchConfig,
    PrefetchedBatch,
    PrefetchPipeline,
)
from flowstate.prism.replay import ReplayConfig, ReplayEngine, ReplayFilter

logger = logging.getLogger(__name__)


@dataclass
class GPUPipelineConfig:
    """Configuration for the end-to-end GPU data pipeline."""

    # Replay
    data_dir: str | Path = ""
    replay_config: ReplayConfig = field(default_factory=ReplayConfig)

    # Alignment
    primary_type: str = "trade"
    secondary_specs: dict[str, list[str] | None] = field(default_factory=dict)
    tolerance_ns: int | None = None

    # Pinned memory
    pinned_config: PinnedBufferConfig = field(default_factory=PinnedBufferConfig)

    # Prefetch
    prefetch_config: PrefetchConfig = field(default_factory=PrefetchConfig)

    # Output
    numeric_columns: list[str] | None = None


@dataclass
class GPUPipelineStats:
    """Aggregated statistics from a pipeline run."""

    replay_batches: int = 0
    aligned_rows: int = 0
    alignment_stats: dict[str, AlignmentStats] = field(default_factory=dict)
    prefetch_stalls: int = 0
    output_batches: int = 0


class GPUDataPipeline:
    """End-to-end pipeline: replay → align → pin → prefetch → GPU tensors.

    Orchestrates the full data path from Parquet on disk to pinned memory
    buffers ready for GPU transfer. Handles the complexity of coordinating
    replay filters, temporal alignment, and async prefetch.

    Example::

        config = GPUPipelineConfig(
            data_dir="/data/market",
            primary_type="trade",
            secondary_specs={"quote": ["bid_price", "ask_price"]},
            tolerance_ns=5_000_000_000,
            numeric_columns=["price", "size", "quote_bid_price", "quote_ask_price"],
        )

        pipeline = GPUDataPipeline(config)

        for batch in pipeline.run(replay_filter):
            prices = batch.column_numpy("price")
            # ... feed to model ...
            batch.release_to(pipeline.pool)
    """

    def __init__(self, config: GPUPipelineConfig) -> None:
        self._config = config
        self._pool = PinnedBufferPool(config.pinned_config)

        self._replay = ReplayEngine(
            config.data_dir,
            config=config.replay_config,
        )

        self._prefetcher = PrefetchPipeline(
            pool=self._pool,
            config=config.prefetch_config,
            numeric_columns=config.numeric_columns,
        )

        self._stats = GPUPipelineStats()

    @property
    def pool(self) -> PinnedBufferPool:
        return self._pool

    @property
    def stats(self) -> GPUPipelineStats:
        return self._stats

    def run(
        self,
        replay_filter: ReplayFilter | None = None,
    ) -> Iterator[PrefetchedBatch]:
        """Run the full pipeline and yield prefetched, aligned batches.

        Args:
            replay_filter: Filter for the replay engine (symbols, time range, etc.).

        Yields:
            PrefetchedBatch instances with numeric columns in pinned memory.
        """
        self._stats = GPUPipelineStats()

        if self._config.secondary_specs:
            source = self._aligned_source(replay_filter)
        else:
            source = self._replay_source(replay_filter)

        for prefetched in self._prefetcher.iter(source):
            self._stats.output_batches += 1
            yield prefetched

        pf_stats = self._prefetcher.stats
        self._stats.prefetch_stalls = pf_stats.stalls

    def run_numpy(
        self,
        replay_filter: ReplayFilter | None = None,
    ) -> Iterator[dict[str, np.ndarray]]:
        """Run the pipeline and yield plain numpy dicts (convenience method).

        Each yielded dict maps column names to numpy arrays. Pinned buffers
        are automatically released after conversion.
        """
        for batch in self.run(replay_filter):
            result = {}
            for name in batch.schema.names:
                col = batch.batch.column(name)
                if (
                    pa.types.is_floating(col.type)
                    or pa.types.is_integer(col.type)
                    or pa.types.is_timestamp(col.type)
                ):
                    result[name] = batch.column_numpy(name)
            batch.release_to(self._pool)
            yield result

    def stop(self) -> None:
        """Stop the prefetcher."""
        self._prefetcher.stop()

    def _replay_source(
        self, replay_filter: ReplayFilter | None
    ) -> Iterator[pa.RecordBatch]:
        """Plain replay without alignment."""
        for batch in self._replay.replay(replay_filter):
            self._stats.replay_batches += 1
            yield batch

    def _aligned_source(
        self, replay_filter: ReplayFilter | None
    ) -> Iterator[pa.RecordBatch]:
        """Replay with temporal alignment across streams.

        Replays each data type separately, feeds them into the TemporalAligner,
        and yields aligned batches.
        """
        aligner = TemporalAligner(
            primary_type=self._config.primary_type,
            secondary_specs=self._config.secondary_specs,
            tolerance_ns=self._config.tolerance_ns,
        )

        # Collect all stream types we need to replay
        all_types = [self._config.primary_type] + list(self._config.secondary_specs.keys())

        # Replay each type and feed into aligner
        for data_type in all_types:
            type_filter = ReplayFilter(
                symbols=replay_filter.symbols if replay_filter else None,
                data_types=[data_type],
                time_range=replay_filter.time_range if replay_filter else None,
            )
            for batch in self._replay.replay(type_filter):
                self._stats.replay_batches += 1
                table = pa.Table.from_batches([batch])
                aligner.add_data(data_type, table)

        # Flush aligned result
        aligned, alignment_stats = aligner.flush()
        self._stats.alignment_stats = alignment_stats

        if aligned is not None:
            self._stats.aligned_rows = aligned.num_rows
            yield from aligned.to_batches(
                max_chunksize=self._config.replay_config.batch_size
            )
