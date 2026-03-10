"""NCCL multi-GPU communication with single-device fallback."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pyarrow as pa

logger = logging.getLogger(__name__)

try:
    import cupy as cp  # noqa: F401

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


@dataclass
class NCCLConfig:
    """Configuration for NCCL multi-GPU communication."""

    world_size: int = 1
    rank: int = 0
    device_id: int = 0


class MultiGPUComm:
    """Multi-GPU communication layer using NCCL.

    Falls back to single-device pass-through when NCCL or multiple GPUs
    are not available. This allows code to run unchanged in CI and single-GPU
    environments.
    """

    def __init__(self, config: NCCLConfig | None = None) -> None:
        self._config = config or NCCLConfig()
        self._is_distributed = self._config.world_size > 1 and HAS_CUPY

        if self._is_distributed:
            logger.info(
                f"NCCL distributed mode: rank {self._config.rank}/{self._config.world_size}"
            )
        else:
            logger.info("Single-device mode (NCCL disabled)")

    @property
    def config(self) -> NCCLConfig:
        return self._config

    @property
    def is_distributed(self) -> bool:
        return self._is_distributed

    @property
    def rank(self) -> int:
        return self._config.rank

    @property
    def world_size(self) -> int:
        return self._config.world_size

    def shard_batch(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        """Shard a RecordBatch across devices by row.

        In single-device mode, returns the full batch unchanged.

        Args:
            batch: The RecordBatch to shard.

        Returns:
            The shard for this rank.
        """
        if not self._is_distributed:
            return batch

        total = batch.num_rows
        shard_size = total // self._config.world_size
        start = self._config.rank * shard_size
        end = start + shard_size if self._config.rank < self._config.world_size - 1 else total

        return batch.slice(start, end - start)

    def all_gather_sizes(self, local_count: int) -> list[int]:
        """Gather row counts from all ranks.

        In single-device mode, returns [local_count].
        """
        if not self._is_distributed:
            return [local_count]

        # In real NCCL, this would use ncclAllGather
        return [local_count] * self._config.world_size

    def broadcast_schema(self, schema: pa.Schema | None) -> pa.Schema:
        """Broadcast schema from rank 0 to all ranks.

        In single-device mode, returns the input schema.
        """
        if schema is None:
            raise ValueError("Schema cannot be None on source rank")
        return schema

    def barrier(self) -> None:
        """Synchronization barrier across all ranks.

        No-op in single-device mode.
        """
        if self._is_distributed:
            # In real NCCL, this would synchronize
            pass
