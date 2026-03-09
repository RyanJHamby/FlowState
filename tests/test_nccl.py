"""Tests for NCCL multi-GPU communication."""

from __future__ import annotations

import pyarrow as pa
import pytest

from flowstate.prism.nccl import MultiGPUComm, NCCLConfig


class TestMultiGPUComm:
    def test_single_device_default(self):
        comm = MultiGPUComm()
        assert not comm.is_distributed
        assert comm.rank == 0
        assert comm.world_size == 1

    def test_shard_batch_single_device(self):
        comm = MultiGPUComm()
        batch = pa.RecordBatch.from_pydict({"x": [1, 2, 3, 4]})
        result = comm.shard_batch(batch)
        assert result.num_rows == 4  # Returns full batch

    def test_all_gather_sizes_single(self):
        comm = MultiGPUComm()
        sizes = comm.all_gather_sizes(100)
        assert sizes == [100]

    def test_broadcast_schema(self):
        comm = MultiGPUComm()
        schema = pa.schema([pa.field("x", pa.int64())])
        result = comm.broadcast_schema(schema)
        assert result == schema

    def test_broadcast_none_raises(self):
        comm = MultiGPUComm()
        with pytest.raises(ValueError):
            comm.broadcast_schema(None)

    def test_barrier_no_op(self):
        comm = MultiGPUComm()
        comm.barrier()  # Should not raise

    def test_config_accessible(self):
        config = NCCLConfig(world_size=4, rank=2)
        comm = MultiGPUComm(config)
        assert comm.config.world_size == 4
        assert comm.config.rank == 2
