"""PyTorch and JAX DataLoader adapters for market data replay."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import pyarrow as pa

from flowstate.prism.replay import ReplayConfig, ReplayEngine, ReplayFilter

logger = logging.getLogger(__name__)

# Optional PyTorch import
try:
    import torch
    from torch.utils.data import IterableDataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class ArrowBatchConverter:
    """Converts Arrow RecordBatches to framework-specific tensors."""

    @staticmethod
    def to_numpy(batch: pa.RecordBatch, columns: list[str] | None = None) -> dict[str, Any]:
        """Convert selected columns to numpy arrays."""

        result = {}
        cols = columns or batch.schema.names
        for name in cols:
            if name not in batch.schema.names:
                continue
            col = batch.column(name)
            if pa.types.is_floating(col.type) or pa.types.is_integer(col.type):
                result[name] = col.to_numpy()
            elif pa.types.is_timestamp(col.type):
                result[name] = col.cast(pa.int64()).to_numpy()
            else:
                result[name] = col.to_pylist()
        return result


class FlowStateIterableDataset:
    """PyTorch-compatible IterableDataset backed by the replay engine.

    If PyTorch is not installed, this still works as a plain Python iterator.
    """

    def __init__(
        self,
        data_dir: str,
        replay_filter: ReplayFilter | None = None,
        replay_config: ReplayConfig | None = None,
        numeric_columns: list[str] | None = None,
    ) -> None:
        self._engine = ReplayEngine(data_dir, config=replay_config)
        self._filter = replay_filter
        self._numeric_columns = numeric_columns
        self._converter = ArrowBatchConverter()

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for batch in self._engine.replay(self._filter):
            yield self._converter.to_numpy(batch, self._numeric_columns)

    def __len__(self) -> int:
        return self._engine.count(self._filter)


# Conditionally inherit from IterableDataset if PyTorch is available
if HAS_TORCH:
    class FlowStateTorchDataset(IterableDataset):  # type: ignore[misc]
        """PyTorch IterableDataset for FlowState data."""

        def __init__(
            self,
            data_dir: str,
            replay_filter: ReplayFilter | None = None,
            replay_config: ReplayConfig | None = None,
            numeric_columns: list[str] | None = None,
            tensor_dtype: Any = None,
        ) -> None:
            super().__init__()
            self._inner = FlowStateIterableDataset(
                data_dir, replay_filter, replay_config, numeric_columns
            )
            self._dtype = tensor_dtype or torch.float32

        def __iter__(self) -> Iterator[dict[str, Any]]:
            import numpy as np

            for batch_dict in self._inner:
                tensor_dict = {}
                for k, v in batch_dict.items():
                    if isinstance(v, np.ndarray):
                        tensor_dict[k] = torch.from_numpy(v).to(self._dtype)
                    else:
                        tensor_dict[k] = v
                yield tensor_dict


class JAXDataIterator:
    """JAX-compatible data iterator for FlowState data.

    Converts Arrow batches to JAX arrays when JAX is available,
    otherwise returns numpy arrays.
    """

    def __init__(
        self,
        data_dir: str,
        replay_filter: ReplayFilter | None = None,
        replay_config: ReplayConfig | None = None,
        numeric_columns: list[str] | None = None,
    ) -> None:
        self._inner = FlowStateIterableDataset(
            data_dir, replay_filter, replay_config, numeric_columns
        )

    def __iter__(self) -> Iterator[dict[str, Any]]:
        try:
            import jax.numpy as jnp

            for batch_dict in self._inner:
                jax_dict = {}
                for k, v in batch_dict.items():
                    try:
                        jax_dict[k] = jnp.array(v)
                    except Exception:
                        jax_dict[k] = v
                yield jax_dict
        except ImportError:
            # Fall back to numpy
            yield from self._inner
