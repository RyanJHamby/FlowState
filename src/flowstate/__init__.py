"""FlowState — Production-grade market data infrastructure for GPU-accelerated ML."""

from flowstate._version import __version__
from flowstate.pipeline import Pipeline, ReplaySession, Schema

__all__ = ["__version__", "Pipeline", "ReplaySession", "Schema"]
