"""Verify the public API surface of FlowState."""

from __future__ import annotations


def test_version_is_string():
    from flowstate import __version__

    assert isinstance(__version__, str)
    parts = __version__.split(".")
    assert len(parts) == 3


def test_top_level_imports():
    import flowstate

    assert hasattr(flowstate, "__version__")
