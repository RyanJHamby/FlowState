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
    assert hasattr(flowstate, "Pipeline")
    assert hasattr(flowstate, "ReplaySession")
    assert hasattr(flowstate, "Schema")


def test_pipeline_importable():
    from flowstate import Pipeline

    assert Pipeline is not None


def test_replay_session_importable():
    from flowstate import ReplaySession

    assert ReplaySession is not None


def test_schema_importable():
    from flowstate import Schema

    assert Schema.trade() is not None
    assert Schema.quote() is not None
    assert Schema.bar() is not None
