"""Shared test fixtures for FlowState."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for test data."""
    return tmp_path / "data"


@pytest.fixture
def tmp_parquet_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for Parquet output."""
    d = tmp_path / "parquet"
    d.mkdir()
    return d
