"""Tests for fsspec-based object storage backend."""

from __future__ import annotations

from pathlib import Path

import pytest

from flowstate.storage.object_store import ObjectStore, ObjectStoreConfig


@pytest.fixture
def store(tmp_path: Path) -> ObjectStore:
    config = ObjectStoreConfig(
        protocol="file",
        bucket=str(tmp_path / "bucket"),
        prefix="data",
    )
    (tmp_path / "bucket" / "data").mkdir(parents=True)
    return ObjectStore(config)


class TestObjectStore:
    def test_upload_and_download(self, store: ObjectStore, tmp_path: Path):
        # Create a source file
        src = tmp_path / "local" / "test.parquet"
        src.parent.mkdir(parents=True)
        src.write_bytes(b"parquet data")

        store.upload(src, "part-00000.parquet")
        assert store.exists("part-00000.parquet")

        dest = tmp_path / "downloaded" / "test.parquet"
        store.download("part-00000.parquet", dest)
        assert dest.exists()
        assert dest.read_bytes() == b"parquet data"

    def test_exists(self, store: ObjectStore):
        assert not store.exists("nonexistent.parquet")

    def test_delete(self, store: ObjectStore, tmp_path: Path):
        src = tmp_path / "local" / "test.parquet"
        src.parent.mkdir(parents=True)
        src.write_bytes(b"data")

        store.upload(src, "to_delete.parquet")
        assert store.exists("to_delete.parquet")
        store.delete("to_delete.parquet")
        assert not store.exists("to_delete.parquet")

    def test_list_keys(self, store: ObjectStore, tmp_path: Path):
        for i in range(3):
            src = tmp_path / "local" / f"file{i}.parquet"
            src.parent.mkdir(parents=True, exist_ok=True)
            src.write_bytes(b"data")
            store.upload(src, f"file{i}.parquet")

        keys = store.list_keys()
        assert len(keys) == 3

    def test_list_keys_empty(self, store: ObjectStore):
        keys = store.list_keys("nonexistent/")
        assert keys == []

    def test_size(self, store: ObjectStore, tmp_path: Path):
        src = tmp_path / "local" / "test.parquet"
        src.parent.mkdir(parents=True)
        src.write_bytes(b"x" * 42)
        store.upload(src, "sized.parquet")
        assert store.size("sized.parquet") == 42

    def test_config_accessible(self, store: ObjectStore):
        assert store.config.protocol == "file"
