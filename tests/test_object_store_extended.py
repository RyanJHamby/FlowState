"""Extended tests for fsspec object storage — overwrite, delete, prefix handling."""

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


def _write_file(path: Path, content: bytes = b"test") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


class TestUploadOverwrite:
    def test_upload_overwrites_existing(self, store: ObjectStore, tmp_path: Path):
        src1 = _write_file(tmp_path / "src" / "v1.parquet", b"version1")
        store.upload(src1, "file.parquet")

        src2 = _write_file(tmp_path / "src" / "v2.parquet", b"version2")
        store.upload(src2, "file.parquet")

        dest = tmp_path / "dl" / "file.parquet"
        store.download("file.parquet", dest)
        assert dest.read_bytes() == b"version2"

    def test_upload_creates_nested_dirs(self, store: ObjectStore, tmp_path: Path):
        src = _write_file(tmp_path / "src" / "f.parquet", b"data")
        store.upload(src, "deep/nested/path/file.parquet")
        assert store.exists("deep/nested/path/file.parquet")


class TestDeleteEdgeCases:
    def test_delete_nonexistent_no_error(self, store: ObjectStore):
        # Should not raise
        store.delete("ghost.parquet")

    def test_delete_then_exists(self, store: ObjectStore, tmp_path: Path):
        src = _write_file(tmp_path / "src" / "f.parquet")
        store.upload(src, "f.parquet")
        assert store.exists("f.parquet")
        store.delete("f.parquet")
        assert not store.exists("f.parquet")


class TestListKeys:
    def test_list_with_subprefix(self, store: ObjectStore, tmp_path: Path):
        for i in range(3):
            src = _write_file(tmp_path / "src" / f"f{i}.parquet")
            store.upload(src, f"sub/f{i}.parquet")
        for i in range(2):
            src = _write_file(tmp_path / "src" / f"g{i}.parquet")
            store.upload(src, f"other/g{i}.parquet")

        sub_keys = store.list_keys("sub/")
        assert len(sub_keys) == 3

    def test_list_returns_relative_keys(self, store: ObjectStore, tmp_path: Path):
        src = _write_file(tmp_path / "src" / "test.parquet")
        store.upload(src, "myfile.parquet")
        keys = store.list_keys()
        assert any("myfile.parquet" in k for k in keys)


class TestDownload:
    def test_download_creates_parent_dirs(self, store: ObjectStore, tmp_path: Path):
        src = _write_file(tmp_path / "src" / "f.parquet", b"content")
        store.upload(src, "f.parquet")

        dest = tmp_path / "deep" / "nested" / "output.parquet"
        store.download("f.parquet", dest)
        assert dest.exists()
        assert dest.read_bytes() == b"content"


class TestConfig:
    def test_no_bucket_no_prefix(self, tmp_path: Path):
        config = ObjectStoreConfig(protocol="file")
        store = ObjectStore(config)
        # Should be able to create and use the store
        assert store.config.protocol == "file"
        assert store.config.bucket == ""

    def test_fs_property(self, store: ObjectStore):
        assert store.fs is not None
