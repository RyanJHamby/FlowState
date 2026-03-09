"""fsspec-based object storage backend for S3/GCS/Azure."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fsspec

logger = logging.getLogger(__name__)


@dataclass
class ObjectStoreConfig:
    """Configuration for object storage backend."""

    protocol: str = "file"  # "s3", "gcs", "az", "file"
    bucket: str = ""
    prefix: str = ""
    storage_options: dict[str, Any] = field(default_factory=dict)


class ObjectStore:
    """Abstraction over cloud object storage using fsspec.

    Supports S3, GCS, Azure Blob Storage, and local filesystem
    with a unified interface.
    """

    def __init__(self, config: ObjectStoreConfig) -> None:
        self._config = config
        self._fs = fsspec.filesystem(config.protocol, **config.storage_options)

    @property
    def config(self) -> ObjectStoreConfig:
        return self._config

    @property
    def fs(self) -> fsspec.AbstractFileSystem:
        return self._fs

    def _remote_path(self, key: str) -> str:
        parts = [self._config.bucket, self._config.prefix, key]
        return "/".join(p for p in parts if p)

    def upload(self, local_path: Path, key: str) -> str:
        """Upload a local file to object storage.

        Args:
            local_path: Path to the local file.
            key: Object key/path in the storage backend.

        Returns:
            The full remote path.
        """
        remote = self._remote_path(key)
        self._fs.makedirs(self._fs._parent(remote), exist_ok=True)
        self._fs.put(str(local_path), remote)
        logger.debug(f"Uploaded {local_path} -> {remote}")
        return remote

    def download(self, key: str, local_path: Path) -> Path:
        """Download a file from object storage.

        Args:
            key: Object key/path in the storage backend.
            local_path: Local destination path.

        Returns:
            The local path.
        """
        remote = self._remote_path(key)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self._fs.get(remote, str(local_path))
        logger.debug(f"Downloaded {remote} -> {local_path}")
        return local_path

    def exists(self, key: str) -> bool:
        """Check if an object exists."""
        return self._fs.exists(self._remote_path(key))

    def delete(self, key: str) -> None:
        """Delete an object."""
        remote = self._remote_path(key)
        if self._fs.exists(remote):
            self._fs.rm(remote)

    def list_keys(self, prefix: str = "") -> list[str]:
        """List object keys under a prefix.

        Args:
            prefix: Key prefix to filter by.

        Returns:
            List of matching keys.
        """
        remote_prefix = self._remote_path(prefix)
        try:
            paths = self._fs.ls(remote_prefix, detail=False)
            base = self._remote_path("")
            return [p.removeprefix(base).lstrip("/") for p in paths]
        except FileNotFoundError:
            return []

    def size(self, key: str) -> int:
        """Get the size of an object in bytes."""
        return self._fs.size(self._remote_path(key))
