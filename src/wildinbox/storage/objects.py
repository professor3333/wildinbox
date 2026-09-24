"""Object storage for originals (and later thumbnails and model artifacts).

Originals are content-addressed (`originals/<sha256>`), so storing the same
file twice is harmless and never duplicates data.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol

from wildinbox.settings import Settings


class ObjectNotFoundError(KeyError):
    pass


class ObjectStore(Protocol):
    def put(self, key: str, data: bytes, content_type: str) -> None: ...
    def get(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...


def original_key(sha256: str) -> str:
    return f"originals/{sha256[:2]}/{sha256}"


class LocalStore:
    """Directory-backed store for tests and single-machine development."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if self.root.resolve() not in path.parents:
            raise ValueError(f"invalid object key {key!r}")
        return path

    def put(self, key: str, data: bytes, content_type: str) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    def get(self, key: str) -> bytes:
        try:
            return self._path(key).read_bytes()
        except FileNotFoundError:
            raise ObjectNotFoundError(key) from None

    def exists(self, key: str) -> bool:
        return self._path(key).exists()


class S3Store:
    """Any S3-compatible service (SeaweedFS in the local docker compose stack)."""

    def __init__(self, settings: Settings) -> None:
        import boto3
        from botocore.config import Config

        self.bucket = settings.object_store_bucket
        self.client: Any = boto3.client(
            "s3",
            endpoint_url=settings.object_store_url,
            aws_access_key_id=settings.object_store_access_key,
            aws_secret_access_key=settings.object_store_secret_key,
            region_name=settings.object_store_region,
            config=Config(
                retries={"max_attempts": 5, "mode": "standard"}, s3={"addressing_style": "path"}
            ),
        )

    def ensure_bucket(self) -> None:
        from botocore.exceptions import ClientError

        try:
            self.client.head_bucket(Bucket=self.bucket)
        except ClientError:
            self.client.create_bucket(Bucket=self.bucket)

    def put(self, key: str, data: bytes, content_type: str) -> None:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)

    def get(self, key: str) -> bytes:
        from botocore.exceptions import ClientError

        try:
            body: bytes = self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
            return body
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                raise ObjectNotFoundError(key) from None
            raise

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False


def store_from_settings(settings: Settings) -> ObjectStore:
    if settings.object_store_backend == "local":
        return LocalStore(settings.local_store_dir)
    store = S3Store(settings)
    store.ensure_bucket()
    return store
