"""S3Store.ensure_bucket: the API and workers start together against a fresh
object store, so bucket creation must tolerate losing the race."""

from __future__ import annotations

from typing import Any

import pytest
from botocore.exceptions import ClientError

from wildinbox.storage.objects import S3Store


def _error(code: str, operation: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": code}}, operation)


class FakeClient:
    def __init__(self, head: str | None, create: str | None) -> None:
        self.head, self.create = head, create
        self.created = 0

    def head_bucket(self, **_: Any) -> None:
        if self.head:
            raise _error(self.head, "HeadBucket")

    def create_bucket(self, **_: Any) -> None:
        self.created += 1
        if self.create:
            raise _error(self.create, "CreateBucket")


def _store(client: FakeClient) -> S3Store:
    store = S3Store.__new__(S3Store)  # no boto3 client, no network
    store.bucket = "wildinbox"
    store.client = client
    return store


def test_existing_bucket_is_left_alone() -> None:
    client = FakeClient(head=None, create=None)
    _store(client).ensure_bucket()
    assert client.created == 0


def test_missing_bucket_is_created() -> None:
    client = FakeClient(head="404", create=None)
    _store(client).ensure_bucket()
    assert client.created == 1


def test_losing_the_creation_race_is_success() -> None:
    # Seen in CI: api and worker both got 404, the worker created the bucket
    # first, and the API crashed on BucketAlreadyOwnedByYou.
    client = FakeClient(head="404", create="BucketAlreadyOwnedByYou")
    _store(client).ensure_bucket()
    assert client.created == 1


@pytest.mark.parametrize("code", ["BucketAlreadyExists", "AccessDenied", "InvalidBucketName"])
def test_other_creation_errors_still_fail(code: str) -> None:
    with pytest.raises(ClientError):
        _store(FakeClient(head="404", create=code)).ensure_bucket()
