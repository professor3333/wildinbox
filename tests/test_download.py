from __future__ import annotations

import hashlib
import io
import tarfile
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from wildinbox.ingestion import download as dl
from wildinbox.ingestion.download import DownloadError, download, extract
from wildinbox.ingestion.sources import Archive

PAYLOAD = bytes(range(256)) * 4096  # 1 MiB


class _Server:
    """Serves PAYLOAD with Range support; can cut the connection early."""

    def __init__(self) -> None:
        self.cut_after: int | None = None  # bytes to send before dropping, once
        self.honor_range = True
        self.requests: list[str | None] = []

    def handler(self) -> type[BaseHTTPRequestHandler]:
        state = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                pass

            def do_GET(self) -> None:
                rng = self.headers.get("Range")
                state.requests.append(rng)
                start = 0
                if rng and state.honor_range:
                    start = int(rng.removeprefix("bytes=").split("-")[0])
                    self.send_response(206)
                    self.send_header(
                        "Content-Range", f"bytes {start}-{len(PAYLOAD) - 1}/{len(PAYLOAD)}"
                    )
                else:
                    self.send_response(200)
                body = PAYLOAD[start:]
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if state.cut_after is not None:
                    self.wfile.write(body[: state.cut_after])
                    state.cut_after = None
                    self.close_connection = True
                    return
                self.wfile.write(body)

        return H


@pytest.fixture
def server() -> Iterator[tuple[_Server, str]]:
    state = _Server()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), state.handler())
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield state, f"http://127.0.0.1:{httpd.server_address[1]}/a.bin"
    httpd.shutdown()


def _archive(url: str, md5: str | None = None) -> Archive:
    # Archive requires https in configs; tests build it unchecked for a local server.
    return Archive.model_construct(
        url=url,
        filename="a.bin",
        size=len(PAYLOAD),
        md5=md5 or hashlib.md5(PAYLOAD).hexdigest(),
    )


def test_interrupted_download_resumes_from_partial_file(
    server: tuple[_Server, str], tmp_path: Path
) -> None:
    state, url = server
    state.cut_after = 300_000
    with pytest.raises(DownloadError, match="re-run to resume"):
        download(_archive(url), tmp_path, retries=1, backoff=0)
    part = tmp_path / "a.bin.part"
    assert part.stat().st_size == 300_000
    assert not (tmp_path / "a.bin").exists()  # never exposed under the final name

    out = download(_archive(url), tmp_path, retries=1, backoff=0)
    assert out.read_bytes() == PAYLOAD
    assert state.requests[-1] == "bytes=300000-"
    assert not part.exists()


def test_retry_within_one_call_resumes(server: tuple[_Server, str], tmp_path: Path) -> None:
    state, url = server
    state.cut_after = 500_000
    out = download(_archive(url), tmp_path, retries=3, backoff=0)
    assert out.read_bytes() == PAYLOAD
    assert state.requests == [None, "bytes=500000-"]


def test_completed_download_is_not_fetched_again(
    server: tuple[_Server, str], tmp_path: Path
) -> None:
    state, url = server
    download(_archive(url), tmp_path, backoff=0)
    download(_archive(url), tmp_path, backoff=0)
    assert len(state.requests) == 1


def test_server_ignoring_range_restarts_cleanly(
    server: tuple[_Server, str], tmp_path: Path
) -> None:
    state, url = server
    state.honor_range = False
    (tmp_path / "a.bin.part").write_bytes(PAYLOAD[:1000])
    out = download(_archive(url), tmp_path, backoff=0)
    assert out.read_bytes() == PAYLOAD


def test_checksum_mismatch_is_rejected_and_discarded(
    server: tuple[_Server, str], tmp_path: Path
) -> None:
    _, url = server
    with pytest.raises(DownloadError, match="MD5 mismatch"):
        download(_archive(url, md5="0" * 32), tmp_path, backoff=0)
    assert not (tmp_path / "a.bin").exists()
    assert not (tmp_path / "a.bin.part").exists()


def _tar(tmp_path: Path, files: dict[str, bytes]) -> Path:
    path = tmp_path / "x.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path


def test_extract_repairs_truncated_files_and_is_idempotent(tmp_path: Path) -> None:
    files = {"imgs/a.jpg": b"a" * 100, "imgs/b.jpg": b"b" * 200}
    archive = _tar(tmp_path, files)
    dest = tmp_path / "out"
    # Simulate an interrupted earlier extraction: one truncated file, one leftover tmp.
    (dest / "imgs").mkdir(parents=True)
    (dest / "imgs" / "a.jpg").write_bytes(b"a" * 10)
    (dest / "imgs" / "b.jpg.tmp").write_bytes(b"junk")
    assert extract(archive, dest) == 2
    assert (dest / "imgs" / "a.jpg").read_bytes() == files["imgs/a.jpg"]
    assert extract(archive, dest) == 0  # marker: nothing to do


def test_extract_rejects_path_traversal(tmp_path: Path) -> None:
    archive = _tar(tmp_path, {"../evil.txt": b"x"})
    with pytest.raises(DownloadError, match="unsafe"):
        extract(archive, tmp_path / "out")


def test_md5_helper(tmp_path: Path) -> None:
    p = tmp_path / "f"
    p.write_bytes(PAYLOAD)
    assert dl.md5_of(p) == hashlib.md5(PAYLOAD).hexdigest()
