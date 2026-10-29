"""HttpRangeFile against a local range-capable server, plus the 200 guard."""

from __future__ import annotations

import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import stream_r2

PAYLOAD = os.urandom(3 * stream_r2.BLOCK + 12345)


class RangeHandler(BaseHTTPRequestHandler):
    ignore_range = False
    hits: list[str] = []

    def log_message(self, *args) -> None:  # keep pytest output clean
        pass

    def do_HEAD(self) -> None:
        self.send_response(200)
        self.send_header("Content-Length", str(len(PAYLOAD)))
        self.send_header("ETag", '"test-etag"')
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

    def do_GET(self) -> None:
        rng = self.headers.get("Range")
        RangeHandler.hits.append(rng or "")
        if rng is None or RangeHandler.ignore_range:
            self.send_response(200)
            self.send_header("Content-Length", str(len(PAYLOAD)))
            self.end_headers()
            # A real server would stream 445 GB here; send a few bytes only.
            self.wfile.write(PAYLOAD[:16])
            return
        start, end = rng.removeprefix("bytes=").split("-")
        start, end = int(start), int(end)
        body = PAYLOAD[start : end + 1]
        self.send_response(206)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Range", f"bytes {start}-{end}/{len(PAYLOAD)}")
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture(scope="module")
def server_url():
    httpd = HTTPServer(("127.0.0.1", 0), RangeHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}/file.h5"
    httpd.shutdown()


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch):
    monkeypatch.setattr(stream_r2, "RETRIES", 2)
    monkeypatch.setattr(stream_r2.time, "sleep", lambda s: None)
    RangeHandler.ignore_range = False
    RangeHandler.hits.clear()


def test_small_reads_hit_block_cache_and_cross_boundaries(server_url):
    f = stream_r2.HttpRangeFile(server_url)
    assert f.size == len(PAYLOAD)
    f.seek(stream_r2.BLOCK - 100)
    assert f.read(300) == PAYLOAD[stream_r2.BLOCK - 100 : stream_r2.BLOCK + 200]
    first = f.requests_made
    f.seek(stream_r2.BLOCK + 10)
    assert f.read(50) == PAYLOAD[stream_r2.BLOCK + 10 : stream_r2.BLOCK + 60]
    assert f.requests_made == first  # served from the cached block
    f.close()


def test_large_read_uses_parallel_pieces_and_reads_to_eof(server_url, monkeypatch):
    monkeypatch.setattr(stream_r2, "PIECE", 1 << 20)
    f = stream_r2.HttpRangeFile(server_url)
    f.seek(5)
    data = f.read()  # to EOF, larger than SMALL_READ
    assert data == PAYLOAD[5:]
    assert f.requests_made > 2
    assert f.read(10) == b""
    f.close()
    assert f.closed


def test_status_200_is_rejected_without_reading_the_body(server_url):
    f = stream_r2.HttpRangeFile(server_url)
    RangeHandler.ignore_range = True
    with pytest.raises(OSError, match="200"):
        f.read(100)
    # A 200 means the remote file changed or Range is ignored; retrying cannot help.
    assert len(RangeHandler.hits) == 1
    f.close()


def test_remote_h5_close_closes_reader(server_url, monkeypatch):
    class FakeReader:
        closed = False

        def close(self):
            self.closed = True

    reader = FakeReader()
    monkeypatch.setattr(stream_r2.h5py.File, "__init__", lambda self, *a, **k: None)
    monkeypatch.setattr(stream_r2.h5py.File, "close", lambda self: None)
    h5 = stream_r2.RemoteH5(reader)
    h5.close()
    assert reader.closed
