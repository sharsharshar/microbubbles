#!/usr/bin/env python3
"""Run ultratrace-ulm against the R2-hosted beamformed reference without a local copy.

The 445 GB file is contiguous and uncompressed, so every acquisition is one
byte range. This wraps the URL in a seekable file object that serves h5py
through parallel HTTP range requests, then patches ``open_h5`` so the normal
CLI (``track``, ``volume``, ...) reads the remote file as if it were local.

    uv run --with requests python stream_r2.py selftest
    uv run --with requests python stream_r2.py track --beamformed R2 --tracks out/tracks.pkl --num-acqs 2 ...

``requests`` is not a project dependency, hence ``--with requests``.

Pass the literal path ``R2`` (or the file's basename) wherever the CLI wants
the beamformed file. Only read paths are patched: ``track-viewer --beamformed``
samples intensities with os.pread on a local file and still needs one.
"""

from __future__ import annotations

import io
import sys
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import requests

URL = "https://pub-9c1be6312b2441eb8732660783d9ee81.r2.dev/beamformed_reference_216acq.h5"
USER_AGENT = "beamformed-reference-downloader/1.0"
REMOTE_NAMES = {"R2", "r2", "beamformed_reference_216acq.h5"}

# r2.dev throttles a single connection to ~45 MB/s from the UT network; 16
# connections of 32 MB pieces reach a few hundred MB/s on a 2 GB acquisition.
WORKERS = 16
PIECE = 32 << 20
# h5py metadata traffic is thousands of reads of a few KB; serve those from
# 4 MB aligned blocks kept in a small LRU so each B-tree walk costs one request.
BLOCK = 4 << 20
SMALL_READ = 1 << 20
CACHE_BLOCKS = 256
TIMEOUT = (30, 60)
RETRIES = 8


class HttpRangeFile(io.RawIOBase):
    """Read-only, seekable file over HTTP range requests."""

    def __init__(self, url: str = URL) -> None:
        super().__init__()
        self.url = url
        self._pos = 0
        self._lock = threading.Lock()
        self._local = threading.local()
        self._cache: OrderedDict[int, bytes] = OrderedDict()
        self._pool = ThreadPoolExecutor(max_workers=WORKERS)
        self.bytes_fetched = 0
        self.requests_made = 0
        head = self._session().head(url, timeout=TIMEOUT, allow_redirects=True)
        head.raise_for_status()
        self.size = int(head.headers["Content-Length"])
        self.etag = head.headers.get("ETag")

    def _session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers["User-Agent"] = USER_AGENT
            self._local.session = session
        return session

    def _fetch(self, start: int, end: int) -> bytes:
        """Bytes [start, end) with retries; If-Range pins the ETag we opened."""
        headers = {"Range": f"bytes={start}-{end - 1}"}
        if self.etag:
            headers["If-Range"] = self.etag
        delay = 2.0
        last: Exception | None = None
        for _ in range(RETRIES):
            try:
                # stream=True: a 200 (If-Range mismatch, or a server ignoring Range)
                # would otherwise read the whole 445 GB object into memory.
                with self._session().get(self.url, headers=headers, timeout=TIMEOUT, stream=True) as resp:
                    if resp.status_code == 200:
                        raise OSError("server answered 200 instead of 206: remote file changed or Range ignored")
                    length = int(resp.headers.get("Content-Length") or -1)
                    if resp.status_code == 206 and length == end - start:
                        data = resp.content
                        if len(data) == end - start:
                            with self._lock:
                                self.bytes_fetched += end - start
                                self.requests_made += 1
                            return data
                        last = OSError(f"short body {len(data)} of {end - start} for {headers['Range']}")
                    else:
                        last = OSError(f"HTTP {resp.status_code}, Content-Length {length} for {headers['Range']}")
            except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError) as exc:
                last = exc
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
        raise OSError(f"range {start}-{end} failed after {RETRIES} attempts: {last}")

    def _block(self, index: int) -> bytes:
        with self._lock:
            data = self._cache.get(index)
            if data is not None:
                self._cache.move_to_end(index)
                return data
        start = index * BLOCK
        data = self._fetch(start, min(start + BLOCK, self.size))
        with self._lock:
            self._cache[index] = data
            while len(self._cache) > CACHE_BLOCKS:
                self._cache.popitem(last=False)
        return data

    def _read_range(self, start: int, end: int, out: memoryview) -> None:
        if end - start <= SMALL_READ:
            pos = start
            while pos < end:
                block = self._block(pos // BLOCK)
                lo = pos - (pos // BLOCK) * BLOCK
                n = min(end - pos, len(block) - lo)
                out[pos - start : pos - start + n] = block[lo : lo + n]
                pos += n
            return
        pieces = [(s, min(s + PIECE, end)) for s in range(start, end, PIECE)]
        for (s, e), data in zip(pieces, self._pool.map(lambda p: self._fetch(*p), pieces)):
            out[s - start : e - start] = data

    # io.RawIOBase interface
    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self.size + offset
        else:
            raise ValueError(f"bad whence {whence}")
        return self._pos

    def readinto(self, buffer) -> int:
        view = memoryview(buffer).cast("B")
        start = self._pos
        end = min(start + len(view), self.size)
        if end <= start:
            return 0
        self._read_range(start, end, view[: end - start])
        self._pos = end
        return end - start

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self.size - self._pos
        out = bytearray(min(size, max(0, self.size - self._pos)))
        n = self.readinto(out)
        return bytes(out[:n])

    def close(self) -> None:
        if not self.closed:
            self._pool.shutdown(wait=False)
            with self._lock:
                self._cache.clear()
        super().close()


class RemoteH5(h5py.File):
    """h5py.File that closes its HTTP reader (and the reader's pool) with the file."""

    def __init__(self, reader: HttpRangeFile) -> None:
        self._reader = reader
        super().__init__(reader, "r")

    def close(self) -> None:
        try:
            super().close()
        finally:
            self._reader.close()


def open_h5_patched(path) -> h5py.File:
    name = Path(str(path)).name
    if name in REMOTE_NAMES and not Path(str(path)).exists():
        return RemoteH5(HttpRangeFile())
    return h5py.File(Path(path).expanduser().resolve(), "r")


def patch() -> None:
    import ultratrace_ulm.h5_io as h5_io
    import ultratrace_ulm.tracking as tracking
    import ultratrace_ulm.track_viewer_export as track_viewer_export
    import ultratrace_ulm.volume_export as volume_export

    for module in (h5_io, tracking, track_viewer_export, volume_export):
        module.open_h5 = open_h5_patched


def selftest() -> None:
    t0 = time.time()
    f = HttpRangeFile()
    h5 = h5py.File(f, "r")
    keys = sorted(int(k) for k in h5["acquisitions"].keys())
    print(f"open+keys: {len(keys)} acqs in {time.time() - t0:.1f}s, {f.requests_made} requests")
    t1 = time.time()
    grid = h5["acquisitions/0/meta/grid/x"][()]
    print(f"grid x {grid.shape} in {time.time() - t1:.1f}s")
    t2 = time.time()
    vol = h5["acquisitions/0/meta/compound_image"][()]
    dt = time.time() - t2
    print(f"compound {vol.shape} {vol.dtype} {vol.nbytes / 1e9:.2f} GB in {dt:.1f}s = {vol.nbytes / 1e6 / dt:.0f} MB/s")
    print(f"total fetched {f.bytes_fetched / 1e9:.2f} GB in {f.requests_made} requests")
    print("frame_rate_hz", float(h5.attrs["frame_rate_hz"]))


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        selftest()
        return
    patch()
    from ultratrace_ulm.cli import main as cli_main

    sys.argv = ["ultratrace-ulm", *sys.argv[1:]]
    cli_main()


if __name__ == "__main__":
    main()
