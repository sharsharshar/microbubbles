#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["requests", "tqdm"]
# ///
"""Resumable, retrying download of the ~445 GB beamformed reference dataset.

Data is streamed in fixed-size chunks to ``<name>.part`` and appended in
place; if the connection drops (or the script is killed), the next attempt --
or the next run -- sends ``Range: bytes=<size of .part>-`` and continues from
there. ``If-Range`` with the server's ETag guarantees we never splice bytes
from two different versions of the remote file. The ``.part`` file is only
renamed to the final name once its size matches the server's Content-Length.

Usage:
    uv run download_beamformed_reference.py            # deps resolved by uv
    python3 download_beamformed_reference.py           # needs requests + tqdm
    python3 download_beamformed_reference.py --help    # all options
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import signal
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

DEFAULT_URL = (
    "https://pub-9c1be6312b2441eb8732660783d9ee81.r2.dev/"
    "beamformed_reference_216acq.h5"
)
DEFAULT_DIR = Path(__file__).resolve().parent

MB = 1024 * 1024
GB = 1000**3  # decimal GB, matching how `df -H` and the 445 GB file size are quoted
READ_SIZE = 1 * MB  # socket read size; data is written to disk in --chunk-mb blocks

# Cloudflare R2's public endpoint rejects some default library user agents.
USER_AGENT = "beamformed-reference-downloader/1.0"
# (connect, read) seconds. The read timeout applies to each socket read, not
# to the whole transfer, so a stalled connection is detected within a minute.
TIMEOUT = (30, 60)
BACKOFF_BASE = 2.0  # seconds; doubles per consecutive failure
BACKOFF_CAP = 120.0
HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"

RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524}
RETRYABLE_EXC = (
    requests.exceptions.ConnectionError,  # DNS failure, refused, reset, dropped
    requests.exceptions.Timeout,  # connect or read timeout
    requests.exceptions.ChunkedEncodingError,  # connection cut mid-body
    requests.exceptions.ContentDecodingError,
)


class RetryableHTTPError(Exception):
    """Transient HTTP status (5xx, 429, ...) worth retrying."""

    def __init__(self, status: int, retry_after: float | None = None):
        super().__init__(f"HTTP {status}")
        self.retry_after = retry_after


class DownloadError(Exception):
    """Unrecoverable problem; retrying would not help."""


def fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1000 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1000
    raise AssertionError("unreachable")


def describe(exc: BaseException, limit: int = 120) -> str:
    text = str(exc)
    if len(text) > limit:
        text = text[: limit - 3] + "..."
    return f"{type(exc).__name__}: {text}"


def parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None  # HTTP-date form; fall back to normal backoff


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            # Byte offsets must refer to the file itself, never to a
            # compressed transfer encoding of it.
            "Accept-Encoding": "identity",
        }
    )
    return session


def check_status(resp: requests.Response) -> None:
    if resp.status_code in RETRYABLE_STATUS:
        raise RetryableHTTPError(
            resp.status_code, parse_retry_after(resp.headers.get("Retry-After"))
        )
    if resp.status_code >= 400:
        raise DownloadError(f"HTTP {resp.status_code} {resp.reason} for {resp.url}")


def probe(session: requests.Session, url: str) -> tuple[int, str | None, bool]:
    """Return (total size, ETag, server advertises byte ranges)."""
    resp = session.head(url, timeout=TIMEOUT, allow_redirects=True)
    check_status(resp)
    length = resp.headers.get("Content-Length")
    if length is None:
        raise DownloadError("Server did not report Content-Length; cannot resume safely.")
    etag = resp.headers.get("ETag")
    accepts_ranges = resp.headers.get("Accept-Ranges", "").lower() == "bytes"
    return int(length), etag, accepts_ranges


def backoff_delay(failures: int, retry_after: float | None) -> float:
    if retry_after is not None:
        return min(retry_after, BACKOFF_CAP * 5)
    delay = min(BACKOFF_CAP, BACKOFF_BASE * 2 ** (failures - 1))
    return delay * random.uniform(0.75, 1.25)  # jitter avoids synchronized retries


def with_retries(fn, what: str, max_retries: int):
    """Call ``fn()`` retrying transient network failures with backoff."""
    failures = 0
    while True:
        try:
            return fn()
        except (*RETRYABLE_EXC, RetryableHTTPError) as exc:
            failures += 1
            if failures > max_retries:
                raise DownloadError(f"{what} failed after {max_retries} retries: {exc}") from exc
            delay = backoff_delay(failures, getattr(exc, "retry_after", None))
            print(
                f"{what}: {describe(exc)} -- retry {failures}/{max_retries} "
                f"in {delay:.0f}s",
                file=sys.stderr,
            )
            time.sleep(delay)


def file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def check_disk_space(directory: Path, already_have: int, total: int, min_free: int) -> None:
    # The requirement is ``min_free`` for a fresh download. On a resume the
    # bytes already in the .part file count toward it; otherwise a download
    # that is 90% done could never be resumed.
    free = shutil.disk_usage(directory).free
    needed = max(min_free - already_have, total - already_have)
    if free < needed:
        raise DownloadError(
            f"Not enough disk space in {directory}: {fmt_bytes(free)} free, "
            f"{fmt_bytes(needed)} required "
            f"(min {fmt_bytes(min_free)} total, {fmt_bytes(already_have)} already downloaded)."
        )
    print(f"Disk space OK: {fmt_bytes(free)} free, {fmt_bytes(needed)} required.")


def stream_once(
    session: requests.Session,
    url: str,
    part: Path,
    offset: int,
    total: int,
    state: dict,
    chunk_size: int,
    bar: tqdm,
) -> None:
    """One HTTP request: append bytes from ``offset`` until EOF or an error.

    ``state["etag"]`` is the version the .part file belongs to; it is updated
    whenever the server sends a full (200) body, i.e. we start a new version.
    """
    etag = state["etag"]
    headers = {}
    if offset:
        headers["Range"] = f"bytes={offset}-"
        if etag:
            # Server answers 206 only if the file is unchanged; otherwise it
            # sends the whole new file with 200 and we start over.
            headers["If-Range"] = etag

    with session.get(url, headers=headers, stream=True, timeout=TIMEOUT) as resp:
        if resp.status_code == 416:
            raise DownloadError(
                f"Server rejected resume offset {offset} (HTTP 416); the remote file "
                f"may have changed. Delete {part} and re-run."
            )
        check_status(resp)

        if resp.status_code == 200:
            # A full body starts the .part file over, so later resumes must
            # validate against this response's version, not the old one.
            state["etag"] = resp.headers.get("ETag", etag)
            if offset:
                bar.write(
                    "Server ignored the Range request (or the file changed); restarting from 0."
                )
                offset = 0
                bar.reset(total=total)
        elif resp.status_code == 206:
            content_range = resp.headers.get("Content-Range", "")
            expected = f"bytes {offset}-{total - 1}/{total}"
            if content_range != expected:
                raise DownloadError(
                    f"Unexpected Content-Range {content_range!r} (expected {expected!r}); "
                    "refusing to append possibly misaligned data."
                )

        # Read the socket in small pieces but write to disk in ``chunk_size``
        # blocks. urllib3 discards a partially filled read when the
        # connection drops, so reading whole 16 MiB chunks directly would lose
        # up to 16 MiB per drop (and never progress on a link that drops more
        # often than that). Bytes already buffered here are valid and in
        # order, so they are always flushed, even on error.
        buf = bytearray()
        with open(part, "ab" if offset else "wb") as fh:
            try:
                for piece in resp.iter_content(chunk_size=READ_SIZE):
                    buf += piece
                    bar.update(len(piece))
                    if len(buf) >= chunk_size:
                        fh.write(buf)
                        buf.clear()
            finally:
                if buf:
                    fh.write(buf)


def download(
    url: str,
    dest: Path,
    chunk_size: int,
    min_free: int,
    max_retries: int,
) -> Path:
    part = dest.with_name(dest.name + ".part")
    meta = dest.with_name(dest.name + ".part.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    session = make_session()

    total, etag, accepts_ranges = with_retries(
        lambda: probe(session, url), "Fetching file info", max_retries
    )
    print(f"Remote file: {fmt_bytes(total)} ({total:,} bytes), ETag {etag}")
    if not accepts_ranges:
        print("Warning: server does not advertise byte ranges; resume may not work.")

    if file_size(dest) == total:
        print(f"Already complete: {dest}")
        return dest

    # Discard a partial file that cannot belong to the current remote file.
    have = file_size(part)
    if have:
        try:
            saved_etag = json.loads(meta.read_text()).get("etag")
        except (OSError, ValueError):
            saved_etag = None
        if have > total or (etag and saved_etag and saved_etag != etag):
            print("Existing partial download does not match the remote file; starting over.")
            part.unlink()
            have = 0
        elif etag and not saved_etag:
            # The sidecar is written before any data, so this only happens if
            # it was deleted. If-Range below can't catch a change made before
            # this run, so say we are trusting the partial file.
            print(
                f"Warning: no record of which remote version {part.name} came from "
                f"({meta.name} missing); assuming it is the current one."
            )

    check_disk_space(dest.parent, have, total, min_free)
    meta.write_text(json.dumps({"url": url, "etag": etag, "total": total}))
    if have:
        print(f"Resuming from {fmt_bytes(have)} ({100 * have / total:.2f}%).")

    state = {"etag": etag}
    failures = 0
    with tqdm(
        total=total,
        initial=have,
        unit="B",
        unit_scale=True,
        unit_divisor=1000,
        desc=dest.name,
        dynamic_ncols=True,
        smoothing=0.05,  # steadier speed/ETA readout
        miniters=1,
    ) as bar:
        while True:
            offset = file_size(part)
            if offset == total:
                break
            if offset > total:
                raise DownloadError(f"{part} grew past the remote size; delete it and re-run.")

            retry_after = None
            try:
                stream_once(session, url, part, offset, total, state, chunk_size, bar)
                error: Exception | None = None
            except (*RETRYABLE_EXC, RetryableHTTPError) as exc:
                error = exc
                retry_after = getattr(exc, "retry_after", None)
            finally:
                if state["etag"] != etag:
                    etag = state["etag"]
                    meta.write_text(json.dumps({"url": url, "etag": etag, "total": total}))

            size = file_size(part)
            if size == total:
                break
            # Count only consecutive attempts that made no progress, so a long
            # download survives any number of well-separated network blips.
            failures = 1 if size > offset else failures + 1
            if failures > max_retries:
                raise DownloadError(
                    f"Gave up after {max_retries} consecutive attempts without progress "
                    f"(last error: {error or 'connection closed early'}). Re-run to resume."
                )
            delay = backoff_delay(failures, retry_after)
            reason = describe(error) if error else "connection closed early"
            bar.write(
                f"{reason} at {fmt_bytes(size)} -- retry {failures}/{max_retries} in {delay:.0f}s"
            )
            bar.refresh()
            time.sleep(delay)

    with open(part, "rb") as fh:
        if fh.read(len(HDF5_SIGNATURE)) != HDF5_SIGNATURE:
            print("Warning: file does not start with the HDF5 signature.", file=sys.stderr)
    os.replace(part, dest)
    meta.unlink(missing_ok=True)
    print(f"Done: {dest} ({fmt_bytes(total)})")
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=f"destination file (default: <script dir>/<name from URL>, i.e. {DEFAULT_DIR})",
    )
    parser.add_argument("--chunk-mb", type=int, default=16, help="chunk size in MiB (default 16)")
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=450,
        help="required free space in GB for a fresh download (default 450)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=30,
        help="consecutive failed attempts without progress before giving up (default 30)",
    )
    args = parser.parse_args()

    # Treat `kill`, a closed tmux pane, or a shutdown like Ctrl+C: stop
    # cleanly with the partial file intact.
    def on_sigterm(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, on_sigterm)

    dest = args.output or DEFAULT_DIR / args.url.rstrip("/").rsplit("/", 1)[-1]
    try:
        download(
            args.url,
            dest.expanduser().resolve(),
            chunk_size=args.chunk_mb * MB,
            min_free=int(args.min_free_gb * GB),
            max_retries=args.max_retries,
        )
    except KeyboardInterrupt:
        print("\nInterrupted. Partial data is kept; re-run the same command to resume.")
        return 130
    except DownloadError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:  # disk full, permissions, ...
        print(f"File system error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
