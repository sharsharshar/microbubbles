"""Export ULM tracks to the compact binary the 3D track viewer reads.

Produces ``tracks.bin`` in the viewer's v3 layout:

    Header (64 bytes):
        0:  uint32  magic         0x554C4D54 ("ULMT")
        4:  uint32  version       3
        8:  uint32  n_tracks
        12: uint32  total_points
        16: float32 max_speed     (mm/frame, after smoothing)
        20: 3xf32   bounds_min    (x, y, z) mm
        32: 3xf32   bounds_max    (x, y, z) mm
        44: padding to 64 bytes
    Track table (n_tracks x 8 bytes): uint32 point_offset, uint32 length
    Point data (total_points x 24 bytes): f32 x, y, z, frame, speed, intensity

Optional per-point B-mode intensity is sampled from a beamformed ultratrace H5
using this package's own SVD clutter filter. The runtime stays self-contained:
no external dependencies beyond numpy and scipy.
"""

from __future__ import annotations

import os
import shutil
import struct
from importlib import resources
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates

from .h5_io import compound_dataset, grid_arrays, acq_keys, open_h5
from .runtime import load_pickle, thread_map
from .svd import filter_svd_3d

MAGIC = 0x554C4D54
VERSION = 3
# Concurrent point reads; os.pread releases the GIL, so these overlap their I/O.
_READ_WORKERS = 16


def _track_length(track: dict) -> int:
    return int(track.get("length", len(track["positions"])))


def smooth_track_speeds(positions: np.ndarray, frames: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    """Gaussian-smoothed speed (mm/frame) at each point along a track."""
    n = len(positions)
    if n < 2:
        return np.zeros(n, dtype=np.float32)

    dp = np.diff(positions, axis=0)
    df = np.diff(frames).astype(np.float64)
    df[df == 0] = 1.0
    vel = dp / df[:, np.newaxis]

    vel_full = np.empty((n, 3), dtype=np.float64)
    vel_full[0] = vel[0]
    vel_full[-1] = vel[-1]
    vel_full[1:-1] = 0.5 * (vel[:-1] + vel[1:])

    if n > 2 * sigma:
        for axis in range(3):
            vel_full[:, axis] = gaussian_filter1d(vel_full[:, axis], sigma=sigma)

    return np.linalg.norm(vel_full, axis=1).astype(np.float32)


def _grids(data: dict, h5_path: Path | None):
    """Return (grid_x, grid_y, grid_z) in mm, each shaped (elev, z, x)."""
    if all(k in data for k in ("grid_x", "grid_y", "grid_z")):
        return data["grid_x"], data["grid_y"], data["grid_z"]
    if h5_path is None:
        raise ValueError("Pickle has no grid arrays; pass --beamformed to read the grid from H5")
    with open_h5(h5_path) as h5:
        acq = acq_keys(h5)[0]
        return grid_arrays(h5, acq)


def _is_plain_complex64(stored: h5py.h5t.TypeID) -> bool:
    """Whether the stored type's bytes are packed little-endian float32 (real, imag)
    pairs: h5py's own complex64 compound, or HDF5 2's native complex type."""
    if stored.equal(h5py.h5t.py_create(np.dtype("<c8"))):
        return True
    native = getattr(h5py.h5t, "COMPLEX_IEEE_F32LE", None)
    return native is not None and stored.equal(native)


def _raw_offset(dset: h5py.Dataset, h5: h5py.File) -> int | None:
    """Byte offset of ``dset`` in ``h5``'s file when its raw bytes there are exactly
    what h5py would return as a (frames, elev, z, x) complex64 array, else None.

    Each check rules out a layout where reading at get_offset() would not match
    h5py: a dataset reached through an external link lives in another file;
    chunked, compact or external-storage data is not one run of bytes; h5py
    reports padded or non-IEEE {r, i} compounds as complex64 although their bytes
    differ; unallocated storage reads as fill values; os.pread is Unix-only.
    """
    if not hasattr(os, "pread") or dset.ndim != 4 or dset.file != h5:
        return None
    if dset.chunks is not None or dset.external or not _is_plain_complex64(dset.id.get_type()):
        return None
    if dset.id.get_storage_size() != dset.nbytes:
        return None
    offset = dset.id.get_offset()
    return None if offset is None else int(offset)


def _interp_bounds(coord: float, n: int) -> tuple[int, int]:
    """Index range [lo, hi) holding every voxel that order-1, mode="nearest"
    interpolation at ``coord`` reads, plus a voxel of margin either side.

    mode="nearest" maps out-of-range taps to the edge voxel, so a block that
    reaches the array edge wherever the taps pass it gives the full-array value.
    scipy takes floor(coord) as a 64-bit integer, so for non-finite coordinates
    and beyond +-2**63 the taps are arbitrary: take the whole axis there, which
    reproduces whatever map_coordinates does on the full frame.
    """
    if not np.isfinite(coord) or abs(coord) >= 2.0**63:
        return 0, n
    base = int(np.floor(min(max(coord, 0.0), n - 1.0)))
    return max(0, base - 1), min(n, base + 3)


def _sample_raw(fd: int, offset: int, shape: tuple[int, ...], frame: int, coords: np.ndarray) -> np.float32:
    """``map_coordinates(|compound[frame]|, coords, order=1, mode="nearest")[0]``
    computed from only the voxels it reads.

    One ``os.pread`` per elevation plane fetches the needed (z, x) rows, instead
    of reading the whole acquisition. Coordinates are shifted by the block origin;
    the shift is exact for whole-number offsets, so the value is bit-identical.
    """
    _, n_elev, n_z, n_x = shape
    (e0, e1), (z0, z1), (x0, x1) = (
        _interp_bounds(float(c), n) for c, n in zip(coords[:, 0], (n_elev, n_z, n_x))
    )
    row_bytes = n_x * 8
    nbytes = (z1 - z0) * row_bytes
    block = np.empty((e1 - e0, z1 - z0, x1 - x0), dtype=np.complex64)
    for i, elev in enumerate(range(e0, e1)):
        start = offset + ((frame * n_elev + elev) * n_z + z0) * row_bytes
        raw = os.pread(fd, nbytes, start)
        if len(raw) != nbytes:
            raise OSError(f"short read: {len(raw)} of {nbytes} bytes at offset {start}")
        block[i] = np.frombuffer(raw, dtype="<c8").reshape(z1 - z0, n_x)[:, x0:x1]
    local = coords - np.array([[e0], [z0], [x0]], dtype=np.float64)
    return map_coordinates(np.abs(block), local, order=1, mode="nearest")[0]


def _sample_raw_points(
    fd: int, offset: int, shape: tuple[int, ...], points: list[tuple[int, np.ndarray]]
) -> list[np.float32]:
    """_sample_raw over (frame, coords) points, with their reads overlapped on threads."""
    return thread_map(lambda point: _sample_raw(fd, offset, shape, *point), points, max_workers=_READ_WORKERS)


def lookup_intensities(tracks, all_speeds, h5_path, grids, pkl_n_acquisitions=None, svd_cutoff=0.0,
                       selected_acq_ids=None):
    """Sample SVD-filtered B-mode magnitude at each track point, normalized to [0, 1].

    ``selected_acq_ids`` (the tracked acquisitions, in order) maps the global frame
    index back to its acquisition; without it the last ``pkl_n_acquisitions`` are assumed.
    """
    grid_x, grid_y, grid_z = grids
    n_elev, n_z, n_x = grid_x.shape
    x_min, x_max = float(grid_x[0, 0, 0]), float(grid_x[0, 0, -1])
    y_min, y_max = float(grid_y[0, 0, 0]), float(grid_y[-1, 0, 0])
    z_min, z_max = float(grid_z[0, 0, 0]), float(grid_z[0, -1, 0])

    all_intensities = [np.ones(len(s), dtype=np.float32) for s in all_speeds]

    with open_h5(h5_path) as h5, open(Path(h5_path).expanduser().resolve(), "rb", buffering=0) as raw:
        # Read-write intent on a read-only open means a writer in this process
        # shares the file and may still hold writes in HDF5's buffers.
        shared_with_writer = bool(h5.id.get_intent() & h5py.h5f.ACC_RDWR)
        all_acq_ids = acq_keys(h5)
        n_h5_acqs = len(all_acq_ids)
        if selected_acq_ids:
            acq_ids = [int(a) for a in selected_acq_ids]
        elif pkl_n_acquisitions is not None and pkl_n_acquisitions < n_h5_acqs:
            acq_ids = all_acq_ids[-pkl_n_acquisitions:]
        else:
            acq_ids = all_acq_ids

        acq_frame_offsets = []
        global_offset = 0
        for acq_id in acq_ids:
            n_frames = compound_dataset(h5, acq_id).shape[0]
            acq_frame_offsets.append((acq_id, global_offset, n_frames))
            global_offset += n_frames

        records = []
        for ti, t in enumerate(tracks):
            pos, frames = t["positions"], t["frames"]
            for pi in range(len(frames)):
                records.append((ti, pi, int(frames[pi]),
                                float(pos[pi, 0]), float(pos[pi, 1]), float(pos[pi, 2])))
        records.sort(key=lambda r: r[2])

        rec_idx = 0
        for acq_id, acq_start, acq_n in acq_frame_offsets:
            acq_end = acq_start + acq_n
            acq_points = []
            while rec_idx < len(records) and records[rec_idx][2] < acq_end:
                if records[rec_idx][2] >= acq_start:
                    acq_points.append(records[rec_idx])
                rec_idx += 1
            if not acq_points:
                continue

            points = []
            for ti, pi, gframe, x_mm, y_mm, z_mm in acq_points:
                x_idx = (x_mm - x_min) / (x_max - x_min) * (n_x - 1) if n_x > 1 else 0.0
                elev_idx = (y_mm - y_min) / (y_max - y_min) * (n_elev - 1) if n_elev > 1 else 0.0
                z_idx = (z_mm - z_min) / (z_max - z_min) * (n_z - 1) if n_z > 1 else 0.0
                points.append((ti, pi, gframe - acq_start, np.array([[elev_idx], [z_idx], [x_idx]])))

            dset = compound_dataset(h5, acq_id)  # (frames, elev, z, x)
            offset = None if svd_cutoff > 0 else _raw_offset(dset, h5)
            # Each point reads up to 4 elevation planes x 4 rows. Scattered row
            # reads cost more per byte than one sequential read, and on the
            # reference grid the cold-cache times cross near half the dataset's
            # size (about 30k points per acquisition): past that, read it whole.
            sparse_bytes = len(points) * 16 * dset.shape[-1] * dset.dtype.itemsize
            if offset is not None and sparse_bytes < dset.nbytes // 2:
                if shared_with_writer:
                    # HDF5 keeps a dataset's buffered writes with the open dataset,
                    # so flush now that this handle has it open, before raw reads.
                    h5.flush()
                # Raw magnitude needs a few rows per point, not the whole
                # acquisition (~2 GB each, so a full pass re-reads the entire file).
                values = _sample_raw_points(
                    raw.fileno(), offset, dset.shape, [(fia, coords) for _, _, fia, coords in points]
                )
            else:
                compound_raw = np.asarray(dset)
                if compound_raw.ndim == 3:
                    # (frames, z, x) is one elevation plane, as tracking reads it.
                    compound_raw = compound_raw[:, None]
                if svd_cutoff > 0:
                    # Stable full SVD: the covariance ("fast") variant is numerically
                    # unstable in float32 and not reproducible across backends.
                    compound = np.abs(filter_svd_3d(compound_raw, low_cutoff=svd_cutoff, method="full"))
                else:
                    compound = np.abs(compound_raw)
                values = [
                    map_coordinates(compound[fia], coords, order=1, mode="nearest")[0]
                    for _, _, fia, coords in points
                ]
            for (ti, pi, _, _), value in zip(points, values):
                all_intensities[ti][pi] = value

    all_vals = np.concatenate(all_intensities)
    p1, p99 = np.percentile(all_vals, [1, 99])
    if p99 > p1:
        for i in range(len(all_intensities)):
            all_intensities[i] = np.clip((all_intensities[i] - p1) / (p99 - p1), 0.0, 1.0).astype(np.float32)
    else:
        all_intensities = [np.ones_like(a) for a in all_intensities]
    return all_intensities


def export_tracks_bin_v3(
    pickle_path: Path,
    output_path: Path,
    min_length: int = 35,
    sigma: float = 10.0,
    use_smoothed: bool = True,
    beamformed: Path | None = None,
    svd_cutoff: float = 0.0,
) -> Path:
    data = load_pickle(pickle_path)
    key = "tracks_smoothed" if use_smoothed and data.get("tracks_smoothed") else "tracks"
    tracks = [t for t in data[key] if _track_length(t) >= min_length]
    if not tracks:
        raise SystemExit(f"No tracks with length >= {min_length} in {pickle_path}")

    n_tracks = len(tracks)
    total_points = sum(_track_length(t) for t in tracks)

    all_speeds, all_mins, all_maxs, max_speed = [], [], [], 0.0
    for t in tracks:
        pos = t["positions"]  # keep original dtype for speed/bounds (cast only at write)
        all_mins.append(np.asarray(pos).min(axis=0))
        all_maxs.append(np.asarray(pos).max(axis=0))
        s = smooth_track_speeds(np.asarray(pos), np.asarray(t["frames"]), sigma=sigma)
        all_speeds.append(s)
        if len(s):
            max_speed = max(max_speed, float(s.max()))
    bounds_min = np.min(all_mins, axis=0).astype(np.float32)
    bounds_max = np.max(all_maxs, axis=0).astype(np.float32)

    if beamformed is not None:
        intensities = lookup_intensities(
            tracks, all_speeds, beamformed, _grids(data, beamformed),
            pkl_n_acquisitions=data.get("n_acquisitions"), svd_cutoff=svd_cutoff,
            selected_acq_ids=(data.get("params") or {}).get("selected_acq_ids"),
        )
    else:
        intensities = [np.ones(len(s), dtype=np.float32) for s in all_speeds]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as fp:
        header = struct.pack("<IIIIf3f3f", MAGIC, VERSION, n_tracks, total_points,
                             max_speed, *bounds_min.tolist(), *bounds_max.tolist())
        fp.write(header + b"\x00" * (64 - len(header)))
        offset = 0
        for t in tracks:
            length = _track_length(t)
            fp.write(struct.pack("<II", offset, length))
            offset += length
        for t, speeds, ints in zip(tracks, all_speeds, intensities):
            pos = np.asarray(t["positions"], dtype=np.float32)
            frames = np.asarray(t["frames"], dtype=np.float32)
            point_data = np.column_stack([pos, frames[:, None], speeds[:, None], ints[:, None]])
            fp.write(point_data.astype(np.float32).tobytes())

    print(f"Exported {n_tracks} tracks ({total_points} points) -> {output_path}")
    return output_path


def _copy_web_assets(output_dir: Path) -> None:
    asset_root = resources.files("ultratrace_ulm.web.track_viewer")
    with resources.as_file(asset_root / "index.html") as src:
        shutil.copyfile(src, output_dir / "index.html")


def write_track_viewer(
    pickle_path: Path,
    output_dir: Path,
    min_length: int = 35,
    sigma: float = 10.0,
    use_smoothed: bool = True,
    beamformed: Path | None = None,
    svd_cutoff: float = 0.0,
) -> Path:
    """Write a self-contained 3D track viewer bundle (index.html + data/tracks.bin)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    export_tracks_bin_v3(
        pickle_path, output_dir / "data" / "tracks.bin",
        min_length=min_length, sigma=sigma, use_smoothed=use_smoothed,
        beamformed=beamformed, svd_cutoff=svd_cutoff,
    )
    _copy_web_assets(output_dir)
    print(f"Wrote 3D track viewer bundle to {output_dir}")
    return output_dir / "index.html"
