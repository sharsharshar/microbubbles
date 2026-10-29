#!/usr/bin/env python3
"""Per-acquisition track counts and flow statistics for a tracks pickle.

    uv run track_stats.py runs/baseline216/tracks.pkl [--min-length N] [--json out.json]

A track is assigned to the acquisition of its first frame. "Flowing" means the
net displacement between first and last point is at least ``--flow-mm`` (one
voxel on the reference grid is about 0.19 mm axially), which separates bubble
tracks from tissue speckle that jitters in place.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def load(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def summarize(data: dict, min_length: int, flow_mm: float) -> dict:
    tracks = [t for t in data["tracks"] if len(t["frames"]) >= min_length]
    frames_per_acq = int(data["frames_per_acq"])
    n_acq = int(data["n_acquisitions"])
    params = data.get("params", {})
    frame_rate = float(params.get("frame_rate_hz") or 0.0)

    starts = np.array([int(t["frames"][0]) // frames_per_acq for t in tracks], dtype=np.int64)
    lengths = np.array([len(t["frames"]) for t in tracks], dtype=np.int64)
    net = np.array(
        [float(np.linalg.norm(t["positions"][-1] - t["positions"][0])) for t in tracks], dtype=np.float64
    )
    path_len = np.array(
        [float(np.sum(np.linalg.norm(np.diff(t["positions"], axis=0), axis=1))) for t in tracks],
        dtype=np.float64,
    )
    duration_s = np.array(
        [(int(t["frames"][-1]) - int(t["frames"][0]) + 1) / frame_rate if frame_rate else np.nan for t in tracks]
    )
    mean_speed = np.where(duration_s > 0, path_len / duration_s, np.nan)
    flowing = net >= flow_mm
    # Straightness (net / path length) separates directed bubble motion from
    # random-walk links between noise detections, which my flow threshold alone
    # cannot: a 5-frame noise chain can also wander 0.5 mm within the gate.
    straight = np.where(path_len > 0, net / np.maximum(path_len, 1e-9), 0.0)
    cross = np.array([int(t["frames"][0]) // frames_per_acq != int(t["frames"][-1]) // frames_per_acq for t in tracks])
    per_acq = np.bincount(starts, minlength=n_acq)
    per_acq_flowing = np.bincount(starts[flowing], minlength=n_acq)

    det = data.get("detections", {})
    n_det = int(len(det.get("frame_indices", [])))

    return {
        "min_length": min_length,
        "n_acquisitions": n_acq,
        "frames_per_acq": frames_per_acq,
        "n_detections": n_det,
        "detections_per_frame": n_det / max(1, data.get("n_frames", 1)),
        "n_tracks": int(len(tracks)),
        "tracks_per_acq_mean": float(per_acq.mean()),
        "tracks_per_acq_median": float(np.median(per_acq)),
        "tracks_per_acq_min": int(per_acq.min()),
        "tracks_per_acq_max": int(per_acq.max()),
        "flowing_fraction": float(flowing.mean()) if len(tracks) else 0.0,
        "flowing_tracks_per_acq_mean": float(per_acq_flowing.mean()),
        "cross_boundary_tracks": int(cross.sum()),
        "straightness_median": float(np.median(straight)) if len(tracks) else 0.0,
        "straight_fraction": float((straight >= 0.8).mean()) if len(tracks) else 0.0,
        "length_median": float(np.median(lengths)) if len(tracks) else 0.0,
        "length_mean": float(lengths.mean()) if len(tracks) else 0.0,
        "net_displacement_mm_median": float(np.median(net)) if len(tracks) else 0.0,
        "mean_speed_mm_s_median": float(np.nanmedian(mean_speed)) if len(tracks) else 0.0,
        "mean_speed_mm_s_p90": float(np.nanpercentile(mean_speed, 90)) if len(tracks) else 0.0,
        "per_acq": per_acq.tolist(),
        "params": {k: v for k, v in params.items() if k != "selected_acq_ids"},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pickle", type=Path)
    ap.add_argument("--min-length", type=int, nargs="+", default=[5, 15, 35])
    ap.add_argument("--flow-mm", type=float, default=0.5)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    data = load(args.pickle)
    out = {}
    for n in args.min_length:
        s = summarize(data, n, args.flow_mm)
        out[str(n)] = s
        print(
            f"min_length>={n:>3}: tracks={s['n_tracks']:>7}  per_acq mean={s['tracks_per_acq_mean']:.1f} "
            f"median={s['tracks_per_acq_median']:.0f} [{s['tracks_per_acq_min']}-{s['tracks_per_acq_max']}]  "
            f"flowing(>={args.flow_mm}mm)={100 * s['flowing_fraction']:.0f}%  "
            f"len med={s['length_median']:.0f}  net disp med={s['net_displacement_mm_median']:.2f}mm  "
            f"speed med={s['mean_speed_mm_s_median']:.1f} p90={s['mean_speed_mm_s_p90']:.1f} mm/s  "
            f"straight med={s['straightness_median']:.2f} (>=0.8: {100 * s['straight_fraction']:.0f}%)  "
            f"cross-acq={s['cross_boundary_tracks']}"
        )
    s5 = out[str(args.min_length[0])]
    print(f"detections: {s5['n_detections']} ({s5['detections_per_frame']:.1f}/frame), acqs={s5['n_acquisitions']}")
    print("params:", json.dumps(s5["params"], default=str))
    if args.json:
        args.json.write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
