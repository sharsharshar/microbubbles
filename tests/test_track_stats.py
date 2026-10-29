"""track_stats.summarize on hand-built tracks with known geometry."""

from __future__ import annotations

import numpy as np

import track_stats


def _track(frames, positions):
    return {"frames": np.array(frames, dtype=np.int32), "positions": np.array(positions, dtype=np.float32)}


def _data(tracks, frames_per_acq=10, n_acq=2):
    return {
        "tracks": tracks,
        "frames_per_acq": frames_per_acq,
        "n_acquisitions": n_acq,
        "n_frames": frames_per_acq * n_acq,
        "params": {"frame_rate_hz": 100.0},
        "detections": {"frame_indices": np.zeros(7)},
    }


def test_known_speed_straight_track():
    # 1 mm per frame at 100 Hz for 5 frames: elapsed 4 frames, 4 mm path -> 100 mm/s.
    t = _track([0, 1, 2, 3, 4], [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]])
    s = track_stats.summarize(_data([t]), min_length=5, flow_mm=0.5)
    assert s["n_tracks"] == 1
    assert s["mean_speed_mm_s_median"] == 100.0
    assert s["straightness_median"] == 1.0
    assert s["flowing_fraction"] == 1.0
    assert s["per_acq"] == [1, 0]


def test_gapped_frames_use_elapsed_time():
    # Same 4 mm path spread over frames 0..8 (gaps): 8 elapsed frames -> 50 mm/s.
    t = _track([0, 2, 4, 6, 8], [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]])
    s = track_stats.summarize(_data([t]), min_length=5, flow_mm=0.5)
    assert s["mean_speed_mm_s_median"] == 50.0


def test_acquisition_assignment_and_cross_boundary():
    a = _track([0, 1, 2, 3, 4], [[0, 0, 0]] * 5)  # stationary, acq 0
    b = _track([8, 9, 10, 11, 12], [[0, 0, i] for i in range(5)])  # starts acq 0, ends acq 1
    c = _track([12, 13, 14, 15, 16], [[i, 0, 0] for i in range(5)])  # acq 1
    s = track_stats.summarize(_data([a, b, c]), min_length=5, flow_mm=0.5)
    assert s["per_acq"] == [2, 1]
    assert s["cross_boundary_tracks"] == 1
    assert abs(s["flowing_fraction"] - 2 / 3) < 1e-9
    assert np.isnan(s["mean_speed_mm_s_median"]) or s["mean_speed_mm_s_median"] >= 0


def test_min_length_filter_and_empty_result():
    short = _track([0, 1, 2], [[0, 0, 0]] * 3)
    s = track_stats.summarize(_data([short]), min_length=5, flow_mm=0.5)
    assert s["n_tracks"] == 0
    assert s["tracks_per_acq_mean"] == 0.0
    assert s["flowing_fraction"] == 0.0
