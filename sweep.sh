#!/usr/bin/env bash
# Tracking parameter sweep on the first N acquisitions of the R2 reference.
#   bash sweep.sh <out_root> <num_acqs> <name:extra args> [<name:extra args> ...]
# Each config runs the README recipe plus its extra args (later flags win),
# then track_stats.py writes <out_root>/<name>/stats.json.
set -uo pipefail
failed=0
export PATH="$HOME/.local/bin:$PATH"
root="$1"; n="$2"; shift 2
recipe="--beamformed R2 --svd-method adaptive --knee-filter --temporal-sigma 0 --sigma-threshold 2.0 --svd-low-cutoff 0.1 --min-distance 2 --smoothing-sigma 1.0 --subpixel centroid --tracking kalman --max-gap 3 --min-track-length 5 --max-cost 10 --smooth-sigma 2.0 --min-lengths 5 15 35"
for spec in "$@"; do
  name="${spec%%:*}"; extra="${spec#*:}"
  dir="$root/$name"; mkdir -p "$dir"
  if [ -f "$dir/stats.json" ]; then echo "[sweep] $name done, skipping"; continue; fi
  echo "[sweep] $name: $extra  ($(date +%H:%M:%S))"
  # shellcheck disable=SC2086
  nice -n 15 uv run --quiet --with requests python -u stream_r2.py track-export \
    --tracks "$dir/tracks.pkl" --export-dir "$dir" --num-acqs "$n" $recipe $extra > "$dir/log.txt" 2>&1 \
    || { echo "[sweep] $name FAILED, see $dir/log.txt"; failed=$((failed + 1)); continue; }
  nice -n 15 uv run --quiet python track_stats.py "$dir/tracks.pkl" --min-length 5 15 35 --json "$dir/stats.json" | tee "$dir/stats.txt" \
    || { echo "[sweep] $name stats FAILED"; failed=$((failed + 1)); }
done
echo "[sweep] all done $(date +%H:%M:%S), $failed failed"
exit $(( failed > 0 ))
