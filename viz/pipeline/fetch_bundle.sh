#!/usr/bin/env bash
# Pull a built viz bundle from the inference box for local viewing.
# The bundle is a build artifact (see docs/PROVENANCE.md) — never committed as blobs into this
# repo, same policy as the index/*.faiss files. Regenerate with build_bundle.py, or fetch an
# already-built one with this script.
#
# Usage: ./fetch_bundle.sh [dest_dir] [remote_host] [remote_path]
set -euo pipefail
DEST="${1:-../viewer}"
HOST="${2:-rohit@100.86.142.125}"
REMOTE="${3:-/fast/atlas_v109/viz_bin}"

FILES=(meta.json positions.bin positions_lda.bin figure_meta.json novelty.bin domain.bin \
       density.bin cell.bin family_idx.bin genus_idx.bin species_idx.bin \
       neighbors.bin neighbor_dist_u8.bin theta.bin theta_r.bin accessions.json)

mkdir -p "$DEST"
for f in "${FILES[@]}"; do
  scp -o BatchMode=yes "$HOST:$REMOTE/$f" "$DEST/$f" && echo "  got $f" || echo "  MISSING $f"
done
echo "bundle -> $DEST ($(du -sh "$DEST" | awk '{print $1}'))"
