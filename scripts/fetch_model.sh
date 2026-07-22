#!/usr/bin/env bash
# Downloads the exported TorchScript model from a GitHub Release asset at
# build time, since the file (~112MB) exceeds GitHub's 100MB per-file push
# limit and was never committed to this repo's git history.
#
# Required env var:
#   MODEL_DOWNLOAD_URL — the GitHub Release asset URL, e.g.
#     https://github.com/<owner>/<repo>/releases/download/<tag>/model.torchscript.pt
#
# Optional:
#   ML_MODEL_PATH — where to place it (default matches serve.py's own default)
#
# Run from the ml/ directory: bash scripts/fetch_model.sh

set -euo pipefail

TARGET="${ML_MODEL_PATH:-outputs/export/model.torchscript.pt}"
MIN_BYTES=10000000  # ~10MB — far below the real ~112MB, far above an HTML error page

if [ -f "$TARGET" ] && [ "$(stat -c%s "$TARGET" 2>/dev/null || stat -f%z "$TARGET")" -ge "$MIN_BYTES" ]; then
  echo "Model already present at $TARGET, skipping download."
  exit 0
fi

if [ -z "${MODEL_DOWNLOAD_URL:-}" ]; then
  echo "ERROR: MODEL_DOWNLOAD_URL is not set. Create a GitHub Release with" >&2
  echo "model.torchscript.pt attached as a binary asset, then set this env var" >&2
  echo "to its release-asset URL (see ml/README.md)." >&2
  exit 1
fi

mkdir -p "$(dirname "$TARGET")"
echo "Downloading model from $MODEL_DOWNLOAD_URL to $TARGET ..."
curl -fL --retry 3 -o "$TARGET" "$MODEL_DOWNLOAD_URL"

ACTUAL_BYTES=$(stat -c%s "$TARGET" 2>/dev/null || stat -f%z "$TARGET")
if [ "$ACTUAL_BYTES" -lt "$MIN_BYTES" ]; then
  echo "ERROR: downloaded file is only $ACTUAL_BYTES bytes (expected >= $MIN_BYTES)." >&2
  echo "MODEL_DOWNLOAD_URL likely points to a login/error page, not the real asset" >&2
  echo "(e.g. the release or repo is private and needs an authenticated request)." >&2
  rm -f "$TARGET"
  exit 1
fi

echo "Model downloaded successfully ($ACTUAL_BYTES bytes)."
