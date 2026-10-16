#!/usr/bin/env bash
# Fetch the ONNX models for ALive. They are GitHub release assets on the
# repo (GitHub's 100MB file limit keeps them out of git).
# Usage: ALIVE_REPO=owner/repo ./fetch.sh   (requires: gh auth login)
set -euo pipefail
cd "$(dirname "$0")"

REPO="${ALIVE_REPO:?Set ALIVE_REPO=owner/repo (the repo holding the models release)}"
TAG="${ALIVE_MODELS_TAG:-models-v1}"

gh release download "$TAG" --repo "$REPO" --pattern "*.onnx" --dir . --clobber

echo "Verifying checksums..."
sha256sum -c SHA256SUMS
echo "Done."
