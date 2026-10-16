#!/usr/bin/env bash
# Fetch the Muse build-history files (conversation + trajectory) from the
# repo's muse-history-v1 release. Usage:
#   ALIVE_REPO=owner/repo ./context/fetch.sh   (requires: gh auth login)
set -euo pipefail
cd "$(dirname "$0")"

REPO="${ALIVE_REPO:?Set ALIVE_REPO=owner/repo}"
TAG="${ALIVE_HISTORY_TAG:-muse-history-v1}"

gh release download "$TAG" --repo "$REPO" --dir . --clobber
echo "Fetched. Read the conversation tail-to-head; parse the trajectory selectively."
