#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/nccl-v2.28.9-1-source" >&2
  exit 2
fi
NCCL_SRC="$(cd "$1" && pwd)"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TAG="$(git -C "$NCCL_SRC" describe --tags --exact-match 2>/dev/null || true)"
if [[ "$TAG" != "v2.28.9-1" ]]; then
  echo "Expected exact tag v2.28.9-1, got: \${TAG:-not-an-exact-tag}" >&2
  exit 3
fi
git -C "$NCCL_SRC" apply --check "$ROOT/patches/nccl-v2.28.9-1-host-trace.patch"
git -C "$NCCL_SRC" apply "$ROOT/patches/nccl-v2.28.9-1-host-trace.patch"
echo "Patch applied. Build in this separate trace tree with:"
echo "  make -C \\"$NCCL_SRC\\" -j src.build NCCL_ENABLE_HOST_TRACE=1"
