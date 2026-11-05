#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/workspace" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKSPACE="$(mkdir -p "$1" && cd "$1" && pwd)"
BASELINE="$WORKSPACE/nccl-v2.28.9-1-baseline"
TRACE="$WORKSPACE/nccl-v2.28.9-1-host-trace"

if [[ -e "$BASELINE" || -e "$TRACE" ]]; then
  echo "Refusing to overwrite existing NCCL source directories in $WORKSPACE" >&2
  exit 3
fi

git clone --depth 1 --branch v2.28.9-1 https://github.com/NVIDIA/nccl.git "$BASELINE"
git -C "$BASELINE" rev-parse HEAD | tee "$WORKSPACE/upstream-commit.txt"

cp -a "$BASELINE" "$TRACE"
git -C "$TRACE" apply --check "$ROOT/patches/nccl-v2.28.9-1-host-trace.patch"
git -C "$TRACE" apply "$ROOT/patches/nccl-v2.28.9-1-host-trace.patch"

make -C "$BASELINE" -j"$(nproc)" src.build
make -C "$TRACE" -j"$(nproc)" src.build NCCL_ENABLE_HOST_TRACE=1

cat <<EOF
Builds completed.
Baseline NCCL: $BASELINE/build
Trace NCCL:    $TRACE/build

Use baseline build/lib for uninstrumented end-to-end runs.
Use trace build/lib only for 8-byte Host-stage tracing.
EOF
