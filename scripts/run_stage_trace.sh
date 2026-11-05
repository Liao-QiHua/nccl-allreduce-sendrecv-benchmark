#!/usr/bin/env bash
set -euo pipefail
: "\${NCCL_TRACE_HOME:?Set NCCL_TRACE_HOME to nccl-v2.28.9-1-host-trace/build}"
: "\${BENCH_BIN:=./reduce_sendrecv_bench}"
: "\${MPIEXEC:=mpirun}"
: "\${OUTDIR:=trace-results}"
mkdir -p "$OUTDIR"
export LD_LIBRARY_PATH="$NCCL_TRACE_HOME/lib:\${LD_LIBRARY_PATH:-}"
export BENCH_ONLY_8B=1
export NCCL_HOST_TRACE=1
export NCCL_HOST_TRACE_BYTES=8
export NCCL_P2P_USE_CUDA_MEMCPY=0
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,COLL,TUNING,P2P
for proto in Simple LL; do
  export NCCL_PROTO="$proto"
  export RESULT_CSV="$OUTDIR/8B-$proto.csv"
  export NCCL_HOST_TRACE_FILE="$OUTDIR/host-$proto.csv"
  export NCCL_DEBUG_FILE="$OUTDIR/nccl-%h-%p-$proto.log"
  "$MPIEXEC" -np 2 --bind-to none "$BENCH_BIN"
done
