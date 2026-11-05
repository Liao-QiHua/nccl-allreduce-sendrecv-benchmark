#!/usr/bin/env bash
set -euo pipefail
BIN="\${BIN:-./reduce_sendrecv_bench}"
MPIEXEC="\${MPIEXEC:-mpirun}"
OUTDIR="\${OUTDIR:-results}"
mkdir -p "\${OUTDIR}"
unset NCCL_HOST_TRACE NCCL_HOST_TRACE_BYTES NCCL_HOST_TRACE_FILE
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_DEBUG="\${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="\${NCCL_DEBUG_SUBSYS:-INIT,COLL,TUNING,P2P}"
for proto in Simple LL; do
  export NCCL_PROTO="\${proto}"
  export RESULT_CSV="\${OUTDIR}/\${proto}.csv"
  export NCCL_DEBUG_FILE="\${OUTDIR}/nccl_%h_%p.log"
  echo "=== NCCL_PROTO=\${proto}; NCCL_ALGO automatic ==="
  "\${MPIEXEC}" -np 2 --bind-to none "\${BIN}"
done
