#!/usr/bin/env bash
set -euo pipefail
: "\${CUDA_HOME:?Set CUDA_HOME}"
: "\${NCCL_HOME:?Set NCCL_HOME}"
"\${CUDA_HOME}/bin/nvcc" -std=c++17 -O2 -lineinfo -ccbin mpicxx \
  -I"\${NCCL_HOME}/include" benchmark/reduce_sendrecv_bench.cu \
  -L"\${NCCL_HOME}/lib" -lnccl -L"\${CUDA_HOME}/lib64" -lcudart \
  -o reduce_sendrecv_bench
echo "Built ./reduce_sendrecv_bench"
