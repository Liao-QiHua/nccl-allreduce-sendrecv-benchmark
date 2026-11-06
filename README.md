# NCCL Reduce / SendRecv Benchmark

NVIDIA NCCL v2.28.9-1; two local MPI ranks; one PCIe-connected GPU per rank.

Repository: https://github.com/Liao-QiHua/nccl-allreduce-sendrecv-benchmark

## Primary goals
1. Measure uninstrumented end-to-end latency for AllReduce and paired bidirectional Send/Recv at 8 B, 16 KiB, and 8 MiB, forcing SIMPLE and LL separately.
2. Break down the 8 B path into Host function durations and complete GPU Kernel durations, with measured values printed directly after every function in AllReduce and Send/Recv hierarchical text call trees. The report must show parent/child relations and serial/parallel overlap.

## Fixed test parameters
- Two MPI ranks; each rank uses one GPU.
- Two RTX PRO 5000 GPUs on the same PCIe-connected host.
- FP32; AllReduce uses ncclSum.
- Payloads: 8 B, 16 KiB, 8 MiB.
- Protocols: SIMPLE and LL, tested separately.
- 20 warmups and 100 measured iterations per case.
- Stage tracing is limited to 8 B.

8 bytes means two FP32 elements, not eight billion elements.

## Repository layout
- benchmark/: CUDA/MPI benchmark source.
- scripts/: build, run, PCIe topology checks, and statistics (stage_stats.py, kernel_stats.py).
- docs/: Chinese test design, call chains, measured call-tree report template, and AI Agent prompt.
- instrumentation/: Host instrumentation design and coverage.
- patches/: optional Host Trace patch.

## Statistics

Do not aggregate the raw CSVs by hand. Two scripts own the arithmetic and both
have self-tests:

    python3 scripts/stage_stats.py <trace_dir>              # Host call trees (8 B)
    python3 scripts/stage_stats.py <trace_dir> --percentiles # add P50/P90/P99
    python3 scripts/stage_stats.py <trace_dir> --format csv  # one row per node
    python3 scripts/kernel_stats.py <nsys_dir_or_sqlite>     # GPU kernels
    python3 scripts/test_stats.py                            # verify both

They enforce the four things that are easy to get wrong:

1. `duration_ns` is nanoseconds; every reported value is microseconds.
2. The 20 warmup operation windows are dropped per (pid, operation) before any
   averaging. Leaving them in once produced "groupLaunch mean=4999 us", because
   the first window contains the one-off P2P preconnect stall (~0.5 s).
3. Exclusive time is `duration - union(nested intervals)`, never a sum of
   children, and not inclusive if the trace has a partial overlap.
4. Conditional paths that never ran print `N/A`, never `0 us`.

Windows whose anchor is far above the median are reported explicitly instead of
being averaged away.

## Important
- Baseline and Trace NCCL builds must be isolated.
- End-to-end results must use the uninstrumented official NCCL build.
- GPU Device source is not instrumented; use CUPTI Activity / Nsight Systems for complete Kernel duration.
- Every measured Host function node must show actual timing statistics directly after its name in the final hierarchy text tree.
- Unexecuted conditional functions must be marked N/A, not 0 us.
- The source instrumentation is not considered verified until applied, built, and run on the target GPU host.

See:
- docs/测试设计与实施文档.md
- docs/调用链耗时报告模板.md
- docs/执行链路与计时边界.md
- docs/AI测试Agent提示词.md
- docs/源码插桩方案-待确认.md
