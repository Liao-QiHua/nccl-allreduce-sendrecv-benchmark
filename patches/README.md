# NCCL v2.28.9-1 Host Trace Patch

This patch adds optional Host-side function duration tracing only. It does not modify any CUDA Device source file and does not instrument GPU kernel internals.

## Apply to exact upstream source

    git clone https://github.com/NVIDIA/nccl.git
    cd nccl
    git checkout v2.28.9-1
    git rev-parse HEAD
    git apply /path/to/nccl-v2.28.9-1-host-trace.patch

## Build

Build in a separate source worktree/copy from the unmodified baseline:

    make -j src.build NCCL_ENABLE_HOST_TRACE=1

The ordinary baseline build omits the option and compiles the trace macro to an empty statement.

## Enable at runtime

    export NCCL_HOST_TRACE=1
    export NCCL_HOST_TRACE_BYTES=8
    export NCCL_HOST_TRACE_FILE=/tmp/nccl-host-trace.csv

Each process writes to a separate file by appending its PID to the configured file prefix. NCCL_HOST_TRACE_BYTES=8 filters known-payload scopes to 8 bytes; planner/group scopes with unknown payload (0) are retained.

Fields: event, rank, pid, tid, bytes, start_ns, end_ns, duration_ns. Time uses CLOCK_MONOTONIC_RAW. The fixed buffer is flushed at process exit; overflow silently drops records after 32768 events per translation unit.

## Instrumented Host functions

- Public entry: ncclAllReduce, ncclSend, ncclRecv.
- Group API: ncclGroupStart, ncclGroupEnd.
- Enqueue: ncclEnqueueCheck.
- Group internals: ncclGroupEndInternal, groupLaunch.
- Planner: ncclPrepareTasks, ncclLaunchPrepare.
- Upload/launch: ncclLaunchKernelBefore_NoUncapturedCuda, ncclLaunchKernel.
- Post-launch: ncclLaunchKernelAfter_NoCuda, ncclLaunchFinish.

The durations are inclusive function wall times. Nested scopes overlap and must not be summed. Use the trace as a call timeline; do not interpret each event as exclusive CPU time.

## GPU kernel duration

No NCCL Device source is modified. Use CUPTI Activity API or Nsight Systems from the benchmark process to capture each kernel's GPU start/end/duration. If one NCCL operation emits multiple kernels, report them separately and account for overlap.

## Important

- This patch has been checked against the v2.28.9-1 source file contexts, but it has not been applied and built on the remote RTX PRO 5000 host.
- The uninstrumented end-to-end benchmark must link against a separately built upstream NCCL without this patch.
- Use the trace build only for 8-byte stage tests.
- For direct PCIe P2P, keep NCCL_P2P_USE_CUDA_MEMCPY=0 and verify the selected transport from NCCL logs before assuming no proxy work.
