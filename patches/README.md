# NCCL v2.28.9-1 Host Trace Patch

This patch adds optional Host-side function duration tracing only. It does not modify any CUDA Device source file and does not instrument GPU kernel internals.

## Apply and build

    git clone https://github.com/NVIDIA/nccl.git
    cd nccl
    git checkout v2.28.9-1
    git rev-parse HEAD
    git apply /path/to/nccl-v2.28.9-1-host-trace.patch
    make -j src.build NCCL_ENABLE_HOST_TRACE=1

Build in a separate source tree from the uninstrumented baseline.

## Runtime

    export NCCL_HOST_TRACE=1
    export NCCL_HOST_TRACE_BYTES=8
    export NCCL_HOST_TRACE_FILE=/tmp/nccl-host-trace.csv

Each process appends its PID to the output prefix. Events contain event, rank, pid, tid, bytes, start_ns, end_ns, duration_ns. Time uses CLOCK_MONOTONIC_RAW. Fixed buffers flush at process exit.

## Instrumented Host functions

Public APIs:
- ncclAllReduce
- ncclSend
- ncclRecv
- ncclGroupStart
- ncclGroupEnd

Enqueue / task append:
- ncclEnqueueCheck
- taskAppend
- collTaskAppend
- p2pTaskAppend
- ncclGroupEndInternal

Group / prepare / registration:
- groupLaunch
- ncclP2PPreconnectFunc (conditional)
- ncclPrepareTasksAndCollPreconnect
- ncclPrepareTasksAndCollPreconnectFunc (conditional async path)
- ncclPrepareTasks
- ncclTasksRegAndEnqueue
- doLaunches

Planning / launch:
- ncclLaunchPrepare
- scheduleCollTasksToPlan
- scheduleP2pTasksToPlan
- ncclLaunchKernelBefore_NoUncapturedCuda
- ncclLaunchKernel
- ncclLaunchKernelAfter_NoCuda
- ncclLaunchFinish

The inline ncclGroupStartInternal helper is not separately instrumented; its tiny duration is included in its caller.

## Interpretation

All Scope durations are inclusive wall durations. Nested intervals overlap and must not be summed. Exclusive time must be computed by subtracting the union of direct child intervals on the same thread, not the sum of child durations. Async callbacks on other TIDs must be shown as async job execution, not synchronous call-stack children.

The trace CSV does not currently carry an explicit operation_sequence or parent_event_id. Reconstruct measured operation windows using PID/TID, API/group event order, timestamps, and source call flow. Exclude the 20 warmup operations before calculating the formal 100-iteration statistics.

Functions with rank=-1 are group-wide / API scopes without a single communicator argument. Attribute them using PID/TID and timeline; do not invent a Rank.

## GPU Kernel duration

No NCCL Device source is modified. Use CUPTI Activity API or Nsight Systems to capture each Kernel's GPU start/end/duration, name and stream. Report each Kernel separately and identify overlap.

## Validation status

The patch hunk ranges and context have been statically checked against NVIDIA/nccl v2.28.9-1 source. This does not prove git apply --check, compilation, or GPU execution on the remote RTX PRO 5000 host. Mark the patch validated only after those steps succeed.
