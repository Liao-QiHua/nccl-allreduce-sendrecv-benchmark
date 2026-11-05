# NCCL Host Trace 构建与阶段运行

## 两套 NCCL 必须隔离

- Baseline：官方 NVIDIA/nccl v2.28.9-1 原始源码，不应用 Patch；用于正式端到端测试。
- Trace：同一 Tag 的独立源码副本，应用 patches/nccl-v2.28.9-1-host-trace.patch；用于 8 B Host 阶段测试。

构建脚本：

    bash scripts/prepare_nccl_builds.sh /workspace/nccl-bench

执行前确认目录不存在，脚本会拒绝覆盖已有源码目录。

## Baseline 端到端

    export NCCL_HOME=/workspace/nccl-bench/nccl-v2.28.9-1-baseline/build
    export LD_LIBRARY_PATH="$NCCL_HOME/lib:$LD_LIBRARY_PATH"
    unset NCCL_HOST_TRACE NCCL_HOST_TRACE_BYTES NCCL_HOST_TRACE_FILE
    bash scripts/build.sh
    bash scripts/run_all.sh

确认 ldd 指向 Baseline libnccl.so。运行 8 B、16 KiB、8 MiB，SIMPLE/LL，20 warmup/100 measured。

## Trace 阶段测试

    export NCCL_TRACE_HOME=/workspace/nccl-bench/nccl-v2.28.9-1-host-trace/build
    export BENCH_BIN=./reduce_sendrecv_bench
    bash scripts/run_stage_trace.sh

脚本只运行 8 B 的 AllReduce 和双向 SendRecv，分别测试 SIMPLE/LL。设置 NCCL_HOST_TRACE=1、NCCL_HOST_TRACE_BYTES=8、NCCL_P2P_USE_CUDA_MEMCPY=0。必须检查实际 NCCL 日志中的 Transport 和协议。

## 本次 Patch 新增 Host 计时点

- taskAppend
- collTaskAppend
- p2pTaskAppend
- scheduleCollTasksToPlan
- scheduleP2pTasksToPlan
- ncclTasksRegAndEnqueue
- ncclPrepareTasksAndCollPreconnect
- ncclPrepareTasksAndCollPreconnectFunc（条件路径）
- ncclP2PPreconnectFunc（条件路径）
- doLaunches

既有 API、Group、Prepare、Launch 计时点仍保留。

## GPU Kernel

不修改 NCCL Device 源码。使用 CUPTI Activity API 或 Nsight Systems 采集 Kernel 名称、GPU Start/End/Duration、Stream 和可用的 Grid/Block 信息。每个 Kernel 独立统计；重叠 Kernel 不得简单相加。

## 结果分析要求

- 每个 Rank / PID / TID 分别分析 Host 事件。
- 从 120 次调用中排除 20 次 warmup，只统计正式 100 次。
- 每个函数节点在层级文本树中直接标注 calls、mean inclusive、mean exclusive、P50、P90、P99、min、max。
- 父子函数 Inclusive 不相加；Exclusive 通过子区间并集计算。
- Group API rank=-1 时按 PID/TID 关联，不伪造 Rank。
- 未执行的条件路径标记 N/A。
- 输出 AllReduce 和 Send/Recv 两棵带实测耗时的调用树，以及 Rank/Host/GPU 串行并行时间线。

## 验证状态

新增插桩 hunk 已与官方 v2.28.9-1 源码上下文静态核对。尚未在远端 RTX PRO 5000 主机完成 git apply --check、编译及 GPU 运行；必须完成这些步骤后才能标记为验证通过。
