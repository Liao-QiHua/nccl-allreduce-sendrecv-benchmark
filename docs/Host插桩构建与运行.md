# NCCL Host Trace 实施与构建

## 当前状态

- 已生成针对 NVIDIA/nccl v2.28.9-1 的 Host-only 源码 Patch。
- Patch 只修改 Host C++ 文件和 makefiles/common.mk；不修改 src/device 下任何文件。
- GPU Kernel 完整执行时间由测试侧 CUPTI Activity / Nsight Systems 获取。
- 该 Patch 尚未在远端 RTX PRO 5000 机器编译和验证；请先执行 scripts/prepare_nccl_builds.sh。

## 两套源码树必须分离

scripts/prepare_nccl_builds.sh 创建两个独立目录：
- nccl-v2.28.9-1-baseline：官方原始源码，不应用 Patch，用于 8B / 16KiB / 8MiB 端到端。
- nccl-v2.28.9-1-host-trace：应用 Host Trace Patch，用于 8B 阶段计时。

两个目录分别编译到各自 build 目录。不要把 Trace 的 libnccl.so 放入 Baseline 的 LD_LIBRARY_PATH。

## 编译

```bash
bash scripts/prepare_nccl_builds.sh /workspace/nccl-bench
```

## 端到端（无插桩）

```bash
export NCCL_HOME=/workspace/nccl-bench/nccl-v2.28.9-1-baseline/build
export LD_LIBRARY_PATH="$NCCL_HOME/lib:$LD_LIBRARY_PATH"
unset NCCL_HOST_TRACE
unset NCCL_HOST_TRACE_BYTES
bash scripts/build.sh
bash scripts/run_all.sh
```

确认 ldd reduce_sendrecv_bench 指向 baseline build/lib/libnccl.so。

## 阶段测试（Host 插桩 + 外部 Kernel Activity）

```bash
export NCCL_TRACE_HOME=/workspace/nccl-bench/nccl-v2.28.9-1-host-trace/build
export BENCH_BIN=./reduce_sendrecv_bench
bash scripts/run_stage_trace.sh
```

run_stage_trace.sh 固定 BENCH_ONLY_8B=1，分别运行 SIMPLE / LL；测试程序每种协议只跑 8B AllReduce 和双向 SendRecv。Host Trace 输出按进程 PID 后缀区分，避免两个 MPI rank 写同一个文件。

CUPTI Activity 或 Nsight Systems 需要在目标机安装的 CUDA Toolkit 中可用。它们从测试侧采集每个 kernel 的 GPU start/end/duration，不需要修改 NCCL Device 源码。

## Host Trace 环境变量

- NCCL_HOST_TRACE=1：运行时开启 Host trace。
- NCCL_HOST_TRACE_BYTES=8：只记录已知 payload 为 8 字节的 scope；planner/group 事件的 bytes=0，保留用于执行链路分析。
- NCCL_HOST_TRACE_FILE：输出文件前缀；每个进程自动追加 PID。
- Trace buffer 固定容量为每个编译单元 32768 条，超出后丢弃后续事件。

Host 时间使用 CLOCK_MONOTONIC_RAW。父子函数区间存在嵌套，不能把 inclusive duration 简单求和。groupLaunch / groupEnd 的 rank 记录为 -1，因为该函数边界本身不一定对应单一 communicator。

## Proxy

当前阶段测试脚本显式设置 NCCL_P2P_USE_CUDA_MEMCPY=0，目标是默认 P2P direct path。仍须从 NCCL INFO 日志确认实际 transport；如果运行时选择 NET/SHM 或出现 Proxy 工作，则不能把它解释为 direct P2P 无 Proxy。

## 验证清单

1. Patch 对 v2.28.9-1 执行 git apply --check。
2. Baseline 编译成功。
3. Trace 编译成功。
4. ldd 确认两种测试各自链接正确的 libnccl.so。
5. Baseline 端到端运行期间没有 NCCL_HOST_TRACE 环境变量，且没有 trace CSV。
6. Trace 运行仅执行 8B，且生成 Host CSV。
7. CUPTI/Nsight 时间线能关联各 rank 的 NCCL kernel。
8. 比较 Trace 与 Baseline 8B 延迟，报告插桩扰动。
