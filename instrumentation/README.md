# NCCL v2.28.9-1 源码插桩实施说明

当前已生成 Host-only Patch：`patches/nccl-v2.28.9-1-host-trace.patch`。

## 已实施的源码改动设计

Patch 修改：
- `makefiles/common.mk`：通过 `NCCL_ENABLE_HOST_TRACE=1` 单独启用 Host trace 编译宏。
- `src/include/host_trace.h`：固定容量事件缓冲区、CLOCK_MONOTONIC_RAW 时间戳和进程退出时 CSV 导出。
- `src/collectives.cc`：AllReduce、Send、Recv API 入口时间。
- `src/enqueue.cc`：Enqueue、Prepare、Plan、Work Upload、Kernel Launch、Launch Finish 时间。
- `src/group.cc`：GroupEnd 和 groupLaunch 时间。

没有修改任何 `src/device/*` 文件，也没有加入 Proxy 计时。

## 运行时控制

编译时：
```bash
make -j src.build NCCL_ENABLE_HOST_TRACE=1
```

运行时：
```bash
NCCL_HOST_TRACE=1 NCCL_HOST_TRACE_BYTES=8 \
NCCL_HOST_TRACE_FILE=/tmp/nccl-host-trace.csv \
./reduce_sendrecv_bench
```

每个进程会自动在文件名前缀后追加 PID，避免双 Rank 竞争写同一个 CSV。未知 payload 的 planner/group 事件 bytes=0，会保留用于时间线分析。

## GPU Kernel 时间

不修改 Device 源码。Kernel 完整执行时间通过测试侧 CUPTI Activity API 或 Nsight Systems 采集。每个 Kernel 独立报告 GPU start/end/duration；多 Kernel 或重叠 Kernel 不应简单求和。

## Proxy

阶段脚本设置 `NCCL_P2P_USE_CUDA_MEMCPY=0`，目标为默认 direct P2P。仍需查看 NCCL 日志确认实际 Transport。若运行时使用 NET/SHM 或出现 Proxy 工作，不得将结果标注为纯 direct P2P。

## 验证状态

Patch 的 hunk 和上下文已按 NVIDIA/nccl `v2.28.9-1` 对应源码文件核对。当前执行环境无法连接 GitHub 网络克隆源码，也没有 RTX PRO 5000 远端构建环境，因此尚未完成 `git apply --check`、编译和 GPU 运行验证。必须在目标机器执行 `scripts/prepare_nccl_builds.sh` 并保存日志后，才能标记为已验证。
