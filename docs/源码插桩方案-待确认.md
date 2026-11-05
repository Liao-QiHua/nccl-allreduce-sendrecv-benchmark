# NCCL v2.28.9-1 Host Trace 插桩说明

## 1. 范围

本 Patch 用于 8 B 阶段分析。只在 NCCL Host C++ 路径添加函数 wall-time 事件，不修改 src/device 下的任何文件，不在 GPU Kernel 内增加计时代码。GPU Kernel 完整执行时间由测试侧 CUPTI Activity 或 Nsight Systems 获取。

端到端正式结果必须使用未插桩 Baseline NCCL；Trace NCCL 仅用于 8 B 阶段拆解。

## 2. 新增的任务入队与规划插桩

在原有 API、Group、Prepare 和 Launch 事件之外，Patch 新增：

- taskAppend：AllReduce / SendRecv 的公共 task 分发入口。
- collTaskAppend：Collective task 入队，主要对应 AllReduce。
- p2pTaskAppend：P2P Send/Recv task 入队。
- scheduleCollTasksToPlan：Collective task 转换为 Kernel Plan。
- scheduleP2pTasksToPlan：P2P Send/Recv task 配对、拆分并转换为 Kernel Plan。
- ncclTasksRegAndEnqueue：Work 注册与入队。
- ncclPrepareTasksAndCollPreconnect：任务准备与连接准备。
- ncclPrepareTasksAndCollPreconnectFunc：条件异步准备回调。
- ncclP2PPreconnectFunc：条件 P2P 连接预热回调。
- doLaunches：Launch 准备、launch rounds 和收尾调度。

同时保留：
- ncclAllReduce / ncclSend / ncclRecv。
- ncclGroupStart / ncclGroupEnd / ncclGroupEndInternal / groupLaunch。
- ncclEnqueueCheck / ncclPrepareTasks / ncclLaunchPrepare。
- ncclLaunchKernelBefore_NoUncapturedCuda / ncclLaunchKernel / ncclLaunchKernelAfter_NoCuda / ncclLaunchFinish。

## 3. 输出与解释

Host Trace 使用 CLOCK_MONOTONIC_RAW，记录 event、rank、pid、tid、bytes、start_ns、end_ns、duration_ns。不同 translation unit 有各自的固定缓冲区，退出时追加写入当前进程的 PID 文件。

NCCL_HOST_TRACE_BYTES=8 过滤已知非 8B payload；bytes=0 的阶段事件保留。Group / group-wide 函数可能 rank=-1，按 PID/TID 与时间线分析，不要强行分配 Rank。

所有 Scope 都是 inclusive wall duration。父子函数区间重叠，不得直接求和。Exclusive 应通过同一线程中直接子区间的并集计算；如果事件数据不足以可靠推导，报告 N/A。

当前事件格式没有显式 parent_event_id / operation_sequence。分析时需要结合 PID/TID、API 调用顺序、时间包含关系和源码调用关系重建操作窗口。20 次 warmup 必须从正式 100 次统计中排除。

## 4. 条件路径

- ncclPrepareTasksAndCollPreconnectFunc 仅在 SingleProcMemRegEnable 对应异步路径触发。
- ncclP2PPreconnectFunc 仅在存在待处理的 P2P preconnect job 时触发，通常可能发生在初始化/预热阶段。
- scheduleCollTasksToPlan 用于 Collective 任务规划。
- scheduleP2pTasksToPlan 用于 P2P 任务规划。
- 没有执行的路径写 N/A，不写 0 us。

## 5. 验证状态

Patch 适用基线为 NVIDIA/nccl v2.28.9-1。当前 Patch 已对新增和既有 hunk 的行号、上下文及排序进行静态核对；这不等于目标机 git apply --check、编译或 GPU 运行验证。必须在远端执行并保留日志后，才能标记为已验证。
