# NCCL AllReduce / SendRecv 性能测试 Agent 提示词

## 1. 仓库入口与任务目标

测试代码仓库：

https://github.com/Liao-QiHua/nccl-allreduce-sendrecv-benchmark

目标分支：

perf/nccl-2.28.9-reduce-sendrecv

先获取仓库最新提交，检查分支、工作区和已有结果。阅读 README.md、docs/测试设计与实施文档.md、docs/执行链路与计时边界.md、docs/调用链耗时报告模板.md、docs/源码插桩方案-待确认.md、docs/Host插桩构建与运行.md、instrumentation/README.md、patches/README.md、benchmark 源码和 scripts 下的脚本。以实际代码为准，指出文档与代码不一致之处。

本任务只有两个核心交付目标：

1. **端到端通信延迟**：两进程、两张 RTX PRO 5000、同机 PCIe P2P，测试 AllReduce 和双向配对 Send/Recv 在 8 B、16 KiB、8 MiB 下的真实完成延迟。
2. **8 B 阶段延迟拆解**：对 AllReduce 和 Send/Recv 的 Host 函数及 GPU Kernel 进行计时，将实测耗时直接写在层级调用树的每个函数名称后面，并解释父子、串行、并行和重叠关系。

不要把任务扩展成泛化的 NCCL 功能研究。所有检查和分析都服务于上述两个交付目标。

## 2. 固定测试环境

- NCCL：NVIDIA/nccl v2.28.9-1。
- GPU：2 × NVIDIA RTX PRO 5000。
- MPI：2 个进程，每个进程绑定一张 GPU。
- GPU 互联：同一台机器，通过 PCIe；运行时必须检查实际 P2P 能力和 Transport。
- 数据类型：FP32。
- AllReduce 操作：ncclSum。
- Send/Recv：每个 Rank 在同一 Group 内向对端发送，同时从对端接收，属于双向配对 SendRecv。
- 协议：SIMPLE 和 LL 分开强制。
- 每组：预热 20 次，正式测量 100 次。

8 B 指两个 FP32 元素；16 KiB 为 16,384 B；8 MiB 为 8,388,608 B。

测试前记录 GPU 型号、Driver、CUDA、MPI、编译器、NCCL 上游 Tag 和 commit SHA、PCIe 拓扑、P2P 能力及 GPU 进程占用。确认 Rank 到 GPU 的映射。必须通过 NCCL INFO 日志确认实际 Transport 和协议；不能只凭环境变量推断。

如果实际没有使用目标 PCIe P2P 路径，停止该配置并报告原因。不得自动切换到 NET、SHM 或其他路径后仍将结果称为 PCIe P2P。

## 3. 测试一：Baseline 端到端延迟

使用官方未插桩 NCCL v2.28.9-1。不得启用 Host Trace，不得将 Trace 版本的 libnccl.so 混入 Baseline 运行环境。

测试矩阵：

| 操作 | 数据量 | SIMPLE | LL |
|---|---|---|---|
| AllReduce | 8 B | 测试 | 测试 |
| AllReduce | 16 KiB | 测试 | 测试 |
| AllReduce | 8 MiB | 测试 | 测试 |
| SendRecv | 8 B | 测试 | 测试 |
| SendRecv | 16 KiB | 测试 | 测试 |
| SendRecv | 8 MiB | 测试 | 测试 |

每种协议单独运行：

- SIMPLE：NCCL_PROTO=Simple
- LL：NCCL_PROTO=LL

只强制协议，不强制算法；除非另有明确实验要求，否则不要设置 NCCL_ALGO。记录 NCCL 实际选择的算法。指定协议不可用或实际协议不匹配时，该组标记失败，不得静默回退。

每组预热 20 次，测量 100 次。每轮开始前的 Rank 同步必须放在计时区间外。

端到端计时定义：

1. 记录 Host monotonic clock 开始时间。
2. 提交该轮 NCCL 操作。
3. 对对应 Stream 执行同步，等待该轮通信完成。
4. 记录 Host monotonic clock 结束时间。
5. 对同一轮两个 Rank 的耗时取 MAX，作为该轮整体通信完成延迟；同时保留两个 Rank 的原始耗时。

AllReduce 的计时范围为 ncclAllReduce 提交前到对应 Stream 完成。SendRecv 的计时范围为 Group 中 ncclSend/ncclRecv 提交前到对应 Stream 完成。Comm 初始化、连接建立和每轮 MPI Barrier 不计入 E2E。

记录每轮原始样本，并计算 mean、P50、P90、P99、min、max。另行记录 Host API 函数驻留时间，但不得将 Host API 时间与 E2E 相加。

运行前检查测试程序实际加载的 libnccl.so 路径。

## 4. 测试二：8 B 阶段耗时

仅使用独立编译的 Trace NCCL：

- 上游仍为 NVIDIA/nccl v2.28.9-1。
- 应用仓库的 patches/nccl-v2.28.9-1-host-trace.patch。
- 编译时启用 NCCL_ENABLE_HOST_TRACE=1。
- 运行时启用 NCCL_HOST_TRACE=1。
- 使用 NCCL_HOST_TRACE_BYTES=8。
- 使用 NCCL_P2P_USE_CUDA_MEMCPY=0。
- 只测试 8 B 的 AllReduce/SendRecv × SIMPLE/LL 四种组合。
- 仍然预热 20 次、正式测量 100 次。

如果 Trace 数据没有可靠方式区分预热和正式测量，先检查原始事件顺序，按每个 Rank/Host Thread 的前 20 个操作窗口排除预热。不得把预热记录混入正式统计。

必须分别保留每个 Rank、PID、TID 的原始事件。Group API 和 group-wide 函数可能记录 rank=-1；不要伪造 Rank 归属，应按 PID/TID 和时间线关联。

### 4.1 必须计时的 Host 函数

AllReduce 和 Send/Recv 公共 API：
- ncclAllReduce
- ncclSend
- ncclRecv
- ncclGroupStart
- ncclGroupEnd
- ncclEnqueueCheck
- ncclGroupEndInternal
- groupLaunch

任务入队与规划：
- taskAppend
- collTaskAppend
- p2pTaskAppend
- ncclPrepareTasksAndCollPreconnect
- ncclPrepareTasksAndCollPreconnectFunc（条件异步路径）
- ncclPrepareTasks
- ncclTasksRegAndEnqueue
- scheduleCollTasksToPlan（Collective 路径）
- scheduleP2pTasksToPlan（P2P Send/Recv 路径）
- doLaunches

Launch：
- ncclLaunchPrepare
- ncclLaunchKernelBefore_NoUncapturedCuda
- ncclLaunchKernel
- ncclLaunchKernelAfter_NoCuda
- ncclLaunchFinish

连接预热相关：
- ncclP2PPreconnectFunc（仅实际执行时记录；如果只在预热/初始化阶段执行，应在正式 100 次阶段结果中标记 N/A 或单独列为初始化开销）。

新增的 taskAppend、collTaskAppend、p2pTaskAppend、scheduleP2pTasksToPlan 必须真实存在于 Patch 中并成功生成事件；AllReduce 的 scheduleCollTasksToPlan 也必须计时。不能只在文档中列出而没有插桩。

对于未执行的条件分支，标记 N/A（未走该路径），不得写成 0 us。

ncclGroupStartInternal 是 inline helper；如果没有独立事件，不要虚构其耗时。应说明其开销包含在调用方函数的 Exclusive 区间中。

### 4.2 GPU Kernel 完整执行时间

不修改任何 NCCL Device 源码，不在 GPU Kernel 内加入 clock64() 插桩。

使用 CUPTI Activity API 或 Nsight Systems 采集每个 GPU Kernel 的完整执行区间：

- Rank / GPU
- Kernel 名称
- CUDA Stream
- GPU Start / End
- Duration
- Grid / Block（工具可提供时）

按实际采集到的 Kernel 数量报告。不要预设每个操作一定只有一个 Kernel。多个 Kernel 有重叠时，不得把 Duration 直接求和。

不同 GPU 的绝对设备时间戳未经校准不能直接比较；可分别计算每张 GPU 的 Kernel Duration，并使用可关联的 Host/CUPTI 时间线分析重叠。

## 5. 最重要的验收要求：带实测耗时的层级文本调用树

最终报告必须有 AllReduce 和 Send/Recv 两棵层级文本图。

**每个被计时的函数节点，必须把真实统计值直接写在函数名称后面。** 不接受只在另一个表格中给时间、调用树却没有耗时的报告。

统一格式：

函数名()
  [calls=N, mean inclusive=X us, mean exclusive=Y us, P50=..., P90=..., P99=..., min=..., max=...]

示例中的 X/Y 只是格式占位符，最终必须替换为实际测量值，不能保留 XX、不能编造。

### 5.1 AllReduce 层级图模板

~~~text
Rank N Host Thread
└── ncclAllReduce() [实测统计]
    └── ncclEnqueueCheck() [实测统计]
        ├── taskAppend() [实测统计]
        │   └── collTaskAppend() [实测统计]
        └── ncclGroupEndInternal() [实测统计]
            └── groupLaunch() [实测统计]
                ├── ncclP2PPreconnectFunc() [条件执行；否则 N/A]
                ├── ncclPrepareTasksAndCollPreconnect() [实测统计]
                │   ├── ncclPrepareTasks() [实测统计]
                │   └── ncclPrepareTasksAndCollPreconnectFunc() [异步分支；否则 N/A]
                ├── ncclTasksRegAndEnqueue() [实测统计]
                └── doLaunches() [实测统计]
                    ├── ncclLaunchPrepare() [实测统计]
                    │   └── scheduleCollTasksToPlan() [实测统计]
                    └── Launch Round Loop
                        ├── ncclLaunchKernelBefore_NoUncapturedCuda() [实测统计]
                        ├── ncclLaunchKernel() [实测统计]
                        ├── ncclLaunchKernelAfter_NoCuda() [实测统计]
                        └── ncclLaunchFinish() [实测统计]
~~~

该图为逻辑模板。最终应以实际执行分支和源代码为准。函数的直接父子关系必须核实，不能把 Group 逻辑包含关系误画成 C++ 函数调用关系。

### 5.2 Send/Recv 层级图模板

~~~text
Rank N Host Thread

ncclGroupStart() [实测统计；Group 作用域开始]

ncclSend() [实测统计]
└── ncclEnqueueCheck() [实测统计]
    └── taskAppend() [实测统计]
        └── p2pTaskAppend() [实测统计]

ncclRecv() [实测统计]
└── ncclEnqueueCheck() [实测统计]
    └── taskAppend() [实测统计]
        └── p2pTaskAppend() [实测统计]

ncclGroupEnd() [实测统计；Group 作用域结束]
└── ncclGroupEndInternal() [实测统计]
    └── groupLaunch() [实测统计]
        ├── ncclP2PPreconnectFunc() [条件执行；否则 N/A]
        ├── ncclPrepareTasksAndCollPreconnect() [实测统计]
        │   ├── ncclPrepareTasks() [实测统计]
        │   └── ncclPrepareTasksAndCollPreconnectFunc() [异步分支；否则 N/A]
        ├── ncclTasksRegAndEnqueue() [实测统计]
        └── doLaunches() [实测统计]
            ├── ncclLaunchPrepare() [实测统计]
            │   └── scheduleP2pTasksToPlan() [实测统计]
            └── Launch Round Loop
                ├── ncclLaunchKernelBefore_NoUncapturedCuda() [实测统计]
                ├── ncclLaunchKernel() [实测统计]
                ├── ncclLaunchKernelAfter_NoCuda() [实测统计]
                └── ncclLaunchFinish() [实测统计]
~~~

Rank 0 和 Rank 1 必须分别输出，不得把两个 Rank 的函数耗时混在一起。ncclGroupStart/ncclGroupEnd 是 Group API 作用域边界，不是 ncclSend/ncclRecv 的函数调用父节点。

### 5.3 每个函数必须给出的统计值

对每个实际执行的函数节点给出：

- calls：正式测量窗口内的调用次数。
- mean inclusive：函数完整区间平均时间。
- mean exclusive：排除子函数区间并集后的自身时间。
- P50 / P90 / P99。
- min / max。

若无法可靠推导 Exclusive，明确写 N/A，并说明缺少的父子/事件信息。不能用父函数减去所有子函数 duration 的简单方法计算 Exclusive；子区间可能重叠，应计算区间并集。

父函数与子函数的 Inclusive 不可相加。异步线程上的回调不能仅因时间重叠就当作同步函数子调用，应标记为异步提交/执行关系。

### 5.4 串行与并行时间线

除层级树外，必须根据实际时间戳生成文本时间线，至少包含：

- Rank 0 Host Thread
- Rank 1 Host Thread
- 两个 Rank 的异步 Host Thread（如果存在）
- GPU 0 Stream / Kernel
- GPU 1 Stream / Kernel

明确标记：
- 同一 Host Thread 的串行调用顺序。
- 两个 Rank 的并行执行。
- Group 结束和 Launch 阶段。
- Host Kernel Launch 与 GPU Kernel 的重叠。
- GPU 0 与 GPU 1 Kernel 的并行/等待关系。
- 一次通信最终完成的关键路径。

不能只画理想示意图；最终图必须使用采集时间戳。

## 6. Proxy 和传输路径

本轮不单独计量 Proxy Progress，也不修改 Proxy 源码。

目标是 PCIe Direct P2P，并设置 NCCL_P2P_USE_CUDA_MEMCPY=0。但必须从 NCCL 日志确认实际 Transport。若存在 Proxy 工作、NET/SHM 或其他路径，如实报告。不能把 Proxy 时间归入 Host API 或 GPU Kernel。

## 7. 执行步骤

1. Clone 仓库并 checkout 目标分支。
2. 阅读仓库文档和实际源码。
3. 检查环境、GPU 拓扑和 P2P 能力。
4. 先检查 Host Patch 是否与官方 v2.28.9-1 精确匹配。
5. 分别构建未插桩 Baseline 和 Host Trace NCCL。
6. 先完成全部 Baseline 端到端测试。
7. 再完成 8 B Trace 测试和 GPU Kernel Activity 采集。
8. 生成带耗时的 AllReduce/SendRecv 层级文本树和串并行时间线。
9. 计算统计数据并解释主要延迟来源。

任何 Patch 应用失败、构建失败、协议未按预期选择、P2P 路径不符合要求或正确性检查失败，都必须停止对应测试并保留日志。

如果需要修改测试代码或 NCCL 源码，先说明修改文件、原因和对计时的影响，等待用户确认后再修改。不得擅自改变测试方案。

如果当前 Agent 无法访问目标 GPU 服务器，明确说明只能完成仓库分析或构建准备，不能假装已经运行测试。

## 8. 最终交付物

在仓库新增独立结果目录，例如 results/YYYYMMDD-HHMMSS/，至少包含：

1. environment.md：实际测试环境和拓扑。
2. end_to_end.md：两种操作、三种数据量、两种协议的端到端结果。
3. stage_analysis.md：8 B 各函数实测耗时；AllReduce/SendRecv 两棵带耗时的层级文本树。
4. timeline.md：Rank/Host Thread/GPU Stream 的串行并行时间线。
5. raw/：原始 CSV、Host Trace、CUPTI/Nsight 文件、NCCL 日志。
6. build/：Baseline 和 Trace 构建日志及上游 commit SHA。

最终报告的首要内容必须是：
- 端到端延迟结果；
- 8 B 阶段耗时层级树；
- 串行/并行时间线；
- 主要延迟瓶颈结论。

所有结论必须来自实际数据。没有执行的项目标记为未验证，未执行的函数标记 N/A，不得伪造数值。
