# NCCL v2.28.9-1 源码插桩实施说明

本目录当前是插桩设计，不是已验证 Patch。必须在精确 checkout 的 NVIDIA/nccl v2.28.9-1 源码上完成修改、编译和目标机器验证后，才能生成正式 Patch。

## 覆盖范围
- AllReduce：8 Byte 的 SIMPLE、LL。
- Send/Recv：8 Byte 的 SIMPLE、LL。
- 端到端测试覆盖 8 Byte、16 KiB、8 MiB。
- 详细 Host/Device 阶段计时仅覆盖 8 Byte。

## Host 探针
按 v2.28.9-1 实际调用链确认后插桩：
1. Collective API / P2P API 入口和返回。
2. enqueue、task append。
3. group end、任务调度。
4. 算法和协议选择。
5. plan、work upload、kernel launch。
6. Proxy op 创建、提交和 progress（仅当实际路径产生 Proxy 工作）。

记录 rank、thread、operation sequence、event id、begin/end CLOCK_MONOTONIC_RAW 时间戳，以及可获取的 channel/work id。使用固定容量 thread-local buffer；热路径禁止 printf、动态分配和锁。

父子函数区间不可直接相加。父区间用于时间线；互斥叶子区间用于耗时归因。

## Device 探针
先确认实际算法、协议、模板特化和执行分支。记录 Kernel 入口、work batch 装载、dispatch、算法/P2P 阶段、peer/flag wait、数据读取与传输、Reduce（仅 AllReduce 适用）、写回、fence/notify 和 Kernel 完成。

使用 clock64() 计算单 GPU 本地 duration，不跨 GPU 比较绝对设备时间戳。不同 Primitive 的控制流不同，不假设它们执行同一套阶段。

## Proxy 判定
PCIe P2P 直连不等于必然无 Proxy。核对 Connector 的 proxyProgress、Transport 和 CUDA memcpy/CE 配置。无 Proxy 工作时记为 N/A，不记为 0。

## 验证
1. 记录上游 Tag 和 commit SHA。
2. 构建未插桩基线。
3. 应用插桩并构建。
4. 验证 AllReduce/SendRecv × SIMPLE/LL 四种 8 Byte 组合。
5. 检查正确性、实际算法/协议/Transport。
6. 对比插桩前后 E2E，量化插桩开销。
7. 保存 CUDA、Driver、编译器、GPU 拓扑和日志。
