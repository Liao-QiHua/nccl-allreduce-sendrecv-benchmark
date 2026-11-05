# NCCL Reduce / SendRecv Benchmark

NVIDIA NCCL v2.28.9-1; two local MPI ranks; one PCIe-connected GPU per rank.

## Scope
- AllReduce and paired Send/Recv.
- FP32; payloads 8 bytes, 16 KiB, 8 MiB.
- Force protocol separately with NCCL_PROTO=Simple and NCCL_PROTO=LL.
- Leave NCCL_ALGO unset for runtime algorithm selection; record actual selection from NCCL logs.
- 20 warmups and 100 measured iterations for every operation/protocol/payload.
- Detailed Host and Device stage timing is limited to 8-byte runs.

8 bytes means two FP32 elements, not eight billion elements.

## Repository layout
- benchmark/: CUDA/MPI benchmark source.
- scripts/: build, run, and PCIe topology checks.
- docs/: Chinese test design and execution plan.
- instrumentation/: exact-source instrumentation plan.

The source instrumentation is not considered a verified NCCL patch until applied to the exact v2.28.9-1 source, built, and validated on the target GPU host.

See docs/测试设计与实施文档.md.
