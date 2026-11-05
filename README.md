# NCCL Reduce / SendRecv Benchmark

NCCL v2.28.9-1 benchmark project for two local MPI ranks / two PCIe-connected GPUs.

Planned coverage:
- AllReduce: 8 B, 16 KiB, 8 MiB; FP32 SUM.
- Send/Recv: bidirectional peer exchange, same payload sizes.
- Protocol runs: NCCL_PROTO=Simple and NCCL_PROTO=LL, separately.
- 20 warmups and 100 measured iterations per operation and payload.
- Host API time and end-to-end completion time.
- 8-byte-only detailed Host and Device instrumentation.

See docs/ for the test design. Source instrumentation must be validated against the exact
NCCL v2.28.9-1 checkout before being described as a verified patch.
