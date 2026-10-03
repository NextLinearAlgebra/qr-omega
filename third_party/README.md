# Third-party code

* `include/qr_omega/group_mma_multistage.h` is a modified copy of CUTLASS
  `include/cutlass/gemm/threadblock/mma_multistage.h` at commit `c900acd605e907f88b4554d59924893dcc19a3c3`. It keeps
  NVIDIA's copyright and BSD-3-Clause notice (`CUTLASS-LICENSE.txt`); the changes are described at the top of the
  file.
* `include/qr_omega/panel_register.cuh` adapts the register-resident panels of gau.nernst's submission to the GPU
  MODE QR-v2 leaderboard.
* `include/qr_omega/carrier_kami.cuh` follows the communication-avoiding GEMM design of KAMI (Wang et al., SC'25).
* The transport between GPUs uses the `LLBuffer` primitives of the low-latency NCCL of Shen et al. (commit `5357eff`).

CUTLASS, nlohmann/json, CUDA, NCCL, NVSHMEM, MPI, and the reference libraries (cuSOLVERMp, MAGMA, SLATE, MKL) are
fetched or installed separately and keep their own licenses.
