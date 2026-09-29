# Third-party code

* `group_mma_multistage.h` and `gmma_cooperative_2cta.cuh` (in `single-gpu/include` and `multi-gpu/include`) are
  modified copies of CUTLASS files at commit `c900acd605e907f88b4554d59924893dcc19a3c3`
  (`include/cutlass/gemm/threadblock/mma_multistage.h` and `sm90_gemm_tma_warpspecialized_cooperative.hpp`). They
  keep NVIDIA's copyright and BSD-3-Clause notice (`CUTLASS-LICENSE.txt`); the changes are described at the top
  of each file.
* The `gau_*` panel kernels adapt the register-resident panels of gau.nernst's submission 844219 to the GPU MODE
  QR-v2 leaderboard; the headers describe what was kept and what was changed.
* The `kami_*` kernels follow the communication-avoiding GEMM design of KAMI (Wang et al., SC'25).
* The inter-GPU transport uses the `LLBuffer` primitives of the low-latency NCCL of Shen et al. (commit `5357eff`).

CUTLASS, nlohmann/json, CUDA, NCCL, NVSHMEM, MPI, and the reference libraries (cuSOLVERMp, MAGMA, SLATE, MKL) are
fetched or installed separately and keep their own licenses.
