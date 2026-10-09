#pragma once
// Low-latency NCCL primitives from Shen et al., arXiv:2607.16100, pinned at 5357eff.
// Ordered traffic uses ncclLLBuffer itself, so NCCL's host collective dispatch cannot silently
// select a different protocol. NCCL windows also hold the buffer-reuse credits; NVSHMEM is not
// involved in these ordered exchanges. Every consumed LL slot is reset, including variable-size
// tails, so epoch wrap cannot make an old payload appear ready.
#include "common.hpp"
#include <nccl.h>
#include <nccl_device.h>
#include <nccl_device/ll_buffer.h>
#include <nccl_device/impl/ll_buffer__funcs.h>

namespace qr_omega {
inline void nccl_check(ncclResult_t result) {
    if (result != ncclSuccess)
        throw std::runtime_error(std::string("low-latency NCCL: ") + ncclGetErrorString(result));
}

__device__ inline uint64_t acquire_load(const uint64_t *p) {
    uint64_t value;
    asm volatile("ld.acquire.sys.global.u64 %0,[%1];" : "=l"(value) : "l"(p) : "memory");
    return value;
}
__device__ inline void release_store(uint64_t *p, uint64_t value) {
    asm volatile("st.release.sys.global.u64 [%0],%1;" ::"l"(p), "l"(value) : "memory");
}

// One registered window and device communicator. Construction and destruction happen outside
// factorization timing; all communication within it is submitted to the caller's CUDA stream.
class LowLatencyWindow {
    ncclComm_t comm;
    void *memory = nullptr;

  public:
    ncclWindow_t window = nullptr;
    ncclDevComm device{};
    LowLatencyWindow(ncclComm_t comm_, size_t bytes) : comm(comm_) {
        nccl_check(ncclMemAlloc(&memory, bytes));
        CU(cudaMemset(memory, 0, bytes));
        nccl_check(ncclCommWindowRegister(comm, memory, bytes, &window, NCCL_WIN_COLL_SYMMETRIC));
        ncclDevCommRequirements req = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
        req.lsaBarrierCount = 0;
        nccl_check(ncclDevCommCreate(comm, &req, &device));
        if (ncclTeamLsa(device).nRanks != device.nRanks)
            throw std::runtime_error(
                "low-latency NCCL requires one load/store-accessible GPU team");
    }
    LowLatencyWindow(const LowLatencyWindow &) = delete;
    ~LowLatencyWindow() {
        if (std::uncaught_exceptions())
            return;
        ncclDevCommDestroy(comm, &device);
        ncclCommWindowDeregister(comm, window);
        ncclMemFree(memory);
    }
};

enum class CollectiveOp { Broadcast, Sum, Maximum, Gather };

template <class T, CollectiveOp Op>
__global__ void low_latency_collective(ncclDevComm comm, ncclWindow_t window, int pitch,
                                       size_t progress_at, uint64_t sequence, const T *input,
                                       T *output, int count, int root) {
    const auto team = ncclTeamLsa(comm);
    const ncclSymPtr<uint64_t> progress(window, progress_at);
    // Two LL buffers: writers may advance one episode ahead, but cannot overwrite a buffer
    // until all readers of its previous use have retired. No host rendezvous is needed.
    if (threadIdx.x == 0 && sequence > 2)
        for (int p = 0; p < comm.nRanks; ++p)
            while (acquire_load(progress.localPtr() + p) < sequence - 2) {
            }
    __syncthreads();
    ncclLLBuffer<ncclLL, false> ll(ncclSymPtr<char>(window, 0), pitch, 0, 2, ncclMultimemHandle{});
    ll.setSubBuffer(unsigned(sequence & 1));
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
        if constexpr (Op == CollectiveOp::Broadcast) {
            if (comm.rank == root)
                for (int p = 0; p < comm.nRanks; ++p)
                    ll.send<T>(team, p, i, input[i]);
            output[i] = ll.recv<T, true>(i);
        } else {
            const T value = input[i];
            for (int p = 0; p < comm.nRanks; ++p)
                ll.send<T>(team, p, comm.rank * count + i, value);
            T sum = T(0);
            for (int p = 0; p < comm.nRanks; ++p) {
                const T part = ll.recv<T, true>(p * count + i);
                if constexpr (Op == CollectiveOp::Gather)
                    output[p * count + i] = part;
                else if constexpr (Op == CollectiveOp::Sum)
                    sum = p == 0 ? part : sum + part;
                else
                    sum = p == 0 || part > sum ? part : sum;
            }
            if constexpr (Op != CollectiveOp::Gather)
                output[i] = sum;
        }
    }
    // Reset stores must be globally visible before returning the credit. Every thread fences
    // its own writes; the last CTA publishes completion only after all CTAs have done so.
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        auto *done =
            reinterpret_cast<unsigned *>(progress.localPtr() + comm.nRanks) + (sequence & 1);
        if (atomicAdd(done, 1u) == gridDim.x - 1) {
            *done = 0;
            for (int p = 0; p < comm.nRanks; ++p)
                release_store(progress.peerPtr(team, p) + comm.rank, sequence);
        }
    }
}

class CollectiveTeam {
    const int ranks;
    const size_t capacity, progress_at;
    const int pitch;
    LowLatencyWindow arena;
    uint64_t sequence = 0;
    Event last;

  public:
    CollectiveTeam(ncclComm_t comm, int ranks_, size_t bytes, bool broadcast)
        : ranks(ranks_), capacity(bytes), progress_at(4 * bytes * (broadcast ? 1 : ranks)),
          pitch(int(bytes * (broadcast ? 1 : ranks))),
          arena(comm, progress_at + size_t(ranks) * sizeof(uint64_t) + 16) {
        if (bytes > size_t(INT_MAX / 2) / ranks || bytes % 8)
            throw std::runtime_error(
                "low-latency collective capacity must fit its aligned LL window");
    }
    template <class T, CollectiveOp Op>
    void run(const T *input, T *output, size_t count, int root, cudaStream_t stream) {
        if (count * sizeof(T) > capacity || count > size_t(INT_MAX / ranks))
            throw std::runtime_error("low-latency collective exceeds its window");
        if (!count)
            return;
        if (sequence)
            last.wait(stream);
        ++sequence;
        low_latency_collective<T, Op><<<std::min(32, ceildiv(int(count), 256)), 256, 0, stream>>>(
            arena.device, arena.window, pitch, progress_at, sequence, input, output, int(count),
            root);
        CU(cudaGetLastError());
        last.record(stream);
    }
};

// Ordered point-to-point exchanges use a distinct inbox per sender. A receiver resets the
// consumed slots and then acknowledges them through the NCCL window. Unrelated pairs need
// not participate, and there is no global epoch or global barrier.
template <bool Send>
__global__ void ordered_exchange(ncclDevComm comm, ncclWindow_t window, int capacity,
                                 size_t credits_at, int sender, int receiver, uint64_t sequence,
                                 const uint32_t *input, uint32_t *output, int count) {
    const auto team = ncclTeamLsa(comm);
    const ncclSymPtr<uint64_t> credits(window, credits_at);
    if constexpr (Send) {
        if (threadIdx.x == 0)
            while (acquire_load(credits.localPtr() + receiver) < sequence - 1) {
            }
        __syncthreads();
    }
    ncclLLBuffer<ncclLL, false> ll(ncclSymPtr<char>(window, size_t(sender) * 2 * capacity),
                                   capacity, 0, 1, ncclMultimemHandle{});
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
        if constexpr (Send)
            ll.send<uint32_t>(team, receiver, i, input[i]);
        else
            output[i] = ll.recv<uint32_t, true>(i);
    }
    if constexpr (!Send) {
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) {
            auto *done = reinterpret_cast<unsigned *>(credits.localPtr() + comm.nRanks) + sender;
            if (atomicAdd(done, 1u) == gridDim.x - 1) {
                *done = 0;
                release_store(credits.peerPtr(team, sender) + receiver, sequence);
            }
        }
    }
}

class OrderedLinks {
    const int rank, ranks, capacity;
    const size_t credits_at;
    LowLatencyWindow arena;
    std::vector<uint64_t> sent, received;
    Event last;
    bool used = false;

  public:
    OrderedLinks(ncclComm_t comm, int rank_, int ranks_, size_t bytes)
        : rank(rank_), ranks(ranks_), capacity(int(bytes)), credits_at(2 * bytes * ranks),
          arena(comm, credits_at + size_t(ranks) * 16), sent(ranks, 0), received(ranks, 0) {}
    void exchange(int sender, int receiver, const void *source, void *destination, size_t bytes,
                  cudaStream_t stream) {
        if (bytes > size_t(capacity) || bytes % sizeof(uint32_t))
            throw std::runtime_error("ordered exchange exceeds its window or has unaligned bytes");
        if (rank != sender && rank != receiver)
            return;
        if (used)
            last.wait(stream);
        used = true;
        if (sender == receiver)
            CU(cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDeviceToDevice, stream));
        else {
            const int count = int(bytes / sizeof(uint32_t)),
                      blocks = std::max(1, std::min(16, ceildiv(count, 256)));
            if (rank == sender)
                ordered_exchange<true><<<blocks, 256, 0, stream>>>(
                    arena.device, arena.window, capacity, credits_at, sender, receiver,
                    ++sent[receiver], static_cast<const uint32_t *>(source), nullptr, count);
            else
                ordered_exchange<false><<<blocks, 256, 0, stream>>>(
                    arena.device, arena.window, capacity, credits_at, sender, receiver,
                    ++received[sender], nullptr, static_cast<uint32_t *>(destination), count);
            CU(cudaGetLastError());
        }
        last.record(stream);
    }
};
} // namespace qr_omega
