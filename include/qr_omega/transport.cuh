#pragma once
// Communication between the GPUs of a node. Gathers and publications are one-sided: payloads travel
// through NVSHMEM puts or the LLBuffer primitives of the low-latency NCCL over a symmetric window,
// and signals act as credits that order the reuse of every buffer on the device, without host
// barriers. The all-reduce of W is ncclAllReduce, whose ring sums the partials in a fixed order.
#include "kernels.cuh"
#ifdef TQR_MULTI
#include <mpi.h>
#define NVSHMEMI_HOST_ONLY
#include <nvshmem_host.h>
#include <nccl.h>
#include <nccl_device.h>
#include <nccl_device/ll_buffer.h>
#include <nccl_device/impl/ll_buffer__funcs.h>
#endif
namespace tqr {
// The GPUs among which the columns of an h x q block are divided: member z owns the columns
// [q z / size, q (z + 1) / size).
struct ColumnTeam {
    int size = 0, rank[8] = {};
    __host__ __device__ int index(int r) const {
        for (int z = 0; z < size; ++z)
            if (rank[z] == r)
                return z;
        return -1;
    }
    __host__ __device__ int begin(int z, int q) const {
        return int((long long)q * z / size);
    }
    __host__ __device__ int owner(int j, int q) const {
        return int(((long long)(j + 1) * size - 1) / q);
    }
};

// One process per GPU. Under MPI every rank must run on the same node and on a different GPU.
struct Context {
    int rank = 0, size = 1, device = 0;
    Context() {
#ifdef TQR_MULTI
        int provided;
        if (MPI_Init_thread(nullptr, nullptr, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS)
            throw std::runtime_error("MPI_Init_thread failed");
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm node;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
        int node_rank, node_size;
        MPI_Comm_rank(node, &node_rank);
        MPI_Comm_size(node, &node_size);
        MPI_Comm_free(&node);
        if (node_size != size)
            throw std::runtime_error("all ranks must run on one node");
        int devices;
        CU(cudaGetDeviceCount(&devices));
        device = devices == 1 ? 0 : node_rank;
        if (device >= devices)
            throw std::runtime_error("fewer visible GPUs than ranks");
#endif
        CU(cudaSetDevice(device));
#ifdef TQR_MULTI
        const cudaDeviceProp &prop = device_properties();
        std::vector<char> ids(size * sizeof(prop.uuid));
        MPI_Allgather(prop.uuid.bytes, sizeof(prop.uuid), MPI_CHAR, ids.data(), sizeof(prop.uuid),
                      MPI_CHAR, MPI_COMM_WORLD);
        for (int r = 0; r < size; ++r)
            if (r != rank && std::equal(prop.uuid.bytes, prop.uuid.bytes + sizeof(prop.uuid),
                                        ids.data() + r * sizeof(prop.uuid)))
                throw std::runtime_error("two ranks share a GPU");
#endif
    }
    ~Context() {
#ifdef TQR_MULTI
        // While an exception unwinds, main reports it and aborts the job instead.
        if (!std::uncaught_exceptions())
            MPI_Finalize();
#endif
    }
    void barrier() const {
#ifdef TQR_MULTI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
    }
    // Largest value over the ranks.
    double max(double value) const {
#ifdef TQR_MULTI
        MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
        return value;
    }
    int max(int value) const {
#ifdef TQR_MULTI
        MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
        return value;
    }
    double min(double value) const {
        return -max(-value);
    }
    uint64_t sum(uint64_t value) const {
#ifdef TQR_MULTI
        MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
#endif
        return value;
    }
    // Ends the job on every rank after an error.
    static void abort() {
#ifdef TQR_MULTI
        int initialized = 0, finalized = 0;
        MPI_Initialized(&initialized);
        if (initialized)
            MPI_Finalized(&finalized);
        if (initialized && !finalized)
            MPI_Abort(MPI_COMM_WORLD, 2);
#endif
    }
};

#ifdef TQR_MULTI
inline void nccl_check(ncclResult_t e, const char *where) {
    if (e != ncclSuccess)
        throw std::runtime_error(std::string(where) + ": " + ncclGetErrorString(e));
}
#define NC(x) ::tqr::nccl_check((x), #x)

// Every rank counts its LLBuffer kernels with the same sequence number. The window holds two
// sub-buffers, used alternately; when the last block of a rank's kernel `seq` retires, the rank
// stores `seq` into its slot of every peer's progress array, and a sender of kernel `seq` first
// waits, on the device, until its receivers have finished kernel seq - 2, the previous user of the
// same sub-buffer.
__device__ inline unsigned long long transport_ld_acquire(const unsigned long long *p) {
    unsigned long long v;
    asm volatile("ld.acquire.sys.global.u64 %0,[%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}
__device__ inline void transport_st_release(unsigned long long *p, unsigned long long v) {
    asm volatile("st.release.sys.global.u64 [%0],%1;" ::"l"(p), "l"(v) : "memory");
}
__global__ inline void transport_wait_progress(const unsigned long long *progress, int peers,
                                               unsigned long long target) {
    if (threadIdx.x == 0)
        for (int p = 0; p < peers; ++p)
            while (transport_ld_acquire(progress + p) < target) {
            }
}
// The second half of a progress array acknowledges the clearing of a window before its epochs
// are reused.
__global__ inline void transport_clear_ack(unsigned long long *const *peers, int ranks, int rank,
                                           unsigned long long generation) {
    if (threadIdx.x == 0) {
        __threadfence_system();
        for (int p = 0; p < ranks; ++p)
            transport_st_release(peers[p] + ranks + rank, generation);
    }
}
// The last block of a rank's kernel `seq` publishes `seq` to every peer.
__device__ inline void transport_retire(unsigned *done, unsigned long long seq,
                                        unsigned long long *const *peer_progress, int rank,
                                        int ranks) {
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system(); // the block's writes to the peers complete first
        unsigned *d = done + (seq & 3);
        if (atomicAdd(d, 1u) == gridDim.x - 1) {
            *d = 0;
            __threadfence_system();
            for (int p = 0; p < ranks; ++p)
                transport_st_release(peer_progress[p] + rank, seq);
        }
    }
}
// Copies `count` elements from the root to every GPU.
template <class T>
__global__ void ll_broadcast(ncclDevComm comm, ncclWindow_t win, const T *input, T *output,
                             int count, int root, uint8_t epoch, int pitch, unsigned long long seq,
                             unsigned long long *const *peer_progress,
                             const unsigned long long *my_progress, unsigned *done) {
    if (threadIdx.x == 0 && seq >= 2 && comm.rank == root)
        for (int p = 0; p < comm.nRanks; ++p)
            while (transport_ld_acquire(my_progress + p) + 2 < seq) {
            }
    __syncthreads();
    ncclLLBuffer<ncclLL, false> ll(ncclSymPtr<char>(win, 0), pitch, 0, uint8_t(2),
                                   ncclMultimemHandle{});
    ll.setEpochValue(epoch);
    ll.setSubBuffer(uint32_t(seq & 1));
    auto team = ncclTeamLsa(comm);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        if (comm.rank == root)
            for (int p = 0; p < comm.nRanks; ++p)
                ll.template send<T>(team, p, i, input[i]);
        output[i] = ll.template recv<T, false>(i);
    }
    transport_retire(done, seq, peer_progress, comm.rank, comm.nRanks);
}
// Reduce: every member sends its rows x columns partial and the owner of each column sums the
// partials of the members in rank order. Otherwise every member sends its own columns to all
// members. GPUs outside the team move no data and only advance the progress protocol.
template <class T, bool Reduce>
__global__ void ll_columns(ncclDevComm comm, ncclWindow_t win, const T *input, T *output, int rows,
                           int columns, bool transposed, ColumnTeam members, uint8_t epoch,
                           int pitch, unsigned long long seq,
                           unsigned long long *const *peer_progress,
                           const unsigned long long *my_progress, unsigned *done) {
    const int mine = members.index(comm.rank), count = rows * columns;
    if (mine >= 0) {
        if (threadIdx.x == 0)
            for (int z = 0; z < members.size; ++z)
                while (transport_ld_acquire(my_progress + members.rank[z]) + 2 < seq) {
                }
        __syncthreads();
        ncclLLBuffer<ncclLL, false> ll(ncclSymPtr<char>(win, 0), pitch, 0, uint8_t(2),
                                       ncclMultimemHandle{});
        ll.setEpochValue(epoch);
        ll.setSubBuffer(uint32_t(seq & 1));
        auto team = ncclTeamLsa(comm);
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
             i += blockDim.x * gridDim.x) {
            const int row = i % rows, j = i / rows, z = members.owner(j, columns);
            const int first = members.begin(z, columns),
                      width = members.begin(z + 1, columns) - first;
            if constexpr (Reduce) {
                ll.template send<T>(team, members.rank[z], comm.rank * count + i, input[i]);
                if (mine == z) {
                    T sum = T(0);
                    bool initial = true;
                    for (int r = 0; r < comm.nRanks; ++r)
                        if (members.index(r) >= 0) {
                            T v = ll.template recv<T, false>(r * count + i);
                            sum = initial ? v : sum + v;
                            initial = false;
                        }
                    output[row + (j - first) * rows] = sum;
                }
            } else {
                if (mine == z) {
                    const T value =
                        input[transposed ? j - first + row * width : row + (j - first) * rows];
                    for (int peer = 0; peer < members.size; ++peer)
                        ll.template send<T>(team, members.rank[peer], i, value);
                }
                output[transposed ? j + row * columns : i] = ll.template recv<T, false>(i);
            }
        }
    }
    transport_retire(done, seq, peer_progress, comm.rank, comm.nRanks);
}
class Transport {
    Context &ctx;
    size_t capacity, publication_capacity;
    Stream own;
    cudaStream_t stream;
    uint64_t sequence = 0;
    ncclComm_t comm = nullptr;
    ncclDevComm device_comm{};
    ncclWindow_t window = nullptr;
    void *window_memory = nullptr, *staged_input = nullptr, *staged_output = nullptr;
    size_t window_bytes = 0;
    // The window is cleared before its 254 epochs are reused.
    uint64_t next_clear = 254;
    char *inbox = nullptr, *outbox = nullptr;
    uint64_t *signals = nullptr, *credits = nullptr;
    unsigned long long *progress = nullptr, **peer_progress = nullptr, kernels = 0;
    unsigned *progress_done = nullptr;
    // Credit of this rank's last publication, awaited before the outbox is staged again.
    uint64_t pending_credit = 0;
    int pending_receiver = 0;

    static constexpr int min_blocks = 32, elements_per_thread = 32;

    template <class Kernel> static int resident_blocks(Kernel kernel) {
        int per_sm = 0;
        CU(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, 128, 0));
        return std::max(1, per_sm * device_properties().multiProcessorCount);
    }
    static int blocks_for(size_t count, int resident) {
        const size_t sized = std::min<size_t>(
            resident, std::max<size_t>(min_blocks, (count + 128 * elements_per_thread - 1) /
                                                       (128 * elements_per_thread)));
        return int(std::max<size_t>(1, std::min(sized, (count + 127) / 128)));
    }
    // Every reader of the previous epochs finishes, then every rank clears its window and waits
    // for the clears of the others, all on the device.
    void clear_window_on_wrap() {
        if (sequence < next_clear)
            return;
        transport_wait_progress<<<1, 1, 0, stream>>>(progress, ctx.size, kernels);
        CU(cudaMemsetAsync(window_memory, 0, window_bytes, stream));
        transport_clear_ack<<<1, 1, 0, stream>>>(peer_progress, ctx.size, ctx.rank, sequence);
        transport_wait_progress<<<1, 1, 0, stream>>>(progress + ctx.size, ctx.size, sequence);
        CU(cudaGetLastError());
        next_clear = sequence + 254;
    }
    uint8_t epoch() const {
        return uint8_t(2 + sequence % 254);
    }

  public:
    // `bytes` bounds a collective payload, `publication_bytes` a published block.
    Transport(Context &c, size_t bytes, size_t publication_bytes)
        : ctx(c), capacity(std::max(bytes, size_t(64))),
          publication_capacity(std::max(publication_bytes, size_t(64))), stream(own.s) {
        nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
        MPI_Comm world = MPI_COMM_WORLD;
        attr.mpi_comm = &world;
        if (nvshmemx_hostlib_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr) != 0)
            throw std::runtime_error("NVSHMEM initialization failed");
        if (nvshmem_my_pe() != ctx.rank || nvshmem_n_pes() != ctx.size)
            throw std::runtime_error("NVSHMEM and MPI disagree on the ranks");
        inbox = (char *)nvshmem_malloc(publication_capacity * size_t(ctx.size));
        outbox = (char *)nvshmem_malloc(publication_capacity);
        signals = (uint64_t *)nvshmem_calloc(ctx.size, sizeof(uint64_t));
        credits = (uint64_t *)nvshmem_calloc(ctx.size, sizeof(uint64_t));
        progress = (unsigned long long *)nvshmem_calloc(2 * ctx.size, sizeof(unsigned long long));
        peer_progress =
            (unsigned long long **)nvshmem_malloc(sizeof(unsigned long long *) * ctx.size);
        progress_done = (unsigned *)nvshmem_calloc(4, sizeof(unsigned));
        if (!inbox || !outbox || !signals || !credits || !progress || !peer_progress ||
            !progress_done)
            throw std::runtime_error("NVSHMEM allocation failed");
        std::vector<unsigned long long *> peers(ctx.size);
        for (int p = 0; p < ctx.size; ++p) {
            peers[p] = (unsigned long long *)nvshmem_ptr(progress, p);
            if (!peers[p])
                throw std::runtime_error("a peer's symmetric heap is not directly addressable");
        }
        CU(cudaMemcpy(peer_progress, peers.data(), sizeof(unsigned long long *) * ctx.size,
                      cudaMemcpyHostToDevice));
        nvshmem_barrier_all();

        ncclUniqueId id;
        if (ctx.rank == 0)
            NC(ncclGetUniqueId(&id));
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
        NC(ncclCommInitRank(&comm, ctx.size, id, ctx.rank));
        window_bytes = checked_mul(capacity, size_t(ctx.size) * 4);
        NC(ncclMemAlloc(&window_memory, window_bytes));
        NC(ncclMemAlloc(&staged_input, capacity));
        NC(ncclMemAlloc(&staged_output, capacity));
        CU(cudaMemsetAsync(window_memory, 0, window_bytes, stream));
        CU(cudaStreamSynchronize(stream));
        ctx.barrier();
        NC(ncclCommWindowRegister(comm, window_memory, window_bytes, &window,
                                  NCCL_WIN_COLL_SYMMETRIC));
        ncclDevCommRequirements req = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
        req.lsaBarrierCount = 0;
        req.lsaMultimem = false;
        NC(ncclDevCommCreate(comm, &req, &device_comm));
        if (ncclTeamLsa(comm).nRanks != ctx.size)
            throw std::runtime_error("the GPUs do not form one load/store-accessible team");
    }
    ~Transport() {
        if (std::uncaught_exceptions())
            return;
        cudaDeviceSynchronize();
        ncclDevCommDestroy(comm, &device_comm);
        ncclCommWindowDeregister(comm, window);
        ncclMemFree(window_memory);
        ncclMemFree(staged_input);
        ncclMemFree(staged_output);
        ncclCommDestroy(comm);
        nvshmem_free(peer_progress);
        nvshmem_free(progress_done);
        nvshmem_free(progress);
        nvshmem_free(credits);
        nvshmem_free(signals);
        nvshmem_free(outbox);
        nvshmem_free(inbox);
        nvshmemx_hostlib_finalize();
    }
    // Operations are issued on the bound stream, in order with the arithmetic that produces their
    // operands and consumes their results. Unbound, the transport uses a stream of its own.
    void bind(cudaStream_t s) {
        stream = s ? s : own.s;
    }
    cudaStream_t bound() const {
        return stream;
    }
    // Copies `bytes` from `source` on the sender to `destination` on the receiver. The sender waits
    // for the credit of its previous publication before it stages the next one, and every sender
    // owns its region of the receiver's inbox.
    void publish(int sender, int receiver, const void *source, void *destination, size_t bytes) {
        const uint64_t generation = ++sequence;
        if (bytes > publication_capacity)
            throw std::runtime_error("publication exceeds the transport capacity");
        char *slot = inbox + size_t(sender) * publication_capacity;
        if (ctx.rank == sender) {
            if (pending_credit)
                nvshmemx_signal_wait_until_on_stream(credits + pending_receiver, NVSHMEM_CMP_EQ,
                                                     pending_credit, stream);
            CU(cudaMemcpyAsync(outbox, source, bytes, cudaMemcpyDeviceToDevice, stream));
            nvshmemx_putmem_signal_on_stream(slot, outbox, bytes, signals + sender, generation,
                                             NVSHMEM_SIGNAL_SET, receiver, stream);
            pending_credit = generation;
            pending_receiver = receiver;
        }
        if (ctx.rank == receiver) {
            nvshmemx_signal_wait_until_on_stream(signals + sender, NVSHMEM_CMP_EQ, generation,
                                                 stream);
            CU(cudaMemcpyAsync(destination, slot, bytes, cudaMemcpyDeviceToDevice, stream));
            nvshmemx_signal_op_on_stream(credits + receiver, generation, NVSHMEM_SIGNAL_SET, sender,
                                         stream);
        }
    }
    // Copies `count` elements from `source` on the root to `destination` on every GPU.
    template <class T> void broadcast(int root, const T *source, T *destination, int count) {
        ++sequence;
        const size_t bytes = size_t(count) * sizeof(T);
        if (bytes > capacity)
            throw std::runtime_error("broadcast exceeds the transport capacity");
        if (!count)
            return;
        clear_window_on_wrap();
        if (ctx.rank == root)
            CU(cudaMemcpyAsync(staged_input, source, bytes, cudaMemcpyDeviceToDevice, stream));
        static const int resident = resident_blocks(ll_broadcast<T>);
        ll_broadcast<T><<<blocks_for(count, resident), 128, 0, stream>>>(
            device_comm, window, (T *)staged_input, (T *)staged_output, count, root, epoch(),
            int(capacity * ctx.size), ++kernels, peer_progress, progress, progress_done);
        CU(cudaGetLastError());
        CU(cudaMemcpyAsync(destination, staged_output, bytes, cudaMemcpyDeviceToDevice, stream));
    }
    // destination = sum over the GPUs of source, identical on every GPU.
    template <class T> void allreduce(const T *source, T *destination, size_t count) {
        ++sequence;
        if (count)
            NC(ncclAllReduce(source, destination, count, sizeof(T) == 4 ? ncclFloat : ncclDouble,
                             ncclSum, comm, stream));
    }
    // The team's columns of a rows x columns block (see ll_columns): the reduction of the
    // members' partials to the owners of the columns, or the gather of every owner's columns.
    template <class T, bool Reduce>
    void columns(const ColumnTeam &team, const T *source, T *destination, int rows, int columns,
                 bool transposed) {
        const size_t count = checked_mul(size_t(rows), size_t(columns));
        if (count * sizeof(T) > capacity)
            throw std::runtime_error("column exchange exceeds the transport capacity");
        ++sequence;
        if (!count)
            return;
        clear_window_on_wrap();
        const bool member = team.index(ctx.rank) >= 0;
        static const int resident = resident_blocks(ll_columns<T, Reduce>);
        ll_columns<T, Reduce><<<member ? blocks_for(count, resident) : 1, 128, 0, stream>>>(
            device_comm, window, source, destination, rows, columns, transposed, team, epoch(),
            int(capacity * ctx.size), ++kernels, peer_progress, progress, progress_done);
        CU(cudaGetLastError());
    }
};
#else
// A build without MPI runs on one GPU and never creates a transport.
class Transport {
  public:
    Transport(Context &, size_t, size_t) {}
    void bind(cudaStream_t) {}
    cudaStream_t bound() const {
        return nullptr;
    }
    void publish(int, int, const void *, void *, size_t) {}
    template <class T> void broadcast(int, const T *, T *, int) {}
    template <class T> void allreduce(const T *, T *, size_t) {}
    template <class T, bool Reduce>
    void columns(const ColumnTeam &, const T *, T *, int, int, bool) {}
};
#endif
} // namespace tqr
