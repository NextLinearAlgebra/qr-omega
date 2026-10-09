#pragma once
// Communication between GPUs of a node: the NCCL implementation of arXiv:2607.16100 carries
// collectives and ordered exchanges. Independent one-sided publications use NVSHMEM. Both paths
// retain device-side producer/consumer and buffer-reuse dependencies without host rendezvous.
#include "kernels.cuh"
#ifdef QR_OMEGA_MULTI_GPU
#include <mpi.h>
#define NVSHMEMI_HOST_ONLY
#include <nvshmem_host.h>
#include <nccl.h>
#include "low_latency_collectives.cuh"
#endif
namespace qr_omega {

// One process per GPU. Under MPI every rank must run on the same node and on a different GPU.
struct Context {
    int rank = 0, size = 1, device = 0;
    Context() {
#ifdef QR_OMEGA_MULTI_GPU
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
#ifdef QR_OMEGA_MULTI_GPU
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
#ifdef QR_OMEGA_MULTI_GPU
        // While an exception unwinds, main reports it and aborts the job instead.
        if (!std::uncaught_exceptions())
            MPI_Finalize();
#endif
    }
    void barrier() const {
#ifdef QR_OMEGA_MULTI_GPU
        MPI_Barrier(MPI_COMM_WORLD);
#endif
    }
    // Largest value over the ranks.
    double max(double value) const {
#ifdef QR_OMEGA_MULTI_GPU
        MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
        return value;
    }
    int max(int value) const {
#ifdef QR_OMEGA_MULTI_GPU
        MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
        return value;
    }
    double min(double value) const {
        return -max(-value);
    }
    uint64_t sum(uint64_t value) const {
#ifdef QR_OMEGA_MULTI_GPU
        MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
#endif
        return value;
    }
    double sum(double value) const {
#ifdef QR_OMEGA_MULTI_GPU
        MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
        return value;
    }
    // Ends the job on every rank after an error.
    static void abort() {
#ifdef QR_OMEGA_MULTI_GPU
        int initialized = 0, finalized = 0;
        MPI_Initialized(&initialized);
        if (initialized)
            MPI_Finalized(&finalized);
        if (initialized && !finalized)
            MPI_Abort(MPI_COMM_WORLD, 2);
#endif
    }
};

#ifdef QR_OMEGA_MULTI_GPU
inline void nccl_check(ncclResult_t e, const char *where) {
    if (e != ncclSuccess)
        throw std::runtime_error(std::string(where) + ": " + ncclGetErrorString(e));
}
#define NC(x) ::qr_omega::nccl_check((x), #x)

template <class T> ncclDataType_t nccl_type() {
    return sizeof(T) == 4 ? ncclFloat : ncclDouble;
}
class Transport {
    // Two local operand slots. A completed NCCL broadcast has consumed its input and installed
    // the receiver's copy; the engine's far-update event orders reuse of the local operands.
    struct Channel {
        char *slots = nullptr;
        size_t bytes = 0;
    };

    Context &ctx;
    const int grid_row, grid_col, grid_cols;
    size_t publication_capacity;
    Stream own;
    cudaStream_t stream;
    std::vector<uint64_t> unordered_sent, unordered_received;
    ncclComm_t comm = nullptr, column_comm = nullptr, row_comm = nullptr;
    int column_size = 1;
    std::unique_ptr<CollectiveTeam> row_team, column_team;
    std::unique_ptr<OrderedLinks> links;
    uint64_t ordered_copies = 0, collectives = 0, unordered_copies = 0;
    char *inbox = nullptr, *outbox = nullptr;
    uint64_t *signals = nullptr, *credits = nullptr;
    // Credit of this rank's last publication, awaited before the outbox is staged again.
    uint64_t pending_credit = 0;
    int pending_receiver = 0;
    std::vector<Channel> channels;

    int row_rank(int col) const {
        return grid_row * grid_cols + col;
    }
    template <class T>
    void reduce_column(const T *source, T *destination, size_t count, ncclRedOp_t op) {
        if (!count)
            return;
        if (column_size > 1) {
            ++collectives;
            if (op == ncclSum)
                column_team->run<T, CollectiveOp::Sum>(source, destination, count, 0, stream);
            else
                column_team->run<T, CollectiveOp::Maximum>(source, destination, count, 0, stream);
        } else if (source != destination)
            CU(cudaMemcpyAsync(destination, source, count * sizeof(T), cudaMemcpyDeviceToDevice,
                               stream));
    }

  public:
    // This GPU sits at (row, col) of a grid of `cols` columns. `publication_bytes` bounds a
    // published block and `channel_bytes` gives the slot of every share channel, a multiple of 256
    // bytes.
    Transport(Context &c, size_t publication_bytes, size_t collective_bytes, int row, int col,
              int cols, const std::vector<size_t> &channel_bytes)
        : ctx(c), grid_row(row), grid_col(col), grid_cols(cols),
          publication_capacity(std::max(publication_bytes, size_t(64))), stream(own.s) {
        unordered_sent.resize(ctx.size);
        unordered_received.resize(ctx.size);
        // NVSHMEM holds only the independent publication inboxes. Ordered channels and their
        // reuse metadata belong to the low-latency NCCL path.
        size_t heap = publication_capacity * (size_t(ctx.size) + 1) + (size_t(64) << 20);
        if (heap > (size_t(1) << 30))
            setenv("NVSHMEM_SYMMETRIC_SIZE", std::to_string(heap).c_str(), 0);
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
        if (!inbox || !outbox || !signals || !credits)
            throw std::runtime_error("NVSHMEM allocation failed");
        for (size_t bytes : channel_bytes) {
            if (bytes % 8)
                throw std::runtime_error("a share slot must hold whole aligned scalar words");
            Channel ch;
            ch.bytes = bytes;
            CU(cudaMalloc(&ch.slots, 2 * bytes));
            channels.push_back(ch);
        }
        for (int j = 0; j < grid_cols; ++j)
            if (j != grid_col && !nvshmem_ptr(inbox, row_rank(j)))
                throw std::runtime_error("a GPU of the grid row is not directly addressable");
        nvshmem_barrier_all();

        ncclUniqueId id;
        if (ctx.rank == 0)
            NC(ncclGetUniqueId(&id));
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
        NC(ncclCommInitRank(&comm, ctx.size, id, ctx.rank));
        // The GPUs of a grid column, in grid-row order.
        NC(ncclCommSplit(comm, grid_col, grid_row, &column_comm, nullptr));
        NC(ncclCommCount(column_comm, &column_size));
        NC(ncclCommSplit(comm, grid_row, grid_col, &row_comm, nullptr));
        if (grid_cols > 1)
            row_team = std::make_unique<CollectiveTeam>(
                row_comm, grid_cols, *std::max_element(channel_bytes.begin(), channel_bytes.end()),
                true);
        if (column_size > 1)
            column_team = std::make_unique<CollectiveTeam>(column_comm, column_size,
                                                           (collective_bytes + 7) / 8 * 8, false);
        links = std::make_unique<OrderedLinks>(comm, ctx.rank, ctx.size, publication_capacity);
        ctx.barrier();
    }
    ~Transport() {
        if (std::uncaught_exceptions())
            return;
        cudaDeviceSynchronize();
        links.reset();
        column_team.reset();
        row_team.reset();
        ncclCommDestroy(row_comm);
        ncclCommDestroy(column_comm);
        ncclCommDestroy(comm);
        for (auto it = channels.rbegin(); it != channels.rend(); ++it) {
            cudaFree(it->slots);
        }
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
    // Slot s of a share channel on this GPU.
    void *slot(int channel, int s) const {
        return channels[channel].slots + size_t(s) * channels[channel].bytes;
    }
    // The first `bytes` of slot s, from the GPU of grid column `root` to the other GPUs of this
    // grid row. Every GPU of the row makes every share of the row, in the same order.
    void row_share(int channel, int s, int root, size_t bytes) {
        Channel &ch = channels[channel];
        if (bytes > ch.bytes)
            throw std::runtime_error("a share exceeds its slot");
        if (grid_cols == 1 || !bytes)
            return;
        if (bytes % sizeof(uint32_t))
            throw std::runtime_error("row share requires whole scalar words");
        ++collectives;
        auto *mine = static_cast<uint32_t *>(slot(channel, s));
        row_team->run<uint32_t, CollectiveOp::Broadcast>(mine, mine, bytes / sizeof(uint32_t), root,
                                                         stream);
    }
    // Matching ordered exchange. Only the sender and receiver launch a kernel.
    void publish(int sender, int receiver, const void *source, void *destination, size_t bytes) {
        if (ctx.rank == sender || ctx.rank == receiver)
            ++ordered_copies;
        links->exchange(sender, receiver, source, destination, bytes, stream);
    }
    // Copies `bytes` from `source` on the sender to `destination` on the receiver. The sender waits
    // for the credit of its previous publication before it stages the next one, and every sender
    // owns its region of the receiver's inbox. Every rank makes every call, in the same order.
    void publish_unordered(int sender, int receiver, const void *source, void *destination,
                           size_t bytes) {
        if (ctx.rank != sender && ctx.rank != receiver)
            return;
        ++unordered_copies;
        if (bytes > publication_capacity)
            throw std::runtime_error("publication exceeds the transport capacity");
        if (sender == receiver) {
            CU(cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDeviceToDevice, stream));
            return;
        }
        const uint64_t generation =
            ctx.rank == sender ? ++unordered_sent[receiver] : ++unordered_received[sender];
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
    // destination = sum (or maximum) of `source` over the GPUs of this grid column.
    template <class T> void column_allreduce(const T *source, T *destination, size_t count) {
        reduce_column(source, destination, count, ncclSum);
    }
    template <class T> void column_max(const T *source, T *destination, size_t count) {
        reduce_column(source, destination, count, ncclMax);
    }
    // The `count` elements of every GPU of this grid column, in grid-row order; `source` may be
    // this GPU's place in `destination`.
    template <class T> void column_allgather(const T *source, T *destination, size_t count) {
        if (!count)
            return;
        if (column_size > 1) {
            ++collectives;
            column_team->run<T, CollectiveOp::Gather>(source, destination, count, 0, stream);
        } else if (source != destination)
            CU(cudaMemcpyAsync(destination, source, count * sizeof(T), cudaMemcpyDeviceToDevice,
                               stream));
    }
    json report() const {
        return {{"collectives_backend", "low-latency-nccl-LLBuffer"},
                {"ordered_backend", "low-latency-nccl-LLBuffer"},
                {"unordered_backend", "NVSHMEM"},
                {"paper", "https://arxiv.org/abs/2607.16100"},
                {"nccl_commit", "5357eff325eddf978137de7140195a5568fa8a11"},
                {"collectives", collectives},
                {"ordered_exchanges", ordered_copies},
                {"unordered_publications", unordered_copies}};
    }
};
#else
// A build without MPI runs on one GPU and never creates a transport.
class Transport {
  public:
    Transport(Context &, size_t, size_t, int, int, int, const std::vector<size_t> &) {}
    void bind(cudaStream_t) {}
    cudaStream_t bound() const {
        return nullptr;
    }
    void *slot(int, int) const {
        return nullptr;
    }
    void row_share(int, int, int, size_t) {}
    void publish(int, int, const void *, void *, size_t) {}
    void publish_unordered(int, int, const void *, void *, size_t) {}
    template <class T> void column_allreduce(const T *, T *, size_t) {}
    template <class T> void column_max(const T *, T *, size_t) {}
    template <class T> void column_allgather(const T *, T *, size_t) {}
    json report() const {
        return json::object();
    }
};
#endif
} // namespace qr_omega
