// Exact GPU checks for the low-latency NCCL LLBuffer adapter and independent NVSHMEM publications.
// Enqueue hundreds of variable-size episodes without a host/device rendezvous between them.
#include "plan.hpp"
#include "transport.cuh"

using namespace qr_omega;

template <class T> __device__ T value(int epoch, int rank, int i) {
    return T(100 * epoch + 3 * rank + i % 13);
}
template <class T> __global__ void fill_transport(T *p, int count, int epoch, int rank) {
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < count; i += blockDim.x * gridDim.x)
        p[i] = value<T>(epoch, rank, i);
}
// kind: 0 copy, 1 column sum, 2 column max, 3 column all-gather.
template <class T>
__global__ void check_transport(const T *p, int count, int epoch, int source, int rows, int cols,
                                int column, int kind, int *errors) {
    const int total = kind == 3 ? count * rows : count;
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < total; i += blockDim.x * gridDim.x) {
        T expected = value<T>(epoch, source, i);
        if (kind == 1) {
            expected = 0;
            for (int r = 0; r < rows; ++r)
                expected += value<T>(epoch, r * cols + column, i);
        } else if (kind == 2)
            expected = value<T>(epoch, (rows - 1) * cols + column, i);
        else if (kind == 3)
            expected = value<T>(epoch, (i / count) * cols + column, i % count);
        if (p[i] != expected)
            atomicAdd(errors, 1);
    }
}

template <class T> void check(Context &ctx, Grid grid) {
    constexpr int capacity = 1024;
    const int row = grid.row(ctx.rank), col = grid.col(ctx.rank);
    Transport transport(ctx, capacity * sizeof(T), capacity * sizeof(T), row, col, grid.pc,
                        {capacity * sizeof(T)});
    Buffer<T> source(capacity), destination(size_t(capacity) * grid.pr);
    Buffer<int> errors(1);
    errors.zero(0);
    Stream first, second;
    Event finished;
    const int counts[] = {1024, 1, 7, 255, 513, 9, 1023};
    for (int epoch = 1; epoch <= 300; ++epoch) {
        const int count = counts[epoch % 7], root = epoch % grid.pc, sender = epoch % ctx.size,
                  receiver = (sender + 1) % ctx.size;
        cudaStream_t stream = epoch % 2 ? first.s : second.s;
        if (epoch > 1)
            finished.wait(stream);
        transport.bind(stream);
        auto verify = [&](const T *p, int kind, int from) {
            check_transport<<<4, 128, 0, stream>>>(p, count, epoch, from, grid.pr, grid.pc, col,
                                                   kind, errors.p);
        };
        T *slot = static_cast<T *>(transport.slot(0, epoch & 1));
        fill_transport<<<4, 128, 0, stream>>>(slot, count, epoch, ctx.rank);
        transport.row_share(0, epoch & 1, root, count * sizeof(T));
        verify(slot, 0, grid.rank(row, root));
        fill_transport<<<4, 128, 0, stream>>>(source.p, count, epoch, ctx.rank);
        transport.column_allreduce(source.p, destination.p, count);
        verify(destination.p, 1, 0);
        transport.column_max(source.p, destination.p, count);
        verify(destination.p, 2, 0);
        transport.column_allgather(source.p, destination.p, count);
        verify(destination.p, 3, 0);
        transport.publish(sender, receiver, source.p, destination.p, count * sizeof(T));
        if (ctx.rank == receiver)
            verify(destination.p, 0, sender);
        // Only the two peers call the unordered operation. No world-wide sequence is needed.
        if (ctx.rank == sender || ctx.rank == receiver) {
            transport.publish_unordered(sender, receiver, source.p, destination.p,
                                        count * sizeof(T));
            if (ctx.rank == receiver)
                verify(destination.p, 0, sender);
        }
        finished.record(stream);
    }
    first.sync();
    second.sync();
    if (ctx.max(errors.download()[0]))
        throw std::runtime_error("transport payload mismatch");
    if (ctx.rank == 0)
        std::cout << precision<T>() << " " << grid.pr << "x" << grid.pc
                  << " 300 ordered/unordered episodes passed " << transport.report() << '\n';
}

int main(int argc, char **argv) {
    try {
        Context ctx;
        const Grid grid = make_grid(ctx.size, argc > 1 ? std::stoi(argv[1]) : 0);
        check<float>(ctx, grid);
        check<double>(ctx, grid);
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "transport check: " << e.what() << '\n';
        Context::abort();
        return 1;
    }
}
