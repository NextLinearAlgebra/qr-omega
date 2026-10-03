#pragma once
// The update of Algorithm 2 on one strip of columns: W = V^T X, Z = T^T W and X -= V Z. Every
// product runs on the carrier of its arithmetic and reports the contraction replication c it used.
#include "carrier_block.cuh"
#include "carrier_kami.cuh"
#include "carrier_simt.cuh"
#include "carrier_wgmma.cuh"
#include <cublas_v2.h>

namespace tqr {
inline void blas_check(cublasStatus_t s) {
    if (s != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuBLAS error " + std::to_string(s));
}
// C_i = alpha op(A_i) op(B_i) + beta C_i for `batches` operands a fixed stride apart.
template <class T>
void gemm_strided(cublasHandle_t handle, cublasOperation_t opa, cublasOperation_t opb, int m, int n,
                  int k, T alpha, const T *A, int lda, long long sa, const T *B, int ldb,
                  long long sb, T beta, T *C, int ldc, long long sc, int batches) {
    if constexpr (sizeof(T) == 4)
        blas_check(cublasSgemmStridedBatched(handle, opa, opb, m, n, k, &alpha, A, lda, sa, B, ldb,
                                             sb, &beta, C, ldc, sc, batches));
    else
        blas_check(cublasDgemmStridedBatched(handle, opa, opb, m, n, k, &alpha, A, lda, sa, B, ldb,
                                             sb, &beta, C, ldc, sc, batches));
}
// Device pointers of the operands of `peers` batched products: peer z uses a + z pa, b + z pb and
// c + z pc.
template <class T>
__global__ void tile_carrier_pointers(T **pp, T *a, long long pa, T *b, long long pb, T *c,
                                      long long pc, int peers) {
    const int z = blockIdx.x * blockDim.x + threadIdx.x;
    if (z >= peers)
        return;
    pp[z] = a + z * pa;
    pp[peers + z] = b + z * pb;
    pp[2 * peers + z] = c + z * pc;
}
// Sums the c partials of a product in place over the first one, in a fixed order.
template <class T> __global__ void tile_combine_partials(T *w, size_t n, int c) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += size_t(blockDim.x) * gridDim.x) {
        T s = w[i];
        for (int z = 1; z < c; ++z)
            s += w[i + size_t(z) * n];
        w[i] = s;
    }
}

// Products issued by a factorization and how many of them ran with c > 1.
struct Tally {
    uint64_t products = 0, carried = 0;
    void add(int c) {
        ++products;
        carried += c > 1;
    }
};

// Requested replications of W and Z; D always splits its contraction in two.
struct Replication {
    int w = 2, z = 2;
    static constexpr int d = 2;
};

// The CUTLASS carriers of the TF32 modes split a k-tile of 32 between two slices of 16, so a
// contraction shorter than a k-tile would leave a slice empty. Those products keep their c peers on
// the short block carrier, whose intervals can be as short as one index.
template <class T> bool short_carrier_path(int c, int h, bool d) {
    if constexpr (!std::is_same_v<T, float>)
        return false;
    else
        return fp32_math() != Fp32Math::IEEE && (c == 2 || c == 4) && h >= c &&
               !(d ? carrier_d_supported<T>(h) : carrier_z_supported<T>(c, h));
}
// FP64 W and Z, and the TF32 and 3xTF32 products that the wgmma kernels do not admit, run on the
// CUTLASS carriers of carrier_cutlass.cuh; IEEE FP32 W and Z run as cuBLAS peers.
template <class T> bool has_native_wz() {
    return std::is_same_v<T, double> || fp32_math() != Fp32Math::IEEE;
}
template <class T> bool native_z_admits(int c, int h) {
    return has_native_wz<T>() && c > 1 &&
           (carrier_z_supported<T>(c, h) || short_carrier_path<T>(c, h, false));
}
template <class T> bool native_d_admits(int h) {
    return carrier_d_supported<T>(h) || short_carrier_path<T>(Replication::d, h, true);
}
// The replication a product runs with: the requested one when the contraction admits the cut.
template <class T> int w_replication(int requested, int rows) {
    return rows >= requested && (!has_native_wz<T>() || carrier_w_supported(requested, rows))
               ? requested
               : 1;
}
template <class T> int z_replication(int requested, int h) {
    return h >= requested && (!has_native_wz<T>() || native_z_admits<T>(requested, h)) ? requested
                                                                                       : 1;
}
template <class T> int d_replication(int h) {
    return native_d_admits<T>(h) ? Replication::d : 1;
}

// Stream, cuBLAS handle and buffers of one strip in flight.
template <class T> struct Lane {
    cudaStream_t stream = nullptr;
    cublasHandle_t blas = nullptr;
    T *w = nullptr, *z = nullptr; // W followed by its partial slices, and Z
    size_t w_words = 0;
    T **pointers = nullptr; // operands of the batched cuBLAS peers
    T *vr = nullptr;        // FP32: the row-major or split copy of V read by the wgmma kernels
    // Far update under look-ahead: its wgmma products release their SMs tile by tile to the panel.
    bool far = false;
    // What `vr` holds when it is the row-major copy of an aggregated V.
    const T *packed = nullptr;
    int packed_rows = 0, packed_h = 0;
    uint64_t packed_version = 0;
    // Side streams of the concurrent cuBLAS peers of W, created on first use.
    static constexpr int max_peers = 7;
    std::vector<cudaStream_t> peer_stream;
    std::vector<cublasHandle_t> peer_blas;
    std::vector<cudaEvent_t> peer_done;
    cudaEvent_t fork = nullptr;
};

template <class T> class Updater {
    int b, aggregate, strip, max_rows, d_tiles;
    Replication c;
    Tally &tally;
    Buffer<T> w_, z_, vr_;
    Buffer<T *> pointers_;
    std::vector<Buffer<unsigned char>> workspace_;
    std::vector<Lane<T>> lanes_;

    static constexpr size_t blas_workspace_bytes = size_t(32) << 20;

    // Reflector columns one update can carry, the leading dimension of T, W and Z.
    int width() const {
        return b * aggregate;
    }
    size_t vr_words() const {
        const size_t packed = size_t(max_rows) * width();
        return fp32_math() == Fp32Math::X3 ? 3 * packed + 3 * size_t(width()) * strip : packed;
    }
    static cublasHandle_t make_blas(cudaStream_t stream) {
        cublasHandle_t h;
        blas_check(cublasCreate(&h));
        blas_check(cublasSetStream(h, stream));
        blas_check(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH));
        blas_check(cublasSetAtomicsMode(h, CUBLAS_ATOMICS_NOT_ALLOWED));
        return h;
    }
    void make_peers(Lane<T> &lane) {
        int priority = 0;
        CU(cudaStreamGetPriority(lane.stream, &priority));
        CU(cudaEventCreateWithFlags(&lane.fork, cudaEventDisableTiming));
        lane.peer_stream.resize(Lane<T>::max_peers);
        lane.peer_blas.resize(Lane<T>::max_peers);
        lane.peer_done.resize(Lane<T>::max_peers);
        for (int z = 0; z < Lane<T>::max_peers; ++z) {
            CU(cudaStreamCreateWithPriority(&lane.peer_stream[z], cudaStreamNonBlocking, priority));
            lane.peer_blas[z] = make_blas(lane.peer_stream[z]);
            blas_check(cublasSetWorkspace(lane.peer_blas[z], nullptr, 0));
            CU(cudaEventCreateWithFlags(&lane.peer_done[z], cudaEventDisableTiming));
        }
    }

    // W = V^T X with the rows cut into c ranges, each a cuBLAS GEMM on its own stream; the partials
    // are summed in HBM.
    void w_peers(Lane<T> &lane, int c, const T *v, int ldv, const T *x, int ldx, T *w, int ldw,
                 int rows, int h, int q) {
        if (lane.peer_stream.empty())
            make_peers(lane);
        const int part = rows / c;
        const long long slice = (long long)ldw * q;
        CU(cudaEventRecord(lane.fork, lane.stream));
        for (int z = 1; z < c; ++z)
            CU(cudaStreamWaitEvent(lane.peer_stream[z - 1], lane.fork, 0));
        for (int z = c - 1; z >= 0; --z) {
            const int k0 = z * part, kz = z == c - 1 ? rows - k0 : part;
            gemm_strided<T>(z ? lane.peer_blas[z - 1] : lane.blas, CUBLAS_OP_T, CUBLAS_OP_N, h, q,
                            kz, T(1), v + k0, ldv, 0, x + k0, ldx, 0, T(0), w + size_t(z) * slice,
                            ldw, slice, 1);
            if (z)
                CU(cudaEventRecord(lane.peer_done[z - 1], lane.peer_stream[z - 1]));
        }
        for (int z = 1; z < c; ++z)
            CU(cudaStreamWaitEvent(lane.stream, lane.peer_done[z - 1], 0));
        tile_combine_partials<T><<<264, 256, 0, lane.stream>>>(w, size_t(slice), c);
        CU(cudaGetLastError());
    }
    // The same cut as one batched cuBLAS call on the lane's stream.
    void w_batched(Lane<T> &lane, int c, const T *v, int ldv, const T *x, int ldx, T *w, int ldw,
                   int rows, int h, int q) {
        const int part = rows / c;
        const long long slice = (long long)ldw * q;
        gemm_strided<T>(lane.blas, CUBLAS_OP_T, CUBLAS_OP_N, h, q, part, T(1), v, ldv, part, x, ldx,
                        part, T(0), w, ldw, slice, c);
        if (part * c < rows)
            gemm_strided<T>(lane.blas, CUBLAS_OP_T, CUBLAS_OP_N, h, q, rows - part * c, T(1),
                            v + size_t(part) * c, ldv, 0, x + size_t(part) * c, ldx, 0, T(1), w,
                            ldw, slice, 1);
        tile_combine_partials<T><<<256, 256, 0, lane.stream>>>(w, size_t(slice), c);
    }
    // Z = op(T) W, or Z^T when `transposed`, with the reflectors cut into c ranges: one batched
    // cuBLAS call over the peers and a fixed-order sum.
    void z_batched(Lane<T> &lane, int c, const T *t, int ldt, const T *w, int ldw, T *z, int ldz,
                   int h, int q, bool transpose, bool transposed) {
        const int part = h / c, rest = h - part * c;
        const long long slice = (long long)ldw * q,
                        t_step = transpose ? part : (long long)part * ldt;
        T *const *pp = lane.pointers;
        const T alpha = 1, zero = 0;
        if (transposed) {
            const cublasOperation_t opt = transpose ? CUBLAS_OP_N : CUBLAS_OP_T;
            tile_carrier_pointers<T><<<1, 128, 0, lane.stream>>>(
                lane.pointers, const_cast<T *>(w), part, const_cast<T *>(t), t_step, z, slice, c);
            if constexpr (sizeof(T) == 4)
                blas_check(cublasSgemmBatched(lane.blas, CUBLAS_OP_T, opt, q, h, part, &alpha, pp,
                                              ldw, pp + c, ldt, &zero, pp + 2 * c, ldz, c));
            else
                blas_check(cublasDgemmBatched(lane.blas, CUBLAS_OP_T, opt, q, h, part, &alpha, pp,
                                              ldw, pp + c, ldt, &zero, pp + 2 * c, ldz, c));
            if (rest)
                gemm_strided<T>(lane.blas, CUBLAS_OP_T, opt, q, h, rest, T(1), w + size_t(part) * c,
                                ldw, 0, t + size_t(c) * t_step, ldt, 0, T(1), z, ldz, 0, 1);
        } else {
            const cublasOperation_t opt = transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
            tile_carrier_pointers<T><<<1, 128, 0, lane.stream>>>(
                lane.pointers, const_cast<T *>(t), t_step, const_cast<T *>(w), part, z, slice, c);
            if constexpr (sizeof(T) == 4)
                blas_check(cublasSgemmBatched(lane.blas, opt, CUBLAS_OP_N, h, q, part, &alpha, pp,
                                              ldt, pp + c, ldw, &zero, pp + 2 * c, ldz, c));
            else
                blas_check(cublasDgemmBatched(lane.blas, opt, CUBLAS_OP_N, h, q, part, &alpha, pp,
                                              ldt, pp + c, ldw, &zero, pp + 2 * c, ldz, c));
            if (rest)
                gemm_strided<T>(lane.blas, opt, CUBLAS_OP_N, h, q, rest, T(1),
                                t + size_t(c) * t_step, ldt, 0, w + size_t(part) * c, ldw, 0, T(1),
                                z, ldz, 0, 1);
        }
        tile_combine_partials<T><<<256, 256, 0, lane.stream>>>(z, size_t(slice), c);
    }

    // Whether D runs on wgmma, which reads V row-major from the lane and Z as it is.
    bool d_on_wgmma(const Lane<T> &lane, T *x, int ldx, int rows, int h, int q, int ld) const {
        if constexpr (std::is_same_v<T, float>) {
            if (!lane.vr || !native_d_admits<T>(h))
                return false;
            if (fp32_math() == Fp32Math::TF32)
                return h <= width() &&
                       gmma_d_admits(lane.vr, width(), lane.z, ld, x, ldx, rows, h, q);
            if (fp32_math() == Fp32Math::X3) {
                const int hs = gmma_x3_seg(h);
                return hs <= width() && q <= strip &&
                       gmma_d_admits(lane.vr, 3 * hs, lane.vr + 3 * size_t(max_rows) * width(),
                                     3 * hs, x, ldx, rows, 3 * hs, q);
            }
        }
        return false;
    }
    void pack_v(Lane<T> &lane, const T *v, int ldv, int rows, int h, int hs, int ldvr, bool x3,
                uint64_t version) {
        if (version && lane.packed == v && lane.packed_rows == rows && lane.packed_h == h &&
            lane.packed_version == version)
            return;
        if constexpr (std::is_same_v<T, float>)
            launch_pack_vr(v, ldv, rows, h, hs, lane.vr, ldvr, x3, lane.stream);
        lane.packed = v;
        lane.packed_rows = rows;
        lane.packed_h = h;
        lane.packed_version = version;
    }
    // W of a strip of the trailing matrix. Wide FP64 products on tall panels run as concurrent
    // cuBLAS peers and skinny ones on the native carrier, with the rows also cut across groups of
    // blocks so that they fill the GPU; TF32 and 3xTF32 run on wgmma when the shape admits it.
    int strip_w(Lane<T> &lane, const T *v, int ldv, const T *x, int ldx, int rows, int h, int q,
                int ld) {
        T *w = lane.w;
        cudaStream_t st = lane.stream;
        const long long slice = (long long)ld * q;
        const int tiles = lane.far ? 1 : 0;
        const int wc = w_replication<T>(c.w, rows);
        if (wc == 1)
            return stage_w(lane, v, ldv, x, ldx, w, ld, rows, h, q);
        if constexpr (std::is_same_v<T, double>) {
            if (q >= 1024 && rows >= 8192 && rows / wc >= 2048) {
                w_peers(lane, wc, v, ldv, x, ldx, w, ld, rows, h, q);
                return wc;
            }
            const size_t room = lane.w_words / std::max<size_t>(1, size_t(slice));
            const int groups =
                int(std::max<size_t>(1, std::min<size_t>(carrier_w_gpu_split<T>(wc, q, h, rows),
                                                         room > 1 ? room - 1 : 1)));
            launch_carrier_w<T>(wc, v, ldv, x, ldx, w, ld, rows, h, q, st, groups,
                                groups > 1 ? w + slice : nullptr);
            return wc * groups;
        } else {
            float *part = w + slice;
            if (fp32_math() == Fp32Math::TF32 &&
                gmma_w_admits(wc, v, ldv, x, ldx, w, ld, rows, h, q, part, slice)) {
                launch_gmma_w(wc, v, ldv, x, ldx, w, ld, rows, h, q, st, part, slice, tiles);
                return wc;
            }
            if (fp32_math() == Fp32Math::X3 &&
                gmma_w3rs_admits(wc, v, ldv, x, ldx, w, ld, rows, h, q, lane.vr)) {
                // The split V shares the lane's buffer with the row-major copy D packs afterwards.
                lane.packed_version = 0;
                launch_gmma_w3rs(wc, v, ldv, x, ldx, w, ld, rows, h, q, st, lane.vr, tiles);
                return wc;
            }
            return stage_w(lane, v, ldv, x, ldx, w, ld, rows, h, q);
        }
    }

  public:
    // `rows` bounds the rows of an update and `b aggregate` its reflectors; `streams` are the
    // lanes.
    Updater(int b_, int aggregate_, int strip_, int rows, Replication c_, int d_tiles_,
            const std::vector<cudaStream_t> &streams, int first_far, Tally &tally_)
        : b(b_), aggregate(aggregate_), strip(strip_), max_rows(rows), d_tiles(d_tiles_), c(c_),
          tally(tally_), workspace_(streams.size()), lanes_(streams.size()) {
        const size_t n = streams.size(), block = size_t(width()) * strip;
        const size_t peers = size_t(std::max(c.w, c.z));
        w_.alloc(n * c.w * block);
        z_.alloc(n * c.z * block);
        pointers_.alloc(n * 3 * peers);
        if (gmma_enabled<T>())
            vr_.alloc(n * vr_words());
        for (size_t i = 0; i < n; ++i) {
            Lane<T> &lane = lanes_[i];
            lane.stream = streams[i];
            lane.blas = make_blas(streams[i]);
            workspace_[i].alloc(blas_workspace_bytes);
            blas_check(cublasSetWorkspace(lane.blas, workspace_[i].p, workspace_[i].n));
            lane.w = w_.p + i * c.w * block;
            lane.w_words = c.w * block;
            lane.z = z_.p + i * c.z * block;
            lane.pointers = pointers_.p + i * 3 * peers;
            lane.vr = vr_.n ? vr_.p + i * vr_words() : nullptr;
            lane.far = first_far >= 0 && int(i) >= first_far;
        }
    }
    ~Updater() {
        for (auto &lane : lanes_) {
            cublasDestroy(lane.blas);
            for (auto h : lane.peer_blas)
                cublasDestroy(h);
            for (auto s : lane.peer_stream)
                cudaStreamDestroy(s);
            for (auto e : lane.peer_done)
                cudaEventDestroy(e);
            if (lane.fork)
                cudaEventDestroy(lane.fork);
        }
    }
    Lane<T> &lane(int i) {
        return lanes_[i];
    }
    // The three products with explicit operands, on the carriers that cut the contraction inside a
    // block or a cluster (or, in IEEE FP32, as batched cuBLAS peers). `ld` is the leading dimension
    // of T, W and Z. Each returns the replication it ran with.
    int stage_w(Lane<T> &lane, const T *v, int ldv, const T *x, int ldx, T *w, int ld, int rows,
                int h, int q) {
        const int wc = w_replication<T>(c.w, rows);
        if (wc == 1)
            launch_carried_W<T>(v, ldv, x, ldx, w, ld, rows, h, q, lane.stream);
        else if (has_native_wz<T>())
            launch_carrier_w<T>(wc, v, ldv, x, ldx, w, ld, rows, h, q, lane.stream);
        else
            w_batched(lane, wc, v, ldv, x, ldx, w, ld, rows, h, q);
        return wc;
    }
    // Z = op(T) W, written transposed (q x h) when `transposed`.
    int stage_z(Lane<T> &lane, const T *t, const T *w, T *z, int ld, int h, int q, bool transpose,
                bool transposed) {
        cudaStream_t st = lane.stream;
        const int ldz = transposed ? q : ld;
        const int zc = z_replication<T>(c.z, h);
        if (zc == 1)
            launch_carried_Z<T>(t, ld, w, ld, z, ldz, h, q, transpose, st, transposed);
        else if (!native_z_admits<T>(zc, h))
            z_batched(lane, zc, t, ld, w, ld, z, ldz, h, q, transpose, transposed);
        else if (!short_carrier_path<T>(zc, h, false))
            launch_carrier_z<T>(zc, t, ld, w, ld, z, ldz, h, q, transpose, transposed, st);
        else if (transposed)
            launch_short_carrier<T, false, true>(zc, t, ld, w, ld, z, ldz, h, h, q, transpose, st);
        else
            launch_short_carrier<T, false>(zc, t, ld, w, ld, z, ldz, h, h, q, transpose, st);
        return zc;
    }
    // X -= V Z. Z is read transposed whenever the contraction admits c = 2.
    int stage_d(Lane<T> &lane, const T *v, int ldv, const T *z, int ld, T *x, int ldx, int rows,
                int h, int q) {
        cudaStream_t st = lane.stream;
        const int dc = d_replication<T>(h);
        if (dc == 1) {
            launch_carried_D<T>(v, ldv, z, ld, x, ldx, rows, h, q, st);
            return 1;
        }
        if (short_carrier_path<T>(dc, h, true)) {
            launch_short_carrier<T, true>(dc, v, ldv, z, q, x, ldx, rows, h, q, false, st);
            return dc;
        }
        if constexpr (std::is_same_v<T, double>) {
            // Aggregated updates: the (2, 2, 2) block carrier with FP64 tensor-core MMA.
            if (h >= 384 && h % 32 == 0 && kami::d_admits(v, ldv, z, q, x, ldx, h)) {
                kami::launch_dnp16<kami::DN128x64K16>(v, ldv, z, q, x, ldx, rows, h, q, st);
                return dc;
            }
        } else if (fp32_math() == Fp32Math::IEEE) {
            // The two blocks of a cluster own the contraction halves.
            if (h >= 256 && kami::simt_dcl_admits<kami::SF2K16>(v, ldv, z, q, x, ldx, h)) {
                kami::launch_simt_dcl<kami::SF2K16>(v, ldv, z, q, x, ldx, rows, h, q, st);
                return dc;
            }
            if (kami::simt_dcl_admits<kami::SF3>(v, ldv, z, q, x, ldx, h)) {
                kami::launch_simt_dcl<kami::SF3>(v, ldv, z, q, x, ldx, rows, h, q, st);
                return dc;
            }
        }
        launch_carrier_d<T>(v, ldv, z, q, x, ldx, rows, h, q, st);
        return dc;
    }
    // X(rows x q) <- (I - V op(T) V^T) X for V (rows x h) and its triangular factor T, whose
    // leading dimension is max(b, h). `version`, when nonzero, identifies the contents of an
    // aggregated V so that its row-major copy is packed once per lane.
    void apply(Lane<T> &lane, T *x, int ldx, const T *v, int ldv, const T *t, int rows, int h,
               int q, bool transpose, uint64_t version = 0) {
        const int ld = std::max(b, h);
        cudaStream_t st = lane.stream;
        tally.add(strip_w(lane, v, ldv, x, ldx, rows, h, q, ld));
        const bool wgmma = d_on_wgmma(lane, x, ldx, rows, h, q, ld);
        const bool transposed = !wgmma && native_d_admits<T>(h);
        const int zc = z_replication<T>(c.z, h);
        bool z_done = false;
        if constexpr (std::is_same_v<T, float>) {
            if (wgmma) {
                if (transpose && zc == 2 && gmma_z_admits(t, ld, lane.w, ld, lane.z, ld, h, q)) {
                    launch_gmma_z(t, ld, lane.w, ld, lane.z, ld, h, q, st, lane.far ? 1 : 0);
                    tally.add(zc);
                    z_done = true;
                }
            }
        }
        if (!z_done)
            tally.add(stage_z(lane, t, lane.w, lane.z, ld, h, q, transpose, transposed));
        if (!wgmma) {
            tally.add(stage_d(lane, v, ldv, lane.z, ld, x, ldx, rows, h, q));
            return;
        }
        if constexpr (std::is_same_v<T, float>) {
            const int tiles = lane.far ? d_tiles : 0;
            if (fp32_math() == Fp32Math::X3) {
                const int hs = gmma_x3_seg(h);
                float *z3 = lane.vr + 3 * size_t(max_rows) * width();
                pack_v(lane, v, ldv, rows, h, hs, 3 * hs, true, version);
                pack_z3<<<pack_vr_grid((long long)q * hs), 256, 0, st>>>(lane.z, ld, h, hs, q, z3,
                                                                         3 * hs);
                CU(cudaGetLastError());
                launch_gmma_d(lane.vr, 3 * hs, z3, 3 * hs, x, ldx, rows, 3 * hs, q, st, tiles);
            } else {
                pack_v(lane, v, ldv, rows, h, h, width(), false, version);
                launch_gmma_d(lane.vr, width(), lane.z, ld, x, ldx, rows, h, q, st, tiles);
            }
            tally.add(Replication::d);
        }
    }
    // G(h1 x h2) = V1^T V2 over `rows` rows, the product behind every composed T. The rows are cut
    // inside a block or cluster and again across groups of blocks, whose slices go to `partials`.
    // `tf32_groups` selects the wgmma kernel with that many groups in TF32.
    void gram(cudaStream_t st, const T *v1, int ld1, const T *v2, int ld2, T *g, int ldg, int rows,
              int h1, int h2, T *partials, size_t partial_words, int tf32_groups = 0) {
        const int wc = w_replication<T>(c.w, rows);
        if (wc == 1)
            throw std::runtime_error("too few rows for the requested replication of W");
        const long long slice = (long long)ldg * h2;
        const int room = int(std::min<size_t>(128, partial_words / size_t(slice)));
        if constexpr (std::is_same_v<T, float>) {
            if (tf32_groups && gmma_g_admits(tf32_groups, v1, ld1, v2, ld2, g, ldg, rows, h1, h2,
                                             partials, partial_words)) {
                launch_gmma_g(tf32_groups, v1, ld1, v2, ld2, g, ldg, rows, h1, h2, partials, st);
                tally.add(2 * tf32_groups);
                return;
            }
            if (fp32_math() == Fp32Math::IEEE) {
                // Two warp slices in each of 1, 2 or 4 blocks of a cluster.
                const int sc = wc >= 8 ? 8 : wc >= 4 ? 4 : 2;
                if (!simt_w_admits(sc, v1, ld1, v2, ld2, g))
                    throw std::runtime_error("reflector buffers are not 16-byte aligned");
                const int groups = std::max(1, std::min(simt_w_gpu_split(sc, h2, h1, rows), room));
                launch_simt_w(sc, v1, ld1, v2, ld2, g, ldg, rows, h1, h2, st, groups, partials);
                tally.add(sc * groups);
                return;
            }
        }
        const int groups = std::max(1, std::min(carrier_w_gpu_split<T>(wc, h2, h1, rows), room));
        launch_carrier_w<T>(wc, v1, ld1, v2, ld2, g, ldg, rows, h1, h2, st, groups, partials);
        tally.add(wc * groups);
    }
};
} // namespace tqr
