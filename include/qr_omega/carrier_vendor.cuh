#pragma once
// Products C = A^T B over the rows of A and B as peers of one cuBLASLt matmul. The rows are cut into
// c consecutive ranges, the batches of a strided-batched matmul, so that the c peers run at once,
// and their partial outputs are summed in a fixed order. The algorithm is one whose output tile is
// known and that splits the contraction neither across blocks (no split-K, no reduction scheme) nor
// across the warps of a block (cuBLAS kernels tile only the output among their warps): the product
// is carried by (row tiles, column tiles, c) at the GPU level, and c is lowered until c^2 <= tiles.
#include "carrier.hpp"
#include "kernels.cuh"
#include <cublasLt.h>
#include <map>
#include <tuple>

namespace qr_omega {
inline void lt_check(cublasStatus_t s, const char *where) {
    if (s != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(where) + ": cuBLASLt status " +
                                 std::to_string(int(s)));
}
#define LT(x) ::qr_omega::lt_check((x), #x)

// The output tiles of the algorithms the carrier accepts.
struct VendorTile {
    int id, m, n;
};
inline constexpr VendorTile vendor_tiles[] = {
    {CUBLASLT_MATMUL_TILE_32x32, 32, 32},     {CUBLASLT_MATMUL_TILE_32x64, 32, 64},
    {CUBLASLT_MATMUL_TILE_64x32, 64, 32},     {CUBLASLT_MATMUL_TILE_32x128, 32, 128},
    {CUBLASLT_MATMUL_TILE_64x64, 64, 64},     {CUBLASLT_MATMUL_TILE_128x32, 128, 32},
    {CUBLASLT_MATMUL_TILE_64x128, 64, 128},   {CUBLASLT_MATMUL_TILE_128x64, 128, 64},
    {CUBLASLT_MATMUL_TILE_64x256, 64, 256},   {CUBLASLT_MATMUL_TILE_128x128, 128, 128},
    {CUBLASLT_MATMUL_TILE_256x64, 256, 64},   {CUBLASLT_MATMUL_TILE_64x512, 64, 512},
    {CUBLASLT_MATMUL_TILE_128x256, 128, 256}, {CUBLASLT_MATMUL_TILE_256x128, 256, 128},
    {CUBLASLT_MATMUL_TILE_512x64, 512, 64}};

template <class T> class VendorProduct {
    static constexpr int candidates = 16;
    // A matmul of `batches` products m x n over k rows, read with the given leading dimensions and
    // alignment and written `slice` elements apart, accumulating into C when `accumulate`.
    using Key = std::tuple<int, int, int, int, int, int, int, long long, int, bool>;
    struct Plan {
        cublasLtMatmulDesc_t op = nullptr;
        cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
        cublasLtMatmulAlgo_t algo{};
        int tm = 0, tn = 0; // the output tile; 0 when no algorithm qualifies
    };
    cublasLtHandle_t handle = nullptr;
    std::map<Key, Plan> plans;

    static cudaDataType_t data_type() {
        return sizeof(T) == 8 ? CUDA_R_64F : CUDA_R_32F;
    }
    static cublasLtMatrixLayout_t layout(int rows, int cols, int ld, int batches,
                                         long long stride) {
        cublasLtMatrixLayout_t l;
        LT(cublasLtMatrixLayoutCreate(&l, data_type(), rows, cols, ld));
        if (batches > 1) {
            LT(cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batches,
                                                sizeof(batches)));
            LT(cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                &stride, sizeof(stride)));
        }
        return l;
    }
    // The first algorithm of the heuristic, in its order of estimated time, that qualifies.
    const Plan &plan(int m, int n, int k, int lda, int ldb, int ldc, int batches, long long slice,
                     int align, bool accumulate) {
        const Key key{m, n, k, lda, ldb, ldc, batches, slice, align, accumulate};
        auto [it, fresh] = plans.try_emplace(key);
        Plan &p = it->second;
        if (!fresh)
            return p;
        const cublasComputeType_t compute =
            sizeof(T) == 8 ? CUBLAS_COMPUTE_64F : CUBLAS_COMPUTE_32F;
        LT(cublasLtMatmulDescCreate(&p.op, compute, data_type()));
        const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
        LT(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
        LT(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
        p.a = layout(k, m, lda, batches, k);
        p.b = layout(k, n, ldb, batches, k);
        p.c = layout(m, n, ldc, batches, slice);
        cublasLtMatmulPreference_t pref;
        LT(cublasLtMatmulPreferenceCreate(&pref));
        const size_t workspace = 0;
        const uint32_t reduction = CUBLASLT_REDUCTION_SCHEME_NONE, bytes = uint32_t(align);
        LT(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                &workspace, sizeof(workspace)));
        LT(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
                                                &reduction, sizeof(reduction)));
        for (auto attr : {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
                          CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                          CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
                          CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES})
            LT(cublasLtMatmulPreferenceSetAttribute(pref, attr, &bytes, sizeof(bytes)));
        cublasLtMatmulHeuristicResult_t results[candidates];
        int found = 0;
        const cublasStatus_t s = cublasLtMatmulAlgoGetHeuristic(handle, p.op, p.a, p.b, p.c, p.c,
                                                                pref, candidates, results, &found);
        LT(cublasLtMatmulPreferenceDestroy(pref));
        for (int i = 0; s == CUBLAS_STATUS_SUCCESS && i < found && !p.tm; ++i) {
            const cublasLtMatmulAlgo_t &algo = results[i].algo;
            int tile = 0, split = 1;
            uint32_t scheme = CUBLASLT_REDUCTION_SCHEME_NONE;
            if (results[i].state != CUBLAS_STATUS_SUCCESS)
                continue;
            LT(cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile,
                                                    sizeof(tile), nullptr));
            LT(cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &split,
                                                    sizeof(split), nullptr));
            LT(cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                                                    &scheme, sizeof(scheme), nullptr));
            if (split > 1 || scheme != CUBLASLT_REDUCTION_SCHEME_NONE)
                continue;
            for (const VendorTile &t : vendor_tiles)
                if (t.id == tile) {
                    p.algo = algo;
                    p.tm = t.m;
                    p.tn = t.n;
                }
        }
        return p;
    }
    // Largest power of two, up to 256 bytes, that divides every address the matmul reads or writes.
    static int alignment(std::initializer_list<size_t> quantities) {
        size_t a = 256;
        for (size_t q : quantities)
            while (a > sizeof(T) && q % a)
                a /= 2;
        return int(a);
    }

  public:
    VendorProduct() {
        LT(cublasLtCreate(&handle));
    }
    VendorProduct(const VendorProduct &) = delete;
    ~VendorProduct() {
        for (auto &[key, p] : plans) {
            cublasLtMatrixLayoutDestroy(p.c);
            cublasLtMatrixLayoutDestroy(p.b);
            cublasLtMatrixLayoutDestroy(p.a);
            cublasLtMatmulDescDestroy(p.op);
        }
        cublasLtDestroy(handle);
    }
    // C (m x n, leading dimension ldc) = A^T B over the `rows` rows of A (rows x m) and B (rows x n),
    // as up to `peers` batches of consecutive rows. The partial outputs of several batches go to
    // `parts`, `slice` elements apart (`parts` may be C itself), and are summed into C. False,
    // before launching anything, when no algorithm qualifies.
    bool atb(cudaStream_t st, const T *a, int lda, const T *b, int ldb, T *c, int ldc, T *parts,
             long long slice, int rows, int m, int n, int peers, Carrier &carrier) {
        constexpr int row_quantum = 32;
        for (int z = std::min(peers, std::max(1, rows / row_quantum)); z >= 1; --z) {
            const int part = z > 1 ? rows / z / row_quantum * row_quantum : rows,
                      rest = rows - part * z;
            T *out = z > 1 ? parts : c;
            const int align = alignment({size_t(a), size_t(b), size_t(out), size_t(lda) * sizeof(T),
                                         size_t(ldb) * sizeof(T), size_t(ldc) * sizeof(T),
                                         size_t(part) * sizeof(T), size_t(slice) * sizeof(T)});
            const Plan &p = plan(m, n, part, lda, ldb, ldc, z, slice, align, false);
            if (!p.tm || (long long)z * z > (long long)ceildiv(m, p.tm) * ceildiv(n, p.tn))
                continue;
            const Plan *tail = nullptr;
            if (rest) {
                const int tail_align = alignment(
                    {size_t(a + size_t(part) * z), size_t(b + size_t(part) * z), size_t(out),
                     size_t(lda) * sizeof(T), size_t(ldb) * sizeof(T), size_t(ldc) * sizeof(T)});
                tail = &plan(m, n, rest, lda, ldb, ldc, 1, slice, tail_align, true);
                if (!tail->tm)
                    continue;
            }
            const T one = 1, zero = 0;
            LT(cublasLtMatmul(handle, p.op, &one, a, p.a, b, p.b, &zero, out, p.c, out, p.c,
                              &p.algo, nullptr, 0, st));
            if (tail)
                LT(cublasLtMatmul(handle, tail->op, &one, a + size_t(part) * z, tail->a,
                                  b + size_t(part) * z, tail->b, &one, out, tail->c, out, tail->c,
                                  &tail->algo, nullptr, 0, st));
            if (z > 1) {
                tile_combine_partials<T><<<264, 256, 0, st>>>(c, out, size_t(slice), z);
                CU(cudaGetLastError());
            }
            carrier = Carrier{}.set(GpuLevel, ceildiv(m, p.tm), ceildiv(n, p.tn), z);
            return true;
        }
        return false;
    }
};
} // namespace qr_omega
