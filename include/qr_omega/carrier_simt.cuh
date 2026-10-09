#pragma once
// IEEE FP32 carrier of W = V^T X on CUDA cores, used for the Gram products of the T factors. The
// eight warps of a slice split the columns of the 128 x 128 output tile and the two slices of a
// block split every 16-deep k-tile 8 / 8 (the residue tile last): the block is the carrier
// (1, 8, 2). Groups of blocks of the GPU split K again into 16-aligned balanced ranges, within the
// 2.5D bound of the output tiles, and their partials are summed in HBM in a fixed order.
#include "carrier_cutlass.cuh"
namespace qr_omega {
namespace simt {
constexpr int BM = 128, BN = 128, BK = 16, WK = 8, SK = 2, SLICE = 256, THREADS = SLICE * SK;
constexpr int LDS = BM + 4; // SMEM row stride of a k-row (floats): 16-byte aligned rows
// Each slice runs its own pipeline over its 8-deep k-blocks (block j of K_z goes to slice j mod 2)
// with its own stages and named barrier; the slices meet only at the combine.
constexpr int TILE_FLOATS = WK * LDS; // one operand tile (8 k-rows) of one stage of one slice
constexpr int SLICE_FLOATS = 2 * 2 * TILE_FLOATS; // 2 stages x (A, B) per slice
constexpr int PARK_LD = BM + 4;                   // parked partial: column-major [n][m]
constexpr size_t PARK_FLOATS = size_t(BN) * PARK_LD;
constexpr size_t SMEM_MAIN =
    size_t(SK) * SLICE_FLOATS * sizeof(float); // per slice: 2 stages x (A, B)
constexpr size_t SMEM_PARK = size_t(SK) * PARK_FLOATS * sizeof(float);
constexpr size_t SMEM = SMEM_MAIN > SMEM_PARK ? SMEM_MAIN : SMEM_PARK;
// Loader of the tiles of a K-contiguous operand (element (i, k) at p[k + i ld]) for one slice:
// I in [i0, i0 + 128) x K in [k0, k0 + 8), clipped to [0, I) x [k0, ke), written to shared memory
// as s[k LDS + i]. Each of the 256 threads of the slice moves one float4 per tile.
struct Loader {
    float4 r;
    const float *ptr;
    int i_ok;
    int kbase; // this thread's element at the slice's first block
    __device__ __forceinline__ void init(const float *__restrict__ p, long long ld, int I, int i0,
                                         int k0, int tid) {
        const int i = tid >> 1, kq = (tid & 1) * 4, gi = i0 + i;
        i_ok = gi < I;
        kbase = k0 + kq;
        ptr = p + (long long)(i_ok ? gi : 0) * ld + kbase;
    }
    // Block t of this slice: k offset 16 t from kbase.
    __device__ __forceinline__ void load(int t, int ke) {
        const float *q = ptr + t * 2 * WK;
        const int gk = kbase + t * 2 * WK;
        if (i_ok && gk + 3 < ke)
            r = __ldg(reinterpret_cast<const float4 *>(q));
        else {
            float v[4];
#pragma unroll
            for (int e = 0; e < 4; ++e)
                v[e] = (i_ok && gk + e < ke) ? __ldg(q + e) : 0.f;
            r = make_float4(v[0], v[1], v[2], v[3]);
        }
    }
    __device__ __forceinline__ void store(float *s, int tid) const {
        const int i = tid >> 1, kq = (tid & 1) * 4;
        s[(kq + 0) * LDS + i] = r.x;
        s[(kq + 1) * LDS + i] = r.y;
        s[(kq + 2) * LDS + i] = r.z;
        s[(kq + 3) * LDS + i] = r.w;
    }
};
// O(M x N) = A^T B over K, with A = V and B = X both K-contiguous and O column-major; CG > 1 groups
// write partial slices to P, sp apart.
struct Args {
    const float *A;
    long long lda;
    const float *B;
    long long ldb;
    float *O;
    long long ldo;
    int M, N, K;
    int CG;
    float *P;
    long long sp;
};
__global__ void __launch_bounds__(THREADS, 1) simt_carrier_kernel(const __grid_constant__ Args a) {
    extern __shared__ __align__(128) float sm[];
    const int lt = threadIdx.x, sl = threadIdx.y, tid = lt + SLICE * sl;
    const int gz = blockIdx.z, peers = a.CG, pid = gz;
    const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
    const float *Ap = a.A, *Bp = a.B;
    // K_z of group pid among P = CG: [16 floor(floor(K/16) pid/P), 16 floor(floor(K/16)
    // (pid+1)/P)), the last one ending at K; slice w owns the positions ((k - min K_z) mod 16) / 8
    // = w.
    const int K16 = a.K / BK;
    const int kb = BK * int((long long)K16 * pid / peers),
              ke = (pid == peers - 1) ? a.K : BK * int((long long)K16 * (pid + 1) / peers);
    Loader la, lb;
    float acc[8][8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j)
            acc[i][j] = 0.f;
    // Slice sl's SMEM: stage g at sm + sl*SLICE_FLOATS + g*2*TILE_FLOATS (A), + TILE_FLOATS (B).
    // Named barrier 1+sl.
    float *ss = sm + size_t(sl) * SLICE_FLOATS;
    const int nblk = ke > kb ? (ke - kb + WK - 1) / WK
                             : 0; // 8-deep k-blocks of K_z; slice sl takes j = sl, sl+2, ...
    const int mine = nblk > sl ? (nblk - sl + 1) / 2 : 0;
    auto bar = [&] { asm volatile("bar.sync %0, %1;" ::"r"(1 + sl), "r"(SLICE) : "memory"); };
    la.init(Ap, a.lda, a.M, m0, kb + sl * WK, lt);
    lb.init(Bp, a.ldb, a.N, n0, kb + sl * WK, lt);
    if (mine > 0) {
        la.load(0, ke);
        lb.load(0, ke);
        la.store(ss, lt);
        lb.store(ss + TILE_FLOATS, lt);
    }
    bar();
    const int
        tm = lt & 15,
        tn = lt >>
             4; // 16 x 16 threads of the slice; rows tm*4+{0..3}, 64+..; cols tn*4+{0..3}, 64+..
    float4 fa[2][2], fb[2][2];
    auto frag = [&](int f, const float *as, int k) {
        const float *bs = as + TILE_FLOATS;
        fa[f][0] = *reinterpret_cast<const float4 *>(as + k * LDS + tm * 4);
        fa[f][1] = *reinterpret_cast<const float4 *>(as + k * LDS + 64 + tm * 4);
        fb[f][0] = *reinterpret_cast<const float4 *>(bs + k * LDS + tn * 4);
        fb[f][1] = *reinterpret_cast<const float4 *>(bs + k * LDS + 64 + tn * 4);
    };
    auto fma_step = [&](int f) {
        const float av[8] = {fa[f][0].x, fa[f][0].y, fa[f][0].z, fa[f][0].w,
                             fa[f][1].x, fa[f][1].y, fa[f][1].z, fa[f][1].w};
        const float bv[8] = {fb[f][0].x, fb[f][0].y, fb[f][0].z, fb[f][0].w,
                             fb[f][1].x, fb[f][1].y, fb[f][1].z, fb[f][1].w};
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int j = 0; j < 8; ++j)
                acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
    };
    if (mine > 0)
        frag(0, ss, 0);
    for (int t = 0; t < mine; ++t) {
        const int cur = t & 1;
        if (t + 1 < mine) {
            la.load(t + 1, ke);
            lb.load(t + 1, ke);
        }
        const float *as = ss + cur * 2 * TILE_FLOATS;
#pragma unroll
        for (int k = 0; k < WK; ++k) {
            if (k + 1 < WK)
                frag((k + 1) & 1, as, k + 1);
            fma_step(k & 1);
        }
        if (t + 1 < mine) {
            float *nas = ss + (cur ^ 1) * 2 * TILE_FLOATS;
            la.store(nas, lt);
            lb.store(nas + TILE_FLOATS, lt);
        }
        bar();
        if (t + 1 < mine)
            frag(0, ss + (cur ^ 1) * 2 * TILE_FLOATS, 0);
    }
    __syncthreads(); // both slices done: the park region overlaps both pipelines
    // park: slice sl writes its partial column-major into part[sl]
    {
        float *part = sm + size_t(sl) * PARK_FLOATS;
#pragma unroll
        for (int jj = 0; jj < 8; ++jj) {
            const int n = (jj < 4 ? tn * 4 + jj : 64 + tn * 4 + (jj - 4));
            *reinterpret_cast<float4 *>(part + size_t(n) * PARK_LD + tm * 4) =
                make_float4(acc[0][jj], acc[1][jj], acc[2][jj], acc[3][jj]);
            *reinterpret_cast<float4 *>(part + size_t(n) * PARK_LD + 64 + tm * 4) =
                make_float4(acc[4][jj], acc[5][jj], acc[6][jj], acc[7][jj]);
        }
    }
    __syncthreads();
    constexpr int SL = BM;
    const int r0 = 0;
    float *O = a.CG > 1 ? a.P + gz * a.sp : a.O;
    const bool full = (m0 + BM <= a.M) && (n0 + BN <= a.N);
    constexpr int U = 8;
    for (int e0 = tid; e0 < SL * BN; e0 += THREADS * U) {
        float sv[U];
        long long go[U];
        bool in[U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int e = e0 + u * THREADS;
            const int i = r0 + e % SL, j = e / SL;
            go[u] = (m0 + i) + (long long)(n0 + j) * a.ldo;
            in[u] = e < SL * BN && (full || ((m0 + i) < a.M && (n0 + j) < a.N));
            float s = 0.f;
#pragma unroll
            for (int w = 0; w < SK; ++w) // fixed order w = 0..SK-1
                s += (e < SL * BN) ? sm[size_t(w) * PARK_FLOATS + size_t(j) * PARK_LD + i] : 0.f;
            sv[u] = s;
        }
#pragma unroll
        for (int u = 0; u < U; ++u)
            if (in[u])
                O[go[u]] = sv[u]; // the owner commits once (or its GPU-group slice)
    }
}
// The eight warps of a slice hold two of the 16 column phases of the tile each.
inline Carrier carrier(int M, int N, int cg) {
    return Carrier{}
        .set(GpuLevel, ceildiv(M, BM), ceildiv(N, BN), cg)
        .set(BlockLevel, 1, SLICE / 32, SK);
}
inline void launch(const Args &a, cudaStream_t st) {
    if (!a.M || !a.N)
        return;
    if (a.CG > 1 && (!a.P || a.sp < a.ldo * a.N))
        throw std::runtime_error("no room for the partial slices of the GPU groups");
    if ((long long)a.CG * a.CG > (long long)ceildiv(a.M, BM) * ceildiv(a.N, BN))
        throw std::runtime_error("GPU groups beyond the 2.5D bound of the product");
    reserve_shared_memory(simt_carrier_kernel, SMEM);
    simt_carrier_kernel<<<dim3(ceildiv(a.M, BM), ceildiv(a.N, BN), a.CG), dim3(SLICE, SK), SMEM,
                          st>>>(a);
    CU(cudaGetLastError());
    if (a.CG > 1) {
        carrier_g_combine<float><<<std::min(1024, ceildiv(a.M * a.N, 256)), 256, 0, st>>>(
            a.P, a.sp, a.CG, a.O, int(a.ldo), a.N, a.M, 1, 0);
        CU(cudaGetLastError());
    }
}
inline bool aligned16(const void *p, long long ld) {
    return (reinterpret_cast<uintptr_t>(p) % 16 == 0) && (ld % 4 == 0);
}
} // namespace simt
// W(h x q, ldw) = V^T X with the two slices of every block and cg groups of blocks, whose partial
// slices go to `part`.
inline bool simt_w_admits(const float *v, int ldv, const float *x, int ldx, const float *w) {
    return simt::aligned16(v, ldv) && simt::aligned16(x, ldx) && w;
}
// Groups for two waves of the GPU, at least 256 rows each, at most `room`, within the 2.5D bound.
inline int simt_w_groups(int q, int h, int rows, int room) {
    const long long tiles = (long long)ceildiv(h, simt::BM) * ceildiv(q, simt::BN);
    long long cg = std::max(1LL, 2LL * device_properties().multiProcessorCount / tiles);
    cg = std::min<long long>({cg, 128, room, std::max(1, rows / 256)});
    return bounded_groups(tiles, int(std::max(1LL, cg)));
}
inline Carrier launch_simt_w(const float *v, int ldv, const float *x, int ldx, float *w, int ldw,
                             int rows, int h, int q, cudaStream_t st, int cg, float *part) {
    const simt::Args a{
        v, ldv, x, ldx, w, ldw, h, q, rows, std::max(1, cg), part, (long long)ldw * q};
    simt::launch(a, st);
    return simt::carrier(h, q, std::max(1, cg));
}
} // namespace qr_omega
