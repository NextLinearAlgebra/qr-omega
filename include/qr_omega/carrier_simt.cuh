#pragma once
// IEEE FP32 carrier of W = V^T X on CUDA cores, used for the Gram products of the T factors. The
// two slices of a block split every 16-deep k-tile 8 / 8 (the residue tile last), the blocks of a
// cluster and the groups of blocks of the GPU split K into 16-aligned balanced ranges, and block z
// sums its rows of the tile over the pieces (z', w) in the order z' = 0..CL-1, w = 0..SK-1 through
// distributed shared memory before it commits them; group partials are summed in HBM.
#include "carrier_cutlass.cuh"
namespace tqr {
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
template <int CL>
__global__ void __launch_bounds__(THREADS, 1) simt_carrier_kernel(const __grid_constant__ Args a) {
    extern __shared__ __align__(128) float sm[];
    const int lt = threadIdx.x, sl = threadIdx.y, tid = lt + SLICE * sl;
    int z = 0;
    if constexpr (CL > 1)
        z = int(cooperative_groups::this_cluster().block_rank());
    const int gz = blockIdx.z / CL, peers = CL * a.CG, pid = gz * CL + z;
    const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
    const float *Ap = a.A, *Bp = a.B;
    // K_z of peer pid among P = CL CG: [16 floor(floor(K/16) pid/P), 16 floor(floor(K/16)
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
    if constexpr (CL > 1) {
        asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    } else
        __syncthreads();
    constexpr int SL = BM / CL;
    const int r0 = z * SL;
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
            for (int r = 0; r < CL; ++r) { // fixed order z' = 0..CL-1, w = 0..SK-1
                const float *peer = sm;
                if constexpr (CL > 1)
                    peer = cooperative_groups::this_cluster().map_shared_rank(sm, r);
#pragma unroll
                for (int w = 0; w < SK; ++w)
                    s += (e < SL * BN) ? peer[size_t(w) * PARK_FLOATS + size_t(j) * PARK_LD + i]
                                       : 0.f;
            }
            sv[u] = s;
        }
#pragma unroll
        for (int u = 0; u < U; ++u)
            if (in[u])
                O[go[u]] = sv[u]; // the owner commits once (or its GPU-group slice)
    }
    if constexpr (CL > 1) {
        asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
        asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
    }
}
template <int CL> void launch(const Args &a, cudaStream_t st) {
    if (!a.M || !a.N)
        return;
    if (a.CG > 1 && (!a.P || a.sp < a.ldo * a.N))
        throw std::runtime_error("no room for the partial slices of the GPU groups");
    reserve_shared_memory(simt_carrier_kernel<CL>, SMEM);
    launch_clustered(simt_carrier_kernel<CL>, dim3(ceildiv(a.M, BM), ceildiv(a.N, BN), CL * a.CG),
                     dim3(SLICE, SK), SMEM, st, dim3(1, 1, CL), a);
    if (a.CG > 1) {
        carrier_g_combine<float><<<std::min(1024, ceildiv(a.M * a.N, 256)), 256, 0, st>>>(
            a.P, a.sp, a.CG, a.O, int(a.ldo), a.N, a.M);
        CU(cudaGetLastError());
    }
}
inline bool aligned16(const void *p, long long ld) {
    return (reinterpret_cast<uintptr_t>(p) % 16 == 0) && (ld % 4 == 0);
}
} // namespace simt
// W(h x q, ldw) = V^T X with c = SK x CL (SK = 2 slices, CL = c / 2 blocks of a cluster), split
// again over cg groups of blocks whose partial slices go to `part`.
inline bool simt_w_admits(int c, const float *v, int ldv, const float *x, int ldx, const float *w) {
    return (c == 2 || c == 4 || c == 8) && simt::aligned16(v, ldv) && simt::aligned16(x, ldx) && w;
}
inline int simt_w_gpu_split(int c, int q, int h, int rows) {
    const int cl = c / simt::SK;
    const long long tiles = (long long)ceildiv(h, simt::BM) * ceildiv(q, simt::BN) * cl;
    long long cg = std::max(1LL, 2LL * device_properties().multiProcessorCount / tiles);
    cg = std::min<long long>({cg, 128, std::max(1, rows / (256 * cl))});
    return int(std::max(1LL, cg));
}
inline void launch_simt_w(int c, const float *v, int ldv, const float *x, int ldx, float *w,
                          int ldw, int rows, int h, int q, cudaStream_t st, int cg, float *part) {
    const simt::Args a{
        v, ldv, x, ldx, w, ldw, h, q, rows, std::max(1, cg), part, (long long)ldw * q};
    if (c == 2)
        simt::launch<1>(a, st);
    else if (c == 4)
        simt::launch<2>(a, st);
    else
        simt::launch<4>(a, st);
}
} // namespace tqr
