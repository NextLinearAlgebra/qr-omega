#pragma once
// Block and cluster carriers of D = V Z written directly against the hardware, after KAMI (Wang et
// al., SC'25): tensor cores or FMAs compute, registers hold the output block, and shared memory is
// only the staging ring of the operands. Both cut the contraction in two halves: the two warp
// layers of a block (FP64) or the two blocks of a cluster (FP32) each own one half, and the owner
// adds the two partials and commits X -= D once.
#include "kernels.cuh"
namespace tqr {
namespace kami {
__device__ __forceinline__ uint32_t smem_u32(const void *p) {
    return uint32_t(__cvta_generic_to_shared(p));
}
// 16-byte async copy; src_bytes < 16 zero-fills the tail (bounds).
__device__ __forceinline__ void cp16(void *dst, const void *src, int src_bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src),
                 "r"(src_bytes)
                 : "memory");
}
__device__ __forceinline__ void cp_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}
template <int N> __device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}
__device__ __forceinline__ void bar_sync(int id, int n) {
    asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(n) : "memory");
}
// Plain C++ shared-memory accesses: the compiler may schedule fragment loads ahead of the MMAs (the
// barriers above carry memory clobbers, so no load crosses a stage hand-off).
__device__ __forceinline__ void lds128(double &x, double &y, const double *p) {
    const double2 v = *reinterpret_cast<const double2 *>(p);
    x = v.x;
    y = v.y;
}
__device__ __forceinline__ double lds64(const double *p) {
    return *p;
}
__device__ __forceinline__ void sts128(double *p, double x, double y) {
    *reinterpret_cast<double2 *>(p) = make_double2(x, y);
}
// One output tile per block, the X block staged into shared memory with cp.async at the start of
// the tile. PI x PJ warps per layer, S pipeline stages of BK reflectors.
template <int BM_, int BN_, int BK_, int PI_, int PJ_, int S_> struct NCfg {
    static constexpr int BM = BM_, BN = BN_, BK = BK_, PI = PI_, PJ = PJ_, C = 2, S = S_;
    static constexpr int WM = BM / PI, WN = BN / PJ, MT = WM / 16, NT = WN / 8;
    static constexpr int LayerThreads = PI * PJ * 32, Threads = LayerThreads * C;
    static constexpr int LDA = BM + 4, LDB = BN + 4, LDX = BM + 2;
    static constexpr int StageElems = BK * LDA + BK * LDB, LayerElems = S * StageElems,
                         XElems = BN * LDX;
    static constexpr size_t Smem = size_t(C * LayerElems + XElems) * sizeof(double);
    static_assert(WM % 16 == 0 && WN % 8 == 0 && BK % 8 == 0, "tile");
    static_assert(BN * LDX <= LayerElems, "a parked partial fits in its layer's dead stages");
};
__device__ __forceinline__ void dmma16(double (&d)[4], const double (&a)[8], const double (&b)[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f64.f64.f64.f64 {%0,%1,%2,%3}, {%4,%5,%6,%7,%8,%9,%10,%11}, {%12,%13,%14,%15}, {%0,%1,%2,%3};\n"
        : "+d"(d[0]), "+d"(d[1]), "+d"(d[2]), "+d"(d[3])
        : "d"(a[0]), "d"(a[1]), "d"(a[2]), "d"(a[3]), "d"(a[4]), "d"(a[5]), "d"(a[6]), "d"(a[7]),
          "d"(b[0]), "d"(b[1]), "d"(b[2]), "d"(b[3]));
}
template <class Cfg>
__global__ void __launch_bounds__(Cfg::Threads, 1)
    kami_dnp16_kernel(const double *__restrict__ V, int ldv, const double *__restrict__ Zt, int ldz,
                      double *X, int ldx, int rows, int h, int q, int raster) {
    constexpr int BM = Cfg::BM, BN = Cfg::BN, BK = Cfg::BK, S = Cfg::S, LDA = Cfg::LDA,
                  LDB = Cfg::LDB, LDX = Cfg::LDX;
    constexpr int MT = Cfg::MT, NT = Cfg::NT, LT = Cfg::LayerThreads;
    extern __shared__ __align__(128) double kn16_smem[];
    const int tid = threadIdx.x, layer = tid / LT, ltid = tid % LT, warp = ltid / 32,
              lane = tid % 32, g = lane >> 2, t = lane & 3;
    const int wx = warp / Cfg::PJ, wy = warp % Cfg::PJ;
    int tcol, trow;
    raster_tile(blockIdx.x + gridDim.x * blockIdx.y, gridDim.x, gridDim.y, raster, tcol, trow);
    const int n0 = tcol * BN, m0 = trow * BM;
    double *const Ls = kn16_smem + size_t(layer) * Cfg::LayerElems;
    double *const Xs = kn16_smem + size_t(Cfg::C) * Cfg::LayerElems;
    const int nk8 = (h + 7) / 8, per = (nk8 + Cfg::C - 1) / Cfg::C;
    const int kb = min(h, layer * per * 8), ke = min(h, (layer + 1) * per * 8);
    const int nkt = (ke - kb + BK - 1) / BK;
    constexpr int VCH = BM / 2, ZCH = BN / 2;
    static_assert(LT % VCH == 0 && LT % ZCH == 0 && (BK * VCH) % LT == 0 && (BK * ZCH) % LT == 0,
                  "copy plan");
    constexpr int VPT = BK * VCH / LT, VKS = LT / VCH, ZPT = BK * ZCH / LT, ZKS = LT / ZCH;
    const int vmm = (ltid % VCH) * 2, vkk = ltid / VCH, znn = (ltid % ZCH) * 2, zkk = ltid / ZCH;
    const int vnb = min(2, max(0, rows - (m0 + vmm))) * 8, znb = min(2, max(0, q - (n0 + znn))) * 8;
    const double *pV = V + (vnb ? size_t(m0 + vmm) : 0) + size_t(kb + vkk) * ldv;
    const double *pZ = Zt + (znb ? size_t(n0 + znn) : 0) + size_t(kb + zkk) * ldz;
    const size_t vstep = size_t(VKS) * ldv, zstep = size_t(ZKS) * ldz, vtile = size_t(BK) * ldv,
                 ztile = size_t(BK) * ldz;
    const int vso = vkk * LDA + vmm, zso = BK * LDA + zkk * LDB + znn;
    auto load_tile = [&](int kt, int stage) {
        double *St = Ls + stage * Cfg::StageElems;
        const int k0 = kb + kt * BK;
        const double *v = pV + size_t(kt) * vtile;
        const double *z = pZ + size_t(kt) * ztile;
        if (k0 + BK <= ke) {
#pragma unroll
            for (int i = 0; i < VPT; ++i)
                cp16(St + vso + i * VKS * LDA, v + i * vstep, vnb);
#pragma unroll
            for (int i = 0; i < ZPT; ++i)
                cp16(St + zso + i * ZKS * LDB, z + i * zstep, znb);
        } else {
#pragma unroll
            for (int i = 0; i < VPT; ++i) {
                const bool in = k0 + vkk + i * VKS < ke;
                cp16(St + vso + i * VKS * LDA, in ? v + i * vstep : V, in ? vnb : 0);
            }
#pragma unroll
            for (int i = 0; i < ZPT; ++i) {
                const bool in = k0 + zkk + i * ZKS < ke;
                cp16(St + zso + i * ZKS * LDB, in ? z + i * zstep : Zt, in ? znb : 0);
            }
        }
    };
#pragma unroll
    for (int s = 0; s < S - 1; ++s) {
        if (s < nkt)
            load_tile(s, s);
        cp_commit();
    }
    {
        constexpr int XPT = BN * VCH / Cfg::Threads, XKS = Cfg::Threads / VCH;
        static_assert((BN * VCH) % Cfg::Threads == 0 && Cfg::Threads % VCH == 0, "x plan");
        const int xmm = (tid % VCH) * 2, xnn = tid / VCH,
                  xnb = min(2, max(0, rows - (m0 + xmm))) * 8;
        const double *px = X + (xnb ? size_t(m0 + xmm) : 0) + size_t(n0 + xnn) * ldx;
#pragma unroll 4
        for (int i = 0; i < XPT; ++i) {
            const bool in = n0 + xnn + i * XKS < q;
            cp16(Xs + (xnn + i * XKS) * LDX + xmm, in ? px + size_t(i * XKS) * ldx : X,
                 in ? xnb : 0);
        }
    }
    cp_commit();
    double acc[MT][NT][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0;
        }
    const int am = wx * Cfg::WM, bn = wy * Cfg::WN;
    for (int kt = 0; kt < nkt; ++kt) {
        cp_wait<S - 2>();
        bar_sync(1 + layer, LT);
        {
            const int nt = kt + S - 1;
            if (nt < nkt)
                load_tile(nt, nt % S);
            cp_commit();
        }
        const double *As = Ls + (kt % S) * Cfg::StageElems;
        const double *Bs = As + BK * LDA;
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            double a[MT][8];
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                const double *p = As + (kk + t) * LDA + am + i * 16 + 2 * g;
#pragma unroll
                for (int u = 0; u < 4; ++u)
                    lds128(a[i][2 * u], a[i][2 * u + 1], p + 4 * u * LDA);
            }
            double bv[NT][4];
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                const double *p = Bs + (kk + t) * LDB + bn + j * 8 + g;
#pragma unroll
                for (int u = 0; u < 4; ++u)
                    bv[j][u] = lds64(p + 4 * u * LDB);
            }
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int j = 0; j < NT; ++j)
                    dmma16(acc[i][j], a[i], bv[j]);
        }
    }
    cp_wait<0>();
    bar_sync(1 + layer, LT);
    if (layer > 0) {
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                double *p = Ls + (bn + j * 8 + 2 * t) * LDX + am + i * 16 + 2 * g;
                sts128(p, acc[i][j][0], acc[i][j][2]);
                sts128(p + LDX, acc[i][j][1], acc[i][j][3]);
            }
    }
    __syncthreads();
    if (layer == 0) {
        const bool full = (m0 + BM <= rows) && (n0 + BN <= q);
#pragma unroll
        for (int i = 0; i < MT; ++i) {
            double2 xv[NT][2];
#pragma unroll
            for (int j = 0; j < NT; ++j)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int ml = am + i * 16 + 2 * g, nl = bn + j * 8 + 2 * t + e;
                    lds128(xv[j][e].x, xv[j][e].y, Xs + nl * LDX + ml);
                }
#pragma unroll
            for (int j = 0; j < NT; ++j)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int ml = am + i * 16 + 2 * g, nl = bn + j * 8 + 2 * t + e;
                    double d0 = acc[i][j][e], d1 = acc[i][j][2 + e];
                    {
                        double p0, p1;
                        lds128(p0, p1, kn16_smem + size_t(Cfg::LayerElems) + nl * LDX + ml);
                        d0 += p0;
                        d1 += p1;
                    }
                    const int r = m0 + ml, col = n0 + nl;
                    double *dst = X + size_t(r) + size_t(col) * ldx;
                    const double x0 = xv[j][e].x - d0, x1 = xv[j][e].y - d1;
                    if (full)
                        *reinterpret_cast<double2 *>(dst) = make_double2(x0, x1);
                    else if (col < q) {
                        if (r < rows)
                            dst[0] = x0;
                        if (r + 1 < rows)
                            dst[1] = x1;
                    }
                }
        }
    }
}

// Column tiles per raster group of the D carriers.
inline constexpr int raster_group = 16;
using DN128x64K16 = NCfg<128, 64, 16, 2, 2, 3>;
template <class Cfg>
void launch_dnp16(const double *v, int ldv, const double *zt, int ldz, double *x, int ldx, int rows,
                  int h, int q, cudaStream_t st) {
    static_assert(Cfg::C == 2 && Cfg::BK % 16 == 0, "two contraction peers with k16 tiles");
    if (!rows || !h || !q)
        return;
    if (h % 32)
        throw std::runtime_error("the FP64 D carrier needs two halves of whole k-tiles");
    reserve_shared_memory(kami_dnp16_kernel<Cfg>, Cfg::Smem, true);
    const int nx = ceildiv(q, Cfg::BN), ny = ceildiv(rows, Cfg::BM);
    kami_dnp16_kernel<Cfg><<<dim3(nx, ny), Cfg::Threads, Cfg::Smem, st>>>(
        v, ldv, zt, ldz, x, ldx, rows, h, q, std::min(raster_group, nx));
    CU(cudaGetLastError());
}
// The FP64 carrier needs 16-byte copies of every column start and a nonempty half per layer.
inline bool d_admits(const double *v, int ldv, const double *zt, int ldz, const double *x, int ldx,
                     int h) {
    const uintptr_t bases = uintptr_t(v) | uintptr_t(zt) | uintptr_t(x);
    return h > 8 && !(bases & 15) && !((ldv | ldz | ldx) & 1);
}

__device__ __forceinline__ uint32_t cluster_rank() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(r));
    return r;
}
__device__ __forceinline__ uint32_t map_peer(uint32_t saddr, uint32_t rank) {
    uint32_t r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(r) : "r"(saddr), "r"(rank));
    return r;
}
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar), "r"(bytes)
                 : "memory");
}
__device__ __forceinline__ void mbar_wait_cluster(uint32_t bar, uint32_t parity) {
    asm volatile(
        "{\n .reg .pred P;\n W: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P, [%0], %1;\n @!P bra W;\n}\n" ::
            "r"(bar),
        "r"(parity)
        : "memory");
}
__device__ __forceinline__ void cp16a(uint32_t dst, const void *src, int src_bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src),
                 "r"(src_bytes)
                 : "memory");
}
// Fragments are double-buffered in registers (asm-ordered loads of step k+1 between the FFMAs of
// step k); an S-stage cp.async ring of BK reflectors; the owned half of X is staged by cp.async.
template <int S_, int BK_ = 8> struct FCfg {
    static constexpr int BM = 128, BN = 128, BK = BK_, S = S_, C = 2, HN = 64, Threads = 128;
    static_assert(BK % 4 == 0 && BK >= 8, "copy plan: 4 k-rows per pass");
    static constexpr int AElems = BK * BM, BElems = BK * BN, StageElems = AElems + BElems,
                         RingElems = S * StageElems;
    static constexpr int XLD = BM + 4; // X staging [col][row], padded against bank conflicts
    static constexpr int XElems = HN * XLD;
    static constexpr int RecvElems = 128 * 128 / 2; // the peer's half: 64 lanes-slots x 128
    static constexpr uint32_t RecvBytes = uint32_t(RecvElems) * 4u;
    static constexpr size_t Smem = size_t(RingElems + XElems + RecvElems) * 4 + 16;
};
__device__ __forceinline__ void ldsf4(float (&v)[4], uint32_t a) {
    asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];\n"
                 : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3])
                 : "r"(a));
}
__device__ __forceinline__ void st_async_f4(uint32_t raddr, const float (&v)[4], uint32_t rbar) {
    asm volatile(
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1,%2,%3,%4}, [%5];\n" ::
            "r"(raddr),
        "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]), "r"(rbar)
        : "memory");
}
template <class Cfg>
__global__ void __launch_bounds__(128, 2)
    simt_dcl_kernel(const float *__restrict__ V, int ldv, const float *__restrict__ Zt, int ldz,
                    float *X, int ldx, int rows, int h, int q, int raster) {
    constexpr int BM = Cfg::BM, BN = Cfg::BN, BK = Cfg::BK, S = Cfg::S, HN = Cfg::HN,
                  XLD = Cfg::XLD;
    constexpr uint32_t StageB = uint32_t(Cfg::StageElems) * 4u, ABytes = uint32_t(Cfg::AElems) * 4u;
    extern __shared__ __align__(128) float sf_smem[];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, wm = warp >> 1, wn = warp & 1,
              r = lane >> 3, sc = lane & 7;
    const int z = int(cluster_rank());
    int tcol, trow;
    raster_tile((blockIdx.x >> 1) + (gridDim.x >> 1) * blockIdx.y, gridDim.x >> 1, gridDim.y,
                raster, tcol, trow);
    const int n0 = tcol * BN, m0 = trow * BM;
    const uint32_t sbase = smem_u32(sf_smem);
    const uint32_t xs = sbase + uint32_t(Cfg::RingElems) * 4u;
    const uint32_t rbuf = xs + uint32_t(Cfg::XElems) * 4u, mbar = rbuf + Cfg::RecvBytes;
    if (tid == 0) {
        mbar_init(mbar, 1);
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
        mbar_expect_tx(mbar, Cfg::RecvBytes);
    }
    asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory");
    const int nk = (h + 2 * BK - 1) / (2 * BK);
    const int kb = min(h, z * nk * BK), ke = min(h, (z + 1) * nk * BK);
    const int nkt = (ke - kb) / BK; // admission: h % 16 == 0
    const int cu = tid & 31, ck = tid >> 5;
    const int avb = min(4, max(0, rows - (m0 + 4 * cu))) * 4,
              bvb = min(4, max(0, q - (n0 + 4 * cu))) * 4;
    const char *pa =
        reinterpret_cast<const char *>(V + (avb ? size_t(m0 + 4 * cu) : 0) + size_t(kb + ck) * ldv);
    const char *pb = reinterpret_cast<const char *>(Zt + (bvb ? size_t(n0 + 4 * cu) : 0) +
                                                    size_t(kb + ck) * ldz);
    const uint32_t astep = 4u * uint32_t(ldv) * 4u, bstep = 4u * uint32_t(ldz) * 4u,
                   atile = uint32_t(BK) * uint32_t(ldv) * 4u,
                   btile = uint32_t(BK) * uint32_t(ldz) * 4u;
    const uint32_t aso = uint32_t(ck * BM + 4 * cu) * 4u,
                   bso = ABytes + uint32_t(ck * BN + 4 * cu) * 4u;
    auto load_tile = [&](uint32_t st) {
#pragma unroll
        for (int i = 0; i < BK / 4; ++i) {
            cp16a(st + aso + uint32_t(i * 4 * BM) * 4u, pa + size_t(i) * astep, avb);
            cp16a(st + bso + uint32_t(i * 4 * BN) * 4u, pb + size_t(i) * bstep, bvb);
        }
        pa += atile;
        pb += btile;
    };
#pragma unroll
    for (int s = 0; s < S - 1; ++s) {
        if (s < nkt)
            load_tile(sbase + uint32_t(s) * StageB);
        cp_commit();
    }
    cp_wait<S - 2>();
    __syncthreads();
    if (S - 1 < nkt)
        load_tile(sbase + uint32_t(S - 1) * StageB);
    { // the 64 columns of X this block commits -> smem [col][row]
        const int nh = n0 + z * HN;
        const int xr = 4 * (tid & 31), xvb = min(4, max(0, rows - (m0 + xr))) * 4;
        const float *px = X + (xvb ? size_t(m0 + xr) : 0) + size_t(nh) * ldx;
        for (int c = tid >> 5; c < HN; c += 4) {
            const bool in = nh + c < q;
            cp16a(xs + uint32_t(c * XLD + xr) * 4u, in ? px + size_t(c) * ldx : X, in ? xvb : 0);
        }
    }
    cp_commit();
    float acc[16][8];
#pragma unroll
    for (int i = 0; i < 16; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j)
            acc[i][j] = 0.f;
    // fragment addresses inside a stage (k-row k): A at k*BM + 64wm + 4r + 16c ; B at k*BN + 64wn +
    // 4sc + 32d
    const uint32_t aoff = uint32_t(64 * wm + 4 * r) * 4u,
                   boff = ABytes + uint32_t(64 * wn + 4 * sc) * 4u;
    float fa[2][16], fb[2][8];
    auto ldfrag = [&](float(&a)[16], float(&b)[8], uint32_t st, int k) {
        const uint32_t ak = st + aoff + uint32_t(k * BM) * 4u,
                       bk = st + boff + uint32_t(k * BN) * 4u;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            float v[4];
            ldsf4(v, ak + uint32_t(16 * c) * 4u);
            a[4 * c] = v[0];
            a[4 * c + 1] = v[1];
            a[4 * c + 2] = v[2];
            a[4 * c + 3] = v[3];
        }
#pragma unroll
        for (int d = 0; d < 2; ++d) {
            float v[4];
            ldsf4(v, bk + uint32_t(32 * d) * 4u);
            b[4 * d] = v[0];
            b[4 * d + 1] = v[1];
            b[4 * d + 2] = v[2];
            b[4 * d + 3] = v[3];
        }
    };
    // Column by column: in this order ptxas spreads the loads of the next fragments through the
    // FMAs instead of issuing them in bursts.
    auto fma_step = [&](const float(&a)[16], const float(&b)[8]) {
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int i = 0; i < 16; ++i)
                acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
    };
    uint32_t cur = sbase, nxt = sbase + StageB;
    const uint32_t ring_end = sbase + uint32_t(S) * StageB;
    if (nkt > 0)
        ldfrag(fa[0], fb[0], cur, 0);
    for (int kt = 0; kt < nkt; ++kt) {
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            if (k < BK - 1)
                ldfrag(fa[(k + 1) & 1], fb[(k + 1) & 1], cur, k + 1);
            else if (kt + 1 < nkt) {
                cp_wait<S - 2>();
                __syncthreads(); // tile kt+1 landed; stage cur fully read
                if (kt + S < nkt)
                    load_tile(cur);
                cp_commit();
                ldfrag(fa[(k + 1) & 1], fb[(k + 1) & 1], nxt, 0);
            }
            fma_step(fa[k & 1], fb[k & 1]);
        }
        cur = nxt;
        nxt = nxt + StageB == ring_end ? sbase : nxt + StageB;
    }
    cp_wait<0>();
    __syncthreads(); // X staged
    const bool full = (m0 + BM <= rows) && (n0 + BN <= q);
    // Thread element (i, j): row 64 wm + 4 r + 16 (i / 4) + i % 4, column 64 wn + 4 sc + 32 (j / 4) +
    // j % 4. The block that does not own column half wn sends its partials to the peer.
    asm volatile("barrier.cluster.wait.aligned;\n" ::: "memory");
    // recv layout: slot (c, j) of lane-slot L = 32 wm + lane: float4 at ((c*8 + j)*64 + L)*4
    // floats (conflict-free)
    const int L = 32 * wm + lane;
    if (wn != z) {
        const uint32_t rb = map_peer(rbuf, uint32_t(z ^ 1)), rbar = map_peer(mbar, uint32_t(z ^ 1));
#pragma unroll
        for (int c = 0; c < 4; ++c)
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                float v[4] = {acc[4 * c][j], acc[4 * c + 1][j], acc[4 * c + 2][j],
                              acc[4 * c + 3][j]};
                st_async_f4(rb + uint32_t(((c * 8 + j) * 64 + L) * 4) * 4u, v, rbar);
            }
    } else {
        mbar_wait_cluster(mbar, 0);
#pragma unroll
        for (int j = 0; j < 8; ++j) { // stream: peer float4 + X float4 per (c, j)
            const int cl = 64 * wn + 4 * sc + 32 * (j >> 2) + (j & 3) - z * HN,
                      col = n0 + z * HN + cl;
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const int rl = 64 * wm + 4 * r + 16 * c;
                float x[4], pp[4];
                ldsf4(pp, rbuf + uint32_t(((c * 8 + j) * 64 + L) * 4) * 4u);
                ldsf4(x, xs + uint32_t(cl * XLD + rl) * 4u);
                float o[4];
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const float d = z == 0 ? acc[4 * c + e][j] + pp[e] : pp[e] + acc[4 * c + e][j];
                    o[e] = x[e] - d;
                } // D = P_0 + P_1
                float *dst = X + size_t(m0 + rl) + size_t(col) * ldx;
                if (full)
                    *reinterpret_cast<float4 *>(dst) = make_float4(o[0], o[1], o[2], o[3]);
                else if (col < q) {
#pragma unroll
                    for (int e = 0; e < 4; ++e)
                        if (m0 + rl + e < rows)
                            dst[e] = o[e];
                }
            }
        }
    }
}
using SF3 = FCfg<3>;
using SF2K16 = FCfg<2, 16>;
// The FP32 carrier needs whole k-tiles in each half, 16-byte copies of every column start, and
// 32-bit offsets for a k-tile of operand rows.
template <class Cfg>
bool simt_dcl_admits(const float *v, int ldv, const float *zt, int ldz, const float *x, int ldx,
                     int h) {
    const uintptr_t bases = uintptr_t(v) | uintptr_t(zt) | uintptr_t(x);
    return h >= 2 * Cfg::BK && h % (2 * Cfg::BK) == 0 && !(bases & 15) &&
           !((ldv | ldz | ldx) & 3) && (long long)Cfg::BK * ldv * 4 < (1LL << 31) &&
           (long long)Cfg::BK * ldz * 4 < (1LL << 31);
}
template <class Cfg>
void launch_simt_dcl(const float *v, int ldv, const float *zt, int ldz, float *x, int ldx, int rows,
                     int h, int q, cudaStream_t st) {
    if (!rows || !q)
        return;
    reserve_shared_memory(simt_dcl_kernel<Cfg>, Cfg::Smem, true);
    const int nx = ceildiv(q, Cfg::BN), ny = ceildiv(rows, Cfg::BM);
    launch_clustered(simt_dcl_kernel<Cfg>, dim3(2 * nx, ny), dim3(128), Cfg::Smem, st,
                     dim3(2, 1, 1), v, ldv, zt, ldz, x, ldx, rows, h, q,
                     std::min(raster_group, nx));
}
} // namespace kami
} // namespace tqr