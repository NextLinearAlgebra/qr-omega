#pragma once
// Arithmetic of one 16 x 8 x 16 warp tile. The tree algorithm, tiling, replication and memory
// schedule are shared by all precisions. Only this instruction adapter changes with the math mode.
// In lane (g = lane / 4, t = lane % 4), a[2u + r] holds A(2g + r, 4t + u),
// b[u] holds B(4t + u, g), and d[2r + j] holds C(2g + r, 2t + j).
#include "carrier_kami.cuh"

namespace qr_omega {
template <class T> struct alignas(2 * sizeof(T)) TreePair {
    T x, y;
};
template <class T> __device__ __forceinline__ TreePair<T> tree_pair(const T *p) {
    return *reinterpret_cast<const TreePair<T> *>(p);
}
template <class T> __device__ __forceinline__ void tree_pair(T *p, T x, T y) {
    *reinterpret_cast<TreePair<T> *>(p) = {x, y};
}

template <class T> __device__ __forceinline__ TreePair<T> tree_make_pair(T x, T y) {
    return {x, y};
}
template <class T, int Words>
__device__ __forceinline__ void tree_copy(T *dst, const T *src, int valid) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n" ::"r"(
                     unsigned(__cvta_generic_to_shared(dst))),
                 "l"(src), "n"(Words * sizeof(T)), "r"(int(sizeof(T)) * valid)
                 : "memory");
}

// A logical row permutation and a contraction permutation give the TF32 instruction the same
// register layout as FP64. Its two k=8 instructions consume u=0,1 then u=2,3 of every lane.
__device__ __forceinline__ void tree_tf32(float (&d)[4], const float *a, const float *b) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(__float_as_uint(a[0])), "r"(__float_as_uint(a[1])),
                   "r"(__float_as_uint(a[2])), "r"(__float_as_uint(a[3])),
                   "r"(__float_as_uint(b[0])), "r"(__float_as_uint(b[1])));
}
__device__ __forceinline__ float tree_round_tf32(float x) {
    uint32_t y;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(y) : "f"(x));
    return __uint_as_float(y);
}

template <class T, Fp32Math Math = Fp32Math::IEEE> struct TreeMath {
    __device__ __forceinline__ static void mma(T (&d)[4], const T (&a)[8], const T (&b)[4]) {
        if constexpr (std::is_same_v<T, double>) {
            kami::dmma16(d, a, b);
        } else if constexpr (Math == Fp32Math::IEEE) {
            // FP32 values on the FP64 tensor cores: products of FP32 values are exact in FP64
            // and the 16-term sum is rounded once to FP32, as accurate as FP32 FMAs or more.
            double ad[8], bd[4], dd[4];
#pragma unroll
            for (int i = 0; i < 8; ++i)
                ad[i] = a[i];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                bd[i] = b[i];
                dd[i] = d[i];
            }
            kami::dmma16(dd, ad, bd);
#pragma unroll
            for (int i = 0; i < 4; ++i)
                d[i] = float(dd[i]);
        } else {
            // The 16-term sum forms on the tensor cores from zero and joins d in one FP32
            // addition, as in the FP32 path: the tensor cores truncate their running TF32 sums,
            // so d is promoted out of them after every tile instead of accumulating there.
            float ah[8], bh[4], s[4] = {};
#pragma unroll
            for (int i = 0; i < 8; ++i)
                ah[i] = tree_round_tf32(a[i]);
#pragma unroll
            for (int i = 0; i < 4; ++i)
                bh[i] = tree_round_tf32(b[i]);
            if constexpr (Math == Fp32Math::X3) {
                float al[8], bl[4];
#pragma unroll
                for (int i = 0; i < 8; ++i)
                    al[i] = tree_round_tf32(a[i] - ah[i]);
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    bl[i] = tree_round_tf32(b[i] - bh[i]);
                tree_tf32(s, al, bh);
                tree_tf32(s, al + 4, bh + 2);
                tree_tf32(s, ah, bl);
                tree_tf32(s, ah + 4, bl + 2);
            }
            tree_tf32(s, ah, bh);
            tree_tf32(s, ah + 4, bh + 2);
#pragma unroll
            for (int i = 0; i < 4; ++i)
                d[i] += s[i];
        }
    }
};

// Dispatch only the arithmetic adapter; all modes see exactly the same schedules.
template <class T, class F> void with_tree_math(F &&f) {
    if constexpr (std::is_same_v<T, float>) {
        if (fp32_math() == Fp32Math::TF32) {
            f(TreeMath<T, Fp32Math::TF32>{});
            return;
        }
        if (fp32_math() == Fp32Math::X3) {
            f(TreeMath<T, Fp32Math::X3>{});
            return;
        }
    }
    f(TreeMath<T>{});
}
} // namespace qr_omega
