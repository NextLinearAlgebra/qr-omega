#pragma once
// TF32 and 3xTF32 GEMM on Hopper wgmma with a block-level contraction split between two warp groups.
// Alg 2 "sum the c partials over disjoint K_z", "owner commits exactly once").
//
// Built from CUTLASS 3 sm90 building blocks: TMA loads, the warp-specialized
// cooperative kernel (one producer warp group, two consumer warp groups, persistent tile scheduler,
// TMA epilogue). The carrier semantics are ours and replace the cooperative mainloop's M split:
//  * both consumer warp groups compute the FULL BM x BN output tile of the work unit;
//  * in every k-tile of BK = 32 columns, warp group w contracts columns [16w, 16w + 16) (two tf32
//    k-blocks): c = 2 disjoint K sets whose union is [0, K) -- the law (bk, wk) = (32, 16) that
//  * the two partials are combined on chip in the FIXED order p0 + p1 through a shared-memory
//    exchange (each group receives the other's partial for the half of the tile it owns), and
//    each group hands its owned half to the unchanged cooperative epilogue, which commits the
//    output (D: X <- X - V Z; W: W <- V^T X) exactly once;
//  * one leader thread reports the executed geometry (block c = 2) to the device witness.
// The pipeline protocol is the base's: both groups consume every stage (each its half of the stage's
// k-blocks) and release it with the base's one-stage lag.
//
// The same carrier with error-compensated 3xTF32 arithmetic inside each peer (Ootomo-Yokota: a = big +
// small, big = rn_tf32(a), small = rn_tf32(a - big); per k-block the products small*big, big*small,
// big*big accumulate in that order into the peer's partial). The split is below the moves: each group
// splits only its own K_z columns of the stage (in place for big, into a two-slot small buffer for
// small), so K_z, the fixed-order combine, the single commit and the witness are exactly the TF32
// carrier's. The operand element type is tfloat32_t, which makes the TMA copy raw fp32 bits (with
// float the TMA rounds to tf32 in flight and the small part would be lost).
#include "kernels.cuh"
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/arch/barrier.h"
#include "gmma_cooperative_2cta.cuh"
namespace tqr::gmma {
using namespace cute;
// Base = the standard SS warp-specialized mainloop the builder picks for this tile (cooperative).
// Small-part slots: A and B (mode 1), B only (modes 2, 3; in mode 3 only the promotion exchange uses it).
template<int SmallA,int SmallB,int Align> struct KSplitSmall {
  cute::array_aligned<float,SmallA,Align> smem_As;cute::array_aligned<float,SmallB,Align> smem_Bs;};
template<int SmallB,int Align> struct KSplitSmall<0,SmallB,Align> {cute::array_aligned<float,SmallB,Align> smem_Bs;};
template<int Align> struct KSplitSmall<0,0,Align> {};
// Round to nearest (ties away) at the tf32 mantissa, as cvt.rna.tf32.f32 does for finite values, in
// integer ops (the split is on the stage's critical path).
__device__ __forceinline__ float tf32_rna(float x){return __uint_as_float((__float_as_uint(x)+0x1000u)&0xFFFFE000u);}
// Arithmetic modes (all keep the carrier: K_z halves per k-tile, fixed-order combine, one commit):
//  0 TF32   : operands float (the TMA rounds to tf32), one GMMA per k-block.
//  1 X3     : raw operands; both split in the peer's K_z columns; small*big, big*small, big*big; promotion.
//  2 X3B    : raw operands, A already rounded to tf32 by the caller; B split; A*B_small, A*B_big; promotion.
//  3 RAW    : raw operands, one GMMA (the tensor core truncates both to tf32); promotion.
// Modes 2 + 3 make one 3xTF32 product from two launches: V_big^T (X_big + X_small) then V_small^T X_raw added
// by the epilogue (beta = 1), with V pre-split in HBM.
template<class Base,int XM=0>
struct KSplitMainloop : Base {
  static constexpr bool X3=XM!=0,SplitA=XM==1,SplitB=XM==1||XM==2;
  using typename Base::TiledMma;using typename Base::TileShape;using typename Base::MainloopPipeline;
  using typename Base::PipelineState;using typename Base::SmemLayoutA;using typename Base::SmemLayoutB;
  static constexpr int BM=size<0>(TileShape{}),BN=size<1>(TileShape{});
  static constexpr int WG=128;
  // One warp group covering the whole tile: the same GMMA atom, no M split across groups.
  using TiledMmaFull=decltype(make_tiled_mma(typename TiledMma::Atom{},Layout<Shape<_1,_1,_1>>{}));
  static_assert(size(TiledMma{})==2*WG,"cooperative (two consumer warp groups) only");
  static_assert(size(TiledMmaFull{})==WG,"one warp group per full-tile partial");
  // Exchange: each group parks the half it does not own, thread-major (fragment coordinates are the
  // same in both groups for the same thread index, so the partner reads exactly its elements).
  static constexpr int FullVals=BM*BN/WG,HalfVals=FullVals/2;
  // The exchange uses the smem of the tile's LAST stage, which this group holds (does not release)
  // until the combine is done: A-stage + B-stage = BK (BM + BN) floats >= one half tile of partials.
  static constexpr int BK_=size<2>(TileShape{});
  // X3: two slots (k-tile parity) of small parts, laid out exactly like one stage of A and B so the
  // stage's GMMA descriptors address them with the same coordinates.
  static constexpr int SmallSlots=2;
  static constexpr int SmallAWords=SplitA?SmallSlots*BM*BK_:0,SmallBWords=X3?SmallSlots*BN*BK_:0;
  struct TensorStorage : Base::TensorStorage, KSplitSmall<SmallAWords,SmallBWords,1024> {};
  static constexpr int SmallBytes=(SmallAWords+SmallBWords)*4;
#ifndef TQR_X3_PROMOTE
#define TQR_X3_PROMOTE 16
#endif
// Same carrier, K_z halves and combine; only the TC accumulation chunk of the W launches changes. Default: the
// common interval.
#ifndef TQR_X3_PROMOTE_W
#define TQR_X3_PROMOTE_W TQR_X3_PROMOTE
#endif
  static constexpr int PromoteKTiles=(XM==2||XM==3)?TQR_X3_PROMOTE_W:TQR_X3_PROMOTE;   // X3: k-tiles per promoted chunk (32 PromoteKTiles columns)
  static_assert(!X3||SmallSlots*BN*BK_>=FullVals/2*WG,"the B small slots hold one half-tile exchange");
  static_assert(BK_*(BM+BN)>=HalfVals*WG,"a held stage must hold one half-tile exchange");
  struct SharedStorage {TensorStorage tensors;typename Base::PipelineStorage pipeline;};
  using PipelineStorage=typename Base::PipelineStorage;
  // gpu: GPU-level K peers (the stream-K scheduler's deterministic split-K; 1 = none), reported as
  // the gpu level of the executed decomposition.
  struct Arguments : Base::Arguments {Witness* wit=nullptr;int kind=4;int commit=1;int gpu=1;};
  struct Params : Base::Params {Witness* wit;int kind;int commit;int gpu;};
  template<class ProblemShape>
  static Params to_underlying_arguments(ProblemShape const& ps,Arguments const& args,void* ws){
    Params p{Base::to_underlying_arguments(ps,static_cast<typename Base::Arguments const&>(args),ws),nullptr,4,1,1};
    p.wit=args.wit;p.kind=args.kind;p.commit=args.commit;p.gpu=args.gpu;return p;}
  bool reported_=false;
  template<class FrgTensorC>
  CUTLASS_DEVICE void mma(MainloopPipeline pipeline,PipelineState smem_pipe_read,FrgTensorC& accum,int k_tile_count,
                          int thread_idx,TensorStorage& st,Params const& prm){
    Tensor sA=make_tensor(make_smem_ptr(st.smem_A.data()),SmemLayoutA{});
    Tensor sB=make_tensor(make_smem_ptr(st.smem_B.data()),SmemLayoutB{});
    const int wg=__shfl_sync(0xFFFFFFFF,thread_idx/WG,0),t=thread_idx%WG;
    TiledMmaFull full;auto thr=full.get_slice(t);
    Tensor tCsA=thr.partition_A(sA);Tensor tCsB=thr.partition_B(sB);
    Tensor tCrA=thr.make_fragment_A(tCsA);Tensor tCrB=thr.make_fragment_B(tCsB);
    Tensor accF=partition_fragment_C(full,take<0,2>(TileShape{}));   // (MMA, BM/64, BN/atomN)
    static_assert(decltype(size(accF))::value==FullVals,"full-tile partial");
    if(k_tile_count<1){clear(accF);}
    // K_z of peer w (law bk = BK, wk = BK/2): within EVERY k-tile, warp group w contracts the k-blocks
    // [w KB/2, (w+1) KB/2) (tf32 k-block = 8 columns). Both groups consume every stage together, so
    // the pipeline, its release lag (K_PIPE_MMAS = 1) and the producer are exactly the base's.
    constexpr int KB=decltype(size<2>(tCrA))::value;static_assert(KB%2==0,"two K halves per k-tile");
    const int kb0=wg*(KB/2);
    auto issue1=[&](int stage){
      CUTLASS_PRAGMA_UNROLL
      for(int kb=0;kb<KB/2;++kb){cute::gemm(full,tCrA(_,_,kb0+kb,stage),tCrB(_,_,kb0+kb,stage),accF);full.accumulate_=GMMA::ScaleOut::One;}};
    // X3 small parts: slot (k-tile parity) tensors with the stage layout; descriptors built the same way.
    using TA_=typename Base::TiledMma::ValTypeA;using TB_=typename Base::TiledMma::ValTypeB;
    TA_* pAs=st.smem_A.data();TB_* pBs=st.smem_B.data();   // (unused aliases unless X3)
    if constexpr(SplitA)pAs=reinterpret_cast<TA_*>(st.smem_As.data());
    if constexpr(X3)pBs=reinterpret_cast<TB_*>(st.smem_Bs.data());
    float* pAs_raw=reinterpret_cast<float*>(pAs);float* pBs_raw=reinterpret_cast<float*>(pBs);
    Tensor sAs=make_tensor(make_smem_ptr(pAs),SmemLayoutA{});Tensor sBs=make_tensor(make_smem_ptr(pBs),SmemLayoutB{});
    Tensor tCrAs=thr.make_fragment_A(thr.partition_A(sAs));Tensor tCrBs=thr.make_fragment_B(thr.partition_B(sBs));
    auto issue2=[&](int stage,int slot){   // mode 2: A (pre-rounded) * B_small, then A * B_big
      CUTLASS_PRAGMA_UNROLL
      for(int kb=0;kb<KB/2;++kb){
        cute::gemm(full,tCrA(_,_,kb0+kb,stage),tCrBs(_,_,kb0+kb,slot),accF);full.accumulate_=GMMA::ScaleOut::One;
        cute::gemm(full,tCrA(_,_,kb0+kb,stage),tCrB(_,_,kb0+kb,stage),accF);}};
    auto issue3=[&](int stage,int slot){
      CUTLASS_PRAGMA_UNROLL
      for(int kb=0;kb<KB/2;++kb){
        cute::gemm(full,tCrAs(_,_,kb0+kb,slot),tCrB(_,_,kb0+kb,stage),accF);full.accumulate_=GMMA::ScaleOut::One;
        cute::gemm(full,tCrA(_,_,kb0+kb,stage),tCrBs(_,_,kb0+kb,slot),accF);
        cute::gemm(full,tCrA(_,_,kb0+kb,stage),tCrB(_,_,kb0+kb,stage),accF);}};
    // Split this group's K_z columns [BK/2 wg, BK/2 (wg+1)) of the stage: big in place, small into the
    // slot. 16-byte chunks (4 columns) are contiguous under the 128-byte swizzle.
    auto split=[&](int stage,int slot){
      constexpr int CH=BK_/8,NA_=SplitA?BM*CH:0,NT=NA_+BN*CH;static_assert(NT%WG==0,"whole chunks per thread");
      const int kc0=wg*(BK_/2);
      // Slot reuse: the slot was last read by this group's GMMAs of k-tile kt - 2. wgmma completion is
      // per warp (warpgroup_wait covers the executing warp's share only), so every warp of the group
      // must have passed its wait for kt - 2 (done in iteration kt - 1) before any warp rewrites it.
      cutlass::arch::NamedBarrier::sync(WG,9+wg);
      CUTLASS_PRAGMA_UNROLL
      for(int j=0;j<NT/WG;++j){const int i=j*WG+t;const bool isA=i<NA_;const int li=isA?i:i-NA_,r=li/CH,k=kc0+4*(li%CH);
        float* src=isA?reinterpret_cast<float*>(&sA(r,k,stage)):reinterpret_cast<float*>(&sB(r,k,stage));
        float* dst=isA?reinterpret_cast<float*>(&sAs(r,k,slot)):reinterpret_cast<float*>(&sBs(r,k,slot));
        float4 v=*reinterpret_cast<float4*>(src),b,m;
        b.x=tf32_rna(v.x);b.y=tf32_rna(v.y);b.z=tf32_rna(v.z);b.w=tf32_rna(v.w);
        m.x=tf32_rna(v.x-b.x);m.y=tf32_rna(v.y-b.y);m.z=tf32_rna(v.z-b.z);m.w=tf32_rna(v.w-b.w);
        *reinterpret_cast<float4*>(src)=b;*reinterpret_cast<float4*>(dst)=m;}
      cutlass::arch::fence_view_async_shared();            // generic-proxy writes before the GMMA (async proxy) reads
      cutlass::arch::NamedBarrier::sync(WG,9+wg);};         // the whole group's split before its GMMAs
    // Fixed-order combine p0 + p1 over the owned half (M atoms [wg*MA/2, (wg+1)*MA/2)), through one
    // half-tile exchange buffer used twice (group 1 -> group 0, then group 0 -> group 1). Indices are
    // compile-time so the partial stays in registers. X3 promotion: the tensor core accumulates a
    // chunk's products in its own (truncating) fp32 accumulator, whose error grows linearly with K.
    // Every PromoteKTiles k-tiles both groups drain their GMMAs, combine their chunk partials p0 + p1
    // (fixed order) and the owner adds the sum into its register accumulator (`accum`, the owned half
    // tile) with an IEEE fp32 add; the next chunk restarts the GMMA accumulator at zero. The combine is
    // the same additive combine over the same disjoint K_z, performed per chunk (a pipeline over the
    // contraction); the output is still committed once, by the epilogue.
    constexpr int MA=decltype(size<1>(accF))::value;static_assert(MA%2==0,"two M halves");
    constexpr int NA=decltype(size<2>(accF))::value,VA=decltype(size<0>(accF))::value;
    float* xa=nullptr;float* xb=nullptr;bool first=true;
    auto slot=[&](int e)->float&{return e<BM*BK_?xa[e]:xb[e-BM*BK_];};
    auto put=[&](auto half){constexpr int H0=decltype(half)::value*(MA/2);
      #pragma unroll
      for(int m=0;m<MA/2;++m){
        #pragma unroll
        for(int n=0;n<NA;++n){
          #pragma unroll
          for(int i=0;i<VA;++i)slot(((m*NA+n)*VA+i)*WG+t)=accF(i,H0+m,n);}}};
    auto take=[&](auto half,bool own_first){constexpr int H0=decltype(half)::value*(MA/2);
      #pragma unroll
      for(int m=0;m<MA/2;++m){
        #pragma unroll
        for(int n=0;n<NA;++n){
          #pragma unroll
          for(int i=0;i<VA;++i){const float other=slot(((m*NA+n)*VA+i)*WG+t),own=accF(i,H0+m,n);
            const float sum=own_first?own+other:other+own;   // p0 + p1 either way
            accum(i,m,n)=first?sum:accum(i,m,n)+sum;}}}};
    auto combine=[&](){
      cutlass::arch::NamedBarrier::sync(2*WG,8);              // both groups retired their GMMAs
      if(wg==1)put(Int<0>{});                                  // p1 of the half group 0 owns
      cutlass::arch::NamedBarrier::sync(2*WG,8);
      if(wg==0)take(Int<0>{},true);                            // p0 (own) + p1
      cutlass::arch::NamedBarrier::sync(2*WG,8);
      if(wg==0)put(Int<1>{});                                  // p0 of the half group 1 owns
      cutlass::arch::NamedBarrier::sync(2*WG,8);
      if(wg==1)take(Int<1>{},false);                           // p0 + p1 (own)
      first=false;};
    // X3 exchanges go through the small slots: once both groups have drained, no GMMA reads them, and
    // the trailing barrier keeps the next split from overwriting a slot still being read.
    auto promote=[&](){
      if constexpr(X3){xa=pBs_raw;xb=pBs_raw+BM*BK_;combine();cutlass::arch::NamedBarrier::sync(2*WG,8);}};   // B slots: 2 BN BK >= exchange
    int kt=0;
    // warpgroup_arrive (wgmma.fence) after the split: the group's operand writes precede its GMMAs.
    auto issue=[&](int stage){
      if constexpr(XM==1){split(stage,kt&1);warpgroup_arrive();issue3(stage,kt&1);}
      else if constexpr(XM==2){split(stage,kt&1);warpgroup_arrive();issue2(stage,kt&1);}
      else{warpgroup_arrive();issue1(stage);}
      ++kt;};
    PipelineState smem_pipe_release=smem_pipe_read;
    constexpr int K_PIPE_MMAS=1;
    const int prologue=cute::min(K_PIPE_MMAS,k_tile_count);
    full.accumulate_=GMMA::ScaleOut::Zero;
    warpgroup_fence_operand(accF);
    for(int p=0;p<prologue;++p){
      auto tok=pipeline.consumer_try_wait(smem_pipe_read);pipeline.consumer_wait(smem_pipe_read,tok);
      issue(smem_pipe_read.index());warpgroup_commit_batch();++smem_pipe_read;}
    warpgroup_fence_operand(accF);
    CUTLASS_PRAGMA_NO_UNROLL
    for(int k=prologue;k<k_tile_count;++k){
      auto tok=pipeline.consumer_try_wait(smem_pipe_read);pipeline.consumer_wait(smem_pipe_read,tok);
      warpgroup_fence_operand(accF);issue(smem_pipe_read.index());warpgroup_commit_batch();
      warpgroup_wait<K_PIPE_MMAS>();warpgroup_fence_operand(accF);
      pipeline.consumer_release(smem_pipe_release);
      ++smem_pipe_read;++smem_pipe_release;
      if constexpr(X3){if(kt%PromoteKTiles==0&&k+1<k_tile_count){   // promote a finished chunk (see below)
        warpgroup_wait<0>();warpgroup_fence_operand(accF);promote();full.accumulate_=GMMA::ScaleOut::Zero;}}}
    warpgroup_wait<0>();warpgroup_fence_operand(accF);
    // Hold the last stage (prologue == 1) as the exchange buffer (TF32); release it after the combine.
    const int held=smem_pipe_release.index();
    if constexpr(X3){promote();}
    else{xa=reinterpret_cast<float*>(st.smem_A.data())+size_t(held)*BM*BK_;
         xb=reinterpret_cast<float*>(st.smem_B.data())+size_t(held)*BN*BK_;combine();}
    cutlass::arch::fence_view_async_shared();                 // generic-proxy use of the stage before TMA refills it
    pipeline.consumer_release(smem_pipe_release);
    if(prm.wit&&!reported_&&thread_idx==0&&blockIdx.x==0&&blockIdx.y==0&&blockIdx.z==0){
      reported_=true;Witness*w=prm.wit;
      atomicAdd(&w->device_peer_partials[prm.kind],2ULL*prm.gpu);atomicAdd(&w->device_block_peers[prm.kind],2ULL);
      witness_levels(w,prm.kind,2,1,prm.gpu,1);atomicAdd(&w->device_membership_reports,1ULL);
      if(prm.commit)atomicAdd(&w->device_physical_commits,1ULL);else atomicAdd(&w->device_combines,1ULL);}
  }
  CUTLASS_DEVICE void mma_tail(MainloopPipeline,PipelineState,int){warpgroup_wait<0>();}
};
// C(M x N) = alpha A(M x K) B(K x N) + beta C, A RowMajor (K-major), B ColumnMajor (K-major),
// C/D layout LC. TF32 GMMA from fp32 storage. X3 = false: operands declared float, the TMA rounds
// them to tf32 in flight. X3 = true: operands declared tfloat32_t (pointers are the same fp32 data),
// the TMA copies raw bits and the mainloop splits them (3xTF32).
template<int BM,int BN,class LC,class Sched=void,int XM=0,int CLM=1>
struct KSplitGemm {
  static constexpr bool X3=XM!=0;
  using E=cute::conditional_t<X3,cutlass::tfloat32_t,float>;
  // CLM: CTAs per cluster along M.
  using TS=Shape<Int<BM>,Int<BN>,_32>;using CS=Shape<Int<CLM>,_1,_1>;
  using Epi=typename cutlass::epilogue::collective::CollectiveBuilder<cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,
    TS,CS,cutlass::epilogue::collective::EpilogueTileAuto,float,float,float,LC,4,float,LC,4,
    cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;
  static constexpr int XchgBytes=(XM==1?2*(BM+BN)*32:X3?2*BN*32:0)*4;   // held stage for TF32; small slots otherwise
  using BaseMain=typename cutlass::gemm::collective::CollectiveBuilder<cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,
    E,cutlass::layout::RowMajor,4,E,cutlass::layout::ColumnMajor,4,float,TS,CS,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epi::SharedStorage))+XchgBytes>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
  using Main=KSplitMainloop<BaseMain,XM>;
  static_assert(Main::SmallBytes==XchgBytes,"carve-out matches the small slots");
  using Kernel=cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>,Main,Epi,Sched>;
  using Gemm=cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
  static constexpr int BK=32;
};
// Two CTAs share an SM. Smaller output tile and three TMA stages keep shared memory below half the
// SM budget; the fork supplies the register pool.
template<class LC>
struct KSplitGemm2CTA {
  using TS=Shape<_128,_64,_32>;using CS=Shape<_1,_1,_1>;
  using Epi=typename cutlass::epilogue::collective::CollectiveBuilder<cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,
    TS,CS,Shape<_128,_16>,float,float,float,LC,4,float,LC,4,
    cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;
  using BaseMain=typename cutlass::gemm::collective::CollectiveBuilder<cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,
    float,cutlass::layout::RowMajor,4,float,cutlass::layout::ColumnMajor,4,float,TS,CS,
    cutlass::gemm::collective::StageCount<3>,cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
  using Main=KSplitMainloop<BaseMain,0>;
  using Kernel=cutlass::gemm::kernel::GmmaCooperative2CTA<Shape<int,int,int,int>,Main,Epi,void>;
  using Gemm=cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
  static_assert(Kernel::SharedStorageSize<=116736,"two CTA shared-memory budget");
};
}
