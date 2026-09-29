#pragma once
// 3xTF32 W with the X split done IN REGISTERS (Hopper wgmma RS).
//  W^T (q x h) = X^T (q x rows) V (rows x h). A = X^T: X column-major, K (= rows) contiguous -> K-major tile, TMA copies
//  the raw fp32 bits. Each consumer thread copies its A fragment shared -> registers and splits it there:
//  A_big = rna(x), A_small = rna(x - A_big). B = V_big and V_small (split_v_rn output, column-major = K-major), two
//  TMA tiles per stage. Per k-block: A_big*V_big + A_small*V_big + A_big*V_small (three RS GMMAs) into a tensor-core
//  chunk accumulator that is promoted (IEEE fp32 adds) into the register accumulator every PromoteKTiles k-tiles.
//  Cooperative M split: each warp group owns its 64 output rows, so a promotion needs no exchange between groups.
// Carrier: the M split is not a contraction split. [K] = rows is cut into CG = c contiguous GPU-level
//  slices (L-batched: concurrent CTAs of one launch, disjoint K_z), each committed to its own partial slice; the
//  existing fixed-order combine (g = 0..CG-1) forms W once. Executed decomposition b1.cl1.g<c>.n1: c = the scheduled c.
//  here the split never touches shared memory.
#include "gmma_ksplit.cuh"
#include "cutlass/gemm/collective/sm90_mma_tma_gmma_rs_warpspecialized.hpp"
namespace tqr { namespace gmma {
using namespace cute;
#ifndef TQR_X3RS_PROMOTE
#define TQR_X3RS_PROMOTE 4
#endif
#ifndef TQR_X3RS_STAGES
#define TQR_X3RS_STAGES 3   // 4 x (A + 2 B) stages exceed shared memory with the fp32 epilogue
#endif
template<class Base>
struct X3RSMainloop : Base {
  using typename Base::TiledMma;using typename Base::TileShape;using typename Base::MainloopPipeline;
  using typename Base::PipelineState;using typename Base::SmemLayoutA;using typename Base::SmemLayoutB;
  using typename Base::DispatchPolicy;
  using ClusterShape=typename DispatchPolicy::ClusterShape;
  static constexpr int BM=size<0>(TileShape{}),BN=size<1>(TileShape{}),BK=size<2>(TileShape{});
  static constexpr int PromoteKTiles=TQR_X3RS_PROMOTE;
  static_assert(size(TiledMma{})==256,"cooperative: two consumer warp groups split M");
  struct TensorStorage : Base::TensorStorage {
    cute::array_aligned<typename TiledMma::ValTypeB,cute::cosize_v<SmemLayoutB>,Base::SmemAlignmentB> smem_Bs;};
  struct SharedStorage {TensorStorage tensors;typename Base::PipelineStorage pipeline;};
  using PipelineStorage=typename Base::PipelineStorage;
  static constexpr uint32_t TmaBytes=Base::TmaTransactionBytesMK+2*Base::TmaTransactionBytesNK;
  struct Arguments : Base::Arguments {typename Base::ElementB const* ptr_Bs=nullptr;Witness* wit=nullptr;int kind=2;int gpu=1;};
  using TMA_B=typename Base::Params::TMA_B;
  struct Params : Base::Params {TMA_B tma_load_bs;Witness* wit;int kind;int gpu;};
  template<class ProblemShape>
  static Params to_underlying_arguments(ProblemShape const& ps,Arguments const& args,void* ws){
    Params p{Base::to_underlying_arguments(ps,static_cast<typename Base::Arguments const&>(args),ws),{},nullptr,2,1};
    auto [M,N,K,L]=append<4>(ps,1);(void)M;
    Tensor tb=make_tensor(reinterpret_cast<typename Base::InternalElementB const*>(args.ptr_Bs),make_layout(make_shape(N,K,L),args.dB));
    p.tma_load_bs=make_tma_copy_B_sm90(typename Base::GmemTiledCopyB{},tb,SmemLayoutB{}(_,_,cute::Int<0>{}),TileShape{},ClusterShape{});
    p.tma_transaction_bytes=TmaBytes;p.wit=args.wit;p.kind=args.kind;p.gpu=args.gpu;return p;}
  CUTLASS_DEVICE static void prefetch_tma_descriptors(Params const& p){
    cute::prefetch_tma_descriptor(p.tma_load_a.get_tma_descriptor());cute::prefetch_tma_descriptor(p.tma_load_b.get_tma_descriptor());
    cute::prefetch_tma_descriptor(p.tma_load_bs.get_tma_descriptor());}
  template<class ProblemShape_MNKL>
  CUTLASS_DEVICE auto load_init(ProblemShape_MNKL const& ps,Params const& p) const {
    using X=Underscore;auto [M,N,K,L]=ps;
    Tensor mA=p.tma_load_a.get_tma_tensor(make_shape(M,K,L));Tensor mB=p.tma_load_b.get_tma_tensor(make_shape(N,K,L));
    Tensor mBs=p.tma_load_bs.get_tma_tensor(make_shape(N,K,L));
    Tensor gA=local_tile(mA,TileShape{},make_coord(_,_,_),Step<_1,X,_1>{});
    Tensor gB=local_tile(mB,TileShape{},make_coord(_,_,_),Step<X,_1,_1>{});
    Tensor gBs=local_tile(mBs,TileShape{},make_coord(_,_,_),Step<X,_1,_1>{});
    return cute::make_tuple(gA,gB,gBs);}
  template<class TensorA,class TensorB,class TensorBs,class KTileIterator,class BlockCoord>
  CUTLASS_DEVICE void load(Params const& p,MainloopPipeline pipeline,PipelineState smem_pipe_write,
      cute::tuple<TensorA,TensorB,TensorBs> const& in,BlockCoord const& blk_coord,KTileIterator k_tile_iter,int k_tile_count,
      int thread_idx,uint32_t block_rank_in_cluster,TensorStorage& st){
    if(cute::elect_one_sync()){
      Tensor sA=as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(st.smem_A.data()),SmemLayoutA{}));
      Tensor sB=as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(st.smem_B.data()),SmemLayoutB{}));
      Tensor sBs=as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(st.smem_Bs.data()),SmemLayoutB{}));
      auto [m,n,k,l]=blk_coord;
      Tensor gA=get<0>(in)(_,_,m,_,l);Tensor gB=get<1>(in)(_,_,n,_,l);Tensor gBs=get<2>(in)(_,_,n,_,l);
      auto ta=p.tma_load_a.get_slice(0);auto tb=p.tma_load_b.get_slice(0);auto tbs=p.tma_load_bs.get_slice(0);
      Tensor tAgA=ta.partition_S(gA);Tensor tAsA=ta.partition_D(sA);
      Tensor tBgB=tb.partition_S(gB);Tensor tBsB=tb.partition_D(sB);
      Tensor tBgBs=tbs.partition_S(gBs);Tensor tBsBs=tbs.partition_D(sBs);
      CUTLASS_PRAGMA_NO_UNROLL
      for(;k_tile_count>0;--k_tile_count){
        pipeline.producer_acquire(smem_pipe_write);
        auto* bar=pipeline.producer_get_barrier(smem_pipe_write);const int s=smem_pipe_write.index();
        copy(p.tma_load_a.with(*bar,0),tAgA(_,_,_,*k_tile_iter),tAsA(_,_,_,s));
        copy(p.tma_load_b.with(*bar,0),tBgB(_,_,_,*k_tile_iter),tBsB(_,_,_,s));
        copy(p.tma_load_bs.with(*bar,0),tBgBs(_,_,_,*k_tile_iter),tBsBs(_,_,_,s));
        ++k_tile_iter;++smem_pipe_write;}}}
  // Consumer. accum is this warp group's 64 x BN promoted accumulator (the kernel's fragment); accF the tensor-core chunk.
  template<class FrgTensorC>
  CUTLASS_DEVICE void mma(MainloopPipeline pipeline,PipelineState smem_pipe_read,FrgTensorC& accum,int k_tile_count,
      int thread_idx,TensorStorage& st,Params const& prm){
    Tensor sA=as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(st.smem_A.data()),SmemLayoutA{}));
    Tensor sB=make_tensor(make_smem_ptr(st.smem_B.data()),SmemLayoutB{});
    Tensor sBs=make_tensor(make_smem_ptr(st.smem_Bs.data()),SmemLayoutB{});
    TiledMma tiled_mma;
    const int wg=__shfl_sync(0xFFFFFFFF,thread_idx/128,0);
    auto thr=tiled_mma.get_thread_slice(thread_idx);
    auto wgs=tiled_mma.get_slice(make_layout(Int<2>{},Int<128>{})(wg));
    Tensor tCrA=thr.partition_fragment_A(sA(_,_,Int<0>{}));        // (MMA,MMA_M,MMA_K) raw bits, then A_big in place
    Tensor tCrAs=make_fragment_like(tCrA);                            // A_small
    Tensor tCrB=wgs.make_fragment_B(wgs.partition_B(sB));           // (MMA,MMA_N,MMA_K,PIPE) descriptors
    Tensor tCrBs=wgs.make_fragment_B(wgs.partition_B(sBs));
    auto cpA=make_tiled_copy_A(typename Base::InternalSmemCopyAtomA{},tiled_mma);
    auto thr_cpA=cpA.get_thread_slice(thread_idx);
    Tensor tCsA_v=thr_cpA.partition_S(sA);
    Tensor tCrA_v=thr_cpA.retile_D(tCrA);
    Tensor accF=make_fragment_like(accum);
    constexpr int KB=size<2>(tCrA);
    auto split=[&](int kb){
      CUTLASS_PRAGMA_UNROLL
      for(int i=0;i<size(tCrA(_,_,kb));++i){
        const float x=__uint_as_float(tCrA(_,_,kb)(i).raw());const float b=tf32_rna(x);
        tCrA(_,_,kb)(i)=cutlass::tfloat32_t::bitcast(__float_as_uint(b));
        tCrAs(_,_,kb)(i)=cutlass::tfloat32_t::bitcast(__float_as_uint(tf32_rna(x-b)));}};
    PipelineState release=smem_pipe_read;
    bool fresh=true;   // next GMMA starts a chunk (ScaleOut::Zero)
    bool first_chunk=true;
    warpgroup_fence_operand(accF);warpgroup_fence_operand(accum);
    for(int t=0;t<k_tile_count;++t){
      auto tok=pipeline.consumer_try_wait(smem_pipe_read);pipeline.consumer_wait(smem_pipe_read,tok);
      const int s=smem_pipe_read.index();
      CUTLASS_PRAGMA_UNROLL
      for(int kb=0;kb<KB;++kb){
        if(t>0){warpgroup_wait<KB-1>();if(kb==KB-1){pipeline.consumer_release(release);++release;}}   // (t-1,kb) retired
        copy(cpA,tCsA_v(_,_,kb,s),tCrA_v(_,_,kb));
        split(kb);
        warpgroup_fence_operand(accF);warpgroup_arrive();
        tiled_mma.accumulate_=fresh?GMMA::ScaleOut::Zero:GMMA::ScaleOut::One;fresh=false;
        cute::gemm(tiled_mma,tCrA(_,_,kb),tCrB(_,_,kb,s),accF);
        tiled_mma.accumulate_=GMMA::ScaleOut::One;
        cute::gemm(tiled_mma,tCrAs(_,_,kb),tCrB(_,_,kb,s),accF);
        cute::gemm(tiled_mma,tCrA(_,_,kb),tCrBs(_,_,kb,s),accF);
        warpgroup_commit_batch();warpgroup_fence_operand(accF);}
      ++smem_pipe_read;
      if((t+1)%PromoteKTiles==0&&t+1<k_tile_count){
        warpgroup_wait<0>();warpgroup_fence_operand(accF);
        CUTLASS_PRAGMA_UNROLL
        for(int i=0;i<size(accF);++i)accum(i)=first_chunk?accF(i):accum(i)+accF(i);
        first_chunk=false;fresh=true;warpgroup_fence_operand(accum);}}
    warpgroup_wait<0>();warpgroup_fence_operand(accF);
    if(k_tile_count>0){pipeline.consumer_release(release);++release;}
    CUTLASS_PRAGMA_UNROLL
    for(int i=0;i<size(accF);++i)accum(i)=k_tile_count<1?0.f:(first_chunk?accF(i):accum(i)+accF(i));
    warpgroup_fence_operand(accum);
    // One report per launch: a persistent CTA calls mma() once per output tile (same guard as KSplitMainloop).
    if(prm.wit&&!reported_&&thread_idx==0&&blockIdx.x==0&&blockIdx.y==0&&blockIdx.z==0){
      reported_=true;Witness*w=prm.wit;
      atomicAdd(&w->device_peer_partials[prm.kind],1ULL*prm.gpu);atomicAdd(&w->device_block_peers[prm.kind],1ULL);
      witness_levels(w,prm.kind,1,1,prm.gpu,1);atomicAdd(&w->device_membership_reports,1ULL);atomicAdd(&w->device_combines,1ULL);}
  }
  CUTLASS_DEVICE void mma_tail(MainloopPipeline,PipelineState,int){warpgroup_wait<0>();}
  bool reported_=false;
};
// C (M x N, RowMajor here: W^T view of the column-major W) = A (M x K, K-major) B (N x K, K-major).
template<int BM=128,int BN=128>
struct X3RSGemm {
  using E=cutlass::tfloat32_t;
  using TS=Shape<Int<BM>,Int<BN>,_32>;using CS=Shape<_1,_1,_1>;
  using LC=cutlass::layout::RowMajor;
  using Epi=typename cutlass::epilogue::collective::CollectiveBuilder<cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,
    TS,CS,cutlass::epilogue::collective::EpilogueTileAuto,float,float,float,LC,4,float,LC,4,
    cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;
  using Atom=cute::SM90::GMMA::MMA_64x128x8_F32TF32TF32_RS_TN<>;
  static_assert(BN==128,"RS atom is 64x128x8");
  using TiledMmaRS=decltype(make_tiled_mma(Atom{},Layout<Shape<_2,_1,_1>>{}));
  using SmemAtom=GMMA::Layout_K_SW128_Atom<E>;
  using StrideAB=cute::tuple<int64_t,cute::Int<1>,int64_t>;
  using BaseMain=cutlass::gemm::collective::CollectiveMma<
    cutlass::gemm::MainloopSm90TmaGmmaRmemAWarpSpecialized<TQR_X3RS_STAGES,CS,cutlass::gemm::KernelTmaWarpSpecializedCooperative>,
    TS,E,StrideAB,E,StrideAB,TiledMmaRS,
    cute::SM90_TMA_LOAD,SmemAtom,cute::Copy_Atom<cute::AutoVectorizingCopy,E>,cute::identity,
    cute::SM90_TMA_LOAD,SmemAtom,void,cute::identity>;
  using Main=X3RSMainloop<BaseMain>;
  using Kernel=cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>,Main,Epi,void>;
  using Gemm=cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};
}}
