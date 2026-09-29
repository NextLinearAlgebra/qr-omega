#pragma once
// Workspace sizing of the far-update pipeline (look-ahead windows and aggregated transforms).
#include "common.hpp"
namespace tqr {
inline bool near_enabled(){static const bool v=[]{const char*e=std::getenv("TQR_NEAR");return !(e&&std::string(e)=="0");}();return v;}
inline bool near2_enabled(){static const bool v=[]{const char*e=std::getenv("TQR_NEAR2");return near_enabled()&&!(e&&std::string(e)=="0");}();return v;}
inline size_t p7_gpu_partial_words(int b,int arep){return size_t(128)*size_t(b)*size_t(b)+size_t(std::max(1,arep)-1)*size_t(b)*size_t(b);}
// Extra words owned by aggregate g > 1 (per batch member where noted), exactly what TiledEngine
// allocates: g-1 more V halves, W/Z credits x g on every slot, V_g, T_g and its permuted copy, the
// compose_G blocks (plus the library partials that land beyond them), and the layer permutations.
inline size_t p7_aggregate_words(int b,int leaf,int strip,int radix,int depth,int arep,int zrep,int agg,size_t batch,bool la=false){
 if(agg<=1)return 0;
 // la: two lane groups of max(1,d) slots and two V halves are already charged by the base frame;
 // aggregation adds halves beyond those and x(agg-1) credits per slot.
 const size_t vhalf=batch*std::max(size_t(leaf)*b,size_t(radix)*b*b);
 const size_t halves=size_t(std::max(la?2:1,agg)-(la?2:1));
 const size_t lanes=size_t(std::max(1,depth))*(la?2:1);
 const size_t credits=lanes*size_t(std::max(1,arep)+std::max(1,zrep))*batch*size_t(b)*size_t(agg-1)*size_t(strip);
 const size_t H=size_t(agg)*b;
 // under look-ahead V_g, T_g and its permuted copy exist in TWO generations (the far update of group G-1 reads one
 // while group G composes the other; tiled_engine.cuh agg_gens), charged here as the upper bound.
 const size_t gens=la?2:1;
 // The near slot's compose_G products own a second GPU-partial buffer (GPART_N, 128 b x b slices).
 const size_t near_part=(la&&near2_enabled())?size_t(128)*b*b:0;
 return halves*vhalf+credits+gens*(size_t(leaf)*b*agg+2*H*H)+near_part+size_t(std::max(1,arep)+agg)*size_t(b)*b+size_t(10)*b;
}
// fp32 plans own a per-lane-slot row-major copy of the product's V for the TF32 GMMA D: lanes x leaf
// x (b aggregate) words, with lanes = max(1,d) (x2 under lookahead) exactly as tiled_lane_slots.
inline size_t gmma_vr_words(size_t word,int depth,int la,int leaf,int b,int agg){
 return word==4?size_t(std::max(1,depth))*size_t(la?2:1)*size_t(leaf)*size_t(b)*size_t(std::max(1,agg)):0;}
// The 3xTF32 GMMA D pre-splits both operands into three K segments (VR3 = [V_small | V_big | V_big],
// rows x 3(b aggregate); Z3 = [Z_big; Z_small;
inline size_t gmma_x3_words(size_t word,int depth,int la,int leaf,int b,int agg,int strip){
 const size_t lanes=size_t(std::max(1,depth))*size_t(la?2:1),ba=size_t(b)*size_t(std::max(1,agg));
 return word==4?lanes*(2*size_t(leaf)*ba+3*ba*size_t(strip)):0;}
}
