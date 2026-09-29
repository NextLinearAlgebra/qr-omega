#pragma once
namespace tqr {
// Plans, profiles and execution witnesses change identity together. Old binary-keyed measurements
// remain evidence for their original binaries.
//
// 24: physical execution evidence. Per-product-kind DEVICE completion counts, the
// simultaneous-credit high-water mark, device %globaltimer packet/panel intervals
// (within-elimination overlap and next-panel release), and numeric capacity/structural certificates
// for every product that ran unreplicated or every elimination that issued one packet or used one
// slot. 25: per-level carriers with launch-derived partitions and Eq-compose G. 26: Phase-B
// explicit records. Six named concepts ride alongside the aggregates (which stay the qualification
// basis; full detail must reconcile exactly, never substitute): product identity (factorization,
// panel, elimination, packet, kind/subkind, operand versions, transpose/scalar op, output region);
// carrier specification (level, physical peer mapping, selected vs actual (pi,pj,c), partition
// rule, kernel variant, measurement provenance); partial identity (parent product, output tile,
// peer z, exact contraction interval, input versions, generation); pipeline specification
// (elimination-local packet boundaries, depth, slot ownership, buffer reservations, dependency
// events); execution receipt (actual peers, partial completion, additive combine, physical output
// commit, arithmetic-stage timestamps); refusal (attempted config, actual replacement or error,
// reason, count, independently checkable proof). Device scope: counters below are host-issued
// unless the field says device-observed; vendor- internal block execution is labeled unknown, never
// invented.
inline constexpr int execution_format_version = 33;
// Aggregation is replayed from the digested schedule, never an executor option. With it the engine owns
// two lane groups of d slots (the main group at the device's greatest stream priority factors the next
// unit; lanes d..2d-1 carry window B) and two V halves, charged in the inventory. The device
// packet/panel intervals are the item-4 evidence. D: operands pre-split into three K segments
// (small*big, big*small, big*big) run on the TF32 K-split carrier at K' = 3 roundup(h, 32), law (32,16)
// per segment, so each block peer's reflector set is the TF32 one. W: in-kernel split of each group's
// own K_z columns, three GMMAs per k-block, chunk promotion (fixed-order p0 + p1 added into the owner's
// fp32 registers every 16 k-tiles). fp32 plans also charge gmma_x3_words. No inventory change (formats
// 31 -> 32 add 0 bytes).
inline constexpr int execution_levels = 4;       // block, cluster, gpu, node
inline constexpr int execution_c_buckets = 10;   // c in {1,2,3,4,8,16,32,64,128,other}
inline constexpr int execution_product_kinds = 7;
inline constexpr const char* execution_catalog =
    "native-inplace-ge-tt-radix-coop-perlevel-physical-v26";
}
