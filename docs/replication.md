# Replication, storage and concurrent execution

Hierarchical Householder QR and 2.5D scheduling jointly choose independent eliminations,
operand reuse and contraction splits. A carrier `(p_i, p_j, c)` describes an individual
product at a particular machine level. Its executed contraction split satisfies
`c² <= p_i p_j`; requests are reduced when the product shape cannot support them.
The driver's `carriers` report records the executed choices. Requested replication
and allocated workspace describe different quantities.

## What is stored

The input matrix remains distributed over grid rows and columns. Increasing a
product's contraction split does not allocate that many complete copies of the matrix.
Peers read their operand partitions, form partial outputs and combine those outputs.
An operand reused across output tiles can have copies in registers, shared memory or
device buffers, depending on the carrier. These copies and partials have distinct costs.

For a compact-WY update `W = Vᵀ X`, `Z = Tᵀ W`, `X -= V Z`, let `h` be the
number of carried reflectors, `q` the reserved strip width, `L` the number of strip
lanes and `s` the scalar size in bytes. The updater allocates:

| Buffer | Allocated bytes | Contents |
|---|---:|---|
| W | `L wc h q s` | W's contraction partials and final output; storage is reused by the combine |
| Z | `L zc h q s` | Z's contraction partials and final output |
| Packed operands | Carrier-dependent | Row-major V, or split TF32 operands, when the arithmetic requires them |

At a fixed shape and lane count, increasing `wc` by one reserves `L h q s`
additional bytes; the same formula applies to `zc`. This allocation uses the
requested capacity even when a smaller executed carrier is selected for a short
product. Effective `c` must therefore be read from the carrier report.

Tree updates additionally reserve W, partial and Z workspaces. Each has
`L (D max(1, dc, mc, tree_groups) + 8) b q_tree s` bytes, where `D` is the
maximum domain count, `b` is panel width and `q_tree` is a budget-limited strip
width. Segmented TS updates can also require a row workspace. Increasing a tree
replication request may shrink `q_tree` because the workspace budget is fixed.
Fused tree products keep operands and partial accumulators in registers/shared
memory; their warp replication does not imply extra complete V copies in HBM.

Across GPUs, the grid distributes the matrix and determines which ranks share
reflector blocks and triangular factors. The communication layer reserves operand
slots, reduction/exchange buffers and synchronization state. Their cost depends on
grid shape, factor dimensions and in-flight lanes, rather than a universal `c n²`
formula. A tree-format tail also owns its trailing matrix and update workspace.

The benchmark's `update_workspace` field reports the updater's actual buffer
allocations on rank zero. `total_bytes` sums the current updater's buffers;
`inclusive_bytes` also includes a child tail's buffers and its matrix when present. It is
not a measurement of total process peak memory: matrix storage, retained factors,
transport/runtime allocations and validation buffers must also be accounted for.
Registers/shared memory and HBM have separate capacity constraints.

Measured example (H200, FP64, `n = 8192`, `wc = 4`, `zc = 2`, `dc = mc = 1`, one request varied at a
time, full-process CUDA allocation traces): raising `wc` to 8 adds 64 MiB to the updater and to the
allocation peak, and raising `zc` to 4 adds 32 MiB. The updater holds 708 MiB, or 1670 MiB with its
tail and tail matrix; the process peak is 2340 MiB including the validation buffers. These numbers
describe this schedule, not a universal cost per executed `c`.

## How peers execute

`VendorProduct::atb` represents contraction peers as strided batches of one
cuBLASLt matmul on the strip's stream. A remaining contraction range can require
one additional matmul. A deterministic combine sums the partial outputs afterward.
Other carriers launch a grid with explicit contraction layers and combine at their
designated memory level. These are implementations of the same carried products.

Separate CUDA streams schedule different strips and overlap the critical panel
with far updates. They do not mean that each vendor contraction peer is submitted
on its own stream. Ordered communication and collectives use the low-latency NCCL;
independent local-triangle publications into disjoint merge-owner slots use NVSHMEM. Each
publication signals that its payload is ready, and reuse credits protect the staging buffer.
The merged V/T packets and reconstruction factors return through ordered NCCL exchanges.

## Scope of claims

The tunable carrier admits the replication that each level's capacity and measured costs support;
it does not establish superiority over 3D schedules in general. Borrowed kernels and the
reconstruction and communication methods keep their original attribution.
