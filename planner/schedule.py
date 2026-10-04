"""The schedules of the QR-Omega engine, which of them are admissible, and their driver options.

A schedule is admissible when it is faithful to the algorithm and fits the machine:
  * every carried product (W, Z, D) and the panel product split their contraction among c >= 2
    peers;
  * the replicas, partial results and pipeline buffers fit the memory level that holds them: the
    device memory for the strips in flight, the shared memory of one block for a panel window.
The elimination list is the engine's: one GE per panel on every GPU and, across GPUs, one TT
elimination of arity P.
"""

from dataclasses import asdict, dataclass, fields
from typing import NamedTuple


class Mode(NamedTuple):
    arithmetic: str  # arithmetic of the vendor product the probe measured
    panel: str  # arithmetic of the panel kernels
    word: int  # bytes
    passes: int  # products per logical product


MODES = {
    "fp64": Mode("fp64", "fp64", 8, 1),
    "fp32": Mode("fp32", "fp32", 4, 1),
    "tf32": Mode("tf32", "fp32", 4, 1),
    "3xtf32": Mode("tf32", "fp32", 4, 3),
}

REGISTER_ROWS = 8192  # tallest panel the register kernels hold
MAX_GROUPS = 128  # most blocks a cooperative panel is split among
WINDOWS = (64, 32, 16, 8)
BLAS_WORKSPACE = 32 << 20  # cuBLAS workspace of every lane


@dataclass(frozen=True)
class Schedule:
    b: int = 128  # panel width
    aggregate: int = 1  # panels composed into one far update
    lookahead: bool = False  # factor the next panel while the far update runs
    strip: int = 4096  # columns of a strip of the trailing update
    depth: int = 1  # strips in flight
    wc: int = 2  # contraction replication of W = V^T X
    zc: int = 2  # and of Z = T^T W (D = V Z always splits in two)
    groups: int = 32  # blocks of a cooperative panel
    minipanel: int = 16  # columns a cooperative panel holds in registers
    window: int = 64  # columns it holds in shared memory (0: the whole panel)
    late_groups: int = 0  # blocks of the panels with fewer than late_rows rows
    late_rows: int = 0
    compose_groups: int = 0  # TF32: blocks of the Gram products of an aggregated T
    d_tiles: int = 1  # 3xTF32: output tiles per block of the far D

    def replace(self, **changes):
        return Schedule(**{**asdict(self), **changes})

    def options(self):
        """Driver options, in the format of reproducers/presets.json."""
        options = {key.replace("_", "-"): value for key, value in asdict(self).items()}
        defaults = Schedule()
        for key in ("late_groups", "late_rows", "compose_groups", "d_tiles", "lookahead"):
            if getattr(self, key) == getattr(defaults, key):
                del options[key.replace("_", "-")]
        return options

    @classmethod
    def from_options(cls, options):
        names = {field.name for field in fields(cls)}
        values = {key.replace("-", "_"): value for key, value in options.items()}
        unknown = set(values) - names
        if unknown:
            raise ValueError(f"unknown driver options {sorted(unknown)}")
        return cls(**values)


def static_shared(kernels, mode, minipanel):
    """Bytes of shared memory the cooperative panel kernel declares statically."""
    return kernels["panel"][MODES[mode].panel]["static_shared_bytes"][str(minipanel)]


def panel_window(machine, kernels, mode, rows, groups, minipanel, window, h=128):
    """Width of the window the engine gives a cooperative panel: the widest candidate whose slab,
    ceil(rows / groups) (window + 1) + minipanel window words, fits the shared memory of a block.
    Zero when none does."""
    local = -(-rows // groups)
    limit = machine.capacity("shared") - static_shared(kernels, mode, minipanel)
    widest = min(h, window or h)
    for width in [widest] + [w for w in WINDOWS if minipanel <= w < widest]:
        if (local * (width + 1) + minipanel * width) * MODES[mode].word <= limit:
            return width
    return 0


def register_panel(rows, h):
    return h <= 128 and rows <= REGISTER_ROWS


def local_rows(m, gpus, b):
    """Rows of the GPU that holds the most in a block-cyclic distribution of blocks of b rows."""
    blocks = -(-m // b)
    return min(m, -(-blocks // gpus) * b)


def memory_bytes(mode, m, n, s, gpus=1):
    """Device memory one GPU needs for an m x n factorization: the matrix, the buffers of the
    engine (engine.cuh, update.cuh, transport.cuh) and the validation of the driver."""
    b, h, q = s.b, s.b * s.aggregate, s.strip
    rows = local_rows(m, gpus, b)
    ldv = -(-rows // 128) * 128
    panels = -(-min(m, n) // b)
    lanes = s.depth * (2 if s.lookahead else 1)
    generations = 2 if s.lookahead else 1
    tiles = panels if gpus == 1 else panels + -(-panels // gpus)
    words = rows * n + tiles * b * b + n + q
    words += max(generations, s.aggregate) * ldv * b
    words += 130 * b * b + b
    if s.groups:
        groups = max(s.groups, s.late_groups)
        words += groups * max(4 + 4 * s.minipanel, b) + (groups + 1) * s.minipanel * b
    if s.aggregate > 1:
        words += generations * (ldv * h + 2 * h * h) + 2 * s.aggregate * b * b
    words += lanes * (s.wc + s.zc) * h * q
    if mode == "tf32":
        words += lanes * ldv * h
    elif mode == "3xtf32":
        words += lanes * (3 * ldv * h + 3 * h * q)
    if gpus > 1:
        payload = max(b * b, b * q)
        words += (4 * gpus + 1) * b * b + lanes * b * q + (4 * gpus + 2) * payload
    words += 2 * rows * min(q, 512)
    return words * MODES[mode].word + lanes * BLAS_WORKSPACE


def faithful(s):
    """c >= 2 on every carried product and on the panel product."""
    return s.wc >= 2 and s.zc >= 2 and s.groups != 1 and s.late_groups != 1


def admissible(machine, kernels, mode, m, n, s, gpus=1, reserve=0.02):
    """Faithful and within capacity. Returns (ok, reason)."""
    if not faithful(s):
        return False, "a carried product would run with c = 1"
    if min(s.strip, s.depth, s.aggregate) < 1 or s.b > (512 if gpus > 1 else 128):
        return False, "outside the engine"
    if s.groups > MAX_GROUPS or s.late_groups > MAX_GROUPS:
        return False, "more panel blocks than the kernel partitions"
    rows = local_rows(m, gpus, s.b)
    late = [(s.late_rows - 1, s.late_groups)] if s.late_groups > 1 and s.late_rows > 1 else []
    for height, groups in [(rows, s.groups)] + late:
        if register_panel(height, s.b):
            continue
        if not groups or not panel_window(
            machine, kernels, mode, height, groups, s.minipanel, s.window, s.b
        ):
            return False, "the panel window does not fit the shared memory of a block"
    if memory_bytes(mode, m, n, s, gpus) > (1 - reserve) * machine.memory_free:
        return False, "replicas and pipeline buffers exceed the device memory"
    return True, ""
