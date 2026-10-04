"""Cost model of a QR-Omega factorization on one GPU, from a machine profile.

The factorization is a sequence of units of `aggregate` panels. A unit costs

    M  the main lane: its panel factorizations, the applies of earlier panels of the unit to the
       columns of later ones (near), and the composition of its reflectors into one transform;
    A  the far update of the columns the next unit factors (look-ahead only);
    F  the far update of the remaining columns.

Without look-ahead a unit takes M + F. With look-ahead the next unit's main lane overlaps this
unit's F on the SMs the panel leaves free, so a cycle takes A + overlap(M, F). Every term is a
count of latency events, bytes over a bandwidth, or operations over a rate of the profile, weighted
by the dimensionless kernel constants of kernels.json.
"""

import math
from typing import NamedTuple

from .schedule import MODES, panel_window, register_panel


class Unit(NamedTuple):
    panel: float  # seconds of its panels
    other: float  # other main-lane seconds: near applies and composition
    window: float  # update of the next unit's columns (look-ahead)
    far: float  # update of the remaining columns
    sms: int  # SMs the panels occupy
    launches: int  # panel launches
    strip: float  # seconds of one strip of the far update
    near: float
    compose: float


def sync_latency(machine):
    """One handshake between thread blocks: a block barrier and a round trip through L2."""
    return machine.barrier(512) + machine.latency("l2")


def register_events(table, rows):
    """Handshakes per column of a register panel, interpolated in log2(rows) between heights."""
    knots = sorted((int(rows), events) for rows, events in table.items())
    if rows <= knots[0][0]:
        return knots[0][1]
    for (r0, e0), (r1, e1) in zip(knots, knots[1:]):
        if rows <= r1:
            return e0 + (e1 - e0) * math.log2(rows / r0) / math.log2(r1 / r0)
    return knots[-1][1]


def panel_time(machine, kernels, mode, rows, b, groups, minipanel, window):
    """Seconds of one panel factorization of b columns, the SMs it occupies and its launches.

    A register panel is a chain of handshakes per column. A cooperative panel splits its rows
    among `groups` blocks: the same chain, plus the work on the window each block holds in shared
    memory, plus the update of the rest of the panel at every window boundary, which is bound by
    the bandwidth of device memory."""
    k = kernels["panel"][MODES[mode].panel]
    sync = sync_latency(machine)
    if register_panel(rows, b):
        sms = min(machine.sms, max(16, rows // 128))
        return b * register_events(k["register"], rows) * sync, sms, 1
    c = k["cooperative"]
    width = panel_window(machine, kernels, mode, rows, groups, minipanel, window, b)
    if not width:
        return math.inf, groups, 1
    word = MODES[mode].word
    boundaries = b / width - 1
    seconds = (
        b * c["handshakes"] * sync
        + boundaries * c["launches"] * machine.launch_sync
        + c["shared"]
        * (rows / groups)
        * b
        * width
        * word
        / machine.levels["shared"].bandwidth_per_sm
        + c["boundary"] * rows * b * boundaries * word / machine.levels["memory"].bandwidth_per_sm
    )
    return seconds, groups, -(-b // width)


def apply_time(machine, kernels, mode, rows, k, q, s, depth=1):
    """Seconds to apply a transform of rank k to q columns of `rows` rows, in strips of s.strip:
    per strip W = V^T X with s.wc peers, Z = T^T W, X -= V Z.

    The arithmetic costs gamma per operation, weighted by `eta`. The product law adds the traffic
    of a short contraction (kappa / k) and of a small output (kappa xi (1/m + 1/n)), weighted by
    `shape`; the peers of W share its small output. The remaining terms are the share of W that
    only peers recover, the reads of Z that leave L2, the additive combines in device memory, and
    the launches of every further strip."""
    if min(q, rows, k) <= 0:
        return 0.0
    word, passes = MODES[mode].word, MODES[mode].passes
    law = machine.product(MODES[mode].arithmetic)
    c = kernels["apply"][mode]
    strips = -(-q // s.strip)
    width = q / strips
    fw = fd = 2.0 * k * width * rows
    fz = k * width * k
    small = law.xi * (1.0 / k + 1.0 / width)
    traffic = (
        fw * (s.wc / rows + small / s.wc)
        + fz * (1.0 / k + small)
        + fd * (1.0 / k + law.xi * (1.0 / rows + 1.0 / width))
    )
    unsplit = c["peers"] * fw / s.wc
    work = (
        passes
        * law.gamma
        * (c["eta"] * (fw + fz + fd) + c["shape"] * law.kappa * traffic + unsplit)
    )
    # Every block of D reads the strip of Z it multiplies, rows / 128 times. While k x width words
    # fit the quarter of L2 that Z shares with the blocks of V and X, those reads stay in the cache.
    z_bytes = k * width * word
    miss = max(0.0, 1.0 - 0.25 * machine.capacity("l2") / z_bytes)
    reread = (
        c["reread"]
        * (rows / 128.0)
        * z_bytes
        * miss
        * (1.0 / machine.bandwidth("memory") - 1.0 / machine.bandwidth("l2"))
    )
    combine = (s.wc + s.zc) * k * width * word / machine.copy_bandwidth
    strip = (work + passes * reread + combine) * (1.0 + c["depth"] * (depth - 1))
    return strips * strip + (strips - 1) * c["strip"] * machine.launch_sync


def compose_time(machine, kernels, mode, rows, b):
    """Seconds to add one panel to a composed transform: a Gram block and its triangular products."""
    law = machine.product(MODES[mode].arithmetic)
    c = kernels["apply"][mode]
    gram = MODES[mode].passes * (law.time(b, b, rows) - law.call)
    return c["compose"] * gram + c["compose_launches"] * machine.launch_sync


def overlap(unit, far, machine, o):
    """Elapsed time of a unit's main lane beside the far update of the previous unit.

    A panel beside a far update runs `panel_slowdown` times slower, and the far update proceeds on
    the share `free` of the SMs the panel leaves. Every panel launch first waits for the kernel of
    the far update in flight to release the SMs it needs, a share `drain` of a strip for all of
    them. The other main-lane kernels use the whole device, so the far update waits for them; what
    is left of it when the main lane ends runs alone. Forking and joining the lanes costs `joins`
    joins."""
    panel, other = unit.panel, unit.other
    if far <= 0.0:
        return panel + other
    free = max(0.0, 1.0 - unit.sms / machine.sms)
    slowdown = o["panel_slowdown"]
    hidden = slowdown * panel * free * o["efficiency"]
    fork = o["joins"] * machine.join
    wait = o["drain"] * unit.launches * (1.0 - free) * min(unit.strip, far)
    if far >= hidden:
        return slowdown * panel + other + far - hidden + fork + wait
    return panel * (1.0 + (slowdown - 1.0) * far / hidden) + other + fork + wait


def unit_terms(machine, kernels, mode, m, n, s, first, panels):
    """The terms of the unit that starts at panel `first`."""
    b, group = s.b, min(s.aggregate, panels - first)
    rows = m - first * b
    panel = near = compose = 0.0
    sms = launches = 0
    for i in range(group):
        height = m - (first + i) * b
        groups = s.late_groups if s.late_groups > 1 and height < s.late_rows else s.groups
        seconds, used, count = panel_time(
            machine,
            kernels,
            mode,
            height,
            min(b, n - (first + i) * b),
            groups,
            s.minipanel,
            s.window,
        )
        panel += seconds * kernels["apply"][mode]["panel"]
        sms = max(sms, used)
        launches += count
        if i:
            near += i * apply_time(machine, kernels, mode, rows, b, b, s)
            compose += i * compose_time(machine, kernels, mode, rows, b)
    end = min(n, (first + group) * b)
    rest = n - end
    rank = group * b
    window = 0.0
    if s.lookahead and first + group < panels:
        columns = min(rest, min(s.aggregate, panels - first - group) * b)
        window = apply_time(machine, kernels, mode, rows, rank, columns, s)
        rest -= columns
    far = apply_time(machine, kernels, mode, rows, rank, rest, s, s.depth)
    strip = apply_time(machine, kernels, mode, rows, rank, min(s.strip, rest), s)
    return Unit(panel, near + compose, window, far, sms, launches, strip, near, compose)


def factor_time(machine, kernels, mode, m, n, s, detail=False, samples=0):
    """Predicted seconds of the factorization of an m x n matrix with schedule s. With samples > 0
    about that many units, evenly spaced, stand for all of them."""
    panels = -(-min(m, n) // s.b)
    units = -(-panels // s.aggregate)
    stride = max(1, units // samples) if samples else 1
    total = 0.0
    parts = dict.fromkeys(("panel", "near", "compose", "window", "far", "hidden"), 0.0)
    pending = 0.0
    for index in range(0, units, stride):
        weight = min(stride, units - index)
        unit = unit_terms(machine, kernels, mode, m, n, s, index * s.aggregate, panels)
        if s.lookahead:
            if stride > 1 and index:
                pending = unit_terms(
                    machine, kernels, mode, m, n, s, (index - 1) * s.aggregate, panels
                ).far
            cycle = overlap(unit, pending, machine, kernels["overlap"][mode])
            parts["hidden"] += weight * (unit.panel + unit.other + pending - cycle)
            total += weight * (cycle + unit.window)
            pending = unit.far
        else:
            total += weight * (unit.panel + unit.other + unit.far)
        for key in ("panel", "near", "compose", "window", "far"):
            parts[key] += weight * getattr(unit, key)
    return (total, parts) if detail else total
