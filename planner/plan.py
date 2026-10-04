"""Select the schedule of a factorization from a machine profile.

    schedule* = argmin { T(schedule; machine, problem) : schedule admissible }

T is the cost model of model.py and admissibility is schedule.py: c >= 2 on every carried product,
replicas and buffers within the capacity of their memory level. The parameters that interact
(aggregation, look-ahead, panel blocks, strip) are searched exhaustively, the others by coordinate
descent from the best points. Across GPUs the node-level carriers are fixed by the algorithm and
the machine selects the panel blocks of every GPU.
"""

import json
from pathlib import Path

from machine import Tuner

from .model import factor_time
from .schedule import (
    MAX_GROUPS,
    MODES,
    Schedule,
    admissible,
    local_rows,
    panel_window,
    static_shared,
)

KERNELS = Path(__file__).resolve().with_name("kernels.json")
SAMPLES = 48  # units of the factorization the model evaluates while searching
STRIPS = (4096, 8192, 16384, 32768)
GROUPS = (32, 48, 64, 80, 96, MAX_GROUPS)

# Across GPUs: panels and block-cyclic blocks of 512 rows, one look-ahead lane of two strips.
NODE = Schedule(b=512, lookahead=True, strip=8192, depth=2, minipanel=8, window=32)


def load_kernels(path=KERNELS):
    return json.loads(Path(path).read_text())


def panel_width(machine, mode, n):
    """Largest power of two b <= 128 whose b x b triangular factor fits the shared memory of a
    block."""
    b = 1
    while 2 * b <= min(128, n) and (2 * b) ** 2 * MODES[mode].word <= machine.capacity("shared"):
        b *= 2
    return b


def late_heights(machine, kernels, mode, m, b):
    """Panels below these heights still hold a window of a mini-panel with 32 blocks."""
    heights = set()
    for minipanel in (8, 16):
        for window in (16, 32, 64):
            if window < minipanel:
                continue
            limit = int(machine.capacity("shared")) - static_shared(kernels, mode, minipanel)
            local = (limit // MODES[mode].word - minipanel * window) // (window + 1)
            heights.add(local * 32 // b * b)
    return sorted(h for h in heights if 8192 < h < m)


def space(machine, kernels, mode, m, n):
    """Candidate values of every parameter for this problem."""
    b = panel_width(machine, mode, n)
    panels = -(-n // b)
    tall = m > 8192
    strips = {s for s in STRIPS if s < n - b}
    if n - b <= 16384:
        strips.add(max(64, n - b))
    late = (
        [(0, 0)] + [(32, rows) for rows in late_heights(machine, kernels, mode, m, b)]
        if tall
        else [(0, 0)]
    )
    return {
        "b": [b],
        "aggregate": [a for a in (1, 2, 4, 8) if a == 1 or 2 * a <= panels],
        "lookahead": [False, True] if panels >= 2 else [False],
        "groups": list(GROUPS) if tall else [0],
        "late": late,
        "strip": sorted(strips),
        "depth": [1, 2],
        "wc": [2, 4, 8],
        "zc": [2, 4],
        "minipanel": [16, 8] if tall else [16],
        "window": [64, 32, 0] if tall else [64],
    }


def to_schedule(config, mode, m):
    values = dict(config)
    late_groups, late_rows = values.pop("late")
    compose = 64 if mode == "tf32" and m >= 48000 else 0
    return Schedule(late_groups=late_groups, late_rows=late_rows, compose_groups=compose, **values)


def tuner(machine, kernels, mode, m, n, samples=SAMPLES):
    """The search problem: parameters, restrictions and the model as the objective."""

    def restriction(config):
        s = to_schedule(config, mode, m)
        if s.late_groups and s.late_groups >= s.groups:
            return False
        return admissible(machine, kernels, mode, m, n, s)[0]

    def objective(config):
        return factor_time(
            machine, kernels, mode, m, n, to_schedule(config, mode, m), samples=samples
        )

    return Tuner(space(machine, kernels, mode, m, n), objective, [restriction])


def footprint(s):
    """How much a schedule replicates and buffers: the order that settles ties of the model."""
    return (s.lookahead, s.aggregate, s.depth, s.wc + s.zc, s.groups, s.late_groups)


def plan(machine, mode, m, n, kernels=None, top=1, indifference=0.01):
    """Schedules ranked by predicted time, [(seconds, Schedule)], best first.

    Predictions closer than `indifference` are below the resolution of the model; among them the
    schedule that replicates and buffers least comes first."""
    kernels = kernels or load_kernels()
    t = tuner(machine, kernels, mode, m, n)
    interacting = ("aggregate", "lookahead", "groups", "late", "strip")
    start = {key: values[0] for key, values in t.params.items()}
    start.update(wc=4 if 4 in t.params["wc"] else 2)
    coarse = Tuner(
        {key: values if key in interacting else [start[key]] for key, values in t.params.items()},
        t.objective,
        t.restrictions,
    )
    candidates = {}
    for _, config in coarse.rank(top=max(4 * top, 12)):
        for refined in (config, t.descend(config)[1]):
            s = to_schedule(refined, mode, m)
            if s not in candidates:
                candidates[s] = factor_time(machine, kernels, mode, m, n, s)
    best = min(candidates.values())

    def rank(item):
        s, seconds = item
        return (0, footprint(s)) if seconds <= best * (1 + indifference) else (1, seconds)

    return [(seconds, s) for s, seconds in sorted(candidates.items(), key=rank)[:top]]


def node_groups(machine, kernels, mode, m, gpus):
    """Smallest number of panel blocks whose window fits the shared memory of a block, for the rows
    one GPU holds. Fewer blocks would not hold the window; more would take SMs from the far update
    that runs beside the panel."""
    rows = local_rows(m, gpus, NODE.b)
    for groups in GROUPS:
        width = panel_window(
            machine, kernels, mode, rows, groups, NODE.minipanel, NODE.window, NODE.b
        )
        if width == NODE.window:
            return groups
    return GROUPS[-1]


def node_schedule(machine, kernels, mode, m, gpus):
    """The schedule across GPUs: W = V^T X of the TT update runs with the carrier (1, 1, P) and
    D = V Z with (P, 1, 1), so the machine decides the panel carrier inside each GPU."""
    compose = 64 if mode == "tf32" else 0
    return NODE.replace(groups=node_groups(machine, kernels, mode, m, gpus), compose_groups=compose)


def case(machine, mode, n, gpus=1, reps=3, kernels=None):
    """A runner case (reproducers/presets.json format) with the selected schedule."""
    kernels = kernels or load_kernels()
    if gpus > 1:
        s, predicted = node_schedule(machine, kernels, mode, n, gpus), None
    else:
        predicted, s = plan(machine, mode, n, n, kernels)[0]
    selected = {"gpus": gpus, "mode": mode, "n": n, "reps": reps, "options": s.options()}
    if predicted:
        selected["predicted_time_s"] = predicted
    return selected
