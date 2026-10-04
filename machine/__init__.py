"""Machine profiles and cost primitives for dense linear algebra on GPUs.

A profile is measured once per GPU by ``machine_probe`` and describes the hardware only: capacity, latency and
bandwidth of every level of the memory hierarchy, arithmetic rates, and the latency of the operations that order
work. ``Machine`` turns it into the terms of a cost model, and ``Tuner`` selects the parameters of an algorithm
by minimizing such a model over a restricted search space, with an optional measured refinement.
"""

from .profile import Level, Machine
from .tuner import Tuner

__all__ = ["Machine", "Level", "Tuner"]
