"""Parameter selection by minimizing a cost model over a restricted search space.

The structure follows auto-tuners such as Kernel Tuner: named parameters with candidate values, restrictions that
remove infeasible combinations, and an objective. The difference is the objective: it is a cost model evaluated on
a machine profile, so the whole space is ranked without running anything. `refine` then measures only the few
configurations the model ranks best and keeps the fastest, which bounds the effect of model error.
"""

import itertools
import json
from pathlib import Path


class Tuner:
    def __init__(self, params, objective, restrictions=()):
        """params: {name: [values]}; objective(config) -> predicted seconds (or any cost, smaller is better);
        restrictions: callables config -> bool, or expressions over the parameter names such as "wc >= 2"."""
        self.params = {k: list(v) for k, v in params.items()}
        self.objective = objective
        self.restrictions = [self._compile(r) for r in restrictions]

    @staticmethod
    def _compile(restriction):
        if callable(restriction):
            return restriction
        code = compile(restriction, "<restriction>", "eval")
        return lambda config: bool(
            eval(code, {"__builtins__": {"min": min, "max": max, "abs": abs}}, dict(config))
        )

    def feasible(self, config):
        return all(r(config) for r in self.restrictions)

    def space(self):
        names = list(self.params)
        for values in itertools.product(*(self.params[n] for n in names)):
            config = dict(zip(names, values))
            if self.feasible(config):
                yield config

    def rank(self, top=None):
        """Feasible configurations with their predicted cost, cheapest first."""
        ranked = sorted(
            ((self.objective(c), i, c) for i, c in enumerate(self.space())), key=lambda t: t[:2]
        )
        if not ranked:
            raise ValueError("no configuration satisfies the restrictions")
        return [(cost, config) for cost, _, config in (ranked if top is None else ranked[:top])]

    def best(self):
        return self.rank(top=1)[0]

    def descend(self, start, sweeps=4):
        """Coordinate descent from `start`: minimize one parameter at a time until nothing moves. For spaces too
        large to enumerate; it finds a minimum along every axis, not necessarily the global one."""
        config = dict(start)
        if not self.feasible(config):
            raise ValueError("the starting configuration violates a restriction")
        cost = self.objective(config)
        for _ in range(sweeps):
            moved = False
            for name, values in self.params.items():
                for value in values:
                    trial = dict(config, **{name: value})
                    if value == config[name] or not self.feasible(trial):
                        continue
                    c = self.objective(trial)
                    if c < cost:
                        config, cost, moved = trial, c, True
            if not moved:
                break
        return cost, config

    def refine(self, measure, top=5, within=None, cache=None):
        """Measure the model's best configurations and return (seconds, config, table).

        measure(config) -> measured seconds, or None when the configuration fails. Candidates are the `top`
        cheapest by the model, cut to those predicted within a factor (1 + within) of the best when `within` is
        given. `cache` is a JSON file of earlier measurements, so a repeated refinement costs nothing.
        """
        ranked = self.rank(top=top)
        if within is not None:
            ranked = [x for x in ranked if x[0] <= ranked[0][0] * (1.0 + within)]
        store = {}
        path = Path(cache) if cache else None
        if path and path.exists():
            store = json.loads(path.read_text())
        table = []
        for predicted, config in ranked:
            key = json.dumps(config, sort_keys=True)
            if key not in store:
                store[key] = measure(config)
                if path:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(store, indent=1) + "\n")
            table.append({"config": config, "predicted_s": predicted, "measured_s": store[key]})
        valid = [t for t in table if t["measured_s"] is not None]
        if not valid:
            raise RuntimeError("every candidate failed when measured")
        winner = min(valid, key=lambda t: t["measured_s"])
        return winner["measured_s"], winner["config"], table
