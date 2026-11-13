"""CMA-ES as ask/tell with a checkpoint after every generation (D-P03; remote plan §12 item 2).

D-P03: pycma 4.5.0; popsize 4 + floor(3 ln n) (pycma default); parameters normalized to
[0, 1] with BoundTransform (pycma's boundary handler when "bounds" is given, options_parameters
.py:69-70); x0 uniform in [0, 1]^n from a fixed, recorded seed; sigma0 = 0.3 x range width, i.e.
0.3 in normalized units (Hansen's tutorial, notes/04 O); every other option pycma's default, as in
cma.fmin2 with restarts=0 and no noise handler (notes/04 P). Changed only: output options
(verbose, verb_log, verb_disp: no files, no printing), pycma's own seed (fixed and recorded), and,
when the job gives a generation budget (D-P05), pycma's "maxiter" option.

The loop is the ask/tell form of cma.fmin2(objective, x0, 0.3, options): fmin2 runs
ask_and_eval (ask, then evaluate in order, resampling only non-finite values) and tell until
es.stop(), then evaluates the final mean ("eval_final_mean", default on,
evolution_strategy.py:5197-5204). run() does the same, so the same points are evaluated in the
same order (tests/test_cmaes_loop.py).

RNG: pycma seeds and samples numpy's global RNG (it uses the "seed" option only when "randn" is
np.random.randn, evolution_strategy.py:1122-1131). The strategy keeps that RNG state itself,
restores it before every ask, tell and final evaluation and saves it after, so other code in the
process cannot change the trajectory. Pickling keeps the np.random.randn reference by identity
(pickling the bound method would copy numpy's global RandomState), so a checkpoint resumes the
identical trajectory, in this process or another.
"""
import hashlib
import io
import math
import pickle

import numpy as np

SIGMA0 = 0.3  # D-P03: 0.3 x range width in normalized units
PYCMA_VERSION = "4.5.0"  # D-P03
_RANDN = "numpy.random.randn"


def options(n, cma_seed, max_generations=None):
    """pycma defaults plus bounds [0, 1]^n, a fixed seed, no output and an optional generation cap."""
    opts = {"bounds": [[0.0] * n, [1.0] * n], "seed": int(cma_seed),
            "verbose": -9, "verb_log": 0, "verb_disp": 0}
    if max_generations is not None:
        opts["maxiter"] = int(max_generations)
    return opts


def initial_point(n, x0_seed):
    """x0 uniform in [0, 1]^n from a recorded seed (D-P03; notes/04 O)."""
    return np.random.default_rng(int(x0_seed)).uniform(0.0, 1.0, size=n).tolist()


def default_popsize(n):
    """pycma's default population size, 4 + floor(3 ln n) (options_parameters.py:87)."""
    return 4 + int(math.floor(3 * math.log(n)))


def denormalize(x, parameters):
    """Map a point of [0, 1]^n to parameter values (each parameter: name, low, high)."""
    return {p["name"]: p["low"] + float(v) * (p["high"] - p["low"]) for v, p in zip(x, parameters)}


class _Pickler(pickle.Pickler):
    def persistent_id(self, obj):
        return _RANDN if obj is np.random.randn else None


class _Unpickler(pickle.Unpickler):
    def persistent_load(self, pid):
        if pid == _RANDN:
            return np.random.randn
        raise pickle.UnpicklingError(f"unknown persistent id {pid!r}")


class Strategy:
    """A pycma strategy with its own RNG state, checkpointable after every generation."""

    def __init__(self, n, x0_seed, cma_seed, max_generations=None):
        import cma
        if cma.__version__ != PYCMA_VERSION:
            raise RuntimeError(f"pycma {cma.__version__} installed; D-P03 fixes {PYCMA_VERSION}")
        saved = np.random.get_state()
        self.es = cma.CMAEvolutionStrategy(initial_point(n, x0_seed), SIGMA0,
                                           options(n, cma_seed, max_generations))
        self.rng_state = np.random.get_state()  # pycma seeded the global RNG at construction
        np.random.set_state(saved)
        self.generation = 0
        self.finished = False
        self.final_mean = None
        self.records = []
        self.settings = {"n": n, "x0_seed": int(x0_seed), "cma_seed": int(cma_seed), "sigma0": SIGMA0,
                         "popsize": self.es.popsize, "max_generations": max_generations,
                         "pycma": cma.__version__}

    def _with_rng(self, action):
        saved = np.random.get_state()
        np.random.set_state(self.rng_state)
        try:
            return action()
        finally:
            self.rng_state = np.random.get_state()
            np.random.set_state(saved)

    def stopped(self):
        return bool(self.es.stop())

    def ask(self):
        return [np.asarray(x, dtype=float) for x in self._with_rng(self.es.ask)]

    def tell(self, points, fitness):
        self._with_rng(lambda: self.es.tell(points, [float(f) for f in fitness]))
        self.generation += 1
        self.records.append({"generation": self.generation,
                             "points": [list(map(float, x)) for x in points],
                             "fitness": [float(f) for f in fitness],
                             "sigma": float(self.es.sigma),
                             "best_fitness": float(self.es.result.fbest)})

    def final_mean_point(self):
        """The final mean in phenotype space, as fmin computes it (evolution_strategy.py:5199-5201)."""
        es = self.es
        return self._with_rng(lambda: np.asarray(es.to_phenotype(
            es.mean, copy=True, into_bounds=es.boundary_handler.repair, archive=es.sent_solutions), dtype=float))

    def record_final_mean(self, point, value):
        """fmin's bookkeeping after evaluating the final mean (evolution_strategy.py:5202-5204)."""
        es = self.es
        es.countevals += 1
        self._with_rng(lambda: es.best.update([point], es.sent_solutions, [float(value)], es.countevals))
        self.final_mean = {"point": list(map(float, point)), "fitness": float(value)}
        self.finished = True

    def result(self):
        r = self.es.result
        return {"xbest": [float(v) for v in r.xbest], "fbest": float(r.fbest),
                "evaluations": int(r.evaluations), "iterations": int(r.iterations),
                "xfavorite": [float(v) for v in r.xfavorite], "final_mean": self.final_mean,
                "stop": {k: (v if isinstance(v, (int, float, str)) else repr(v)) for k, v in self.es.stop().items()}}

    def dumps(self):
        buffer = io.BytesIO()
        _Pickler(buffer, protocol=pickle.HIGHEST_PROTOCOL).dump(self)
        return buffer.getvalue()

    @staticmethod
    def loads(data):
        return _Unpickler(io.BytesIO(data)).load()


def save_checkpoint(path, strategy):
    """Write the pickled strategy (with its RNG state); returns its SHA-256."""
    data = strategy.dumps()
    temporary = path.with_name(path.name + ".partial")
    temporary.write_bytes(data)
    temporary.replace(path)
    return hashlib.sha256(data).hexdigest()


def load_checkpoint(path):
    return Strategy.loads(path.read_bytes())


def run(strategy, evaluate, keep_going=lambda: True, after_generation=None, evaluate_final=None,
        after_final=None):
    """ask -> evaluate -> tell until pycma stops, then evaluate the final mean, as cma.fmin2 does.

    keep_going() is asked before every generation and before the final mean; when it says no the
    run returns early and the strategy (checkpointed by after_generation) can be resumed later.
    evaluate(points) returns one fitness per point; evaluate_final(point) defaults to evaluate([point])[0].
    """
    while not strategy.stopped():
        if not keep_going():
            return strategy
        points = strategy.ask()
        fitness = evaluate(points)
        strategy.tell(points, fitness)
        if after_generation:
            after_generation(strategy, points, fitness)
    if strategy.finished:
        return strategy
    if not strategy.es.opts["eval_final_mean"]:
        strategy.finished = True
        return strategy
    if not keep_going():
        return strategy
    point = strategy.final_mean_point()
    value = evaluate_final(point) if evaluate_final else evaluate([point])[0]
    strategy.record_final_mean(point, value)
    if after_final:
        after_final(strategy, point, value)
    return strategy
