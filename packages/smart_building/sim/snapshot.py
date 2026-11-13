"""Mid-episode save and restore of an official simulator environment.

For the PPO trainer's mid-episode checkpoints (learned/ppo_env.py) and the
per-state teacher choice of notes/09: save a running environment (together
with any policy objects and the last observation), restore it later or in
another process, and continue it so that it replays bit-identically.
Feasibility and the three conditions below were established by
tests/test_env_snapshot.py (300 steps of building_train_1/atlanta_2 after a
300-step prefix, gateway-style env.reset / env.step loop, official pid_water;
reviewer round 2, 2026-10-16).

Method (pickle):
  1. The scipy SuperLU factorization cached by the finite-difference solver
     (smart_control/simulator/finite_difference_solver.py:832, 993-1003) cannot
     be pickled. It is dropped; the restored solver finds no factorization and
     refactorizes the same matrix on its next step, which replays bit for bit.
  2. The official bidict (smart_control/utils/bidict.py) is a dict subclass whose
     __setitem__ needs self._inv; plain unpickling sets items before attributes
     and fails, so it is rebuilt through its constructor.
  3. Three process-global random states are not part of the environment object
     and must be saved and restored with it: Python's random module and numba's
     internal generator (both seeded and used by StochasticConvectionSimulator,
     stochastic_convection_simulator.py:12, 22-24, 114-118) and numpy's global
     generator (kept for completeness). Without them a copy diverges from its
     second step.

Risk: numba's generator state is reached through numba's private module
numba._helperlib (rnd_get_np_state_ptr, rnd_get_state, rnd_set_state). It was
verified with numba 0.61.2, the version of the project environment and of the
Kaggle image (docs/環境與執行.md); a different numba version may move or change
that API, so a warning is issued and tests/test_env_snapshot.py must be re-run.

Several copies in one process: the random states are global, so each copy must
step with its own state swapped in. RandomStateSlot does that:

    snapshot = take((env, policy, obs))
    copies = [restore(snapshot, set_random_state=False) for _ in range(2)]
    slots = [RandomStateSlot(snapshot.random_state) for _ in copies]
    for _ in range(steps):
        for (env_i, policy_i, obs_i), slot in zip(copies, slots):
            with slot:
                ...  # step env_i with policy_i

Snapshots must be taken and restored with the same simulator copy (GitHub
starter kit or the Kaggle data mount) on the import path.
"""
from dataclasses import dataclass
import io
import pickle
import random
import warnings

import numpy as np

NUMBA_VERSION_VERIFIED = "0.61.2"
_checked_numba = False


def _helperlib():
    global _checked_numba
    import numba
    import numba._helperlib as helperlib
    if not _checked_numba:
        _checked_numba = True
        if numba.__version__ != NUMBA_VERSION_VERIFIED:
            warnings.warn(f"smart_building.sim.snapshot was verified with numba {NUMBA_VERSION_VERIFIED}; "
                          f"numba {numba.__version__} may change the private random-state API. "
                          "Re-run tests/test_env_snapshot.py.", RuntimeWarning, stacklevel=3)
    return helperlib


def global_random_state():
    """(Python random, numpy global, numba internal) generator states."""
    helperlib = _helperlib()
    return (random.getstate(), np.random.get_state(),
            helperlib.rnd_get_state(helperlib.rnd_get_np_state_ptr()))


def set_global_random_state(state):
    helperlib = _helperlib()
    python_state, numpy_state, numba_state = state
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    helperlib.rnd_set_state(helperlib.rnd_get_np_state_ptr(), numba_state)


def _superlu_type():
    import scipy.sparse
    import scipy.sparse.linalg
    return type(scipy.sparse.linalg.splu(scipy.sparse.eye(2, format="csc")))


def _official_bidict_type():
    from smart_buildings.smart_control.utils.bidict import bidict
    return bidict


def _dropped_factorization():
    return None


class _SnapshotPickler(pickle.Pickler):
    """Drops SuperLU factorizations and rebuilds the official bidict (conditions 1, 2)."""

    def __init__(self, stream):
        super().__init__(stream, protocol=pickle.HIGHEST_PROTOCOL)
        self._superlu = _superlu_type()
        self._bidict = _official_bidict_type()

    def reducer_override(self, obj):
        if isinstance(obj, self._superlu):
            return (_dropped_factorization, ())
        if type(obj) is self._bidict:
            return (self._bidict, (dict(obj),))
        return NotImplemented


@dataclass(frozen=True)
class Snapshot:
    payload: bytes          # pickled objects
    random_state: tuple     # global_random_state() at the moment of the snapshot

    @property
    def megabytes(self):
        return len(self.payload) / 1e6


def take(objects):
    """Snapshot `objects` (for example (env, policy, last_observation)) and the global random state."""
    random_state = global_random_state()
    stream = io.BytesIO()
    _SnapshotPickler(stream).dump(objects)
    return Snapshot(stream.getvalue(), random_state)


def restore(snapshot, *, set_random_state=True):
    """New copies of the snapshotted objects.

    With set_random_state (the default, for one copy) the global random state
    is set to the snapshot's; pass False when several copies will step in turn
    under RandomStateSlot.
    """
    if set_random_state:
        set_global_random_state(snapshot.random_state)
    return pickle.loads(snapshot.payload)


class RandomStateSlot:
    """The global random state of one copy, swapped in for the duration of a `with` block."""

    def __init__(self, state=None):
        self.state = global_random_state() if state is None else state
        self._outer = None

    def __enter__(self):
        self._outer = global_random_state()
        set_global_random_state(self.state)
        return self

    def __exit__(self, *exc_info):
        self.state = global_random_state()
        set_global_random_state(self._outer)
        self._outer = None
        return False
