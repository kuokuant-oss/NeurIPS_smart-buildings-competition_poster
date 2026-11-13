"""Environment side of the B1-RL PPO trainer (WP5 5.1; learned/ppo.py).

What one env slot is. CleanRL `ppo_continuous_action.py` @fe8d8a0 (MIT; D-RL01) steps `num_envs`
gymnasium environments in a vector env that resets an environment as soon as its episode ends
(L87-103 make_env, L220-223 envs.step); this module is that vector env for the official
simulator. Each slot plays one TRAIN episode after another on the Kaggle gateway's loop, the
path the project scores with (smart_building.evaluation.gateway_episode, learned/rollouts.py
load_environment): the scenario is loaded with the working directory at the scenario root
(gateway.py:308-340), env.reset() (gateway.py:186-211), then env.step(action) once per step until
terminated or truncated (gateway.py:213-248); a -inf reward is a refused action.

The rule controller (the B1-R base of D-RL03 (a) and the on/off source of D-RL04 (a)) is a new
instance per episode, built after env.reset() as the inference server builds a policy, and
called exactly once per step with the observation of that step (as the shadow teacher of
D-L14). Which class it is comes from the trainer's settings (rule_controller, REQ-021 (11); for
stage 3 v4 + readback, policies/c3_teachers.py G36PolicyC3LaaOccrowDampfloorReadback, whose
own-command state reads the executed commands back from the observation, the teacher of the
j7b-v5s students). That class is made by a factory (policies/g36.py c3_configured_class), so
its __module__ (smart_building.policies.g36) does not hold it under its name and pickle could not
find it; register_class adds the name to that module in this process before a snapshot or a
restore (the files are not changed).

An episode's end (the official 30-day step limit, or the steps_cap truncation) also returns the
last observation the environment gives with it ("final_obs"), for the bootstrap of REQ-021 (12).

B1-R internal state as network inputs (the j7b-v5 / v5s students, learned/privileged.py; student
anchor, D-RL03 (b), reviewer 2026-10-21). A slot made with privileged=True reads the rule
controller's state (privileged.policy_features) immediately *before* each of its get_action calls,
exactly as the v5s DAgger rollouts (learned/rollouts_v5.py run_rollout: features, then the shadow
teacher's call) and the v5s evaluation and ensemble policies (privileged.StudentPolicyV5,
policies/b1l_ensemble_v5.py: features, then the switches object's call) do; the controller is the
same new-per-episode instance called once per step with that step's observation, so the state is
the same deterministic function of the observation history. The features travel with the base
action ("features", {type: float32 (count, features)}) and, at an episode's end, as
"final_features" next to "final_obs" (the state before a call on the last observation).

Global random state. The simulator seeds and uses Python's random module and numba's generator
(stochastic_convection_simulator.py:12-24, 114-118), which are process globals. Every slot keeps
its own copy (smart_building.sim.snapshot.RandomStateSlot) swapped in around every simulator
call, so slots in one process do not disturb each other and a slot behaves the same in this
process or in a worker process. A slot's state between iterations is a snapshot
(sim/snapshot.py: environment, rule controller, last observation, counters and that random
state), which is how a training segment hands running episodes to the next one (stage45 §4;
tests/test_learned_ppo.py: the save / restore bit-identity tests).

Worker processes use the "spawn" start method (remote plan §3.4 step 0), one slot per process,
commands over a pipe; with in_process the slots live in the trainer's process (tests, smoke).
"""
import contextlib
import ctypes
import ctypes.util
import math
import multiprocessing
import os
from pathlib import Path
import pickle
import sys
import traceback

import numpy as np

REJECTED = -math.inf
# scripts/workspace.py local_environment(): cache and temp locations (absolute there and in the remote runner).
CACHE_VARIABLES = ("TMPDIR", "TEMP", "TMP", "XDG_CACHE_HOME", "MPLCONFIGDIR", "NUMBA_CACHE_DIR", "IPYTHONDIR",
                   "JUPYTER_RUNTIME_DIR")


def absolute_cache_paths(base):
    """Make relative cache and temp variables absolute (against `base`): a slot works with the
    working directory at the scenario root, where a relative cache path would write into the
    (read-only) simulator copy. Must run before numba or matplotlib are imported to take effect."""
    for key in CACHE_VARIABLES:
        value = os.environ.get(key)
        if value and not os.path.isabs(value):
            os.environ[key] = os.path.abspath(os.path.join(base, value))


@contextlib.contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def register_class(cls):
    """Make a factory-made class picklable by reference in this process: pickle finds a class as
    getattr(sys.modules[cls.__module__], cls.__qualname__). policies/g36.py c3_configured_class runs its class
    statement inside g36.py, so the classes of policies/c3_teachers.py report __module__
    smart_building.policies.g36 under a name that module does not have; the name is added there (only if
    it is free). Classes that are already reachable are left alone."""
    module = sys.modules.get(cls.__module__)
    if module is not None and "." not in cls.__qualname__ and getattr(module, cls.__qualname__, None) is None:
        setattr(module, cls.__qualname__, cls)
    return cls


def load_rule_controller(code_root, spec):
    """The rule controller class of a {"file", "class"} spec (rollouts.load_class), registered for pickling."""
    from smart_building.learned.rollouts import load_class
    return register_class(load_class(code_root, spec))


class _RusageInfoV4(ctypes.Structure):
    """macOS <sys/resource.h> struct rusage_info_v4 (proc_pid_rusage flavor RUSAGE_INFO_V4 = 4)."""
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(name, ctypes.c_uint64) for name in (
        "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups", "ri_pageins", "ri_wired_size",
        "ri_resident_size", "ri_phys_footprint", "ri_proc_start_abstime", "ri_proc_exit_abstime",
        "ri_child_user_time", "ri_child_system_time", "ri_child_pkg_idle_wkups", "ri_child_interrupt_wkups",
        "ri_child_pageins", "ri_child_elapsed_abstime", "ri_diskio_bytesread", "ri_diskio_byteswritten",
        "ri_cpu_time_qos_default", "ri_cpu_time_qos_maintenance", "ri_cpu_time_qos_background",
        "ri_cpu_time_qos_utility", "ri_cpu_time_qos_legacy", "ri_cpu_time_qos_user_initiated",
        "ri_cpu_time_qos_user_interactive", "ri_billed_system_time", "ri_serviced_system_time", "ri_logical_writes",
        "ri_lifetime_max_phys_footprint", "ri_instructions", "ri_cycles", "ri_billed_energy", "ri_serviced_energy",
        "ri_interval_max_phys_footprint", "ri_runnable_time")]


def process_memory(pid=None):
    """Memory of a process in MB (default: this one). macOS: the physical footprint now and its lifetime peak
    (proc_pid_rusage RUSAGE_INFO_V4; unlike ru_maxrss it counts compressed pages, docs/plan/reviews/
    WP5-throughput.md §5.1). Elsewhere: this process's peak resident set size (ru_maxrss). Never raises."""
    pid = os.getpid() if pid is None else int(pid)
    try:
        if sys.platform == "darwin":
            library = ctypes.CDLL(ctypes.util.find_library("proc") or ctypes.util.find_library("c"))
            info = _RusageInfoV4()
            if library.proc_pid_rusage(pid, 4, ctypes.byref(info)) == 0:
                return {"pid": pid, "phys_footprint_mb": info.ri_phys_footprint / 2 ** 20,
                        "peak_phys_footprint_mb": info.ri_lifetime_max_phys_footprint / 2 ** 20}
        if pid == os.getpid():
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return {"pid": pid, "peak_rss_mb": peak / 2 ** 20 if sys.platform == "darwin" else peak / 2 ** 10}
    except Exception:  # noqa: BLE001 - a measurement must not stop training
        pass
    return {"pid": pid}


class EnvSlot:
    """One stream of TRAIN episodes (see the module docstring)."""

    def __init__(self, scenario_root, code_root, rule_controller, history, steps_cap, need_base, privileged=False):
        self.scenario_root = str(scenario_root)
        self.code_root = str(code_root)
        self.rule_controller = dict(rule_controller)
        self.history = int(history)
        self.steps_cap = int(steps_cap) if steps_cap else None
        self.privileged = bool(privileged)
        self.need_base = bool(need_base) or self.privileged
        self.env = self.base = self.obs = self.base_action = None
        self.label, self.steps, self.total = None, 0, 0.0
        self.random = None

    # -- simulator calls under this slot's random state and working directory ---------------

    @contextlib.contextmanager
    def _context(self):
        with self.random, working_directory(self.scenario_root):
            yield

    def _features(self):
        """The rule controller's internal state before its next call (privileged.policy_features), or None."""
        if not self.privileged:
            return None
        from smart_building.learned.privileged import policy_features
        return policy_features(self.base)

    def _base(self):
        if self.base is None:
            return None
        features = self._features()                      # before this step's call (rollouts_v5.run_rollout order)
        action = self.base.get_action(self.obs)
        out = {"continuous": np.asarray(action["continuous_action"], dtype=np.float32),
               "discrete": np.asarray(action["discrete_action"], dtype=np.int32)}
        if features is not None:
            out["features"] = features
        return out

    def start(self, label, scenario_dir):
        """Load a scenario, reset it, build a new rule controller; returns the first observation,
        the controller's action for it and the building's token layout tables."""
        from smart_building.learned.rollouts import load_environment
        from smart_building.learned.tokens import TokenLayout
        from smart_building.sim.snapshot import RandomStateSlot
        self.close()
        # The construction seeds the simulator's generators (module docstring), so the state this
        # slot starts from does not matter; everything after it is the slot's own.
        self.random = RandomStateSlot()
        with self._context():
            env, metadata = load_environment(scenario_dir, self.scenario_root)
            spec = env.action_spec()
            observation, _ = env.reset()
            self.env, self.obs = env, np.asarray(observation, dtype=np.float32)
            self.base = (load_rule_controller(self.code_root, self.rule_controller)(metadata, spec)
                         if self.need_base else None)
            self.base_action = self._base()
        self.label, self.steps, self.total = label, 0, 0.0
        layout = TokenLayout(metadata, spec, self.history)
        continuous, discrete = spec["continuous_action"], spec["discrete_action"]
        tables = layout.spec()
        tables.update(low=np.asarray(continuous.low, np.float32), high=np.asarray(continuous.high, np.float32),
                      n_discrete=np.asarray(int(np.prod(discrete.shape))), n_fields=np.asarray(layout.n_fields))
        return {"obs": self.obs, "base": self.base_action, "layout": tables, "label": label}

    def step(self, action):
        """env.step(action); at the episode's end the environment is closed (the trainer starts the
        next scenario) and its last observation is returned as "final_obs" (REQ-021 (12)). steps_cap
        (smoke runs, tests) ends an episode as a truncation."""
        with self._context():
            observation, reward, terminated, truncated, _ = self.env.step(action)
        reward = float(reward)
        if reward == REJECTED:
            raise RuntimeError(f"{self.label} step {self.steps}: the building refused the action")
        self.steps += 1
        self.total += reward
        terminated = bool(terminated)
        truncated = bool(truncated) or (self.steps_cap is not None and self.steps >= self.steps_cap
                                        and not terminated)
        result = {"reward": reward, "terminated": terminated, "truncated": truncated}
        if terminated or truncated:
            fallbacks = getattr(self.base, "diagnostics", {}).get("fallback_steps") if self.base else None
            result["episode"] = {"label": self.label, "steps": self.steps, "total_reward": self.total,
                                 "mean_reward": self.total / self.steps, "cost": -self.total / self.steps,
                                 "terminated": terminated, "truncated": truncated,
                                 "rule_controller_fallback_steps": fallbacks}
            result["final_obs"] = np.asarray(observation, dtype=np.float32)
            if self.privileged:              # the state a call on the last observation would start from
                result["final_features"] = self._features()
            self.close()
            return result
        self.obs = np.asarray(observation, dtype=np.float32)
        with self._context():
            self.base_action = self._base()
        result.update(obs=self.obs, base=self.base_action)
        return result

    def snapshot(self):
        """Pickled sim/snapshot.Snapshot of the running episode (None between episodes)."""
        from smart_building.sim.snapshot import take
        if self.env is None:
            return None
        with self._context():
            state = take((self.env, self.base, self.obs, self.base_action, self.label, self.steps, self.total))
        return pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)

    def restore(self, payload):
        """Continue the episode of a snapshot() payload. The snapshot pickles the rule controller, whose class
        lives in the module rollouts.load_class registers under a generated name (or, for a factory-made class,
        under the name register_class adds); a new process (every later segment of a chain) loads and registers
        it before unpickling (docs/plan/reviews/WP5-throughput.md §6.1)."""
        from smart_building.sim.snapshot import RandomStateSlot, restore
        self.close()
        if self.need_base:
            load_rule_controller(self.code_root, self.rule_controller)
        state = pickle.loads(payload)
        self.random = RandomStateSlot(state.random_state)
        with self._context():
            objects = restore(state, set_random_state=False)
        (self.env, self.base, self.obs, self.base_action, self.label, self.steps, self.total) = objects
        return {"obs": self.obs, "base": self.base_action, "label": self.label, "steps": self.steps}

    def memory(self):
        """process_memory() of the process that holds this slot, and whether torch is loaded there (a worker never
        loads it: the simulator and the rule controller do not use it, so the trainer's thread settings do not
        reach the workers)."""
        return {**process_memory(), "torch_loaded": "torch" in sys.modules}

    def close(self):
        if self.env is not None:
            with self._context():
                self.env.close()
        self.env = self.base = self.obs = self.base_action = None


def worker_main(connection, init):
    """Worker process: one EnvSlot driven by (command, payload) messages."""
    absolute_cache_paths(init.get("cwd", os.getcwd()))
    sys.dont_write_bytecode = True
    for path in reversed(init.get("sys_path", [])):
        if path not in sys.path:
            sys.path.insert(0, path)
    slot = EnvSlot(**init["slot"])
    while True:
        command, payload = connection.recv()
        try:
            if command == "close":
                slot.close()
                connection.send(("ok", None))
                return
            result = getattr(slot, command)(*payload)
            connection.send(("ok", result))
        except Exception as exc:  # noqa: BLE001 - reported to the trainer, which raises it
            connection.send(("error", f"{exc!r}\n{traceback.format_exc()}"))


class _Remote:
    """A worker process holding one slot."""

    def __init__(self, slot_arguments):
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        # A spawned child inherits the environment variables; the import path is passed explicitly.
        init = {"slot": slot_arguments, "sys_path": list(sys.path), "cwd": os.getcwd()}
        self.process = context.Process(target=worker_main, args=(child, init), daemon=True)
        self.process.start()
        child.close()

    def send(self, command, payload):
        self.connection.send((command, payload))

    def receive(self):
        status, value = self.connection.recv()
        if status != "ok":
            raise RuntimeError(f"env worker failed: {value}")
        return value

    def close(self):
        try:
            self.send("close", ())
            self.receive()
        except (EOFError, OSError, RuntimeError):
            pass
        self.process.join(timeout=60)
        if self.process.is_alive():
            self.process.terminate()


class EnvSlots:
    """The trainer's vector of slots: call(commands) runs [(slot, command, args)] and returns the
    results in that order (worker slots work in parallel; results are read in order)."""

    def __init__(self, count, slot_arguments, in_process):
        self.in_process = bool(in_process)
        if self.in_process:
            self.slots = [EnvSlot(**slot_arguments) for _ in range(count)]
        else:
            self.slots = [_Remote(slot_arguments) for _ in range(count)]

    def __len__(self):
        return len(self.slots)

    def call(self, commands):
        if self.in_process:
            return [getattr(self.slots[i], command)(*args) for i, command, args in commands]
        for i, command, args in commands:
            self.slots[i].send(command, args)
        return [self.slots[i].receive() for i, _, _ in commands]

    def close(self):
        for slot in self.slots:
            slot.close()


class RewardNormalizer:
    """CleanRL L99-100: gymnasium NormalizeReward(gamma) then the reward clipped to +-clip (D-RL02 +-10).

    Same arithmetic as gymnasium 1.2.0 wrappers/stateful_reward.py NormalizeReward.step (the
    project's pinned version): discounted = discounted * gamma * (1 - terminated) + reward, the
    running statistics updated with it (gymnasium RunningMeanStd), reward / sqrt(var + epsilon).
    One normaliser per env slot, kept across its episodes, as CleanRL wraps each env (L87-103)."""

    def __init__(self, gamma, clip, epsilon=1e-8):
        from gymnasium.wrappers.utils import RunningMeanStd
        self.gamma, self.clip, self.epsilon = float(gamma), float(clip), float(epsilon)
        self.return_rms = RunningMeanStd(shape=())
        self.discounted_reward = np.array([0.0])

    def __call__(self, reward, terminated):
        self.discounted_reward = self.discounted_reward * self.gamma * (1 - terminated) + float(reward)
        self.return_rms.update(self.discounted_reward)
        normalized = reward / np.sqrt(self.return_rms.var + self.epsilon)
        return float(np.clip(normalized, -self.clip, self.clip))

    def state(self):
        return {"mean": float(self.return_rms.mean), "var": float(self.return_rms.var),
                "count": float(self.return_rms.count), "discounted_reward": float(self.discounted_reward[0])}

    def load(self, state):
        self.return_rms.mean = np.float64(state["mean"])
        self.return_rms.var = np.float64(state["var"])
        self.return_rms.count = state["count"]
        self.discounted_reward = np.array([state["discounted_reward"]])
