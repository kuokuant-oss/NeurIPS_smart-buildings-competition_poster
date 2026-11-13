"""Environment side of the B2-RL trainer: one decision per hour on the final rule (handoff/RUNBOOK-windows-b2rl.md §4.1).

A slot plays TRAIN episodes on the Kaggle gateway's loop (evaluation/gateway_episode.py): the scenario is loaded with
the working directory at the simulator root, metrics_path None, _occupancy_normalization_constant 125.0, metadata
from evaluate.create_building_meta_data, env.reset(), then env.step(action) until terminated or truncated. Every
episode builds a new rule policy (G36PolicyB2R2 unless a pilot names another class) and wraps it in
policies/b2rl.HourlyRule, which owns the features, the action mapping and the order of the calls, exactly as the
competition policy G36PolicyB2RL runs them. decide(action) applies the action's parameters, runs up to 12 steps and
returns the sum of their official rewards, the next features and whether the episode ended (8641 steps = 721
decisions, the last one a single step).

The simulator path follows .tmp/reviewer/rules/diagnose.py on this host: the competition copy of smart_buildings
(data/raw/kaggle) first on sys.path, the working directory at reference/starter-kit (the config.gin asset paths).
One simulator per spawn worker (the simulator seeds and uses process-global random state; learned/ppo_env.py).

TRAIN pool: data/processed/scenarios/split-v1-r1/train (80 buildings, 367 scenarios; REQ-020 redraw of split-v1/train,
whose 3 buildings the simulator constructor rejects; configs/splits/v1-train-r1-manifest.json). Each worker plays
the pool in its own shuffled order, a new permutation per pass, from (seed, worker, pass).
"""
import math
import multiprocessing
import multiprocessing.connection
import os
from pathlib import Path
import sys
import traceback

import numpy as np

REJECTED = -math.inf
ROOT = Path(__file__).resolve().parents[3]
TRAIN_POOL = ROOT / "data/processed/scenarios/split-v1-r1/train"
SCENARIOS = ROOT / "data/processed/scenarios"


def setup_simulator(root=ROOT):
    """Import path and working directory of diagnose.py on a host without the Mac's run folders. With SB_FASTSIM set
    (scripts/b2rl.py --fast), the opt-in fast simulator of packages/smart_building/fastsim is installed in this
    process; unset, nothing changes."""
    competition, simulator = Path(root) / "data/raw/kaggle", Path(root) / "reference/starter-kit"
    for path in (str(Path(root) / "packages"), str(competition)):
        if path not in sys.path:
            sys.path.insert(0, path)
    os.environ["SB_SCENARIO_ROOT"] = str(simulator)
    os.chdir(simulator)
    if os.environ.get("SB_FASTSIM", "").strip():
        from smart_building.fastsim import install_from_env, installed
        if not installed():
            install_from_env()
    return simulator


def scenario_dirs(pool=TRAIN_POOL):
    """Every scenario folder (building/weather with a config.gin) of a pool, sorted."""
    return sorted(str(p.parent) for p in Path(pool).glob("*/*/config.gin") if not p.parent.name.startswith("._")
                  and not p.parent.parent.name.startswith("._"))


def spec_dir(spec):
    """pool/building/weather of configs/b2rl/eval-sets.json -> its scenario folder (pool val = S4; train = the
    TRAIN pool of split-v1-r1)."""
    pool, building, weather = spec.split("/")
    return (TRAIN_POOL if pool == "train" else SCENARIOS / "split-v1" / pool) / building / weather


def scenario_for(scenarios, seed, worker, episode):
    """The scenario of a worker's episode-th episode: pass e = episode // n plays permutation (seed, worker, e)."""
    n = len(scenarios)
    order = np.random.default_rng([int(seed), int(worker), episode // n]).permutation(n)
    return scenarios[int(order[episode % n])]


def label_of(scenario_dir):
    parts = Path(scenario_dir).parts
    return f"{parts[-3]}/{parts[-2]}/{parts[-1]}"


_RULE_CLASSES = {}


def rule_policy_class(name=None):
    """The rule class: G36PolicyB2R2 by default; otherwise a rule-screen name of .tmp/reviewer/rules/diagnose.py (A1
    plus the "+"-joined changes, e.g. a1_cold3+a1_econ) for runs before the screen has written B2R2_CHANGES."""
    if not name or name == "G36PolicyB2R2":
        from smart_building.policies.g36 import G36PolicyB2R2
        return G36PolicyB2R2
    if name in _RULE_CLASSES:
        return _RULE_CLASSES[name]
    from smart_building.policies.g36 import C3_COMBINATIONS, c3_configured_class
    from smart_building.policies.b2_screen import COLD3_SCREEN_CHANGES, SCREEN_CHANGES
    changes = {**{k: v for k, v in SCREEN_CHANGES.items() if "+" not in k}, **COLD3_SCREEN_CHANGES,
               "a1_econ": {"economizer_mode": "supply_air"}, "a1_m25": {"occupied_margin_k": 0.25}}
    overrides = dict(C3_COMBINATIONS["laa_occrow_dampfloor_b2a1"])
    for part in name.split("+"):
        if part == "a1":
            continue
        if part not in changes:
            raise KeyError(f"unknown rule part {part!r} (G36PolicyB2R2, or a1 joined with {sorted(changes)})")
        overrides.update(changes[part])
    cls = c3_configured_class(overrides)
    # a module-level name, as policies/c3_teachers.py gives its factory classes, so snapshots can pickle instances
    cls.__name__ = cls.__qualname__ = "G36PolicyB2Screen_" + "".join(c if c.isalnum() else "_" for c in name)
    from smart_building.learned.ppo_env import register_class
    _RULE_CLASSES[name] = register_class(cls)
    return cls


class HourlyEnv:
    """One stream of TRAIN episodes with one decision per hour."""

    def __init__(self, scenarios, seed, worker, rule=None, steps_cap=None, interval=None, layout="v1",
                 reward_baseline=None, anchor=None):
        from smart_building.policies.b2rl import DECISION_STEPS
        self.scenarios, self.seed, self.worker = list(scenarios), int(seed), int(worker)
        self.rule_name, self.steps_cap = rule, (int(steps_cap) if steps_cap else None)
        self.interval, self.layout = int(interval or DECISION_STEPS), layout
        self.anchor = dict(anchor) if anchor else None          # policies/b2rl.py: a = 0 gives these parameters
        # rule-relative reward (docs/reference/notes/18-building-rl-what-made-it-learn.md §10 item 1): a folder of
        # <building>__<weather>.npy, the rule's reward sum of every decision of the scenario (b2rl.py rule-baseline);
        # each decision returns its reward minus the rule's for the same decision of the same scenario
        self.reward_baseline = Path(reward_baseline) if reward_baseline else None
        self.baseline, self.decision = None, 0
        self.env = self.agent = self.obs = self.random = None
        self.episode = self.steps = 0
        self.label, self.total = None, 0.0
        self.sums = None

    def start(self, episode):
        """Load and reset the worker's episode-th scenario; returns the first decision's features."""
        from smart_buildings import evaluate
        from smart_building.policies.b2rl import HourlyRule
        from smart_building.sim.snapshot import RandomStateSlot
        self.close()
        scenario = Path(scenario_for(self.scenarios, self.seed, self.worker, episode))
        if self.reward_baseline is not None:
            self.baseline = np.load(self.reward_baseline / f"{scenario.parent.name}__{scenario.name}.npy")
        self.decision = 0
        # the construction seeds the simulator's process-global generators; from then on they are this env's own
        # (swapped in around every simulator call, so in-process envs do not disturb each other; ppo_env.EnvSlot)
        self.random = RandomStateSlot()
        with self.random:
            env = evaluate.load_hybrid_environment(scenario / "config.gin")
            env.metrics_path = None
            env._occupancy_normalization_constant = 125.0
            metadata = evaluate.create_building_meta_data(env, scenario)
            observation, _ = env.reset()
            policy = rule_policy_class(self.rule_name)(metadata, env.action_spec())
            self.env, self.obs, self.agent = env, observation, HourlyRule(policy, self.interval, self.layout,
                                                                          anchor=self.anchor)
            self.episode, self.steps, self.total, self.label = int(episode), 0, 0.0, label_of(scenario)
            self.sums = {"regret": 0.0, "energy": 0.0, "carbon": 0.0}
            due = self.agent.observe(self.obs)
            assert due
            features = self.agent.features()
        return {"features": features, "label": self.label, "episode": self.episode,
                "zones": len(metadata["zone_temp_indices"])}

    def decide(self, action):
        """Apply the action's parameters, then run until the next decision or the episode's end."""
        with self.random:
            return self._decide(action)

    def _decide(self, action):
        parameters = self.agent.apply(action)
        reward_sum, steps, done = 0.0, 0, False
        parts = {"regret": 0.0, "energy": 0.0, "carbon": 0.0}
        while True:
            observation, reward, terminated, truncated, _ = self.env.step(self.agent.rule.get_action(self.obs))
            reward = float(reward)
            if reward == REJECTED:
                raise RuntimeError(f"{self.label} step {self.steps}: the building refused the action")
            _, response = self.env.get_reward_info_and_response()
            for key, value in (("regret", response.normalized_productivity_regret),
                               ("energy", response.normalized_energy_cost),
                               ("carbon", response.normalized_carbon_emission)):
                parts[key] += float(value)
                self.sums[key] += float(value)
            reward_sum += reward
            steps += 1
            self.steps += 1
            self.total += reward
            self.obs = observation
            done = bool(terminated or truncated) or (self.steps_cap is not None and self.steps >= self.steps_cap)
            if done:
                break
            if self.agent.observe(self.obs):
                break
        result = {"reward": reward_sum, "steps": steps, "done": done, "parameters": parameters,
                  "parts": {k: v / steps for k, v in parts.items()}}
        if self.baseline is not None:
            if self.decision >= len(self.baseline):
                raise RuntimeError(f"{self.label}: decision {self.decision} is past the rule baseline's "
                                   f"{len(self.baseline)}")
            result["official_reward"] = reward_sum
            result["reward"] = reward_sum - float(self.baseline[self.decision])
        self.decision += 1
        if done:
            result["episode"] = {"label": self.label, "episode": self.episode, "steps": self.steps,
                                 "total_reward": self.total, "mean_reward": self.total / self.steps,
                                 "cost": -self.total / self.steps,
                                 **{f"mean_{k}": v / self.steps for k, v in self.sums.items()},
                                 "fallback_steps": self.agent.rule.diagnostics.get("fallback_steps")}
            self._release()
        else:
            result["features"] = self.agent.features()
        return result

    def memory(self):
        from smart_building.learned.ppo_env import process_memory
        return process_memory()

    def _release(self):
        """Close the simulator (inside this env's random-state slot; RandomStateSlot does not nest)."""
        if self.env is not None:
            self.env.close()
        self.env = self.agent = self.obs = None

    def close(self):
        if self.env is None:
            return
        with self.random:
            self._release()


def worker_main(connection, init):
    """Worker process: one HourlyEnv driven by (command, args) messages."""
    sys.dont_write_bytecode = True
    for path in reversed(init["sys_path"]):
        if path not in sys.path:
            sys.path.insert(0, path)
    setup_simulator(init["root"])
    env = HourlyEnv(**init["env"])
    while True:
        command, args = connection.recv()
        try:
            if command == "close":
                env.close()
                connection.send(("ok", None))
                return
            connection.send(("ok", getattr(env, command)(*args)))
        except Exception as exc:  # noqa: BLE001 - reported to the trainer, which raises it
            connection.send(("error", f"{exc!r}\n{traceback.format_exc()}"))


class HourlyEnvs:
    """The trainer's slots: N spawn workers (or in-process envs for tests); call() runs [(slot, command, args)]
    in parallel and returns the results in order. send(), ready() and receive() let each slot go at its own pace
    (a small building does not wait for a large one)."""

    def __init__(self, count, seed, scenarios, rule=None, steps_cap=None, in_process=False, interval=None,
                 layout="v1", reward_baseline=None, anchor=None):
        self.in_process = bool(in_process)
        arguments = [{"scenarios": list(scenarios), "seed": seed, "worker": w, "rule": rule, "steps_cap": steps_cap,
                      "interval": interval, "layout": layout,
                      "reward_baseline": str(reward_baseline) if reward_baseline else None,
                      "anchor": dict(anchor) if anchor else None} for w in range(count)]
        if self.in_process:
            setup_simulator(ROOT)
            self.slots = [HourlyEnv(**a) for a in arguments]
            self.results = {}
            return
        context = multiprocessing.get_context("spawn")
        self.slots = []
        for a in arguments:
            parent, child = context.Pipe()
            init = {"env": a, "sys_path": list(sys.path), "root": str(ROOT)}
            process = context.Process(target=worker_main, args=(child, init), daemon=True)
            process.start()
            child.close()
            self.slots.append((parent, process))

    def __len__(self):
        return len(self.slots)

    def call(self, commands):
        if self.in_process:
            return [getattr(self.slots[i], command)(*args) for i, command, args in commands]
        for i, command, args in commands:
            self.slots[i][0].send((command, args))
        out = []
        for i, _, _ in commands:
            status, value = self.slots[i][0].recv()
            if status != "ok":
                raise RuntimeError(f"env worker {i} failed: {value}")
            out.append(value)
        return out

    def send(self, i, command, args=()):
        """Start one call on slot i without waiting; ready() and receive() collect it (in-process: it runs now)."""
        if self.in_process:
            self.results[i] = getattr(self.slots[i], command)(*args)
            return
        self.slots[i][0].send((command, args))

    def ready(self, slots):
        """The slots among `slots` whose call is done, in order; waits for at least one (in-process: all are)."""
        if self.in_process:
            return sorted(i for i in slots if i in self.results)
        connections = {self.slots[i][0]: i for i in slots}
        return sorted(connections[c] for c in multiprocessing.connection.wait(list(connections)))

    def receive(self, i):
        if self.in_process:
            return self.results.pop(i)
        status, value = self.slots[i][0].recv()
        if status != "ok":
            raise RuntimeError(f"env worker {i} failed: {value}")
        return value

    def close(self):
        if self.in_process:
            for slot in self.slots:
                slot.close()
            return
        for connection, process in self.slots:
            try:
                connection.send(("close", ()))
                connection.recv()
            except (EOFError, OSError):
                pass
            process.join(timeout=60)
            if process.is_alive():
                process.terminate()
