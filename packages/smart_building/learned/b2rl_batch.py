"""Batched backend of the B2-RL trainer (user 2026-10-27: "正式把加速版RL pipeline做好"; the fast simulator of the
fast-sim work plugs in here once it is validated).

A batch simulator advances many building slots together. Its contract (the fast simulator implements it on the GPU;
OfficialBatch below implements it with the official environments one after the other, for tests and as the
reference):

    Simulator(slots, device)            each slot behaves like evaluate.load_hybrid_environment(config.gin) with
                                        metrics_path None and _occupancy_normalization_constant 125.0
      load(i, scenario_dir)             (re)build slot i from that scenario folder
      action_spec(i), metadata(i)       env.action_spec(), evaluate.create_building_meta_data(env, scenario_dir)
      reset(slots) -> {i: observation}  env.reset()[0]
      step({i: action}) -> {i: (observation, reward, terminated, truncated, (regret, energy, carbon))}
                                        env.step(action), and the normalized productivity regret, energy cost and
                                        carbon emission of env.get_reward_info_and_response()[1]
      close()

The rule controllers (G36PolicyB2R2 inside policies/b2rl.HourlyRule, about 2 ms per building per step in Python)
run in controller processes spread over the CPU cores, each owning the slots i % processes == k (0 processes: in
this process, for tests). BatchedHourlyEnvs gives the trainer the API of b2rl_env.HourlyEnvs (call, send, ready,
receive, close): "start" and "decide" calls are queued and ready() advances every queued slot together, step by step,
up to its next hourly decision or the end of its episode, which is what a batched simulator needs. Each slot's result
is the one b2rl_env.HourlyEnv returns for the same scenario (tests/test_b2rl_batch.py)."""
import multiprocessing
import os
from pathlib import Path
import sys
import traceback

from smart_building.learned.b2rl_env import REJECTED, ROOT, label_of, rule_policy_class, scenario_for, setup_simulator

PARTS = ("regret", "energy", "carbon")


class OfficialBatch:
    """The contract on official environments stepped one after the other (no speed-up: the reference and the
    tests). Each slot keeps its own process-global random state (sim/snapshot.RandomStateSlot), as HourlyEnv does."""

    def __init__(self, slots, device=None):
        self.envs, self.randoms, self._metadata = [None] * slots, [None] * slots, [None] * slots

    def load(self, i, scenario_dir):
        from smart_buildings import evaluate
        from smart_building.sim.snapshot import RandomStateSlot
        self._close(i)
        self.randoms[i] = RandomStateSlot()
        with self.randoms[i]:
            env = evaluate.load_hybrid_environment(Path(scenario_dir) / "config.gin")
            env.metrics_path = None
            env._occupancy_normalization_constant = 125.0
            self._metadata[i] = evaluate.create_building_meta_data(env, Path(scenario_dir))
        self.envs[i] = env

    def action_spec(self, i):
        return self.envs[i].action_spec()

    def metadata(self, i):
        return self._metadata[i]

    def reset(self, slots):
        out = {}
        for i in slots:
            with self.randoms[i]:
                out[i] = self.envs[i].reset()[0]
        return out

    def step(self, actions):
        out = {}
        for i, action in actions.items():
            with self.randoms[i]:
                observation, reward, terminated, truncated, _ = self.envs[i].step(action)
                _, response = self.envs[i].get_reward_info_and_response()
            out[i] = (observation, reward, terminated, truncated,
                      (response.normalized_productivity_regret, response.normalized_energy_cost,
                       response.normalized_carbon_emission))
        return out

    def _close(self, i):
        if self.envs[i] is not None:
            with self.randoms[i]:
                self.envs[i].close()
            self.envs[i] = None

    def close(self):
        for i in range(len(self.envs)):
            self._close(i)


class HeldBatch:
    """The controller-side ceiling (b2rl.py bench --batch held): each slot holds its scenario's first observation
    and returns it at every step with reward 0, so a run measures the controllers and the transfer between processes,
    not the physics. Not for training."""

    _cache = {}

    def __init__(self, slots, device=None):
        self.slots = [None] * slots

    def load(self, i, scenario_dir):
        key = str(scenario_dir)
        if key not in HeldBatch._cache:
            official = OfficialBatch(1)
            official.load(0, scenario_dir)
            HeldBatch._cache[key] = (official.metadata(0), official.action_spec(0), official.reset([0])[0])
            official.close()
        self.slots[i] = HeldBatch._cache[key]

    def action_spec(self, i):
        return self.slots[i][1]

    def metadata(self, i):
        return self.slots[i][0]

    def reset(self, slots):
        return {i: self.slots[i][2] for i in slots}

    def step(self, actions):
        return {i: (self.slots[i][2], 0.0, False, False, (0.0, 0.0, 0.0)) for i in actions}

    def close(self):
        pass


def make_simulator(name, slots, device):
    """"official" (OfficialBatch), "held" (HeldBatch) or "module:Class" of a batch simulator."""
    if not name or name == "official":
        return OfficialBatch(slots, device)
    if name == "held":
        return HeldBatch(slots, device)
    module, _, attribute = name.partition(":")
    import importlib
    return getattr(importlib.import_module(module), attribute)(slots, device)


class ControllerSet:
    """The rule controllers of some slots: HourlyRule(G36PolicyB2R2 or a rule-screen class) and the last observation
    of each."""

    def __init__(self, rule=None, interval=None):
        self.rule, self.interval = rule, interval
        self.agents, self.obs = {}, {}

    def run(self, command, items):
        return getattr(self, command)(items)

    def start(self, items):
        """{slot: (metadata, action_spec, first observation)} -> {slot: first features}."""
        from smart_building.policies.b2rl import DECISION_STEPS, HourlyRule
        out = {}
        for i, (metadata, spec, observation) in items.items():
            agent = HourlyRule(rule_policy_class(self.rule)(metadata, spec), int(self.interval or DECISION_STEPS))
            assert agent.observe(observation)
            self.agents[i], self.obs[i] = agent, observation
            out[i] = agent.features()
        return out

    def decide(self, items):
        """{slot: action of the learner} -> {slot: (applied parameters, first simulator action)}."""
        out = {}
        for i, action in items.items():
            parameters = self.agents[i].apply(action)
            out[i] = (parameters, self.agents[i].rule.get_action(self.obs[i]))
        return out

    def advance(self, items):
        """{slot: (observation after the step, episode over)} -> {slot: ("action", next simulator action) |
        ("decision", features) | ("end", fallback steps)}; the order of HourlyEnv._decide."""
        out = {}
        for i, (observation, done) in items.items():
            agent = self.agents[i]
            self.obs[i] = observation
            if done:
                out[i] = ("end", agent.rule.diagnostics.get("fallback_steps"))
                del self.agents[i], self.obs[i]
            elif agent.observe(observation):
                out[i] = ("decision", agent.features())
            else:
                out[i] = ("action", agent.rule.get_action(observation))
        return out

    def memory(self, items):
        from smart_building.learned.ppo_env import process_memory
        return {i: process_memory() for i in items}


def controller_main(connection, init):
    """Controller process: one ControllerSet driven by (command, {slot: arguments}) messages."""
    sys.dont_write_bytecode = True
    for path in reversed(init["sys_path"]):
        if path not in sys.path:
            sys.path.insert(0, path)
    setup_simulator(init["root"])
    controllers = ControllerSet(init["rule"], init["interval"])
    while True:
        command, items = connection.recv()
        try:
            if command == "close":
                connection.send(("ok", None))
                return
            connection.send(("ok", controllers.run(command, items)))
        except Exception as exc:  # noqa: BLE001 - reported to the trainer, which raises it
            connection.send(("error", f"{exc!r}\n{traceback.format_exc()}"))


class Controllers:
    """The slots' controllers in this process (processes=0) or in controller processes; run() sends each owner its
    slots' items at once and returns {slot: result}."""

    def __init__(self, processes, rule=None, interval=None):
        self.processes = int(processes)
        if self.processes == 0:
            self.local = ControllerSet(rule, interval)
            return
        context = multiprocessing.get_context("spawn")
        self.pipes, self.workers = [], []
        for _ in range(self.processes):
            parent, child = context.Pipe()
            init = {"sys_path": list(sys.path), "root": str(ROOT), "rule": rule, "interval": interval}
            process = context.Process(target=controller_main, args=(child, init), daemon=True)
            process.start()
            child.close()
            self.pipes.append(parent)
            self.workers.append(process)

    def run(self, command, items):
        if self.processes == 0:
            return self.local.run(command, items)
        groups = {}
        for i, value in items.items():
            groups.setdefault(i % self.processes, {})[i] = value
        for k, group in groups.items():
            self.pipes[k].send((command, group))
        out = {}
        for k in groups:
            status, value = self.pipes[k].recv()
            if status != "ok":
                raise RuntimeError(f"controller process {k} failed: {value}")
            out.update(value)
        return out

    def close(self):
        if self.processes == 0:
            return
        for pipe, process in zip(self.pipes, self.workers):
            try:
                pipe.send(("close", {}))
                pipe.recv()
            except (EOFError, OSError):
                pass
            process.join(timeout=60)
            if process.is_alive():
                process.terminate()


class BatchedHourlyEnvs:
    """b2rl_env.HourlyEnvs on a batch simulator: the same per-slot scenario sequence (scenario_for with the slot as
    the worker) and the same results, with every queued slot advanced together."""

    def __init__(self, count, seed, scenarios, rule=None, steps_cap=None, in_process=False, interval=None,
                 simulator="official", controllers=None, device="cuda"):
        self.count, self.seed, self.scenarios = int(count), int(seed), list(scenarios)
        self.steps_cap = int(steps_cap) if steps_cap else None
        setup_simulator(ROOT)
        self.sim = make_simulator(simulator, self.count, device)
        if in_process:
            processes = 0
        elif controllers is not None:
            processes = int(controllers)
        else:
            processes = max(1, min(self.count, (os.cpu_count() or 2) - 2))
        self.controllers = Controllers(min(processes, self.count), rule, interval)
        self.pending, self.results, self.info = {}, {}, [None] * self.count

    def __len__(self):
        return self.count

    def send(self, i, command, args=()):
        if command not in ("start", "decide"):
            raise ValueError(f"unknown command {command!r}")
        self.pending[i] = (command, args[0])

    def ready(self, slots):
        slots = set(slots)
        if not any(i in self.results for i in slots):
            self._advance(slots)
        return sorted(i for i in slots if i in self.results)

    def receive(self, i):
        return self.results.pop(i)

    def call(self, commands):
        out = [None] * len(commands)
        for n, (i, command, args) in enumerate(commands):
            if command == "memory":
                from smart_building.learned.ppo_env import process_memory
                out[n] = {"simulator": process_memory(), "controller": self.controllers.run("memory", {i: None})[i]}
            else:
                self.send(i, command, args)
        self._advance({i for i, command, _ in commands if command != "memory"})
        for n, (i, command, _) in enumerate(commands):
            if command != "memory":
                out[n] = self.receive(i)
        return out

    def _advance(self, slots):
        """Run the queued starts, then the queued decisions, of these slots."""
        starts = {i: a for i, (c, a) in self.pending.items() if c == "start" and i in slots}
        if starts:
            self._start(starts)
        decides = {i: a for i, (c, a) in self.pending.items() if c == "decide" and i in slots}
        if decides:
            self._decide(decides)

    def _start(self, starts):
        for i, episode in starts.items():
            scenario = Path(scenario_for(self.scenarios, self.seed, i, episode))
            self.sim.load(i, scenario)
            self.info[i] = {"label": label_of(scenario), "episode": int(episode), "steps": 0, "total": 0.0,
                            "sums": {k: 0.0 for k in PARTS}}
        observations = self.sim.reset(sorted(starts))
        features = self.controllers.run("start", {i: (self.sim.metadata(i), self.sim.action_spec(i), observations[i])
                                                  for i in starts})
        for i in starts:
            self.results[i] = {"features": features[i], "label": self.info[i]["label"],
                               "episode": self.info[i]["episode"],
                               "zones": len(self.sim.metadata(i)["zone_temp_indices"])}
            del self.pending[i]

    def _decide(self, decides):
        first = self.controllers.run("decide", decides)
        running = {i: first[i][1] for i in decides}
        acc = {i: {"reward": 0.0, "steps": 0, "parts": {k: 0.0 for k in PARTS}, "parameters": first[i][0]}
               for i in decides}
        while running:
            advance = {}
            for i, (observation, reward, terminated, truncated, parts) in self.sim.step(running).items():
                reward = float(reward)
                info, a = self.info[i], acc[i]
                if reward == REJECTED:
                    raise RuntimeError(f"{info['label']} step {info['steps']}: the building refused the action")
                for key, value in zip(PARTS, parts):
                    a["parts"][key] += float(value)
                    info["sums"][key] += float(value)
                a["reward"] += reward
                a["steps"] += 1
                info["steps"] += 1
                info["total"] += reward
                a["done"] = bool(terminated or truncated) or (self.steps_cap is not None
                                                              and info["steps"] >= self.steps_cap)
                advance[i] = (observation, a["done"])
            running = {}
            for i, (kind, value) in self.controllers.run("advance", advance).items():
                if kind == "action":
                    running[i] = value
                    continue
                a, info = acc[i], self.info[i]
                result = {"reward": a["reward"], "steps": a["steps"], "done": a["done"], "parameters": a["parameters"],
                          "parts": {k: v / a["steps"] for k, v in a["parts"].items()}}
                if kind == "end":
                    result["episode"] = {"label": info["label"], "episode": info["episode"], "steps": info["steps"],
                                         "total_reward": info["total"], "mean_reward": info["total"] / info["steps"],
                                         "cost": -info["total"] / info["steps"],
                                         **{f"mean_{k}": v / info["steps"] for k, v in info["sums"].items()},
                                         "fallback_steps": value}
                else:
                    result["features"] = value
                self.results[i] = result
                del self.pending[i]

    def close(self):
        self.controllers.close()
        self.sim.close()
