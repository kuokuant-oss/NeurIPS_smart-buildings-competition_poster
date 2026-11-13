"""Step 0 of the predictive-control / RL main line (user 2026-10-30 01:5x: "先做第0步"; docs/reference/notes/24 §C):
how much could a planner that sees the future still save on top of the current best rule?

A rollout planner (one-step policy improvement over the rule, Bertsekas): on the fast simulator, from a snapshot at each
decision point, every candidate is held for its window and then the plain policy runs on for a short horizon; the
candidate with the largest reward sum is applied on the real trajectory. The simulator's own random generators make the
future (weather, occupants) known, so the result is an optimistic bound for these levers. Following our lesson that
hourly toggles of comfort settings cost more than they save, choices are held for meaningful windows:
  * daily (00:00 on weekdays, held 24 h, then 6 h of the plain policy): warm-up / night levers — unoccupied lead
    0 / 60 / 300 min, optimal-start cap 2 h, cap 2 h with lead 60;
  * per occupied block (three 4-h blocks from the morning start on weekdays, held 4 h, then 4 h of the plain policy):
    supply-air / airflow levers — A12 off (guard 30 K), A12 free (guard 0), occupied margin 0.3 / 0 K, heating airflow
    ratio 0.25, static pressure cap 1000 Pa.
The overrides are rule parameters (SLOW_PARAMETER_RANGES) re-applied after the hourly outside-air schedule; "keep" is
always a candidate, so the planner is never worse than the policy on its own decisions' horizon.

The policy and its rule are classes built inside functions (b2rl_policy_class, c3_configured_class), which plain pickle
cannot store; each scenario runs in its own process and registers such classes so the snapshot rebuilds them there.

  python3 -B pipeline/launch/launch.py pipeline/setup/headroom.py run <model.json> <tag> <workers> <spec> [<spec> ...]
  python3 -B pipeline/launch/launch.py pipeline/setup/headroom.py run <model.json> <tag> <workers> --plain <spec> ...
  python3 -B pipeline/launch/launch.py pipeline/setup/headroom.py run <model.json> <tag> <workers> --rule=<rule.json> <spec> ...
--plain runs no planning (it must reproduce the stored fast-simulator row exactly). Planning runs record every option's
reward sum and the observable features (policies/b2rl.py layout v3) at each decision (2026-10-30, for the
condition-based rule). --rule follows a learned rule instead (no lookahead): per kind, regression trees of each option's
advantage over keep, stored as plain arrays, and a threshold; the option with the largest predicted advantage is taken
if it is above the threshold. Output data/runs/b2rl/headroom/<tag>/
<scenario>.json; a summary line per scenario in data/runs/b2rl/headroom/<tag>/run.log."""
import gc
import io
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/runs/b2rl/headroom"
DAILY = [("keep", {}), ("lead0", {"unoccupied_lead_min": 0}), ("lead60", {"unoccupied_lead_min": 60}),
         ("lead300", {"unoccupied_lead_min": 300}), ("topt2h", {"t_opt_max_s": 7200.0}),
         ("topt2h_lead60", {"t_opt_max_s": 7200.0, "unoccupied_lead_min": 60})]
BLOCK = [("keep", {}), ("a12_off", {"sat_cost_guard_k": 30.0}), ("a12_free", {"sat_cost_guard_k": 0.0}),
         ("margin03", {"occupied_margin_k": 0.3}), ("margin0", {"occupied_margin_k": 0.0}),
         ("heat025", {"heating_max_ratio": 0.25}), ("sp1000", {"sp_max_pa": 1000.0}),
         # zone geometry options (2026-10-30, stage 2B "geometry as ML features"; G36Config zone_heat_*): each one
         # alone was worse than night1_dcv07 on average over whole months (rounds 24-26); the teacher tries them per
         # block, where they may pay off
         ("geo_up", {"zone_heat_ext_power": 0.5}), ("geo_down", {"zone_heat_ext_power": -0.5}),
         ("geo_redist", {"zone_heat_ext_power": 0.5, "zone_heat_redistribute": True})]
GEOMETRY = ("zone_heat_ext_power", "zone_heat_redistribute", "sat_cost_guard_large_q")
DAY_STEPS, DAILY_AFTER, BLOCK_STEPS, BLOCK_AFTER = 288, 72, 48, 48

REGISTRY = {}


def _rebuild(key, state):
    cls = REGISTRY[key]
    obj = cls.__new__(cls)
    if hasattr(obj, "__setstate__"):
        obj.__setstate__(state)
    else:
        obj.__dict__.update(state)
    return obj


def _local(cls):
    return "<locals>" in getattr(cls, "__qualname__", "")


def take(objects):
    from smart_building.sim import snapshot

    class Pickler(snapshot._SnapshotPickler):
        def reducer_override(self, obj):
            reduced = super().reducer_override(obj)
            if reduced is not NotImplemented:
                return reduced
            if isinstance(obj, type) and _local(obj):
                key = f"{obj.__module__}.{obj.__qualname__}:{id(obj)}"
                REGISTRY[key] = obj
                return (REGISTRY.__getitem__, (key,))
            if _local(type(obj)):
                cls = type(obj)
                key = f"{cls.__module__}.{cls.__qualname__}:{id(cls)}"
                REGISTRY[key] = cls
                return (_rebuild, (key, obj.__getstate__() if hasattr(obj, "__getstate__") else obj.__dict__))
            return NotImplemented

    stream = io.BytesIO()
    Pickler(stream).dump(objects)
    return snapshot.Snapshot(stream.getvalue(), snapshot.global_random_state())


def restore(snap):
    from smart_building.sim import snapshot
    snapshot.set_global_random_state(snap.random_state)
    return pickle.loads(snap.payload)


def tree_value(tree, features):
    """A regression tree stored as plain arrays (sklearn's children_left / children_right / feature / threshold /
    value, feature names instead of indices). A list of such trees is a forest (2026-10-30, condition rule round 2,
    fit_condition_rule.py forest): their mean."""
    if isinstance(tree, list):
        return sum(tree_value(t, features) for t in tree) / len(tree)
    node = 0
    while tree["left"][node] != -1:
        node = tree["left"][node] if features[tree["feature"][node]] <= tree["threshold"][node] else tree["right"][node]
    return tree["value"][node]


def choose(kind_rule, features):
    """The learned rule: the option with the largest predicted advantage over keep, if it is above the threshold."""
    advantages = {name: tree_value(tree, features) for name, tree in kind_rule["trees"].items()}
    best = max(advantages, key=lambda name: advantages[name]) if advantages else "keep"
    return (best if advantages and advantages[best] > kind_rule["threshold"] else "keep"), advantages


def apply_overrides(policy, daily, block):
    """Slow parameters through set_rule_parameters; zone geometry switches through the controller's set_geometry,
    back to the model's own values when no option sets them."""
    merged = {**daily, **block}
    geometry = {k: merged.pop(k) for k in GEOMETRY if k in merged}
    policy.rule.set_rule_parameters(**{**policy.slow.parameters, **merged})
    controller = policy.rule._controller
    if not hasattr(controller, "geometry_base"):
        controller.geometry_base = {k: getattr(controller.cfg, k) for k in GEOMETRY}
    controller.set_geometry(**{**controller.geometry_base, **geometry})


def act(policy, obs, daily, block):
    """G36PolicyB2RL.get_action with the planner's overrides re-applied after the hourly schedule."""
    if policy.slow.observe(obs):
        policy.slow.apply(policy.actor(policy.slow.features()))
        if daily or block:
            apply_overrides(policy, daily, block)
    return policy.rule.get_action(obs)


class Episode:
    def __init__(self, spec, model):
        self.spec = spec
        os.environ["SB_FASTSIM"] = "rom"
        from smart_building.learned.b2rl_env import setup_simulator, spec_dir
        setup_simulator(ROOT)
        from smart_buildings import evaluate
        from smart_building.policies.b2rl import b2rl_policy_class
        folder = spec_dir(spec)
        env = evaluate.load_hybrid_environment(folder / "config.gin")
        env.metrics_path = None
        env._occupancy_normalization_constant = 125.0
        metadata = evaluate.create_building_meta_data(env, folder)
        obs, _ = env.reset()
        policy = b2rl_policy_class(ROOT / model)(metadata, env.action_spec())
        reward_fn = env.reward_function
        self.weights = (reward_fn._productivity_weight, reward_fn._energy_cost_weight,
                        reward_fn._carbon_emission_weight)
        self.state = {"env": env, "policy": policy, "obs": obs, "steps": 0, "total": 0.0, "done": False,
                      "daily": {}, "block": {}, "daily_end": None, "block_end": None,
                      "parts": [0.0, 0.0, 0.0]}
        ix = policy.rule._controller.index
        self.morning, self.evening = ix.morning_start_hour * 60, ix.evening_start_hour * 60

    def clock(self, state):
        o = state["policy"].rule._controller.index.parse(state["obs"])
        return int(o.minute_of_day), int(o.weekday)

    @staticmethod
    def step(state, record=False):
        env, policy = state["env"], state["policy"]
        action = act(policy, state["obs"], state["daily"], state["block"])
        obs, reward, terminated, truncated, _ = env.step(action)
        state["obs"], state["steps"], state["total"] = obs, state["steps"] + 1, state["total"] + float(reward)
        if record:
            _, response = env.get_reward_info_and_response()
            parts = state["parts"]
            parts[0] += response.normalized_productivity_regret
            parts[1] += response.normalized_energy_cost
            parts[2] += response.normalized_carbon_emission
        if terminated or truncated:
            state["done"] = True
        for kind in ("daily", "block"):                    # a held choice ends: back to the plain schedule
            if state[f"{kind}_end"] is not None and state["steps"] >= state[f"{kind}_end"]:
                state[kind], state[f"{kind}_end"] = {}, None
                apply_overrides(policy, state["daily"], state["block"])
        return float(reward)

    def evaluate(self, snap, kind, override, hold, after):
        state = restore(snap)
        state[kind], state[f"{kind}_end"] = dict(override), state["steps"] + hold
        apply_overrides(state["policy"], state["daily"], state["block"])
        total = 0.0
        for _ in range(hold + after):
            if state["done"]:
                break
            total += self.step(state)
        state["env"].close()
        del state
        gc.collect()                                       # a restored copy holds about 1 GB until collected
        return total

    def decide(self, kind, candidates, hold, after):
        snap = take(self.state)
        totals = {name: self.evaluate(snap, kind, override, hold, after) for name, override in candidates}
        del snap
        best = max(totals, key=lambda name: totals[name])
        state = self.state
        state[kind], state[f"{kind}_end"] = dict(dict(candidates)[best]), state["steps"] + hold
        apply_overrides(state["policy"], state["daily"], state["block"])
        return best, totals

    def features(self):
        policy = self.state["policy"]
        from smart_building.policies.b2rl import layout_features
        values = policy.slow.tracker.features(policy.slow.map.encode(policy.slow.parameters))
        return dict(zip(layout_features(policy.slow.layout), map(float, values)))

    def geometry(self, names):
        """The building's run-time descriptors (data/runs/stage2/descriptors.json, from the floor plan, zone map
        and AHUs the policy gets in its metadata) as geo_<name> features; a rule fitted with "geo" needs them."""
        if not hasattr(self, "_geometry"):
            pool, building, _ = self.spec.split("/")
            table = json.loads((ROOT / "data/runs/stage2/descriptors.json").read_text(encoding="utf-8"))["buildings"]
            self._geometry = {f"geo_{k}": float(table[f"{pool}/{building}"][k]) for k in names}
        return self._geometry

    def follow(self, kind, candidates, hold, rule):
        """Mode rule: the learned rule's choice from the observable features (no lookahead)."""
        features = self.features()
        if rule[kind].get("geometry_features"):
            features = {**features, **self.geometry(rule[kind]["geometry_features"])}
        best, advantages = choose(rule[kind], features)
        state = self.state
        state[kind], state[f"{kind}_end"] = dict(dict(candidates)[best]), state["steps"] + hold
        apply_overrides(state["policy"], state["daily"], state["block"])
        return best, advantages, features

    def run(self, mode, rule=None):
        """mode "plain" (no decisions), "plan" (lookahead, every option's total and the features recorded) or "rule"
        (the learned rule's choices)."""
        decisions = []
        state = self.state
        while not state["done"]:
            minute, weekday = self.clock(state)
            if mode != "plain" and weekday < 5:
                points = []
                if minute == 0:
                    points.append(("daily", DAILY, DAY_STEPS, DAILY_AFTER, {}))
                if minute in range(self.morning, self.evening, 240):
                    points.append(("block", BLOCK, min(BLOCK_STEPS, (self.evening - minute) // 5), BLOCK_AFTER,
                                   {"minute": minute}))
                for kind, candidates, hold, after, extra in points:
                    entry = {"step": state["steps"], "kind": kind, "weekday": weekday, **extra}
                    if mode == "plan":
                        features = self.features()
                        best, totals = self.decide(kind, candidates, hold, after)
                        entry.update(best=best, gain=totals[best] - totals["keep"], totals=totals, features=features)
                    else:
                        best, advantages, features = self.follow(kind, candidates, hold, rule)
                        entry.update(best=best, predicted=advantages)
                    decisions.append(entry)
            self.step(state, record=True)
        steps = state["steps"]
        scale = sum(self.weights)
        return {"steps": steps, "total_reward": state["total"], "mean_reward": state["total"] / steps,
                "cost": -state["total"] / steps,
                "comfort_part": float(self.weights[0] * state["parts"][0] / scale),
                "energy_part": float(-self.weights[1] * state["parts"][1] / scale),
                "carbon_part": float(-self.weights[2] * state["parts"][2] / scale),
                "decisions": decisions}


def one(model, tag, spec, mode):
    started = time.time()
    rule = None
    if mode.startswith("rule="):
        rule_path, mode = mode.split("=", 1)[1], "rule"
        rule = json.loads((ROOT / rule_path).read_text(encoding="utf-8"))
    row = Episode(spec, model).run(mode, rule)
    row.update(scenario=spec, model=model, mode=mode, plain=mode == "plain", seconds=round(time.time() - started, 1),
               **({"rule": rule_path} if rule is not None else {}))
    out = OUT / tag / (spec.replace("/", "__") + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row, indent=1) + "\n", encoding="utf-8")
    chosen = {}
    for d in row["decisions"]:
        chosen[d["best"]] = chosen.get(d["best"], 0) + 1
    with (OUT / tag / "run.log").open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(time.strftime("%m-%d %H:%M:%S ") + f"{spec} cost {row['cost']:.6f} "
                     f"({len(row['decisions'])} decisions, {row['seconds']} s) chosen {json.dumps(chosen)}\n")


def main():
    if sys.argv[1] == "one":
        _, _, model, tag, spec, mode = sys.argv
        one(model, tag, spec, mode)
        return
    _, _, model, tag, workers, *rest = sys.argv
    mode = "plan"
    if rest and rest[0] == "--plain":
        mode, rest = "plain", rest[1:]
    elif rest and rest[0].startswith("--rule="):
        mode, rest = "rule=" + rest[0].split("=", 1)[1], rest[1:]
    specs = rest
    env = {k: v for k, v in os.environ.items() if k != "SB_FASTSIM"}
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    running, queue = [], list(specs)
    while queue or running:
        while queue and len(running) < int(workers):
            spec = queue.pop(0)
            running.append(subprocess.Popen([sys.executable, "-B", __file__, "one", model, tag, spec, mode],
                                            cwd=ROOT, env=env))
        time.sleep(5)
        for p in [p for p in running if p.poll() is not None]:
            running.remove(p)
            if p.returncode:
                print(f"a scenario failed (exit {p.returncode})", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "packages"))
    main()
