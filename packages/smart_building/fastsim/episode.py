"""One gateway episode with per-step reward parts, the loop of .tmp/reviewer/rules/diagnose.py (and of
evaluation/gateway_episode.py): working directory at the simulator root, metrics_path None,
_occupancy_normalization_constant 125.0, metadata from evaluate.create_building_meta_data, env.reset(), then
policy.get_action -> env.step -> env.get_reward_info_and_response until the episode ends.

Whatever simulator the process has (official, or fastsim.install) is the one that runs."""
from pathlib import Path
import time

import numpy as np

KEYS = ("reward", "regret", "energy", "carbon", "occ")


def run(scenario_dir, policy_factory, steps=None):
    """Returns (arrays, info): per-step reward / normalized regret / energy / carbon / occupancy and timings."""
    from smart_buildings import evaluate
    scenario_dir = Path(scenario_dir)
    started = time.perf_counter()
    env = evaluate.load_hybrid_environment(scenario_dir / "config.gin")
    env.metrics_path = None
    env._occupancy_normalization_constant = 125.0
    metadata = evaluate.create_building_meta_data(env, scenario_dir)
    observation, _ = env.reset()
    policy = policy_factory(metadata, env.action_spec())
    rf = env.reward_function
    weights = np.array([rf._productivity_weight, rf._energy_cost_weight, rf._carbon_emission_weight])
    loaded = time.perf_counter()
    log = {k: [] for k in KEYS}
    done = False
    while not done:
        action = policy.get_action(observation)
        observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        _, response = env.get_reward_info_and_response()
        log["reward"].append(float(reward))
        log["regret"].append(response.normalized_productivity_regret)
        log["energy"].append(response.normalized_energy_cost)
        log["carbon"].append(response.normalized_carbon_emission)
        log["occ"].append(response.total_occupancy)
        if steps is not None and len(log["reward"]) >= steps:
            break
    finished = time.perf_counter()
    simulator = env.building.simulator
    core = getattr(simulator, "_fastsim", None)
    env.close()
    arrays = {k: np.asarray(v, dtype=np.float64) for k, v in log.items()}
    arrays["weights"] = weights
    info = {"steps": len(arrays["reward"]), "setup_s": loaded - started, "run_s": finished - loaded,
            "ms_per_step": 1000.0 * (finished - loaded) / max(1, len(arrays["reward"])),
            "simulator": core.kind if core is not None else "official"}
    if core is not None and core.kind == "rom":
        info["rom"] = {k: v for k, v in core.model.info.items() if k in ("raw", "kept", "seconds", "cached")}
    return arrays, info


def summary(arrays):
    """cost = -mean reward and its comfort / energy / carbon parts (they add up to the cost)."""
    w = np.asarray(arrays["weights"], dtype=np.float64)
    scale = w.sum()
    out = {"cost": float(-np.mean(arrays["reward"]))}
    if "regret" in arrays:
        out["comfort"] = float(-w[0] * np.mean(arrays["regret"]) / scale)
        out["energy"] = float(w[1] * np.mean(arrays["energy"]) / scale)
        out["carbon"] = float(w[2] * np.mean(arrays["carbon"]) / scale)
    return out
