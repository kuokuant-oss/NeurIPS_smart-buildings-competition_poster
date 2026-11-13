"""One episode on the Kaggle gateway's loop (reviewer decision 2026-10-16).

The hidden test is scored by kaggle_evaluation/gateway.py, not by the starter
kit's SimulationTracker. The two paths diverge once the first occupants arrive:
SimulationTracker.step queries occupancy once more per step
(scenario_generator.py, about lines 1714-1720) and the occupancy model caches only
the last interval (randomized_arrival_departure_occupancy.py:285-293), so the
Markov chains advance differently. Selection and S4 estimates therefore use this
path; the S1 reproduction against baseline_returns.json keeps the tracker path
(smart_building.evaluation.episode).

What it mirrors (data/raw/kaggle/kaggle_evaluation/gateway.py):
- SmartBuildingsGateway._load_environment (308-340): evaluate.load_hybrid_environment,
  metrics_path None, _occupancy_normalization_constant 125.0, metadata from
  evaluate.create_building_meta_data, with the working directory at the
  simulator root (the caller's job here);
- SmartBuildingsDriver.start_new_game (186-211): env.reset();
- the driver loop (gym_game_driver.GameDriver.run_game/step and
  SmartBuildingsDriver.step, 213-248): the policy's action goes to env.step
  unchanged, total += float(reward), stop when terminated or truncated (or at the
  gateway's smoke-test step cap), a -inf reward is a refused action;
- generate_game_results (342-357): mean = total / steps actually run (8641 for
  a full scenario).
The policy is built as the inference server does (policy_class(metadata,
action_spec)); actions are not passed through validate_action, as on the gateway.

Reward decomposition: after each step env.get_reward_info_and_response() re-reads
the interval the step just scored; its occupancy queries hit the cache filled by
env.step itself, so the episode is unchanged (checked in tests and DR3).
"""
import math
from pathlib import Path
import time

REJECTED = -math.inf


def run_gateway_episode(scenario_dir, policy_class, *, policy_key, label=None, max_steps=None,
                        record_parts=True):
    from smart_buildings import evaluate

    scenario_dir = Path(scenario_dir)
    env = evaluate.load_hybrid_environment(scenario_dir / "config.gin")
    env.metrics_path = None
    env._occupancy_normalization_constant = 125.0
    metadata = evaluate.create_building_meta_data(env, scenario_dir)
    observation, _info = env.reset()
    policy = policy_class(metadata, env.action_spec())
    reward_fn = env.reward_function
    weights = (reward_fn._productivity_weight, reward_fn._energy_cost_weight, reward_fn._carbon_emission_weight)
    scale = sum(weights)
    sums = {"regret": 0.0, "energy": 0.0, "carbon": 0.0}
    occupied = below_minus1 = steps = 0
    total, worst = 0.0, math.inf
    start = time.perf_counter()
    terminated = truncated = False
    while not (terminated or truncated):
        action = policy.get_action(observation)
        observation, reward, terminated, truncated, _ = env.step(action)
        reward = float(reward)
        if reward == REJECTED:
            raise RuntimeError(f"step {steps}: the building refused the action (gateway: invalid submission)")
        total += reward
        steps += 1
        worst = min(worst, reward)
        if record_parts:
            _, response = env.get_reward_info_and_response()
            sums["regret"] += response.normalized_productivity_regret
            sums["energy"] += response.normalized_energy_cost
            sums["carbon"] += response.normalized_carbon_emission
            occupied += response.total_occupancy > 0
            below_minus1 += response.normalized_productivity_regret < -1
        if max_steps is not None and steps >= max_steps:
            break
    elapsed = time.perf_counter() - start
    env.close()
    row = {
        "scenario": label if label is not None else "/".join(scenario_dir.parts[-2:]),
        "policy": policy_key, "path": "gateway", "steps": steps, "total_reward": total,
        "mean_reward": total / steps if steps else 0.0,
        "worst_step_reward": worst if steps else 0.0,
        "num_zones": len(metadata["zone_temp_indices"]),
        "num_ahus": len(metadata["ahu_zone_assignments"]),
        "terminated": bool(terminated), "truncated": bool(truncated),
        "elapsed_seconds": elapsed,
    }
    if record_parts:
        row.update(comfort_part=float(weights[0] * sums["regret"] / scale),
                   energy_part=float(-weights[1] * sums["energy"] / scale),
                   carbon_part=float(-weights[2] * sums["carbon"] / scale),
                   occupied_steps=int(occupied), steps_regret_below_minus1=int(below_minus1))
    return row
