"""One scored episode on the official evaluator's scoring path.

Shared by scripts/batch_evaluate.py and the remote runner
(docs/plan/baseline-v1-remote.md §3.1, task kind "episode"). It follows
smart_buildings.evaluate.evaluate_policy_for_config: the same environment
loading, metadata, SimulationTracker and run_policy, without charts or per-step
CSV files. The caller must already have made `smart_buildings` importable and
set the working directory to the root that the scenario's config.gin paths are
relative to (the starter kit root, or a remote scenario overlay root).

Score unit (D-E01): the official reward over the full 8640 steps.
"""
from pathlib import Path
import time


def run_episode(scenario_dir, policy_class, steps, *, policy_key, label=None):
    """Run one episode; return totals and the reward decomposition."""
    import numpy as np
    from smart_buildings import evaluate
    from smart_buildings.smart_control.utils.scenario_generator import SimulationTracker

    scenario_dir = Path(scenario_dir)
    env = evaluate.load_hybrid_environment(scenario_dir / "config.gin")
    tracker = SimulationTracker(env, vmin=280, vmax=310, capture_snapshots=False)
    env.metrics_path = None
    env._occupancy_normalization_constant = 125.0
    metadata = evaluate.create_building_meta_data(env, scenario_dir)
    policy = evaluate.initialize_policy(policy_class, metadata, env.action_spec())
    start = time.perf_counter()
    compact = evaluate.run_policy(env, tracker, policy, steps)
    elapsed = time.perf_counter() - start
    parts = {key: np.array([row[key] for row in tracker.reward_components], dtype=float)
             for key in ("normalized_productivity_regret", "normalized_energy_cost",
                         "normalized_carbon_emission", "total_occupancy")}
    rewards = np.asarray(tracker.rewards, dtype=float)
    occupied = parts["total_occupancy"] > 0
    weights = (tracker.productivity_weight, tracker.energy_weight, tracker.carbon_weight)
    scale = sum(weights)
    return {
        "scenario": label if label is not None else "/".join(scenario_dir.parts[-2:]),
        "policy": policy_key,
        "steps": compact["num_steps"], "total_reward": compact["total_reward"],
        "mean_reward": compact["mean_reward"],
        # Weighted contributions to the total reward; they sum to total_reward.
        "comfort_part": float(weights[0] * parts["normalized_productivity_regret"].sum() / scale),
        "energy_part": float(-weights[1] * parts["normalized_energy_cost"].sum() / scale),
        "carbon_part": float(-weights[2] * parts["normalized_carbon_emission"].sum() / scale),
        "occupied_steps": int(occupied.sum()),
        # Regret below -1 means some occupied zone was beyond the 5 K knee.
        "steps_regret_below_minus1": int((parts["normalized_productivity_regret"] < -1).sum()),
        "worst_step_reward": float(rewards.min()) if rewards.size else 0.0,
        "num_zones": len(metadata["zone_temp_indices"]),
        "num_ahus": len(metadata["ahu_zone_assignments"]),
        "elapsed_seconds": elapsed,
    }
