"""Per-step trace of one episode for the paper's example figures (reviewer scratch; writes only into <scratch>/trace).
python3 -B pipeline/launch/launch.py trace.py <scratch dir> <model name> <scenario dir>"""
import json, os, sys
from pathlib import Path
import numpy as np

ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]; DATA = ROOT / "data/raw/kaggle"
SCR, name, scen = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
os.environ["SB_FASTSIM"] = "rom"; os.environ["SB_FASTSIM_NO_BUILD"] = "1"
for p in (str(ROOT / "packages"), str(DATA)):
    sys.path.insert(0, p)
os.chdir(DATA)
from smart_building.fastsim import install_from_env, installed
if not installed():
    install_from_env()
from smart_buildings import evaluate
from smart_buildings.smart_control.utils.scenario_generator import SimulationTracker
from smart_building.policies.b2rl import b2rl_policy_class

path = SCR / f"model-{name}.json"
if not path.is_file():
    path = ROOT / "data/runs/b2rl/search/models" / f"model-{name}.json"
env = evaluate.load_hybrid_environment(scen / "config.gin")
tracker = SimulationTracker(env, vmin=280, vmax=310, capture_snapshots=False)
env.metrics_path = None
env._occupancy_normalization_constant = 125.0
meta = evaluate.create_building_meta_data(env, scen)
policy = evaluate.initialize_policy(b2rl_policy_class(path), meta, env.action_spec())
tracker.reset(); obs = tracker.last_obs
names = list(meta["observation_field_names"])
flow_idx = [i for i, n in enumerate(names) if "supply_air_flowrate_sensor" in n]
flows, choices, oats = [], [], []
inner = getattr(policy, "policy", policy)
for step in range(8640):
    a = evaluate.validate_action(evaluate.get_policy_action(policy, obs), env.action_spec())
    obs, _, term, trunc, _ = tracker.step(a)
    flows.append(np.asarray(obs).ravel()[flow_idx] if flow_idx else np.zeros(0))
    if step % 288 == 0:
        ch = getattr(inner, "room_choice", None)
        choices.append(list(map(int, ch)) if ch is not None else [])
        try:
            t = inner.slow.tracker; oats.append(t.outside_sum / t.outside_n if t.outside_n else None)
        except Exception:
            oats.append(None)
    if term or trunc:
        break
rc = tracker.reward_components
out = SCR / "trace"; out.mkdir(exist_ok=True)
key = f"{name}__{scen.parts[-2]}__{scen.parts[-1]}"
sched = env.building.simulator._hvac.schedule
band = []
for ts in tracker.timestamps:
    try:
        w = sched.get_temperature_window(ts); band.append([float(w[0]), float(w[1])])
    except Exception:
        band.append([np.nan, np.nan])
np.savez_compressed(out / f"{key}.npz", zone=np.asarray(tracker.zone_temps, dtype=np.float32), flow=np.asarray(flows, dtype=np.float32),
                    occ=np.asarray(tracker.occupancy, dtype=np.float32), reward=np.asarray(tracker.rewards, dtype=np.float32),
                    regret=np.array([r["normalized_productivity_regret"] for r in rc], dtype=np.float32),
                    energy=np.array([r["normalized_energy_cost"] for r in rc], dtype=np.float32),
                    carbon=np.array([r["normalized_carbon_emission"] for r in rc], dtype=np.float32),
                    band=np.asarray(band, dtype=np.float32))
feat = getattr(inner, "room_features", None)
(out / f"{key}.json").write_text(json.dumps({
    "timestamps": [tracker.timestamps[0].isoformat(), tracker.timestamps[-1].isoformat()],
    "choices_daily": choices, "oat_daily": oats, "flow_fields": [names[i] for i in flow_idx],
    "room_features": json.loads(json.dumps(feat, default=lambda o: np.asarray(o).tolist())) if feat is not None else None,
    "options": json.loads(path.read_text()).get("room_policy", {}).get("options"),
    "weights": [tracker.productivity_weight, tracker.energy_weight, tracker.carbon_weight],
    "rule_config": json.loads(path.read_text()).get("rule_config")}))
print("DONE", key, "mean cost", -float(np.mean(tracker.rewards)), "zones", np.asarray(tracker.zone_temps).shape, "flow", len(flow_idx), "band", band[100])
