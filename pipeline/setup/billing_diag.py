"""Does a candidate get heat the boiler does not bill (user 2026-10-29 08:5x: "官方這次模擬器有bug 也有人靠bug上分 你檢查
一下你不可置信的點是不是誤觸那個bug")? Kaggle discussions 749531, 751998, 752142 and starter-kit PR #2: in
smart_control/simulator/boiler.py compute_thermal_energy_rate bills the flow term at max(setpoint, return), while the
reheat coils draw heat at the tank's own temperature, which falls toward a lowered setpoint only at the fixed
cooling_rate (0.1 K/min); a cooling tank is also credited. So whenever the tank is above its setpoint, part of the
reheat is not billed as gas.

For each policy and scenario, one whole episode on the OFFICIAL simulator (SB_FASTSIM unset) with
HotWaterSystem.compute_thermal_energy_rate wrapped (the call that feeds the reward): per call the billed rate, the heat
the coils took from the water, c * rho * flow * (T_tank - T_return), and the tank's standing loss, all read from
stored attributes (the legacy tank moves its temperature on every sensor read, so the sensor property is not touched).
Unbilled share = 1 - billed / (delivered + loss + stored change). Writes data/runs/b2rl/billing-diag/<policy>__<spec>.json
(with the episode's official cost parts, for the rom check); prints a table.

python3 -B pipeline/launch/launch.py pipeline/setup/billing_diag.py <workers> <model.json|rule>,... <spec> [<spec> ...]"""
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages"))
OUT = ROOT / "data/runs/b2rl/billing-diag"


def episode(job):
    policy, spec = job
    if os.environ.get("BILLING_SIM") == "rom":   # 2026-10-31: the fast simulator (same boiler code) for pre-submit checks
        os.environ["SB_FASTSIM"] = "rom"
    else:
        os.environ.pop("SB_FASTSIM", None)
    from smart_building.learned.b2rl_env import setup_simulator, spec_dir
    setup_simulator(ROOT)
    from smart_buildings.smart_control.simulator import hot_water_system
    from smart_buildings.smart_control.utils import constants
    from smart_building.evaluation.gateway_episode import run_gateway_episode
    from smart_building.policies.b2rl import b2rl_policy_class
    from smart_building.policies.g36 import G36PolicyB2R2

    cw = constants.WATER_HEAT_CAPACITY * constants.WATER_DENSITY
    acc = {"calls": 0, "billed_j": 0.0, "delivered_j": 0.0, "loss_j": 0.0, "tank_above_setpoint_calls": 0,
           "setpoint_below_return_calls": 0, "unbilled_flow_j": 0.0, "tank_start_k": None, "tank_end_k": None,
           "capacity_j_per_k": None}
    original = hot_water_system.HotWaterSystem.compute_thermal_energy_rate

    def wrapped(self, return_water_temp, outside_temp):
        billed = original(self, return_water_temp, outside_temp)
        src = self._heat_source
        tank = getattr(src, "_current_temperature", None)
        setpoint = getattr(src, "_supply_water_temperature_setpoint", None)
        stamp = getattr(src, "_observation_timestamp", None)
        if tank is None or setpoint is None or (stamp is not None and stamp == acc.get("_stamp")):
            return billed                                           # called twice a step: count each step once
        acc["_stamp"] = stamp
        ret =float(sum(return_water_temp) / len(return_water_temp)) if isinstance(return_water_temp, list) \
            else float(return_water_temp)
        flow = float(self.total_flow_rate)
        dt = src._last_step_duration.total_seconds() or 300.0
        delivered = cw * flow * (tank - ret)
        acc["calls"] += 1
        acc["billed_j"] += billed * dt
        acc["delivered_j"] += delivered * dt
        acc["loss_j"] += src.compute_thermal_dissipation_rate(tank, outside_temp) * dt
        acc["unbilled_flow_j"] += cw * flow * max(tank - max(setpoint, ret), 0.0) * dt
        acc["tank_above_setpoint_calls"] += int(tank > setpoint + 0.5)
        acc["setpoint_below_return_calls"] += int(setpoint < ret)
        if acc["tank_start_k"] is None:
            acc["tank_start_k"] = float(tank)
            acc["capacity_j_per_k"] = cw * float(getattr(src, "_water_capacity", 0.0))
        acc["tank_end_k"] = float(tank)
        return billed

    hot_water_system.HotWaterSystem.compute_thermal_energy_rate = wrapped
    base = G36PolicyB2R2 if policy == "rule" else b2rl_policy_class(ROOT / policy)
    steps = int(os.environ.get("BILLING_MAX_STEPS", "0")) or None                   # smoke runs only
    row = run_gateway_episode(spec_dir(spec), base, policy_key="billing-diag", label=spec, max_steps=steps)
    acc.pop("_stamp", None)
    stored = (acc["tank_end_k"] - acc["tank_start_k"]) * acc["capacity_j_per_k"] if acc["calls"] else 0.0
    need = acc["delivered_j"] + acc["loss_j"] + stored
    return policy, spec, {
        "cost": -row["mean_reward"], "steps": row["steps"], "comfort_part": row["comfort_part"],
        "energy_part": row["energy_part"], "carbon_part": row["carbon_part"], **acc, "stored_j": stored,
        "unbilled_share": 1.0 - acc["billed_j"] / need if need > 0 else 0.0}


def main():
    workers, policies, specs = int(sys.argv[1]), sys.argv[2].split(","), sys.argv[3:]
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(p, s) for p in policies for s in specs]
    with mp.get_context("spawn").Pool(min(workers, len(jobs)), maxtasksperchild=1) as pool:
        for policy, spec, r in pool.imap_unordered(episode, jobs):
            name = "rule" if policy == "rule" else Path(policy).stem.replace("model-", "")
            suffix = "-rom" if os.environ.get("BILLING_SIM") == "rom" else ""
            (OUT / f"{name}__{spec.replace('/', '__')}{suffix}.json").write_text(json.dumps(r, indent=1) + "\n",
                                                                          encoding="utf-8")
            print(f"{name:18s} {spec:36s} cost {r['cost']:.5f} heat delivered {r['delivered_j'] / 3.6e9:9.1f} MWh "
                  f"billed {r['billed_j'] / 3.6e9:9.1f} MWh unbilled {100 * r['unbilled_share']:5.1f}% "
                  f"(tank above setpoint {r['tank_above_setpoint_calls']}/{r['calls']} calls)", flush=True)


if __name__ == "__main__":
    main()
