"""Single-option records (2026-11-03, docs/plan/option-catalogue-20261103.md): one whole 30-day episode per (model,
scenario) on the fast simulator (rom), recorded by pipeline/setup/dash_record.py episode() (comfort / energy / carbon,
per-device parts, hourly series) with the hot-water billing audit of pipeline/setup/billing_diag.py in the same episode
(HotWaterSystem.compute_thermal_energy_rate wrapped, attributes read only, so the episode is unchanged).

Writes data/runs/b2rl/options/records/<name>__<spec>.json and data/runs/b2rl/options/billing/<name>__<spec>.json;
resumable (finished pairs are skipped). Jobs run in the order given (models outer, scenarios inner).

python3 -B pipeline/launch/launch.py pipeline/setup/options/opt_record.py <workers> <model.json>,... [design | <spec> ...]
"""
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "packages"))
sys.path.insert(0, str(ROOT / "pipeline/setup"))
OUT = ROOT / "data/runs/b2rl/options"
LOG = OUT / "record.log"


def name_of(model):
    return Path(model).stem.replace("model-", "")


def log(text):
    line = f"{time.strftime('%m-%d %H:%M:%S')} {text}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")


def episode(job):
    model, spec = job
    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    try:
        import dash_record
        dash_record.OUT = OUT
        os.environ["SB_FASTSIM"] = "rom"
        from smart_building.learned.b2rl_env import setup_simulator
        setup_simulator(ROOT)
        from smart_buildings.smart_control.simulator import hot_water_system
        from smart_buildings.smart_control.utils import constants
        cw = constants.WATER_HEAT_CAPACITY * constants.WATER_DENSITY
        acc = {"calls": 0, "billed_j": 0.0, "delivered_j": 0.0, "loss_j": 0.0, "tank_above_setpoint_calls": 0,
               "unbilled_flow_j": 0.0, "tank_start_k": None, "tank_end_k": None, "capacity_j_per_k": None}
        original = hot_water_system.HotWaterSystem.compute_thermal_energy_rate

        def wrapped(self, return_water_temp, outside_temp):
            billed = original(self, return_water_temp, outside_temp)
            src = self._heat_source
            tank = getattr(src, "_current_temperature", None)
            setpoint = getattr(src, "_supply_water_temperature_setpoint", None)
            stamp = getattr(src, "_observation_timestamp", None)
            if tank is None or setpoint is None or (stamp is not None and stamp == acc.get("_stamp")):
                return billed                                   # called twice a step: count each step once
            acc["_stamp"] = stamp
            ret = float(sum(return_water_temp) / len(return_water_temp)) if isinstance(return_water_temp, list) \
                else float(return_water_temp)
            flow = float(self.total_flow_rate)
            dt = src._last_step_duration.total_seconds() or 300.0
            acc["calls"] += 1
            acc["billed_j"] += billed * dt
            acc["delivered_j"] += cw * flow * (tank - ret) * dt
            acc["loss_j"] += src.compute_thermal_dissipation_rate(tank, outside_temp) * dt
            acc["unbilled_flow_j"] += cw * flow * max(tank - max(setpoint, ret), 0.0) * dt
            acc["tank_above_setpoint_calls"] += int(tank > setpoint + 0.5)
            if acc["tank_start_k"] is None:
                acc["tank_start_k"] = float(tank)
                acc["capacity_j_per_k"] = cw * float(getattr(src, "_water_capacity", 0.0))
            acc["tank_end_k"] = float(tank)
            return billed

        hot_water_system.HotWaterSystem.compute_thermal_energy_rate = wrapped
        try:
            name, _, summary, _ = dash_record.episode((model, spec))
        finally:
            hot_water_system.HotWaterSystem.compute_thermal_energy_rate = original
        acc.pop("_stamp", None)
        stored = (acc["tank_end_k"] - acc["tank_start_k"]) * acc["capacity_j_per_k"] if acc["calls"] else 0.0
        need = acc["delivered_j"] + acc["loss_j"] + stored
        bill = {**acc, "stored_j": stored, "unbilled_share": 1.0 - acc["billed_j"] / need if need > 0 else 0.0,
                "cost": summary["cost"]}
        dash_record.write_json(OUT / "billing" / f"{name}__{spec.replace('/', '__')}.json", bill)
        return name, spec, summary["cost"], summary["elapsed_seconds"], None
    except Exception as error:  # noqa: BLE001 - report and continue
        import traceback
        return name_of(model), spec, None, None, traceback.format_exc()[-1500:]


def design_specs():
    import group_stage4 as g4
    real, gen1, _ = g4.scenario_sets()
    return real + gen1


def main():
    workers, models = int(sys.argv[1]), sys.argv[2].split(",")
    specs = design_specs() if sys.argv[3:] in ([], ["design"]) else sys.argv[3:]
    jobs = []
    for model in models:
        rel = str(Path(model).resolve().relative_to(ROOT))
        for spec in specs:
            if not (OUT / "records" / f"{name_of(model)}__{spec.replace('/', '__')}.json").is_file():
                jobs.append((rel, spec))
    log(f"recording {len(jobs)} episodes ({len(models)} models x {len(specs)} scenarios, {workers} workers)")
    if not jobs:
        return
    done = failed = 0
    started = time.time()
    with mp.get_context("spawn").Pool(min(workers, len(jobs)), maxtasksperchild=1) as pool:
        for name, spec, cost, elapsed, error in pool.imap(episode, jobs):
            done += 1
            if error:
                failed += 1
                log(f"FAILED {name} {spec}: {error}")
            elif done % 22 == 0 or done == len(jobs):
                rate = done / max(time.time() - started, 1.0) * 60.0
                log(f"{done}/{len(jobs)} done ({rate:.1f}/min); last {name} {spec} cost {cost:.6f}")
    log(f"finished: {done - failed} ok, {failed} failed")


if __name__ == "__main__":
    main()
