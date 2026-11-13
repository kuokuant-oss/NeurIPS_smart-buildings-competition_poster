"""run_x.py restricted to the scenes whose 'building/station' key matches the regular expression in SCENE_RE."""
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_x  # noqa: E402

_all = run_x.scenarios
_re = re.compile(os.environ.get("SCENE_RE", "."))
run_x.scenarios = lambda: [s for s in _all() if _re.search("/".join(Path(s[1]).parts[-2:]))]

EXTRA = {"setback_heat_drop_k": (4.0, 5.0, 6.0, 8.0)}   # reviewer dose ladder: in-memory only, nothing written


def _episode(args):
    for p in (str(run_x.ROOT / "packages"), str(run_x.DATA)):
        if p not in sys.path:
            sys.path.insert(0, p)
    from smart_building.policies import g36
    for k, v in EXTRA.items():
        g36.C3_CONFIGURABLE[k] = tuple(g36.C3_CONFIGURABLE.get(k, ())) + tuple(x for x in v if x not in g36.C3_CONFIGURABLE.get(k, ()))
    if os.environ.get("SB_TANK") == "balance":
        _tank_balance()
    return run_x._episode(args)


def _tank_balance():
    """In-memory copy of starter-kit PR #2 (energy-balanced hot-water tank, commit ec27d9e) for this process only:
    the tank follows C dT/dt = Q_burner - Q_coils - Q_loss and gas is billed as Q_burner. Nothing on disk changes."""
    os.environ["SB_FASTSIM"] = "rom"
    os.environ["SB_FASTSIM_NO_BUILD"] = "1"
    os.chdir(run_x.DATA)
    from smart_building.fastsim import install_from_env, installed
    if not installed():
        install_from_env()
    from smart_buildings.smart_control.simulator import boiler as bpy
    from smart_buildings.smart_control.simulator import simulator_flexible_floor_plan as sfg
    cls, sim = bpy.Boiler, sfg.SimulatorFlexibleGeometries
    if getattr(cls, "_reviewer_balance", False):
        return
    c = bpy.constants
    reset0, sensor0, rate0, step0 = cls.reset, cls.supply_water_temperature_sensor.fget, cls.compute_thermal_energy_rate, sim.execute_step_sim

    def reset(self):
        reset0(self)
        self._burner_heat_rate = 0.0
        self.energy_balance = bool(self._has_tank and self._heating_rate > 0)

    def apply_heat_draw(self, coil_heat_w, outside_temp, duration_s):
        if duration_s <= 0.0:
            return
        capacity = c.WATER_HEAT_CAPACITY * c.WATER_DENSITY * self._water_capacity
        loss = self.compute_thermal_dissipation_rate(self._current_temperature, outside_temp)
        needed = capacity * (self._supply_water_temperature_setpoint - self._current_temperature) / duration_s + coil_heat_w + loss
        max_rate = max(coil_heat_w, 0.0) + loss + capacity * self._heating_rate / 60.0
        self._burner_heat_rate = min(max(needed, 0.0), max_rate)
        self._current_temperature += (self._burner_heat_rate - coil_heat_w - loss) * duration_s / capacity

    def rate(self, return_water_temp, outside_temp, total_flow_rate):
        if getattr(self, "energy_balance", False):
            return self._burner_heat_rate
        return rate0(self, return_water_temp, outside_temp, total_flow_rate)

    def step(self, *a, **k):
        ts = self._current_timestamp
        out = step0(self, *a, **k)
        hws = self._hvac.hot_water_system
        src = hws._heat_source
        if getattr(src, "energy_balance", False):
            coil = c.WATER_HEAT_CAPACITY * c.WATER_DENSITY * hws.total_flow_rate * (
                hws.supply_water_temperature_sensor - hws.return_water_temperature_sensor)
            src.apply_heat_draw(coil, self._weather_controller.get_current_temp(ts), self._time_step_sec)
        return out

    cls.reset = reset
    cls.apply_heat_draw = apply_heat_draw
    cls.compute_thermal_energy_rate = rate
    cls.supply_water_temperature_sensor = property(
        lambda self: self._current_temperature if getattr(self, "energy_balance", False) else sensor0(self))
    sim.execute_step_sim = step
    cls._reviewer_balance = True


if __name__ == "__main__":
    import json
    import multiprocessing as mp
    workers, names = int(sys.argv[2]), sys.argv[3:]
    out = run_x.SCR / os.environ.get("RUNS_DIR", "runs")
    out.mkdir(parents=True, exist_ok=True)
    recs = {(n, t): (json.loads((out / f"{t}-{n}.json").read_text()) if (out / f"{t}-{n}.json").is_file() else {"rows": {}})
            for n in names for t in ("off", "new")}
    todo = [(n, s) for s in run_x.scenarios() for n in names if "/".join(Path(s[1]).parts[-2:]) not in recs[(n, s[0])]["rows"]]
    print(f"{len(todo)} episodes on {workers} workers", flush=True)
    with mp.get_context("spawn").Pool(workers, maxtasksperchild=1) as pool:
        for name, tag, key, row in pool.imap_unordered(_episode, todo):
            if "error" in row:
                print(f"ERROR {name} {tag} {key}: {row['error']}", flush=True)
                continue
            recs[(name, tag)]["rows"][key] = row
            (out / f"{tag}-{name}.json").write_text(json.dumps(recs[(name, tag)], indent=1), encoding="utf-8")
            print(f"{name} {tag} {key} cost {row['cost']:.6f} ({row['elapsed_seconds']:.0f} s)", flush=True)
    print("DONE", flush=True)
