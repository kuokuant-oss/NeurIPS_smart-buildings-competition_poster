"""Pipeline 5 episode recorder (2026-11-04, docs/plan/geo-pipeline5.md): one fast-simulator episode per (model,
scenario) with the dashboard recorder (pipeline/setup/dash_record.py episode: cost, comfort / energy / carbon, device
parts, hourly series) and the hot-water billing audit of pipeline/setup/options/opt_record.py (HotWaterSystem
.compute_thermal_energy_rate wrapped, attributes read only, so the episode is unchanged). Writes
<out>/records/[smoke__]<name>__<spec>.json and <out>/billing/[smoke__]<name>__<spec>.json; "smoke__" when max_steps
is set (DASH_MAX_STEPS). Resumable per episode: finished pairs are skipped.

The tree stage's data collection (戊) uses a room policy mode "assign" (room number -> option index) that the
package does not have: it is added here, in the recording worker only (rooms.option_index wrapped; every other mode
goes to the original), and the worker refuses an "assign" episode in which the wrapper was never called (the policy
would have fallen back silently). Never part of a submission."""
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import traceback

from geo5_lib import (ROOT, POLL_S, Monitor, Wake, kill_tree, read_json, worker_count)


def tag(max_steps):
    return "smoke__" if max_steps else ""


def key(name, spec):
    return f"{name}__{spec.replace('/', '__')}.json"


def record_path(out, name, spec, max_steps=None):
    return Path(out) / "records" / f"{tag(max_steps)}{key(name, spec)}"


def billing_path(out, name, spec, max_steps=None):
    return Path(out) / "billing" / f"{tag(max_steps)}{key(name, spec)}"


def _patch_assign():
    import numpy as np
    from smart_building.g36 import rooms
    if getattr(rooms, "_geo5_assign", False):
        return rooms
    original = rooms.option_index

    def option_index(policy, features, room_numbers, oat=None, hour=None, period=None):
        if policy.get("mode") == "assign":
            rooms._geo5_assign_calls += 1
            table, default = policy["assign"], int(policy.get("default", 0))
            return np.array([int(table.get(str(int(n)), default)) for n in room_numbers], dtype=int)
        ov = policy.get("geo5_overlay")
        if ov:                    # stage 戊: base tree's choice + K x the room's code (geo5_tree.overlay_model)
            rooms._geo5_assign_calls += 1
            base = original({**policy, "options": policy["options"][:int(ov["K"])]}, features, room_numbers, oat,
                            hour, period)
            codes = np.array([int(ov["codes"].get(str(int(n)), 0)) for n in room_numbers], dtype=int)
            return np.asarray(base, dtype=int) + int(ov["K"]) * codes
        return original(policy, features, room_numbers, oat, hour, period)

    rooms.option_index = option_index
    rooms._geo5_assign, rooms._geo5_assign_calls = True, 0
    return rooms


def episode(job):
    """job = (model path relative to ROOT, spec, out dir, max_steps or None)."""
    model, spec, out, max_steps = job
    name = Path(model).stem.replace("model-", "")
    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    if max_steps:
        os.environ["DASH_MAX_STEPS"] = str(int(max_steps))
    else:
        os.environ.pop("DASH_MAX_STEPS", None)
    t0 = time.time()
    try:
        import dash_record
        dash_record.OUT = Path(out)
        os.environ["SB_FASTSIM"] = "rom"
        from smart_building.learned.b2rl_env import setup_simulator
        setup_simulator(ROOT)
        rooms = _patch_assign()
        rooms._geo5_assign_calls = 0
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
        rp = read_json(ROOT / model, {}).get("room_policy") or {}
        if (rp.get("mode") == "assign" or rp.get("geo5_overlay")) and rooms._geo5_assign_calls == 0:
            raise RuntimeError("assign / overlay room policy never applied (the policy fell back to the rule)")
        calls = rooms._geo5_assign_calls
        acc.pop("_stamp", None)
        stored = (acc["tank_end_k"] - acc["tank_start_k"]) * acc["capacity_j_per_k"] if acc["calls"] else 0.0
        need = acc["delivered_j"] + acc["loss_j"] + stored
        bill = {**acc, "stored_j": stored, "unbilled_share": 1.0 - acc["billed_j"] / need if need > 0 else 0.0,
                "cost": summary["cost"], "steps": summary["steps"], "room_wrapper_calls": calls}
        dash_record.write_json(billing_path(out, name, spec, max_steps), bill)
        return {"name": name, "spec": spec, "cost": summary["cost"], "seconds": time.time() - t0, "error": None,
                "room_wrapper_calls": calls}
    except Exception:  # noqa: BLE001 - reported to the parent, which wakes the coordinator
        return {"name": name, "spec": spec, "cost": None, "seconds": time.time() - t0,
                "error": traceback.format_exc()[-1500:]}


def record(ctx, job, pairs, out, workers, max_steps=None, stop_after=None):
    """Record every (model rel path, spec) pair not yet on disk, under a Monitor: progress every <= 4 min, memory
    and 2x-estimate checks, a failed episode or a stall wakes the coordinator. stop_after: test hook, a simulated
    interruption after that many episodes. Returns the number recorded."""
    todo = [(m, s) for m, s in pairs if not record_path(out, Path(m).stem.replace("model-", ""), s, max_steps).is_file()]
    kind = f"episode_worker_min_{'smoke' if max_steps else 'full'}"
    if not todo:
        ctx.log("EVENT", f"{job}: all {len(pairs)} episodes already recorded")
        return 0
    n = worker_count(workers)
    per = ctx.estimate(kind)
    est = per * -(-len(todo) // n) if per else None
    stall = max(20.0, 3.0 * per) if per else (20.0 if max_steps else 45.0)
    ctx.log("EVENT", f"{job}: recording {len(todo)} episodes with {n} "
                     f"workers; estimate {'%.0f min (measured %.2f min per episode per worker)' % (est, per) if est else 'after the first episodes'}")
    mon = Monitor(ctx, job, len(todo), estimate_min=est, fix_after=None if est else min(len(todo), n), stall_min=stall)
    done = failed = 0
    errors = []
    pool = mp.get_context("spawn").Pool(n, maxtasksperchild=1)
    try:
        it = pool.imap_unordered(episode, [(m, s, str(out), max_steps) for m, s in todo])
        while done < len(todo):
            try:
                r = it.next(timeout=POLL_S)
            except mp.TimeoutError:
                mon.tick(done)
                continue
            done += 1
            if r["error"]:
                failed += 1
                errors.append(r)
                ctx.log("ERROR", f"{job}: episode {r['name']} {r['spec']} failed: {r['error'][-400:]}")
            mon.tick(done, f"last {r['name']} {r['spec']} cost {r['cost']}" if r["cost"] is not None else "")
            if failed:
                break
            if stop_after and done >= stop_after:
                raise Wake("crash", f"{job}: simulated interruption after {done} episodes (test hook "
                                    "--stop-after-episodes)", {"done": done, "total": len(todo)})
        pool.close()
        pool.join()
    except BaseException:
        pool.terminate()
        kill_tree()
        raise
    el = mon.elapsed_min()
    if done and not failed:
        ctx.measured(kind, el * n / done if done >= n else el)
    mon.tick(done, force=True)
    if failed:
        diag = ctx.dir / "diag" / f"failed-{time.strftime('%Y%m%d-%H%M%S')}.json"
        from geo5_lib import atomic_json
        atomic_json(diag, errors)
        raise Wake("crash", f"{job}: {failed} episode(s) failed, first {errors[0]['name']} {errors[0]['spec']}",
                   {"failed": failed}, [diag])
    return done


if __name__ == "__main__":
    sys.exit("library module; used by pipeline/setup/geo_pipeline5.py")
