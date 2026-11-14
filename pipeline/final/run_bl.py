"""The four reference controllers of the starter kit (bang_bang, fixed_setpoint, pid, pid_water) on the 90 local
validation scenes, corrected simulator (SB_TANK=balance, SB_RETURN_AIR=flow). Writes <scratch>/runs_p2/{off,new}-org_<name>.json.
python3 -B pipeline/launch/launch.py run_bl.py <scratch dir> <workers>"""
import json, os, sys, time
import multiprocessing as mp
from pathlib import Path
os.environ["SB_RETURN_AIR"] = "flow"; os.environ["SB_TANK"] = "balance"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_c  # noqa: E402
run_x = run_c.run_x
NAMES = ["bang_bang", "fixed_setpoint", "pid", "pid_water"]


def _episode(arg):
    name, tag, scen = arg
    for p in (str(run_x.ROOT / "packages"), str(run_x.DATA)):
        if p not in sys.path:
            sys.path.insert(0, p)
    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    run_c._tank_balance()
    from smart_building.evaluation.episode import run_episode
    from smart_buildings import evaluate
    cls = evaluate.POLICY_MAP[name] if hasattr(evaluate, "POLICY_MAP") else None
    t0 = time.perf_counter(); key = "/".join(Path(scen).parts[-2:])
    try:
        r = run_episode(Path(scen), cls, 8640, policy_key="org_" + name, label=key)
    except Exception as error:  # noqa: BLE001
        return name, tag, key, {"error": repr(error)[:300]}
    return name, tag, key, {"cost": -float(r["mean_reward"]), "steps": int(r["steps"]), "comfort_part": float(r["comfort_part"]),
                            "energy_part": float(r["energy_part"]), "carbon_part": float(r.get("carbon_part", 0.0)),
                            "elapsed_seconds": time.perf_counter() - t0}


if __name__ == "__main__":
    workers = int(sys.argv[2]); out = run_x.SCR / "runs_p2"; out.mkdir(parents=True, exist_ok=True)
    f = lambda n, t: out / f"{t}-org_{n}.json"
    recs = {(n, t): (json.loads(f(n, t).read_text()) if f(n, t).is_file() else {"rows": {}}) for n in NAMES for t in ("off", "new")}
    todo = [(n, t, s) for n in NAMES for t, s in run_x.scenarios() if "/".join(Path(s).parts[-2:]) not in recs[(n, t)]["rows"]]
    print(len(todo), "episodes", flush=True)
    with mp.get_context("spawn").Pool(workers, maxtasksperchild=1) as pool:
        for n, tag, key, row in pool.imap_unordered(_episode, todo):
            if "error" in row:
                print("ERROR", n, tag, key, row["error"], flush=True); continue
            recs[(n, tag)]["rows"][key] = row
            f(n, tag).write_text(json.dumps(recs[(n, tag)], indent=1), encoding="utf-8")
            print(n, tag, key, f"{row['cost']:.6f}", flush=True)
    print("DONE", flush=True)
