"""Controllers on the new validation buildings (<scratch>/val2/manifest.json), corrected simulator
(SB_TANK=balance, SB_RETURN_AIR=flow). Writes <scratch>/runs_v2/{off,new}-<name>.json in the format of runs_p2
(off = public stations, new = unseen stations).
Names: model names (b2rl models), g36_default, org_bang_bang, org_fixed_setpoint, org_pid, org_pid_water.
python3 -B pipeline/launch/launch.py v2_run.py <scratch dir> <workers> <limit or 0> <names...>"""
import json, os, sys, time
import multiprocessing as mp
from pathlib import Path
os.environ["SB_RETURN_AIR"] = "flow"; os.environ["SB_TANK"] = "balance"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_c  # noqa: E402
run_x = run_c.run_x


def _episode(arg):
    name, (tag, scen) = arg
    if not (name.startswith("org_") or name == "g36_default"):
        return run_c._episode(arg)
    for p in (str(run_x.ROOT / "packages"), str(run_x.DATA)):
        if p not in sys.path:
            sys.path.insert(0, p)
    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    run_c._tank_balance()
    from smart_building.evaluation.episode import run_episode
    if name == "g36_default":
        from smart_building.policies.g36 import G36PolicyB2R2 as cls
    else:
        from smart_buildings import evaluate
        cls = evaluate.POLICY_MAP[name[4:]]
    t0 = time.perf_counter(); key = "/".join(Path(scen).parts[-2:])
    try:
        r = run_episode(Path(scen), cls, 8640, policy_key=name, label=key)
    except Exception as error:  # noqa: BLE001
        return name, tag, key, {"error": repr(error)[:300]}
    return name, tag, key, {"cost": -float(r["mean_reward"]), "steps": int(r["steps"]), "comfort_part": float(r["comfort_part"]),
                            "energy_part": float(r["energy_part"]), "carbon_part": float(r.get("carbon_part", 0.0)),
                            "elapsed_seconds": time.perf_counter() - t0}


if __name__ == "__main__":
    workers, limit, names = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4:]
    man = json.loads((run_x.SCR / "val2" / "manifest.json").read_text())
    scen = [({"seen": "off", "unseen": "new"}[r["tag"]], r["dir"]) for r in man["scenarios"]]
    if limit: scen = scen[:limit]
    out = run_x.SCR / "runs_v2"; out.mkdir(exist_ok=True)
    f = lambda n, t: out / f"{t}-{n}.json"
    recs = {(n, t): (json.loads(f(n, t).read_text()) if f(n, t).is_file() else {"rows": {}}) for n in names for t in ("off", "new")}
    todo = [(n, s) for n in names for s in scen if "/".join(Path(s[1]).parts[-2:]) not in recs[(n, s[0])]["rows"]]
    print(len(todo), "episodes", flush=True)
    with mp.get_context("spawn").Pool(workers, maxtasksperchild=1) as pool:
        for name, tag, key, row in pool.imap_unordered(_episode, todo):
            if "error" in row:
                print("ERROR", name, tag, key, row["error"], flush=True); continue
            recs[(name, tag)]["rows"][key] = row
            f(name, tag).write_text(json.dumps(recs[(name, tag)], indent=1), encoding="utf-8")
            print(name, tag, key, f"{row['cost']:.6f}", f"({row['elapsed_seconds']:.0f} s)", flush=True)
    print("DONE", " ".join(names), flush=True)
