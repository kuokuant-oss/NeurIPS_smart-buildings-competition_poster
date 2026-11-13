"""Guideline rule with default settings (G36PolicyB2R2, the 0.01441 submission) on the 90 local validation scenes,
corrected simulator (SB_TANK=balance, SB_RETURN_AIR=flow). Writes <scratch>/runs_p2/{off,new}-g36_default.json.
python3 -B pipeline/launch/launch.py run_r2.py <scratch dir> <workers>"""
import json, os, sys, time
import multiprocessing as mp
from pathlib import Path
os.environ["SB_RETURN_AIR"] = "flow"; os.environ["SB_TANK"] = "balance"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_c  # noqa: E402
run_x = run_c.run_x
NAME = "g36_default"


def _episode(arg):
    tag, scen = arg
    for p in (str(run_x.ROOT / "packages"), str(run_x.DATA)):
        if p not in sys.path:
            sys.path.insert(0, p)
    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    run_c._tank_balance()
    from smart_building.evaluation.episode import run_episode
    from smart_building.policies.g36 import G36PolicyB2R2
    t0 = time.perf_counter(); key = "/".join(Path(scen).parts[-2:])
    try:
        r = run_episode(Path(scen), G36PolicyB2R2, 8640, policy_key=NAME, label=key)
    except Exception as error:  # noqa: BLE001
        return tag, key, {"error": repr(error)[:300]}
    return tag, key, {"cost": -float(r["mean_reward"]), "steps": int(r["steps"]), "comfort_part": float(r["comfort_part"]),
                      "energy_part": float(r["energy_part"]), "carbon_part": float(r.get("carbon_part", 0.0)),
                      "elapsed_seconds": time.perf_counter() - t0}


if __name__ == "__main__":
    workers = int(sys.argv[2]); out = run_x.SCR / "runs_p2"
    recs = {t: (json.loads((out / f"{t}-{NAME}.json").read_text()) if (out / f"{t}-{NAME}.json").is_file() else {"rows": {}}) for t in ("off", "new")}
    todo = [s for s in run_x.scenarios() if "/".join(Path(s[1]).parts[-2:]) not in recs[s[0]]["rows"]]
    print(len(todo), "episodes", flush=True)
    with mp.get_context("spawn").Pool(workers, maxtasksperchild=1) as pool:
        for tag, key, row in pool.imap_unordered(_episode, todo):
            if "error" in row:
                print("ERROR", tag, key, row["error"], flush=True); continue
            recs[tag]["rows"][key] = row
            (out / f"{tag}-{NAME}.json").write_text(json.dumps(recs[tag], indent=1), encoding="utf-8")
            print(tag, key, f"{row['cost']:.6f}", flush=True)
    print("DONE", flush=True)
