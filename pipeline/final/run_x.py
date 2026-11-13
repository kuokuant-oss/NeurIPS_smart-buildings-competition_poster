"""Reviewer's own closed loops for the scratch models (fast simulator, same episode code as the workspace's
official30_rom.py / newstations_rom.py). Reads the workspace, writes only into the scratch folder given as argv[1].
No reduced model is built (SB_FASTSIM_NO_BUILD=1): a scene without a cached one is reported as an error.
Each row also records the episode-mean outside air the policy saw (oat_mean_c) when the policy has a slow-layer tracker.
A model name not found in the scratch folder is read from the workspace models folder.
python3 -B pipeline/launch/launch.py <this file> <scratch dir> <workers> <names...>"""
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
DATA = ROOT / "data/raw/kaggle"
SCR = Path(sys.argv[1])
WS_MODELS = ROOT / "data/runs/b2rl/search/models"


def scenarios():
    off = sorted(p.parent.as_posix() for p in (DATA / "smart_buildings/training_scenarios").glob("*/*/config.gin"))
    new = sorted(p.parent.as_posix() for p in (ROOT / "data/processed/newstations").glob("*/*/config.gin"))
    return [("off", s) for s in off if "minneap" not in s.lower()] + [("new", s) for s in new if "minneap" not in s.lower()]


def _episode(args):
    name, (tag, scen) = args
    os.environ["SB_FASTSIM"] = "rom"
    os.environ["SB_FASTSIM_NO_BUILD"] = "1"
    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    for p in (str(ROOT / "packages"), str(DATA)):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(DATA)
    from smart_building.fastsim import install_from_env, installed
    if not installed():
        install_from_env()
    from smart_building.evaluation.episode import run_episode
    from smart_building.policies.b2rl import b2rl_policy_class
    path = SCR / f"model-{name}.json"
    if not path.is_file():
        path = WS_MODELS / f"model-{name}.json"
    base = b2rl_policy_class(path)
    seen = {}

    class Recording(base):
        def get_action(self, obs):
            a = super().get_action(obs)
            try:
                t = self.policy.slow.tracker if hasattr(self, "policy") else self.slow.tracker
                if t.outside_n:
                    seen["oat"] = t.outside_sum / t.outside_n
            except Exception:  # noqa: BLE001
                pass
            return a

    t0 = time.perf_counter()
    key = "/".join(Path(scen).parts[-2:])
    try:
        r = run_episode(Path(scen), Recording, 8640, policy_key=name, label=key)
    except Exception as error:  # noqa: BLE001
        return name, tag, key, {"error": repr(error)[:300]}
    return name, tag, key, {"cost": -float(r["mean_reward"]), "steps": int(r["steps"]), "comfort_part": float(r["comfort_part"]),
                            "energy_part": float(r["energy_part"]), "carbon_part": float(r.get("carbon_part", 0.0)),
                            "oat_mean_c": seen.get("oat"), "elapsed_seconds": time.perf_counter() - t0}


if __name__ == "__main__":
    workers, names = int(sys.argv[2]), sys.argv[3:]
    out = SCR / "runs"
    out.mkdir(parents=True, exist_ok=True)
    recs = {(n, t): (json.loads((out / f"{t}-{n}.json").read_text()) if (out / f"{t}-{n}.json").is_file() else {"rows": {}})
            for n in names for t in ("off", "new")}
    todo = [(n, s) for s in scenarios() for n in names if "/".join(Path(s[1]).parts[-2:]) not in recs[(n, s[0])]["rows"]]
    print(f"{len(todo)} episodes on {workers} workers", flush=True)
    with mp.get_context("spawn").Pool(workers, maxtasksperchild=1) as pool:
        for name, tag, key, row in pool.imap_unordered(_episode, todo):
            if "error" in row:
                print(f"ERROR {name} {tag} {key}: {row['error']}", flush=True)
                continue
            recs[(name, tag)]["rows"][key] = row
            (out / f"{tag}-{name}.json").write_text(json.dumps(recs[(name, tag)], indent=1), encoding="utf-8")
            print(f"{name} {tag} {key} cost {row['cost']:.6f} oat {row['oat_mean_c']} ({row['elapsed_seconds']:.0f} s)", flush=True)
    print("DONE", flush=True)
