"""Builds the reduced thermal models of the fast simulator for the 90 scenarios of the six starter-kit buildings
(30 at the public stations, 60 at the unseen stations) into the default cache data/runs/fastsim/rom-cache, one per
building and convection coefficient (the same construction as prebuild_v2.py).
python3 -B pipeline/launch/launch.py pipeline/final/prebuild_official.py <workers>"""
import multiprocessing as mp
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/raw/kaggle"
os.environ["SB_FASTSIM"] = "rom"
os.environ.pop("SB_FASTSIM_NO_BUILD", None)
os.environ.pop("SB_FASTSIM_CACHE", None)


def scenario_dirs():
    off = sorted(p.parent for p in (DATA / "smart_buildings/training_scenarios").glob("*/*/config.gin"))
    new = sorted(p.parent for p in (ROOT / "data/processed/newstations").glob("*/*/config.gin"))
    return [d for d in off + new if "minneap" not in d.as_posix().lower()]


def _build(scen):
    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "2"
    for p in (str(ROOT / "packages"), str(DATA)):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(DATA)
    from smart_building.fastsim import install_from_env, installed
    if not installed():
        install_from_env()
    from smart_building.evaluation.episode import run_episode
    from smart_building.policies.g36 import G36PolicyB2R2
    t0 = time.perf_counter()
    try:
        run_episode(Path(scen), G36PolicyB2R2, 3, policy_key="prebuild", label=scen)
    except Exception as error:  # noqa: BLE001
        return scen, "ERROR " + repr(error)[:300], time.perf_counter() - t0
    return scen, "ok", time.perf_counter() - t0


if __name__ == "__main__":
    seen, todo = set(), []
    for d in scenario_dirs():
        h = re.search(r"^convection_coefficient = (.*)$", (d / "config.gin").read_text(), flags=re.M).group(1).strip()
        if (d.parent.name, h) not in seen:
            seen.add((d.parent.name, h))
            todo.append(str(d))
    print(len(todo), "models to build", flush=True)
    with mp.get_context("spawn").Pool(int(sys.argv[1]), maxtasksperchild=1) as pool:
        for scen, status, sec in pool.imap_unordered(_build, todo):
            print("/".join(Path(scen).parts[-2:]), status, f"{sec:.0f} s", flush=True)
    print("DONE", flush=True)
