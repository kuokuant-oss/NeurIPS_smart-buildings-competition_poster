"""Builds the reduced thermal models of the fast simulator for the new validation scenarios into <scratch>/rom-cache
(SB_FASTSIM_CACHE), one per building and convection coefficient; the workspace cache is not touched.
python3 -B pipeline/launch/launch.py prebuild_v2.py <scratch> <workers>"""
import json, os, re, sys, time
import multiprocessing as mp
from pathlib import Path
SCR = Path(sys.argv[1]); ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]; DATA = ROOT / "data/raw/kaggle"
os.environ["SB_FASTSIM_CACHE"] = str(SCR / "rom-cache"); os.environ["SB_FASTSIM"] = "rom"; os.environ.pop("SB_FASTSIM_NO_BUILD", None)


def _build(scen):
    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "2"
    for p in (str(ROOT / "packages"), str(DATA)):
        if p not in sys.path: sys.path.insert(0, p)
    os.chdir(DATA)
    from smart_building.fastsim import install_from_env, installed
    if not installed(): install_from_env()
    from smart_building.evaluation.episode import run_episode
    from smart_building.policies.g36 import G36PolicyB2R2
    t0 = time.perf_counter()
    try:
        run_episode(Path(scen), G36PolicyB2R2, 3, policy_key="prebuild", label=scen)
    except Exception as error:  # noqa: BLE001
        return scen, "ERROR " + repr(error)[:300], time.perf_counter() - t0
    return scen, "ok", time.perf_counter() - t0


if __name__ == "__main__":
    (SCR / "rom-cache").mkdir(exist_ok=True)
    man = json.loads((SCR / "val2" / "manifest.json").read_text()); seen, todo = set(), []
    for r in man["scenarios"]:
        h = re.search(r"^convection_coefficient = (.*)$", (Path(r["dir"]) / "config.gin").read_text(), flags=re.M).group(1).strip()
        if (r["building"], h) not in seen:
            seen.add((r["building"], h)); todo.append(r["dir"])
    print(len(todo), "models to build", flush=True)
    with mp.get_context("spawn").Pool(int(sys.argv[2]), maxtasksperchild=1) as pool:
        for scen, status, sec in pool.imap_unordered(_build, todo):
            print("/".join(Path(scen).parts[-2:]), status, f"{sec:.0f} s", flush=True)
    print("DONE", len(list((SCR / "rom-cache").glob("rom-*.npz"))), "models", flush=True)
