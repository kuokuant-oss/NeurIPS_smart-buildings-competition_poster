"""Entry point of the reproduction repository (run with the host python3 from the repository root).

    python3 run.py env                       # Python 3.13 and the locked environment (.venv) with uv
    python3 run.py setup [--workers N]       # starter kit, public weather, validation and training scenarios
    python3 run.py validate [--workers N]    # the paper's controllers on the 270 validation scenarios
    python3 run.py tables                    # the paper's numbers from the validation results
    python3 run.py models                    # rebuild the final models from their recorded predecessors
    python3 run.py submissions               # rebuild the two Kaggle notebooks and compare with the submitted ones
    python3 run.py chain                     # list the stages of the full training chain
    python3 run.py retrain <stage> [--workers N]   # run one stage of the chain (see PIPELINE.md)

Every command only writes inside this repository. Nothing is submitted to Kaggle.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parent
LAUNCH = ["python3", "-B", "pipeline/launch/launch.py"]
STARTER_KIT = "https://github.com/smart-buildings-2026/starter-kit.git"
STARTER_KIT_COMMIT = "23a4560df0e2fe61054dd21066fa0c4b5da2c79c"
NOAA = "https://noaa-global-hourly-pds.s3.amazonaws.com/2023/{}.csv"
NOAA_STATIONS = {"riyadh": "40437099999", "dubai": "41194099999", "seattle": "72793024233",
                 "london": "03772099999", "beijing": "54511099999"}
FINAL = str(ROOT / "data/final")   # absolute: the scripts change their working directory to the simulator's


def sh(cmd, env=None, check=True):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    result = subprocess.run([str(c) for c in cmd], cwd=ROOT, env=env)
    if check and result.returncode:
        raise SystemExit(f"failed ({result.returncode}): {' '.join(str(c) for c in cmd)}")
    return result.returncode


def uv():
    found = shutil.which("uv") or str(Path.home() / ".local/bin/uv")
    if not Path(found).exists():
        raise SystemExit("uv is required: https://docs.astral.sh/uv/getting-started/installation/")
    return found


def launch(script, *args, extra_env=None):
    env = dict(os.environ)
    env.update(extra_env or {})
    return sh(LAUNCH + [script, *args], env=env)


# ------------------------------------------------------------------------------------------------ env
def cmd_env(args):
    env = dict(os.environ, UV_CACHE_DIR=str(ROOT / ".tmp/uv-cache"), UV_PYTHON_INSTALL_DIR=str(ROOT / "tools/python"),
               UV_PROJECT_ENVIRONMENT=str(ROOT / ".venv"))
    sh([uv(), "python", "install", "3.13"], env=env)
    sh([uv(), "sync", "--locked"], env=env)


# ------------------------------------------------------------------------------------------------ setup
def starter_kit():
    kit = ROOT / "reference/starter-kit"
    if not (kit / ".git").is_dir():
        kit.parent.mkdir(parents=True, exist_ok=True)
        sh(["git", "clone", "--quiet", STARTER_KIT, kit])
    sh(["git", "-C", kit, "checkout", "--quiet", STARTER_KIT_COMMIT])
    # The code expects the competition copy at data/raw/kaggle; the public starter kit at the pinned commit has the
    # same simulator (they differ only in comments, a debug-image helper and a path constant).
    link = ROOT / "data/raw/kaggle"
    link.parent.mkdir(parents=True, exist_ok=True)
    if not link.exists():
        link.symlink_to(Path("../../reference/starter-kit"))


def noaa():
    raw = ROOT / "data/raw/external_noaa_isd"
    raw.mkdir(parents=True, exist_ok=True)
    for name, sid in NOAA_STATIONS.items():
        target = raw / f"{sid}_2023.csv"
        if not target.is_file():
            print(f"downloading NOAA ISD 2023 {name} ({sid})", flush=True)
            urllib.request.urlretrieve(NOAA.format(sid), target)
        print(f"  {target.relative_to(ROOT)} sha256 {hashlib.sha256(target.read_bytes()).hexdigest()[:16]}")


def cmd_setup(args):
    w = str(args.workers)
    starter_kit()
    noaa()
    # unseen-station weather files and the 60 scenarios of the six starter-kit buildings at those stations
    launch("pipeline/setup/new_stations_build.py")
    # the TRAIN pool (80 buildings, 367 scenarios); the generator refuses to write a split that differs from the
    # registered configs/splits/v1-train-r1.json, so a successful run reproduces the recorded pool
    launch("scripts/make_split.py", "--train-r1")
    # the 12 new validation buildings (seeds recorded in data/final/val2/manifest.json) and their 180 scenarios
    launch("pipeline/final/gen_val2.py", FINAL, "12", f"{FINAL}/recorded/val2-manifest.json")
    # reduced thermal models of the fast simulator (data/runs/fastsim/rom-cache)
    launch("scripts/fastsim.py", "build", "--pool", "train", "--workers", w)
    launch("pipeline/final/prebuild_v2.py", FINAL, w)
    launch("pipeline/final/prebuild_official.py", w)
    if args.chain:
        # pools only the training chain uses: VAL (S4 sets) and INNER
        launch("scripts/make_split.py")
        launch("scripts/make_split.py", "--inner")
        launch("scripts/fastsim.py", "build", "--pool", "inner", "--workers", w)
        launch("scripts/fastsim.py", "build", "--pool", "val", "--workers", w)


# ------------------------------------------------------------------------------------------------ validation
PAPER_MODELS = ["pr_opt2", "pr_room", "gt_x8", "gm_tree_m1", "gt_t2b", "gt_m1", "gt_x1"]
CORRECTED = {"SB_TANK": "balance", "SB_RETURN_AIR": "flow", "SB_FASTSIM_CACHE": str(ROOT / "data/runs/fastsim/rom-cache")}


def cmd_validate(args):
    w = str(args.workers)
    names = args.models or PAPER_MODELS
    env = dict(CORRECTED, RUNS_DIR="runs_p2", SCENE_RE=args.scenes)
    # 90 scenarios of the six starter-kit buildings (public and unseen stations) -> data/final/runs_p2
    if not args.models:
        launch("pipeline/final/run_bl.py", FINAL, w, extra_env=env)
        launch("pipeline/final/run_r2.py", FINAL, w, extra_env=env)
    launch("pipeline/final/run_c.py", FINAL, w, *names, extra_env=env)
    # 180 scenarios of the 12 new buildings -> data/final/runs_v2 (limit 0 = all)
    v2 = dict(CORRECTED, SB_FASTSIM_CACHE=str(ROOT / FINAL / "rom-cache"))
    refs = [] if args.models else ["org_bang_bang", "org_fixed_setpoint", "org_pid", "org_pid_water", "g36_default"]
    launch("pipeline/final/v2_run.py", FINAL, w, str(args.limit), *refs, *names, extra_env=v2)
    if args.traces:
        sh(["bash", "pipeline/final/go_trace_p2_all.sh"])
        sh(["bash", "pipeline/final/go_v2.sh", "trace", w])


def cmd_tables(args):
    launch("pipeline/final/an12.py", FINAL, "runs_p2,runs_v2", "an12.json")
    sh(["python3", "-B", "pipeline/final/compare_results.py"])


# ------------------------------------------------------------------------------------------------ models
def cmd_models(args):
    sh([sys.executable, "-B", "pipeline/final/rebuild_models.py"])


def cmd_submissions(args):
    sh([sys.executable, "-B", "pipeline/final/rebuild_submissions.py"])


# ------------------------------------------------------------------------------------------------ chain
def stages():
    return json.loads((ROOT / "pipeline/stages.json").read_text())


def cmd_chain(args):
    for s in stages():
        mark = "manual" if s.get("manual") else ("recorded" if s.get("recorded") else "reconstructed")
        print(f"{s['id']:<14} {mark:<13} {s['what']}")
        for c in s["commands"]:
            print(f"{'':<14} $ {' '.join(c)}")


def cmd_retrain(args):
    stage = {s["id"]: s for s in stages()}.get(args.stage)
    if not stage:
        raise SystemExit(f"unknown stage {args.stage}; see python3 run.py chain")
    for c in stage["commands"]:
        c = [x.replace("{workers}", str(args.workers)) for x in c]
        c = [str(ROOT / x) if x.startswith("data/final") else x for x in c]
        if c[0] == "launch":
            launch(*c[1:], extra_env=stage.get("env"))
        else:
            sh(c)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("env")
    s = sub.add_parser("setup")
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--chain", action="store_true", help="also the VAL and INNER pools of the training chain")
    v = sub.add_parser("validate")
    v.add_argument("--workers", type=int, default=4)
    v.add_argument("--traces", action="store_true", help="also the per-step traces of figures 4 and 5 (~1.1 GB)")
    v.add_argument("--models", nargs="*", help="only these models (default: the paper's controllers and references)")
    v.add_argument("--scenes", default=".", help="regular expression on building/station of the 90 scenarios")
    v.add_argument("--limit", type=int, default=0, help="at most this many of the 180 new-building scenarios")
    sub.add_parser("tables")
    sub.add_parser("models")
    sub.add_parser("submissions")
    sub.add_parser("chain")
    r = sub.add_parser("retrain")
    r.add_argument("stage")
    r.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    {"env": cmd_env, "setup": cmd_setup, "validate": cmd_validate, "tables": cmd_tables, "models": cmd_models,
     "submissions": cmd_submissions, "chain": cmd_chain, "retrain": cmd_retrain}[args.command](args)


if __name__ == "__main__":
    main()
