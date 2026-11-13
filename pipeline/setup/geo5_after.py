"""Run a command once a chain's local work is over (2026-11-10): waits until the chain state <geo5 dir>/pipeline.json
has its closed loop done, or the chain's wake.json has been left unhandled for an hour, so that local runs do not
overlap; "{BEST}" in the command becomes the best candidate so far (gt_r3c1 or gt_r3d1 when its chain passed and beat
its bar in the closed loop, else gt_r3a1). Uses the chain's state files only (never process lists).
python3 -B pipeline/setup/geo5_after.py --state r3/stage_c [--state ...] -- <command ...>"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
G = ROOT / "data/runs/b2rl/geo5"


def stages(d):
    p = G / d / "pipeline.json"
    st = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    return next(iter(st.get("attempts", {}).values()), {}).get("stages", {})


def over(d):
    """Closed loop done, the coordinator's end marker <dir>/end (the chain stopped for good), or a wake left unhandled
    for an hour (a handled wake is renamed and the chain rerun)."""
    stg = stages(d)
    w = G / d / "wake.json"
    return bool(stg.get("closed", {}).get("done") or (G / d / "end").is_file()
                or (w.is_file() and w.stat().st_mtime < time.time() - 3600))


def best():
    out, bar = "gt_r3a1", None
    for d, name in (("r3/stage_c", "gt_r3c1"), ("r3/stage_d", "gt_r3d1"), ("r3/stage_e", "gt_r3e1"), ("r3/stage_f", "gt_r3f1")):
        cl = stages(d).get("closed", {})
        if cl.get("done") and cl.get("passed") and cl.get("mean") is not None and \
                cl["mean"] < cl.get("beat_mean", 0.0) and (bar is None or cl["mean"] < bar):
            out, bar = name, cl["mean"]
    return out


def main():
    argv = sys.argv[1:]
    k = argv.index("--")
    p = argparse.ArgumentParser()
    p.add_argument("--state", action="append", default=[])
    p.add_argument("--resolved", action="append", default=[],
                   help="(user rule 11/10: a candidate is fixed, rerun and resolved before the next) a chain state dir "
                        "that must be RESOLVED: its submission done (score stage started or done), or the "
                        "coordinator's end marker <dir>/end; a wake never counts (the coordinator fixes and reruns)")
    p.add_argument("--gate-free", action="append", default=[],
                   help="a chain state dir whose Kaggle gate must be over (its 5 CPU sessions free): gate stage done, "
                        "or a wake / error in the gate stage, or the chain never reached it and ended")
    a = p.parse_args(argv[:k])
    cmd = argv[k + 1:]

    def gate_over(d):
        stg = stages(d)
        g = stg.get("gate", {})
        if g.get("done") or g.get("wake") or g.get("error"):
            return True
        return not g.get("started") and over(d) and not stg.get("closed", {}).get("passed", False)
    def resolved(d):
        stg = stages(d)
        return bool((G / d / "end").is_file() or stg.get("submit", {}).get("done") or stg.get("score", {}).get("started"))
    while not (all(over(d) for d in a.state) and all(gate_over(d) for d in a.gate_free)
               and all(resolved(d) for d in a.resolved)):
        time.sleep(120)
    if any("{AHU_BEST}" in c for c in cmd):            # the per-AHU correction: its comparison's winner
        kp = G / "r3/kinds-ahu.json"
        while not kp.is_file():
            time.sleep(120)
        kj = json.loads(kp.read_text(encoding="utf-8"))
        if not kj.get("beats_current_both"):
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} the best per-AHU kind {kj.get('best')} does not beat the "
                  "current room learner in both validations: not deployed", flush=True)
            (G / "r3/stage_e").mkdir(parents=True, exist_ok=True)
            (G / "r3/stage_e/end").write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')} per-AHU kinds: no winner\n")
            return 0
        cmd = [c.replace("{AHU_BEST}", kj["best"]) for c in cmd]
    b = best()
    cmd = [c.replace("{BEST}", b) for c in cmd]
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} after {a.state}: best so far {b}; running {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=ROOT, env=dict(os.environ))


if __name__ == "__main__":
    sys.exit(main())
