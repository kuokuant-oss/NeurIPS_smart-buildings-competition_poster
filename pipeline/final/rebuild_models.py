"""Rebuild the final models from their recorded predecessors and compare them with the shipped files.

1. The three hand-made files of the chain from their recorded edits (pipeline/manual/edits).
2. gt_x1 -> gt_x5 -> gt_x6 / gt_x7 -> gt_x8 (rule with the learned layer) with build_x.py, build_x345.py, build_x67.py,
   build_x8.py; pr_opt2 (tuned rule) from the coordinate search's end state with build_pr2.py; pr_room with
   build_pr_room.py. Built into data/final/rebuild, compared byte for byte with data/runs/b2rl/search/models.
python3 -B pipeline/final/rebuild_models.py"""
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "data/runs/b2rl/search/models"
OUT = ROOT / "data/final/rebuild"
STEPS = ["build_x.py", "build_x345.py", "build_x67.py", "build_x8.py", "build_pr2.py", "build_pr_room.py"]
COMPARE = ["gt_x1", "gt_x5", "gt_x6", "gt_x7", "gt_x8", "pr_opt2", "pr_room"]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ok = True
    for edit in sorted((ROOT / "pipeline/manual/edits").glob("*.json")):
        ok &= subprocess.run([sys.executable, "-B", str(ROOT / "pipeline/manual/recorded_edit.py"), "check",
                              str(edit)], cwd=ROOT).returncode == 0
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    shutil.copytree(ROOT / "data/final/recorded/pr", OUT / "pr")
    for step in STEPS:
        r = subprocess.run([sys.executable, "-B", str(ROOT / "pipeline/final" / step), str(OUT)], cwd=ROOT,
                           capture_output=True, text=True)
        print(f"--- {step}: exit {r.returncode}")
        if r.returncode:
            print(r.stderr[-2000:])
            ok = False
    for name in COMPARE:
        built, shipped = OUT / f"model-{name}.json", MODELS / f"model-{name}.json"
        same = built.is_file() and sha(built) == sha(shipped)
        ok &= same
        print(f"{name:<8} {'identical (sha256 ' + sha(shipped)[:16] + ')' if same else 'DIFFERENT or missing'}")
    print("ALL IDENTICAL" if ok else "SOME DIFFER")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
