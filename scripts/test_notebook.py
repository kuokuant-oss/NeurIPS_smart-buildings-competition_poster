"""Run a generated submission notebook locally against the Kaggle data copy.

The notebook's code cells are concatenated and the Kaggle input path is replaced
by data/raw/kaggle. The script runs in a fresh run directory, where the local
gateway writes submission.parquet. With --max-steps the gateway's own smoke-test
switch (HEARTH_HOST_SMART_BUILDINGS_MAX_STEPS) truncates every episode.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from workspace import ROOT, local_environment

KAGGLE_ROOT = "/kaggle/input/competitions/neurips-2026-smart-buildings"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook")
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    notebook_path = (ROOT / args.notebook).resolve()
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    local_root = (ROOT / "data/raw/kaggle").resolve()
    cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    # On Kaggle the competition folder makes `kaggle_evaluation` importable; locally the
    # data root is put on sys.path for the same import to resolve.
    prelude = f"import sys\nsys.path.insert(0, {str(local_root)!r})\n"
    code = prelude + "\n\n".join(cells).replace(KAGGLE_ROOT, local_root.as_posix())
    run_dir = ROOT / "data/runs" / f"notebook-{notebook_path.stem}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    script = run_dir / "notebook_code.py"
    script.write_text(code, encoding="utf-8", newline="\n")
    env = os.environ.copy()
    env.update(local_environment())
    if args.max_steps:
        env["HEARTH_HOST_SMART_BUILDINGS_MAX_STEPS"] = str(args.max_steps)
    start = time.perf_counter()
    with (run_dir / "notebook.log").open("w", encoding="utf-8", newline="\n") as log:
        result = subprocess.run([sys.executable, "-B", str(script)], cwd=run_dir, env=env,
                                stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - start
    report = {"notebook": notebook_path.relative_to(ROOT).as_posix(),
              "notebook_sha256": hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
              "max_steps": args.max_steps, "returncode": result.returncode,
              "elapsed_seconds": elapsed, "python": platform.python_version()}
    submission = run_dir / "submission.parquet"
    if submission.exists():
        import polars as pl
        frame = pl.read_parquet(submission)
        report.update(columns=frame.columns, rows=frame.to_dicts())
    (run_dir / "run.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))
    print("Output:", run_dir.relative_to(ROOT).as_posix())
    if result.returncode or not submission.exists():
        raise SystemExit("Notebook run failed; see notebook.log")


if __name__ == "__main__":
    main()
