"""Portable bootstrap: all uv/runtime writes stay beneath this repository."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def local_environment():
    env = os.environ.copy()
    locations = {
        "UV_CACHE_DIR": ".tmp/uv-cache",
        "UV_PYTHON_INSTALL_DIR": "tools/python",
        "UV_PYTHON_BIN_DIR": "tools/bin",
        "UV_TOOL_DIR": "tools/uv-tools",
        "UV_TOOL_BIN_DIR": "tools/bin",
        "UV_PROJECT_ENVIRONMENT": ".venv",
        "TMPDIR": ".tmp/temp", "TEMP": ".tmp/temp", "TMP": ".tmp/temp",
        "XDG_CACHE_HOME": ".tmp/cache",
        "MPLCONFIGDIR": ".tmp/matplotlib",
        "NUMBA_CACHE_DIR": ".tmp/numba",
        "IPYTHONDIR": ".tmp/ipython",
        "JUPYTER_RUNTIME_DIR": ".tmp/jupyter",
    }
    for key, relative in locations.items():
        path = ROOT / relative
        if key != "UV_PROJECT_ENVIRONMENT":
            path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1",
               MPLBACKEND="Agg", UV_PYTHON_DOWNLOADS="never",
               COPYFILE_DISABLE="1")
    env.pop("VIRTUAL_ENV", None)
    env.pop("PYTHONPATH", None)
    # The Kaggle CLI reads the project's own credential when it is there (user 2026-10-26; .secrets/ is ignored by
    # Git), so no computer needs a home-directory ~/.kaggle.
    if (ROOT / ".secrets/kaggle/kaggle.json").is_file():
        env["KAGGLE_CONFIG_DIR"] = str(ROOT / ".secrets/kaggle")
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["sync", "smoke", "verify", "evaluate", "compare", "batch",
                                            "throughput", "notebook-build", "notebook-test", "test",
                                            "remote", "make-split", "cv", "check-gateway", "autopilot", "pipeline", "inner", "notifications",
                                            "ppo", "ppo-health", "local-chain", "resume", "model-dataset"])
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    env = local_environment()
    uv = shutil.which("uv")
    if not uv:
        raise SystemExit("Install uv on this computer first.")
    def run(arguments):
        subprocess.run([uv, *arguments], cwd=ROOT, env=env, check=True)
    marker = ROOT / ".venv/workspace-platform.json"
    # .python-version pins the interpreter of the Kaggle notebook image.
    python_version = (ROOT / ".python-version").read_text(encoding="utf-8").strip()
    identity = {"platform": sys.platform, "machine": platform.machine(),
                "location_hash": hashlib.sha256(str(ROOT).encode()).hexdigest(),
                "python": python_version}
    pending = ROOT / ".tmp/install-platform.json"
    state = marker if marker.exists() else pending
    previous = json.loads(state.read_text(encoding="utf-8")) if state.exists() else None
    if args.command == "sync":
        # Read an existing interpreter; exFAT cannot unpack managed Python symlinks.
        discovery = env.copy()
        discovery.pop("UV_PYTHON_INSTALL_DIR", None)
        interpreter = subprocess.check_output(
            [uv, "python", "find", python_version, "--system", "--no-python-downloads"],
            cwd=ROOT, env=discovery, text=True).strip()
        # Rebuild instead of trying to carry native executables between OSes/mounts.
        if (ROOT / ".venv").exists() and previous != identity:
            if Path(sys.prefix).resolve() == (ROOT / ".venv").resolve():
                raise SystemExit("Run sync using the computer's bootstrap Python, outside .venv.")
            def remove_error(function, path, error):
                # macOS may remove AppleDouble metadata alongside the main file.
                if not isinstance(error[1], FileNotFoundError):
                    raise error[1]
            shutil.rmtree(ROOT / ".venv", onerror=remove_error)
        with pending.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(identity))
        marker.unlink(missing_ok=True)
        from environment_files import reconcile
        # Each wheel with a .data directory can hit the exFAT AppleDouble error once,
        # so a fresh environment may need several repair-and-resume rounds.
        for attempt in range(1, 9):
            result = subprocess.run([uv, "sync", "--python", interpreter, *args.arguments],
                                    cwd=ROOT, env=env, stderr=subprocess.PIPE, text=True)
            print(result.stderr, file=sys.stderr, end="")
            if not result.returncode:
                break
            if sys.platform != "darwin" or "._" not in result.stderr or attempt == 8:
                raise subprocess.CalledProcessError(result.returncode, result.args)
            # Keep one uv environment; repair completed wheel relocations, then resume.
            reconcile(ROOT / ".venv")
        reconcile(ROOT / ".venv")
        with marker.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(identity, indent=2) + "\n")
        pending.unlink(missing_ok=True)
    else:
        if not marker.exists() or previous != identity:
            raise SystemExit("Run workspace.py sync --locked with the host Python first.")
        scripts = {"verify": ("scripts/verify_data.py", []),
                   "compare": ("scripts/compare_simulators.py", []),
                   "batch": ("scripts/batch_evaluate.py", []),
                   "throughput": ("scripts/benchmark_throughput.py", []),
                   "notebook-build": ("scripts/build_notebook.py", []),
                   "notebook-test": ("scripts/test_notebook.py", []),
                   "test": ("scripts/run_tests.py", []),
                   "remote": ("scripts/remote.py", []),
                   "make-split": ("scripts/make_split.py", []),
                   "cv": ("scripts/cv_evaluate.py", []),
                   "check-gateway": ("scripts/check_gateway_path.py", []),
                   "autopilot": ("scripts/remote_autopilot.py", []),
                   "pipeline": ("scripts/remote_pipeline.py", []),
                   "notifications": ("scripts/notification_watcher.py", []),
                   "inner": ("scripts/inner_evaluate.py", []),
                   "ppo-health": ("scripts/ppo_health.py", []),
                   "local-chain": ("scripts/local_chain.py", []),
                   "resume": ("scripts/mac_resume.py", []),
                   "model-dataset": ("scripts/model_dataset.py", [])}
        script, extra = scripts.get(args.command, ("scripts/simulator.py", [args.command]))
        run(["run", "--no-sync", "python", script, *extra, *args.arguments])


if __name__ == "__main__":
    main()
