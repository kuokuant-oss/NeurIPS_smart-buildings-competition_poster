"""Run a development script of the j10 build in the project's .venv with the repository-local environment of
scripts/workspace.py (caches, TMPDIR, ... under the repository). Usage (host python3):

    nice -n 10 python3 -B pipeline/launch/launch.py <script.py> [arguments ...]
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from workspace import local_environment  # noqa: E402

if __name__ == "__main__":
    env = local_environment()
    python = ROOT / ".venv/bin/python"
    raise SystemExit(subprocess.run([str(python), "-B", *sys.argv[1:]], cwd=ROOT, env=env).returncode)
