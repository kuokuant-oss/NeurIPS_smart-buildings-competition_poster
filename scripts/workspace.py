"""Repository root and the process environment of every script (caches and temporary files under the repository)."""
import os
from pathlib import Path

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
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1", MPLBACKEND="Agg", UV_PYTHON_DOWNLOADS="never",
               COPYFILE_DISABLE="1")
    env.pop("VIRTUAL_ENV", None)
    env.pop("PYTHONPATH", None)
    return env
