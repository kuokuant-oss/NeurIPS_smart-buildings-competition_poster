"""Where a shard runs and where its inputs and outputs are (remote plan §3.1).

Nothing below /kaggle/input is hard-coded: the competition data root is the
first folder that holds both smart_buildings/__init__.py and kaggle_evaluation/,
and earlier shard outputs are found by their job-output.json marker. Local dry
runs and tests point every location into the repository through environment
variables, so the same code runs everywhere.
"""
import json
import os
from pathlib import Path
import tempfile

ENV_BACKEND = "SB_REMOTE_BACKEND"
ENV_COMPETITION_ROOT = "SB_REMOTE_COMPETITION_ROOT"
ENV_INPUT_ROOT = "SB_REMOTE_INPUT_ROOT"
ENV_OUTPUT_DIR = "SB_REMOTE_OUTPUT_DIR"
ENV_WORK_DIR = "SB_REMOTE_WORK_DIR"
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
KAGGLE_TEMP = Path("/kaggle/temp")
MARKER = "job-output.json"
IGNORED_PREFIXES = ("._", ".DS_Store", "__pycache__")


def detect_backend(environ=None, kaggle_input=KAGGLE_INPUT):
    """'local' when the caller redirected the output, else 'kaggle' or 'codex'."""
    environ = os.environ if environ is None else environ
    if environ.get(ENV_BACKEND):
        return environ[ENV_BACKEND]
    if environ.get(ENV_OUTPUT_DIR):
        return "local"
    if kaggle_input.is_dir() or environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    return "local"


def _visible(path):
    return not path.name.startswith(IGNORED_PREFIXES)


def walk_dirs(root, max_depth):
    """Directories under root up to max_depth levels, breadth first, sorted."""
    root = Path(root)
    level = [root]
    for depth in range(max_depth + 1):
        following = []
        for directory in level:
            yield directory, depth
            if depth == max_depth:
                continue
            try:
                children = sorted(p for p in directory.iterdir() if p.is_dir() and _visible(p))
            except OSError:
                continue
            following.extend(children)
        level = following


def is_competition_root(path):
    path = Path(path)
    return (path / "smart_buildings/__init__.py").is_file() and (path / "kaggle_evaluation").is_dir()


def find_competition_root(search_root, max_depth=4):
    """First folder below search_root holding smart_buildings/__init__.py and kaggle_evaluation/."""
    for directory, _ in walk_dirs(search_root, max_depth):
        if is_competition_root(directory):
            return directory
    raise FileNotFoundError(f"no competition data root (smart_buildings/__init__.py + "
                            f"kaggle_evaluation/) within {max_depth} levels of {search_root}")


def find_markers(search_root, job_id, max_depth=6):
    """(folder, marker) for every job-output.json below search_root that belongs to job_id."""
    found = []
    for directory, _ in walk_dirs(search_root, max_depth):
        marker = directory / MARKER
        if not marker.is_file():
            continue
        try:
            content = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if content.get("job_id") == job_id:
            found.append((directory, content))
    return found


def inputs_tree(search_root, max_depth=3):
    """Text listing of folders (and files at the last level) below search_root."""
    search_root = Path(search_root)
    lines = [str(search_root)]
    if not search_root.is_dir():
        return f"{search_root} (missing)\n"
    for directory, depth in walk_dirs(search_root, max_depth):
        if directory == search_root:
            continue
        lines.append("  " * depth + directory.name + "/")
        if depth == max_depth:
            continue
        try:
            files = sorted(p.name for p in directory.iterdir() if p.is_file() and _visible(p))
        except OSError:
            files = []
        shown = files[:20]
        lines.extend("  " * (depth + 1) + name for name in shown)
        if len(files) > len(shown):
            lines.append("  " * (depth + 1) + f"... {len(files) - len(shown)} more files")
    return "\n".join(lines) + "\n"


def resolve_locations(job_id, shard, environ=None):
    """Backend and the four locations a shard needs; raises if local runs lack overrides."""
    environ = os.environ if environ is None else environ
    backend = detect_backend(environ)
    if backend == "kaggle":
        input_root = Path(environ.get(ENV_INPUT_ROOT, KAGGLE_INPUT))
        output_dir = Path(environ.get(ENV_OUTPUT_DIR, KAGGLE_WORKING))
        if environ.get(ENV_WORK_DIR):
            work_dir = Path(environ[ENV_WORK_DIR])
        elif KAGGLE_TEMP.is_dir() and os.access(KAGGLE_TEMP, os.W_OK):
            work_dir = KAGGLE_TEMP / f"sb-{job_id}-{shard}"
        else:
            work_dir = Path(tempfile.gettempdir()) / f"sb-{job_id}-{shard}"
    else:
        # Local and Codex runs must say where everything goes, so nothing is
        # written outside the project by accident.
        missing = [name for name in (ENV_INPUT_ROOT, ENV_OUTPUT_DIR, ENV_WORK_DIR)
                   if not environ.get(name)]
        if missing:
            raise RuntimeError(f"{backend} run needs {', '.join(missing)}")
        input_root = Path(environ[ENV_INPUT_ROOT])
        output_dir = Path(environ[ENV_OUTPUT_DIR])
        work_dir = Path(environ[ENV_WORK_DIR])
    if environ.get(ENV_COMPETITION_ROOT):
        competition_root = Path(environ[ENV_COMPETITION_ROOT])
        if not is_competition_root(competition_root):
            raise FileNotFoundError(f"{competition_root} is not a competition data root")
    else:
        competition_root = find_competition_root(input_root)
    return {"backend": backend, "input_root": input_root, "output_dir": output_dir,
            "work_dir": work_dir, "competition_root": competition_root}
