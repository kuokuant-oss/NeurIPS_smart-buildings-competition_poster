"""Scenario overlay root (remote plan §3.1).

Official config.gin files name their assets relative to the working directory
(for example smart_buildings/weather/csv/atlanta.csv), and the competition data
is read-only. The overlay root is a writable folder whose smart_buildings/
entries link to the competition data; the working directory is set to it.

The overlay sits at <work>/reference/starter-kit so that it mirrors this
repository: generated scenarios live at <work>/data/processed/..., which is
"../../data/processed/..." from the overlay root, exactly as it is from the
local starter kit root. Generated config.gin files therefore resolve, and hash,
the same locally and remotely.
"""
import os
from pathlib import Path
import shutil

OVERLAY_RELATIVE = Path("reference/starter-kit")
SKIPPED = ("._", ".DS_Store", "__pycache__")


def build_overlay(competition_root, work_dir, allow_copy=True):
    """Create (or complete) the overlay root; returns its path and how entries were linked."""
    source = Path(competition_root) / "smart_buildings"
    root = Path(work_dir) / OVERLAY_RELATIVE
    target = root / "smart_buildings"
    target.mkdir(parents=True, exist_ok=True)
    methods = {}
    for child in sorted(source.iterdir()):
        if child.name.startswith(SKIPPED):
            continue
        link = target / child.name
        if link.exists() or link.is_symlink():
            methods[child.name] = "existing"
            continue
        try:
            os.symlink(child.resolve(), link, target_is_directory=child.is_dir())
            methods[child.name] = "symlink"
        except OSError:
            if not allow_copy:
                raise
            if child.is_dir():
                shutil.copytree(child, link)
            else:
                shutil.copy2(child, link)
            methods[child.name] = "copy"
    return root, methods


def official_scenario_dir(root, name):
    """An official scenario such as building_train_1/atlanta_2 under the overlay (or starter kit) root."""
    for folder in ("training_scenarios", "evaluation_scenarios"):
        candidate = Path(root) / "smart_buildings" / folder / name
        if (candidate / "config.gin").is_file():
            return candidate
    raise FileNotFoundError(f"official scenario {name} not found under {root}")
