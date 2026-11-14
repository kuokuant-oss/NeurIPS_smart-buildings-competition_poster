"""Interpreter-start hook of `scripts/fastsim.py run` (this folder is put on PYTHONPATH only there).

When SB_FASTSIM asks for a mode, the fast simulator is installed right after the competition's floor-plan
simulator module is first imported, in this process and in every process it starts with the same environment
(multiprocessing spawn workers included). Without SB_FASTSIM, or in any process started without this folder on
PYTHONPATH, nothing happens."""
import importlib.abc
import os
from pathlib import Path
import sys

_TARGET = "smart_buildings.smart_control.simulator.simulator_flexible_floor_plan"


class _InstallAfterImport(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != _TARGET:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(name, path, target)
            if spec is not None:
                break
        else:
            return None
        sys.meta_path.remove(self)
        execute = spec.loader.exec_module

        def exec_module(module):
            execute(module)
            from smart_building import fastsim
            fastsim.install_from_env()

        spec.loader.exec_module = exec_module
        return spec


if os.environ.get("SB_FASTSIM", "").strip().lower() not in ("", "0", "off", "official", "none"):
    packages = str(Path(__file__).resolve().parents[3])
    if packages not in sys.path:
        sys.path.append(packages)
    sys.meta_path.insert(0, _InstallAfterImport())
