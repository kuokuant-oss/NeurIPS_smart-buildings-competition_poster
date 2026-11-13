"""Build the b1_r_v2 submission notebook: G36PolicyV2 and its modules embedded as source (D-K01).

Kaggle runs the notebook without internet and cannot install this repository,
so every module G36PolicyV2 imports is embedded verbatim and, at run time,
registered in sys.modules under its own name in dependency order; the package
layout of packages/smart_building is therefore unchanged inside the notebook.
The wrapper mirrors scripts/build_notebook.py and the organizers' TransferPolicy
Notebook (reference/official/submission-example/transferpolicy-notebook.ipynb):
a smart_buildings_inference_server.Policy subclass served with
COMPETITION_DATA_FOLDER. The competition data folder is also put on sys.path,
because the D-R23 fallback imports the official pid_water from
smart_buildings.baselines.pid_plus, which ships on that mount.

Writes <out>/<slug>.ipynb, kernel-metadata.json (private, internet off) and
embedded-files.json with the SHA-256 of every embedded file and of the
notebook. Nothing is uploaded: pushing and submitting need the user's
authorisation (D-K03).

Since 2026-10-16 the builder targets b1_r_v2 and writes to data/submissions/b1r-v2 by
default. The first submission (b1_r, data/submissions/b1r) is kept as built; its
sources are frozen in smart_building.g36_v1 and smart_building.policies.g36_v1.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time

from workspace import ROOT

KAGGLE_ROOT = "/kaggle/input/competitions/neurips-2026-smart-buildings"
POLICY_NAMES = {"G36PolicyV2": "b1_r_v2", "G36PolicyV3": "b1_r_v3", "G36PolicyV3Ti900": "b1_r_v3_ti900",
                "G36PolicyC3LoopsAlwaysActive": "b1_r_c3_loops_always_active",
                "G36PolicyC3LaaOccrowDampfloor": "b1_r_c3_laa_occrow_dampfloor",
                "G36PolicyC3LaaOccrowDampfloorCold3": "b1_r_c3_laa_occrow_dampfloor_cold3",   # REQ-022
                # baseline v2 rule candidates (docs/plan/baseline-v2.md §2.1)
                "G36PolicyB2A1": "b1_r_c3_laa_occrow_dampfloor_b2a1",
                "G36PolicyB2A4": "b1_r_c3_laa_occrow_dampfloor_b2a4",
                "G36PolicyB2A14": "b1_r_c3_laa_occrow_dampfloor_b2a14",
                "G36PolicyB2A124": "b1_r_c3_laa_occrow_dampfloor_b2a124",
                "G36PolicyB2R2": "b1_r_c3_laa_occrow_dampfloor_b2r2",
                # Baseline v2 B2-RL and the rule extracted from the lookahead search (handoff/RUNBOOK-windows-b2rl.md
                # §8, §2.6): policies/b2rl.py on G36PolicyB2R2 with the exported actor or candidate tree (--embed)
                "G36PolicyB2RL": "b2rl", "G36PolicyB2RX": "b2rx"}
# the module-level line of policies/b2rl.py that --embed fills with the JSON text (the notebook reads no outside file)
EMBED_LINES = {"G36PolicyB2RL": "B2RL_MODEL_JSON", "G36PolicyB2RX": "B2RX_TREE_JSON"}
COMPETITION = "neurips-2026-smart-buildings"
PACKAGE_ROOT = ROOT / "packages"
# Where the learned ensembles' rule controller (on/off, B1-R state, D-T02 fallback: the chain's teacher) is defined, for
# scripts/build_submission_ensemble.py and build_submission_rl_ensemble.py (both embed the two files): the readback
# teachers are made in policies/c3_teachers.py, the fixed-name rule classes are class statements of policies/g36.py
# (REQ-022: cold-inner rule (c) made the cold3 chain's teacher G36PolicyC3LaaOccrowDampfloorCold3 itself, 2026-10-24).
TEACHER_MODULES = (("smart_building.policies.c3_teachers", "smart_building/policies/c3_teachers.py"),
                   ("smart_building.policies.g36", "smart_building/policies/g36.py"))


def rule_classes(source):
    """The policy classes a policies module defines at its top level: class statements and names bound to
    c3_configured_class(...) (policies/c3_teachers.py), G36Policy* only."""
    import ast
    names = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) \
                and getattr(node.value.func, "id", None) == "c3_configured_class":
            names.update(target.id for target in node.targets if isinstance(target, ast.Name))
    return {name for name in names if name.startswith("G36Policy")}


def teacher_module(name, read=None):
    """(module, file relative to packages/) of the embedded module that defines the rule-controller class `name`:
    policies/c3_teachers.py first (every earlier teacher), then policies/g36.py. read(relative) returns a file's
    source (default: the working tree; the RL builder passes the work package's files)."""
    read = read or (lambda relative: (PACKAGE_ROOT / relative).read_text(encoding="utf-8"))
    for module, relative in TEACHER_MODULES:
        if name in rule_classes(read(relative)):
            return module, relative
    raise ValueError(f"{name}: not a G36Policy class of policies/c3_teachers.py or policies/g36.py")


# (module name, is package, file relative to packages/) in import order. The
# policies package is created empty: its __init__ registers other policies.
MODULES = [
    ("smart_building", True, "smart_building/__init__.py"),
    ("smart_building.g36", True, "smart_building/g36/__init__.py"),
    # modes first: v2 zone.py, ahu.py and optimal_start.py import it (main line fix 2026-10-16 16:1x).
    ("smart_building.g36.modes", False, "smart_building/g36/modes.py"),
    ("smart_building.g36.trim_respond", False, "smart_building/g36/trim_respond.py"),
    ("smart_building.g36.zone", False, "smart_building/g36/zone.py"),
    ("smart_building.g36.building", False, "smart_building/g36/building.py"),
    ("smart_building.g36.requests", False, "smart_building/g36/requests.py"),
    ("smart_building.g36.optimal_start", False, "smart_building/g36/optimal_start.py"),
    ("smart_building.g36.ahu", False, "smart_building/g36/ahu.py"),
    ("smart_building.g36.plant", False, "smart_building/g36/plant.py"),
    # per-room geometry switches (2026-10-31, docs/plan/geo-room-stages.md); imported lazily by policies/g36.py
    ("smart_building.g36.rooms", False, "smart_building/g36/rooms.py"),
    # model-based planning layers (pipeline 5 O505 / O544 / O545, 2026-11-04); imported lazily by policies/g36.py
    ("smart_building.g36.planner", False, "smart_building/g36/planner.py"),
    ("smart_building.policies", True, None),
    ("smart_building.policies.g36", False, "smart_building/policies/g36.py"),
]

# The competition data mount: the organizers' example uses /kaggle/input/competitions/<slug>/; a private kernel
# run on 2026-10-18 found it at /kaggle/input/<slug>/ instead, so the first existing candidate is used.
SETUP = f"""import os
import sys
DATA_CANDIDATES = ['{KAGGLE_ROOT}', '/kaggle/input/{COMPETITION}']
DATA_ROOT = next((path for path in DATA_CANDIDATES if os.path.isdir(os.path.join(path, 'smart_buildings'))),
                 DATA_CANDIDATES[0])
sys.path.append(DATA_ROOT + '/kaggle_evaluation')
if DATA_ROOT not in sys.path:
    sys.path.append(DATA_ROOT)

import numpy as np
from kaggle_evaluation import smart_buildings_inference_server

COMPETITION_DATA_FOLDER = DATA_ROOT + '/'
# The D-R23 fallback imports smart_buildings.baselines.pid_plus from the data mount.
if COMPETITION_DATA_FOLDER not in sys.path:
    sys.path.append(COMPETITION_DATA_FOLDER)
"""

LOADER = '''import sys
import types


def install_embedded_module(name, is_package, source):
    """Register one embedded module under its package name, then run its source."""
    module = types.ModuleType(name)
    module.__file__ = f"<embedded {name}>"
    if is_package:
        module.__path__ = []
        module.__package__ = name
    else:
        module.__package__ = name.rpartition(".")[0]
    sys.modules[name] = module
    parent, _, child = name.rpartition(".")
    if parent:
        setattr(sys.modules[parent], child, module)
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module
'''


def cell(kind, source):
    lines = source.strip("\n").split("\n")
    entry = {"cell_type": kind, "metadata": {},
             "source": [line + "\n" for line in lines[:-1]] + [lines[-1]]}
    if kind == "code":
        entry.update(execution_count=None, outputs=[])
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slug", default="sb-b1-r-v2-g36-rule-controller")
    parser.add_argument("--title", default="SB B1-R v2 G36 rule controller")
    parser.add_argument("--username", default="USERNAME",
                        help="Kaggle username for kernel-metadata.json")
    parser.add_argument("--out", default="data/submissions/b1r-v2")
    # REQ-017 (2026-10-17): b1_r_v3 is G36PolicyV3 in the same module; the embedded files are the same.
    parser.add_argument("--policy-class", default="G36PolicyV2", choices=sorted(POLICY_NAMES))
    # 2026-10-18 (reviewer): embed exactly the modules an S4 run used, from its built work package
    # (data/remote/<job>/<kernel folder>: run.py's embedded code, checked against build.json), not the tree.
    parser.add_argument("--from-build", help="kernel folder of the work package whose code is embedded")
    parser.add_argument("--embed", help="G36PolicyB2RL / G36PolicyB2RX: the exported actor or candidate tree (JSON)")
    args = parser.parse_args()
    modules, embed_text = list(MODULES), None
    if args.policy_class in EMBED_LINES:
        if not args.embed or args.from_build:
            parser.error(f"{args.policy_class} needs --embed (and the working tree, not --from-build)")
        embed_text = json.dumps(json.loads((ROOT / args.embed).read_text(encoding="utf-8")))
        modules.append(("smart_building.policies.b2rl", False, "smart_building/policies/b2rl.py"))
    elif args.embed:
        parser.error("--embed is only for " + " and ".join(EMBED_LINES))
    policy_module = "smart_building.policies.b2rl" if args.policy_class in EMBED_LINES else "smart_building.policies.g36"
    sources = None
    if args.from_build:
        import sys
        sys.path.insert(0, str(PACKAGE_ROOT))
        from smart_building.remote import package as work_package
        folder = ROOT / args.from_build
        build = json.loads((folder / "build.json").read_text(encoding="utf-8"))
        code = work_package.embedded_code((folder / "run.py").read_text(encoding="utf-8"))
        if work_package.sha256_bytes(code) != build["code_sha256"]:
            parser.error(f"{args.from_build}: embedded code does not match build.json")
        sources = {"build": args.from_build, "code_sha256": build["code_sha256"], "files": work_package.unzip(code)}
    policy_name = POLICY_NAMES[args.policy_class]
    # Same rule as scripts/build_notebook.py: Kaggle derives the URL from the title.
    if re.sub(r"[^a-z0-9]+", "-", args.title.lower()).strip("-") != args.slug:
        parser.error("--slug must equal the slug Kaggle derives from --title")

    embedded, install_cells = [], []
    for name, is_package, relative in modules:
        if relative is None:
            source, digest, path = "", None, None
        elif sources is not None:
            path = PACKAGE_ROOT / relative
            data = sources["files"][f"packages/{relative}"]
            source = data.decode("utf-8")
            digest = hashlib.sha256(data).hexdigest()
        else:
            path = PACKAGE_ROOT / relative
            source = path.read_text(encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entry = {"module": name, "package": is_package, "file": path.relative_to(ROOT).as_posix() if path else None,
                 "sha256": digest}
        if name == "smart_building.policies.b2rl":
            line = f"{EMBED_LINES[args.policy_class]} = None"
            if source.count(f"\n{line}\n") != 1:
                parser.error(f"policies/b2rl.py: the line {line!r} is not there exactly once")
            source = source.replace(f"\n{line}\n", f"\n{EMBED_LINES[args.policy_class]} = {embed_text!r}\n")
            entry.update(embedded_json=args.embed, embedded_json_sha256=hashlib.sha256(embed_text.encode()).hexdigest())
        embedded.append(entry)
        install_cells.append(cell("code", f"# {name}" + (f" ({path.relative_to(ROOT).as_posix()}, "
                                                          f"sha256 {digest})" if path else " (empty package)")
                                  + f"\ninstall_embedded_module({name!r}, {is_package!r}, {source!r})"))

    listing = "\n".join(f"| `{e['module']}` | `{e['file'] or '-'}` | `{e['sha256'] or '-'}` |" for e in embedded)
    header = f"""# {args.title}

Generated by `scripts/build_submission.py` on {time.strftime("%Y-%m-%d")}.
Policy: `{policy_module}.{args.policy_class}` ({policy_name}, rule-based, Guideline 36 style), default `G36Config`.
On any B1-R exception the step falls back to the official `pid_water` from the competition data (D-R23).

| module | file | SHA-256 |
|---|---|---|
{listing}
"""
    setup = SETUP
    wrapper = """from smart_building.policies.g36 import G36PolicyV2


class SubmissionPolicy(smart_buildings_inference_server.Policy):
    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        self.policy = G36PolicyV2(building_meta_data, action_spec)

    def get_action(self, obs):
        return self.policy.get_action(obs)


smart_buildings_inference_server.SmartBuildingsInferenceServer(SubmissionPolicy).run(
    competition_data_folder=COMPETITION_DATA_FOLDER,
)
""".replace("G36PolicyV2", args.policy_class).replace("smart_building.policies.g36", policy_module)
    notebook = {
        "cells": [cell("markdown", header), cell("code", setup), cell("code", LOADER),
                  *install_cells, cell("code", wrapper)],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.13.15"},
        },
        "nbformat": 4, "nbformat_minor": 4,
    }
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    code_file = f"{args.slug}.ipynb"
    notebook_path = out / code_file
    with open(notebook_path, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    metadata = {
        "id": f"{args.username}/{args.slug}", "title": args.title, "code_file": code_file,
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
        "dataset_sources": [], "competition_sources": [COMPETITION], "kernel_sources": [],
        "model_sources": [],
    }
    with open(out / "kernel-metadata.json", "w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(metadata, indent=2) + "\n")
    record = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "notebook": notebook_path.relative_to(ROOT).as_posix(),
              "notebook_sha256": hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
              "policy": f"{policy_module}.{args.policy_class} (default G36Config)",
              "embedded": embedded}
    if sources is not None:
        record["embedded_from"] = {"build": sources["build"], "code_sha256": sources["code_sha256"]}
    with open(out / "embedded-files.json", "w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
