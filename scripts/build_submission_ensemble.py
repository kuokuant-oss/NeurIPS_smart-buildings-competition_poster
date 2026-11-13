"""Build the B1-L fold-ensemble submission notebook for the stage-4 leaderboard check (D-K04).

    python3 -B scripts/build_submission_ensemble.py --manifest M.json --model-kernel USER/SLUG [--out DIR]

Policy: smart_building.policies.b1l_ensemble.EnsembleStudentPolicy over the 5 fold students
(continuous action = mean of the students' continuous outputs; notes/07 §R, Roberts et al. 2017
Step 5), on/off actions and the D-T02 fallback from the J7b teacher
(policies/c3_teachers.py G36PolicyC3LoopsReadback, D-L07). The modules are embedded as source
the way scripts/build_submission.py embeds B1-R (registered under their package names in import
order). The student files are not embedded: they come from one attached kernel output
(--model-kernel, mounted at /kaggle/input/<slug>/), and the notebook refuses to run unless each
file's SHA-256 equals the manifest's. The manifest is JSON:
{"students": [{"file": "<name in the model kernel's output>", "sha256": ..., "fold": k,
"iteration": i, "chain_kernel": ...}, ...]}. SB_B1L_MODEL_DIR overrides the folder for the local
notebook test (scripts/test_notebook.py).

Writes <out>/<slug>.ipynb, kernel-metadata.json (private, internet off, the model kernel in
kernel_sources) and embedded-files.json. Nothing is uploaded or submitted (the reviewer submits).
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time

from workspace import ROOT
import build_submission as base

MODULES = base.MODULES + [
    ("smart_building.policies.c3_teachers", False, "smart_building/policies/c3_teachers.py"),
    ("smart_building.learned", True, "smart_building/learned/__init__.py"),
    ("smart_building.learned.tokens", False, "smart_building/learned/tokens.py"),
    ("smart_building.learned.model", False, "smart_building/learned/model.py"),
    ("smart_building.policies.b1l_ensemble", False, "smart_building/policies/b1l_ensemble.py"),
]
# j7b-v5 students (plan-direction-20261020 §10) read the B1-R state: learned/privileged.py and the v5 ensemble.
MODULES_V5 = MODULES[:-1] + [
    ("smart_building.learned.privileged", False, "smart_building/learned/privileged.py"),
    ("smart_building.policies.b1l_ensemble_v5", False, "smart_building/policies/b1l_ensemble_v5.py"),
]
ENSEMBLE = {"v4": ("smart_building.policies.b1l_ensemble", "EnsembleStudentPolicy", MODULES),
            "v5": ("smart_building.policies.b1l_ensemble_v5", "EnsembleStudentPolicyV5", MODULES_V5)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, help="JSON with the students (file, sha256, fold, iteration)")
    parser.add_argument("--model-kernel", required=True, help="USER/SLUG of the kernel whose output holds the students")
    parser.add_argument("--slug", default="sb-b1-l-fold-ensemble")
    parser.add_argument("--title", default="SB B1-L fold ensemble")
    parser.add_argument("--username", default="USERNAME")
    parser.add_argument("--out", default="data/submissions/b1l-ensemble")
    parser.add_argument("--student-format", choices=sorted(ENSEMBLE), default="v4",
                        help="v5: j7b-v5 students that read the B1-R state (learned/privileged.py)")
    parser.add_argument("--teacher-class", default="G36PolicyC3LoopsReadback",
                        help="class in policies/c3_teachers.py or policies/g36.py for on/off and D-T02 (the chains' "
                             "teacher; j7b-v4 / j7b-v5: G36PolicyC3LaaOccrowDampfloorReadback; j7b-cold3: "
                             "G36PolicyC3LaaOccrowDampfloorCold3, policies/g36.py)")
    args = parser.parse_args()
    ensemble_module, ensemble_class, modules = ENSEMBLE[args.student_format]
    teacher = args.teacher_class
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", teacher):
        parser.error("--teacher-class must be a class name")
    try:                                                # c3_teachers.py's readback classes, or a g36.py class (cold3)
        teacher_module, teacher_file = base.teacher_module(teacher)
    except ValueError as error:
        parser.error(str(error))
    teacher_source = teacher_file.removeprefix("smart_building/")          # policies/c3_teachers.py or policies/g36.py
    if re.sub(r"[^a-z0-9]+", "-", args.title.lower()).strip("-") != args.slug:
        parser.error("--slug must equal the slug Kaggle derives from --title")
    manifest = json.loads((ROOT / args.manifest).read_text(encoding="utf-8"))
    students = manifest["students"]
    if not students or any(not re.fullmatch(r"[0-9a-f]{64}", s["sha256"]) for s in students):
        parser.error("the manifest needs students with SHA-256 digests")
    model_dir = f"/kaggle/input/{args.model_kernel.split('/')[1]}"

    embedded, install_cells = [], []
    for name, is_package, relative in modules:
        if relative is None:
            source, digest, path = "", None, None
        else:
            path = base.PACKAGE_ROOT / relative
            source = path.read_text(encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        embedded.append({"module": name, "package": is_package,
                         "file": path.relative_to(ROOT).as_posix() if path else None, "sha256": digest})
        install_cells.append(base.cell("code", f"# {name}" + (f" ({path.relative_to(ROOT).as_posix()}, sha256 {digest})"
                                                               if path else " (empty package)")
                                       + f"\ninstall_embedded_module({name!r}, {is_package!r}, {source!r})"))
    listing = "\n".join(f"| `{e['module']}` | `{e['file'] or '-'}` | `{e['sha256'] or '-'}` |" for e in embedded)
    rows = "\n".join(f"| {s.get('fold')} | {s.get('iteration')} | `{s['file']}` | `{s['sha256']}` |" for s in students)
    state_note = (" whose internal state, read before each of its calls, is the students' extra input "
                  "(learned/privileged.py)") if args.student_format == "v5" else ""
    header = f"""# {args.title}

Generated by `scripts/build_submission_ensemble.py` on {time.strftime("%Y-%m-%d")}.
Policy: `{ensemble_module}.{ensemble_class}`: the continuous action is the mean of
the {len(students)} fold students' continuous outputs; on/off actions and the per-step fallback (D-T02) come
from the teacher `{teacher}` ({teacher_source}, D-L07){state_note}.
Student files from the attached kernel `{args.model_kernel}`, each checked against its SHA-256.

| fold | iteration | file | SHA-256 |
|---|---|---|---|
{rows}

| module | file | SHA-256 |
|---|---|---|
{listing}
"""
    setup = base.SETUP
    students_json = json.dumps([{"file": s["file"], "sha256": s["sha256"]} for s in students])
    wrapper = f"""import hashlib
import json
import os
from pathlib import Path

import torch

from {ensemble_module} import ensemble_policy_class
from {teacher_module} import {teacher} as TeacherClass

torch.set_num_threads(1)
MODEL_DIR = Path(os.environ.get('SB_B1L_MODEL_DIR', {model_dir!r}))
STUDENTS = json.loads({students_json!r})
paths = []
for student in STUDENTS:
    path = MODEL_DIR / student['file']
    if hashlib.sha256(path.read_bytes()).hexdigest() != student['sha256']:
        raise RuntimeError(f"{{path}}: SHA-256 differs from the manifest")
    paths.append(path)
Ensemble = ensemble_policy_class(paths, TeacherClass)


class SubmissionPolicy(smart_buildings_inference_server.Policy):
    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        self.policy = Ensemble(building_meta_data, action_spec)

    def get_action(self, obs):
        return self.policy.get_action(obs)


smart_buildings_inference_server.SmartBuildingsInferenceServer(SubmissionPolicy).run(
    competition_data_folder=COMPETITION_DATA_FOLDER,
)
"""
    notebook = {"cells": [base.cell("markdown", header), base.cell("code", setup), base.cell("code", base.LOADER),
                          *install_cells, base.cell("code", wrapper)],
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python", "version": "3.13.15"}},
                "nbformat": 4, "nbformat_minor": 4}
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    code_file = f"{args.slug}.ipynb"
    notebook_path = out / code_file
    with open(notebook_path, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    metadata = {"id": f"{args.username}/{args.slug}", "title": args.title, "code_file": code_file,
                "language": "python", "kernel_type": "notebook", "is_private": True,
                "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
                "dataset_sources": [], "competition_sources": [base.COMPETITION], "kernel_sources": [args.model_kernel],
                "model_sources": []}
    with open(out / "kernel-metadata.json", "w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(metadata, indent=2) + "\n")
    record = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "notebook": notebook_path.relative_to(ROOT).as_posix(),
              "notebook_sha256": hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
              "policy": f"{ensemble_module}.{ensemble_class} (mean of continuous outputs; "
                        f"teacher {teacher} for on/off and D-T02)", "student_format": args.student_format,
              "model_kernel": args.model_kernel, "students": students, "embedded": embedded}
    with open(out / "embedded-files.json", "w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ("notebook", "notebook_sha256", "model_kernel")}, indent=2))


if __name__ == "__main__":
    main()
