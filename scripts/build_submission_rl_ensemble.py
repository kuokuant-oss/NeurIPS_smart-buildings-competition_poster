"""Build the submission notebook of a 5-fold ensemble whose model files come from several Kaggle inputs.

    # 1. The manifest: the files the gate's S4 rows were made with (one per fold), each checked against its chain.json
    python3 -B scripts/build_submission_rl_ensemble.py manifest --kind rl --chain-job j9 --username antonktt \\
        --results data/runs/remote/j9-s4/merged/full/results.jsonl --out data/submissions/b1rl-ensemble/manifest.json
    # 2. Local copies laid out as /kaggle/input, for the notebook test (SB_ENSEMBLE_INPUT_ROOT)
    python3 -B scripts/build_submission_rl_ensemble.py stage --manifest data/submissions/b1rl-ensemble/manifest.json \\
        --out .tmp/submission-input/b1rl
    # 3. The notebook (code embedded from the work package S4 ran, as scripts/build_submission.py --from-build)
    python3 -B scripts/build_submission_rl_ensemble.py build --manifest data/submissions/b1rl-ensemble/manifest.json \\
        --username antonktt --from-build data/remote/j9-s4/<kernel folder> --check-root .tmp/submission-input/b1rl \\
        --out data/submissions/b1rl-ensemble

Kinds:
  rl          B1 stage 3 (docs/plan/b1-rl-stage3.md §8): policies/b1rl_ensemble.py over the 5 selected j9 agents (each
              read as learned/ppo.py policy_class reads one agent; the mean of their mean actions; on/off, B1-R state
              and the D-T02 fallback from one rule controller: the agents' own, v4 + readback for j9, the cold3 class of
              policies/g36.py for j11 (--chain-job j11)).
  student-v5  the j7b-v5s students with policies/b1l_ensemble_v5.py, the policy scripts/build_submission_ensemble.py
              embeds for --student-format v5, for students that sit in several chain kernels' outputs.

Model files (policies/mounted_models.py): a fold trained on the Mac is in the private dataset (scripts/model_dataset.py,
default sb-j9-models, file <fold>-agent-NNNN.pt); a fold trained on Kaggle is in the output of its chain's last kernel
(file dagger/<file the rows name>). The notebook finds each one under /kaggle/input and stops unless its SHA-256 equals
the manifest's. SB_ENSEMBLE_INPUT_ROOT replaces /kaggle/input for the local notebook test (scripts/test_notebook.py).

manifest refuses: a fold with rows of more than one file, a missing or extra fold, a row file whose SHA-256 differs from
its chain.json, a local copy that differs, a dataset file missing from the dataset's latest version (data/remote/
datasets.json) or recorded there with another SHA-256, a kernel attempt without exactly one built kernel.
build refuses: no --username (no USERNAME default), a source owned by another user, a malformed manifest entry, a
duplicate fold, a title over 50 characters, a --from-build package whose code differs from its build.json or lacks a
module, and (with --check-root) any model file that cannot be found with its SHA-256. The notebook is private, internet
off, no GPU. Nothing is uploaded or submitted (the reviewer does both, handoff/README.md §5).
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import sys
import time

from workspace import ROOT
import build_submission as base
import build_submission_ensemble as student_builder

sys.path.insert(0, str(base.PACKAGE_ROOT))
# policies/mounted_models.py is standard library only; loaded by path so the policies package (numpy, G36) is not imported.
_spec = importlib.util.spec_from_file_location("sb_mounted_models", base.PACKAGE_ROOT / "smart_building/policies/mounted_models.py")
mounted_models = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mounted_models)

TEACHER = "G36PolicyC3LaaOccrowDampfloorReadback"
NEW_MODULES = [("smart_building.policies.mounted_models", False, "smart_building/policies/mounted_models.py")]
MODULES_RL = base.MODULES + [
    ("smart_building.policies.c3_teachers", False, "smart_building/policies/c3_teachers.py"),
    ("smart_building.learned", True, "smart_building/learned/__init__.py"),
    ("smart_building.learned.tokens", False, "smart_building/learned/tokens.py"),
    ("smart_building.learned.model", False, "smart_building/learned/model.py"),
    ("smart_building.learned.privileged", False, "smart_building/learned/privileged.py"),
    ("smart_building.learned.ppo", False, "smart_building/learned/ppo.py"),
] + NEW_MODULES + [("smart_building.policies.b1rl_ensemble", False, "smart_building/policies/b1rl_ensemble.py")]
MODULES_STUDENT = student_builder.MODULES_V5 + NEW_MODULES
# kind: (policy module, class, modules, modules that may come from the tree with --from-build, default title, out)
KINDS = {
    "rl": ("smart_building.policies.b1rl_ensemble", "EnsembleRLPolicy", MODULES_RL,
           {"smart_building/policies/mounted_models.py", "smart_building/policies/b1rl_ensemble.py"},
           "SB B1-RL stage 3 fold ensemble", "data/submissions/b1rl-ensemble"),
    "student-v5": ("smart_building.policies.b1l_ensemble_v5", "EnsembleStudentPolicyV5", MODULES_STUDENT,
                   {"smart_building/policies/mounted_models.py"},
                   "SB B1-L v5s fold ensemble", "data/submissions/b1l-v5s-ensemble"),
}
FOLDS = ["f0", "f1", "f2", "f3", "f4"]
TITLE_LIMIT = 50                     # Kaggle refuses longer kernel titles with 400 (handoff/README.md §6)


def sha256_file(path):
    return mounted_models.sha256_file(path)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")


def check_username(parser, username):
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", username or "") or username.lower() == "username":
        parser.error("--username must be the Kaggle user (no USERNAME placeholder)")


def relative(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


# ------------------------------------------------------------------ manifest

def row_models(rows, kind, chain_job):
    """{fold: {(file, sha256, attempt): [row descriptions]}} and the rule controller classes the rows name."""
    by_fold, teachers = {}, set()
    for row in rows:
        if row.get("kind") != "episode":
            continue
        if kind == "rl":
            rl = row.get("rl") or {}
            if (rl.get("chain") or {}).get("job") != chain_job:
                continue
            fold, attempt, file, digest = rl["chain"]["shard"], rl["attempt"], rl["file"], rl["sha256"]
            detail = {"target": rl.get("target"), "iteration": rl.get("iteration"), "global_step": rl.get("global_step")}
            teachers.add((rl.get("rule_controller") or {}).get("class"))
        else:
            student = row.get("student") or {}
            if student.get("job") != chain_job:
                continue
            fold, attempt = student["shard"], student["attempt"]
            file, digest = student["model"]["file"], student["model"]["sha256"]
            detail = {"best_iteration": student.get("best_iteration")}
            teachers.add((student.get("teacher") or {}).get("class"))
        by_fold.setdefault(fold, {}).setdefault((file, digest, attempt), []).append(
            {"task_id": row.get("task_id"), **detail})
    return by_fold, teachers


def chain_record(chain, kind, file):
    """The chain.json record of `file`: the agent snapshot (rl) or the selected student (student-v5)."""
    result = chain.get("result") or {}
    if not chain.get("finished") or not result.get("finished"):
        return None
    if kind == "rl":
        found = [r for r in result.get("agent_snapshots", []) if r.get("file") == file]
        return found[0] if len(found) == 1 else None
    model = result.get("model") or {}
    return model if model.get("file") == file else None


def kernel_ref(builds_root, job, fold, attempt, mode):
    found = []
    for path in sorted(Path(builds_root, job).glob("*/build.json")):
        if path.parent.name.startswith("._"):
            continue
        build = json.loads(path.read_text(encoding="utf-8"))
        if (build.get("job_id"), build.get("shard"), build.get("attempt"), build.get("mode")) == (job, fold, attempt, mode):
            found.append(build["kernel"])
    return found


def chain_folder(state_path, runs_root, job, fold, attempt, mode):
    """The fetched output folder of a chain attempt: data/remote/state.json's last fetched path for it (the newest
    version if several), else <runs_root>/<job>/<fold>/<mode>-<attempt>/v1 (scripts/remote.py fetched_dir)."""
    path = Path(state_path)
    kernels = json.loads(path.read_text(encoding="utf-8")).get("kernels", {}) if path.is_file() else {}
    fetched = [k for k in kernels.values() if (k.get("job_id"), k.get("shard"), k.get("attempt"), k.get("mode"))
               == (job, fold, attempt, mode) and k.get("fetched")]
    if fetched:
        newest = max(fetched, key=lambda k: int(k.get("version") or 0))
        return ROOT / newest["fetched"][-1]["path"]
    return Path(runs_root, job, fold, f"{mode}-{attempt}", "v1")


def latest_dataset_files(record_path, ref):
    """{file: sha256} of the dataset's latest successful version (the one a notebook mounts), or None."""
    path = Path(record_path)
    if not path.is_file():
        return None
    versions = [v for v in json.loads(path.read_text(encoding="utf-8")).get(ref, {}).get("versions", []) if v.get("ok")]
    return versions[-1]["files"] if versions else None


def cmd_manifest(parser, args):
    check_username(parser, args.username)
    rows = []
    for path in args.results:
        for line in (ROOT / path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    by_fold, teachers = row_models(rows, args.kind, args.chain_job)
    if sorted(by_fold) != sorted(args.folds):
        raise SystemExit(f"rows name folds {sorted(by_fold)}, expected {sorted(args.folds)}")
    if len(teachers) != 1 or None in teachers:
        raise SystemExit(f"the rows name rule controllers {sorted(map(str, teachers))} (expected one)")
    models = []
    for fold in args.folds:
        if len(by_fold[fold]) != 1:
            raise SystemExit(f"{fold}: the rows were made with {len(by_fold[fold])} different files: "
                             f"{sorted(k[:2] for k in by_fold[fold])}")
        (file, digest, attempt), used = next(iter(by_fold[fold].items()))
        folder = chain_folder(ROOT / args.state, ROOT / args.runs_root, args.chain_job, fold, attempt, args.mode)
        chain_path = folder / "chain.json"
        if not chain_path.is_file():
            raise SystemExit(f"{fold}: no {relative(chain_path)} to check the file against")
        record = chain_record(json.loads(chain_path.read_text(encoding="utf-8")), args.kind, file)
        if record is None or record.get("sha256") != digest:
            raise SystemExit(f"{fold}: {file} {digest[:12]}… is not the finished chain's record in {relative(chain_path)}")
        local = folder / "dagger" / file
        if local.is_file() and sha256_file(local) != digest:
            raise SystemExit(f"{relative(local)}: SHA-256 differs from the rows")
        if attempt == "local":
            if args.kind != "rl":
                raise SystemExit(f"{fold}: a student chain run on this computer has no Kaggle input")
            ref = f"{args.username}/{args.dataset}"
            name = f"{fold}-{PurePosixPath(file).name}"                      # scripts/model_dataset.py stage_fold
            files = latest_dataset_files(ROOT / args.datasets_record, ref)
            if files is None or files.get(name) != digest:
                raise SystemExit(f"{fold}: {name} {digest[:12]}… is not in the latest version of {ref} "
                                 f"({relative(ROOT / args.datasets_record)})")
            source, inside = {"type": "dataset", "ref": ref}, name
        else:
            refs = kernel_ref(ROOT / args.builds_root, args.chain_job, fold, attempt, args.mode)
            if len(refs) != 1:
                raise SystemExit(f"{fold}: {len(refs)} built kernels for attempt {attempt} ({args.mode})")
            source, inside = {"type": "kernel", "ref": refs[0]}, f"dagger/{file}"
        if source["ref"].split("/")[0] != args.username:
            raise SystemExit(f"{fold}: source {source['ref']} is not owned by {args.username}")
        models.append({"fold": fold, "file": inside, "sha256": digest, "source": source,
                       "chain": {"job": args.chain_job, "shard": fold, "attempt": attempt, "mode": args.mode,
                                 "file": file, "folder": relative(folder), "chain_json": relative(chain_path),
                                 "local_copy": relative(local) if local.is_file() else None},
                       "rows": len(used), "detail": used[0]})
    manifest = {"kind": args.kind, "chain_job": args.chain_job, "teacher_class": teachers.pop(),
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "results": [{"file": p, "sha256": sha256_file(ROOT / p)} for p in args.results], "models": models}
    write_json(ROOT / args.out, manifest)
    print(json.dumps({"manifest": args.out, "models": [{k: m[k] for k in ("fold", "file", "sha256", "source")}
                                                       for m in models]}, indent=2))


def cmd_stage(parser, args):
    """Copy every manifest file from its local copy into <out>/<slug>/<file> (the /kaggle/input layout), checked, for
    the local notebook test with SB_ENSEMBLE_INPUT_ROOT=<out>. Local copies: the one the manifest recorded, the dataset
    staging folder of scripts/model_dataset.py (.tmp/model-dataset/<slug>/), or a full fetch of the chain kernel."""
    import shutil
    manifest = json.loads((ROOT / args.manifest).read_text(encoding="utf-8"))
    out, missing, staged = ROOT / args.out, [], []
    for entry in manifest.get("models") or []:
        _, slug, inside = mounted_models.check_entry(entry)
        chain = entry.get("chain") or {}
        options = [ROOT / chain["local_copy"]] if chain.get("local_copy") else []
        if entry["source"]["type"] == "dataset":
            options.append(ROOT / ".tmp/model-dataset" / slug / inside)
        elif chain.get("folder") and chain.get("file"):
            options.append(ROOT / chain["folder"] / "dagger" / chain["file"])
        source = next((p for p in options if p.is_file() and sha256_file(p) == entry["sha256"]), None)
        if source is None:
            if entry["source"]["type"] == "kernel" and chain.get("folder"):
                pattern = "^" + re.escape(f"dagger/{chain['file']}") + "$"
                hint = f"kaggle kernels output {entry['source']['ref']} -p '{chain['folder']}' -o --file-pattern '{pattern}'"
            else:
                hint = "a local copy"
            missing.append(f"{entry['fold']} {entry['file']}: no local copy with its SHA-256 (get it with: {hint})")
            continue
        target = out / slug / inside
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if sha256_file(target) != entry["sha256"]:
            raise SystemExit(f"{relative(target)}: copy differs")
        staged.append({"fold": entry["fold"], "from": relative(source), "to": relative(target)})
    if missing:
        raise SystemExit("\n".join(missing))
    print(json.dumps({"input_root": relative(out), "staged": staged}, indent=2))


# ------------------------------------------------------------------ notebook

def check_manifest(parser, manifest, kind, username):
    models = manifest.get("models") or []
    if not models:
        parser.error("the manifest has no models")
    folds = [m.get("fold") for m in models]
    if len(set(folds)) != len(folds):
        parser.error(f"duplicate folds in the manifest: {folds}")
    for entry in models:
        try:
            user, _, _ = mounted_models.check_entry(entry)
        except ValueError as error:
            parser.error(str(error))
        if user != username:
            parser.error(f"{entry['fold']}: source {entry['source']['ref']} is not owned by {username}")
    if manifest.get("kind", kind) != kind:
        parser.error(f"the manifest is for kind {manifest.get('kind')!r}, not {kind!r}")
    return models


def embedded_sources(parser, from_build, modules, tree_allowed):
    """{relative: (source text, sha256, origin)} for every module file; from a built work package when given."""
    files = None
    if from_build:
        from smart_building.remote import package as work_package
        folder = ROOT / from_build
        build = json.loads((folder / "build.json").read_text(encoding="utf-8"))
        code = work_package.embedded_code((folder / "run.py").read_text(encoding="utf-8"))
        if work_package.sha256_bytes(code) != build["code_sha256"]:
            parser.error(f"{from_build}: embedded code does not match build.json")
        files = work_package.unzip(code)
        origin_build = {"build": from_build, "code_sha256": build["code_sha256"]}
    out = {}
    for _, _, rel in modules:
        if rel is None:
            continue
        key = f"packages/{rel}"
        if files is not None and key in files:
            data, origin = files[key], {"from": "build", **origin_build}
        elif files is not None and rel not in tree_allowed:
            parser.error(f"{from_build}: the work package has no {key}")
        else:
            data, origin = (base.PACKAGE_ROOT / rel).read_bytes(), {"from": "tree"}
        out[rel] = (data.decode("utf-8"), hashlib.sha256(data).hexdigest(), origin)
    return out


def cmd_build(parser, args):
    check_username(parser, args.username)
    policy_module, policy_class, modules, tree_allowed, default_title, default_out = KINDS[args.kind]
    title = args.title or default_title
    slug = args.slug or re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    if len(title) > TITLE_LIMIT:
        parser.error(f"--title has {len(title)} characters (Kaggle allows {TITLE_LIMIT})")
    if re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") != slug:
        parser.error("--slug must equal the slug Kaggle derives from --title")
    manifest_path = ROOT / args.manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    models = check_manifest(parser, manifest, args.kind, args.username)
    teacher = args.teacher_class or manifest.get("teacher_class") or TEACHER
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", teacher):
        parser.error("--teacher-class must be a class name")
    if manifest.get("teacher_class") not in (None, teacher):
        parser.error(f"the manifest's rows used {manifest['teacher_class']}, not {teacher}")
    checked = None
    if args.check_root:                       # the files as a notebook would find them (refuses on any mismatch)
        try:
            checked = [relative(p) for p in mounted_models.locate_models(models, ROOT / args.check_root)]
        except (RuntimeError, ValueError) as error:
            raise SystemExit(f"model check failed: {error}")
    sources = embedded_sources(parser, args.from_build, modules, tree_allowed)
    try:                       # the embedded module that defines it: c3_teachers.py (readback) or g36.py (cold3)
        rule_module, rule_file = base.teacher_module(teacher, lambda rel: sources[rel][0])
    except (KeyError, ValueError) as error:
        parser.error(f"--teacher-class {teacher}: {error}")
    rule_source = rule_file.removeprefix("smart_building/")                 # policies/c3_teachers.py or policies/g36.py

    embedded, install_cells = [], []
    for name, is_package, rel in modules:
        if rel is None:
            source, digest, origin = "", None, None
        else:
            source, digest, origin = sources[rel]
        file = f"packages/{rel}" if rel else None
        embedded.append({"module": name, "package": is_package, "file": file, "sha256": digest, "origin": origin})
        install_cells.append(base.cell("code", f"# {name}" + (f" ({file}, sha256 {digest})" if rel else " (empty package)")
                                       + f"\ninstall_embedded_module({name!r}, {is_package!r}, {source!r})"))
    entries = [{k: m[k] for k in ("fold", "file", "sha256", "source")} for m in models]
    listing = "\n".join(f"| `{e['module']}` | `{e['file'] or '-'}` | `{e['sha256'] or '-'}` |" for e in embedded)
    rows = "\n".join(f"| {m['fold']} | {m['source']['type']} `{m['source']['ref']}` | `{m['file']}` | `{m['sha256']}` |"
                     for m in models)
    what = ("the mean of the fold agents' mean actions (each as learned/ppo.py policy_class acts)" if args.kind == "rl"
            else "the mean of the fold students' continuous outputs")
    header = f"""# {title}

Generated by `scripts/build_submission_rl_ensemble.py` ({args.kind}) on {time.strftime("%Y-%m-%d")}.
Policy: `{policy_module}.{policy_class}`: the continuous action is {what}; on/off actions, the B1-R
internal state the networks read (before each of its calls) and the per-step fallback (D-T02) come from one
`{teacher}` ({rule_source}). Every model file is found under the mounted inputs and checked against its
SHA-256 before the server starts (policies/mounted_models.py).

| fold | input | file | SHA-256 |
|---|---|---|---|
{rows}

| module | file | SHA-256 |
|---|---|---|
{listing}
"""
    wrapper = f"""import json
import os
from pathlib import Path

import torch

from {policy_module} import ensemble_policy_class
from {rule_module} import {teacher} as RuleController
from smart_building.policies.mounted_models import locate_models

torch.set_num_threads(1)
INPUT_ROOT = Path(os.environ.get('SB_ENSEMBLE_INPUT_ROOT', '/kaggle/input'))
MODELS = json.loads({json.dumps(entries)!r})
MODEL_PATHS = locate_models(MODELS, INPUT_ROOT)        # stops unless every file has the manifest's SHA-256
print('model files:', [str(path) for path in MODEL_PATHS])
Ensemble = ensemble_policy_class(MODEL_PATHS, RuleController)


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
    notebook = {"cells": [base.cell("markdown", header), base.cell("code", base.SETUP), base.cell("code", base.LOADER),
                          *install_cells, base.cell("code", wrapper)],
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python", "version": "3.13.15"}},
                "nbformat": 4, "nbformat_minor": 4}
    out = ROOT / (args.out or default_out)
    out.mkdir(parents=True, exist_ok=True)
    code_file = f"{slug}.ipynb"
    notebook_path = out / code_file
    with open(notebook_path, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    metadata = {"id": f"{args.username}/{slug}", "title": title, "code_file": code_file,
                "language": "python", "kernel_type": "notebook", "is_private": True,
                "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
                "dataset_sources": sorted({m["source"]["ref"] for m in models if m["source"]["type"] == "dataset"}),
                "competition_sources": [base.COMPETITION],
                "kernel_sources": sorted({m["source"]["ref"] for m in models if m["source"]["type"] == "kernel"}),
                "model_sources": []}
    write_json(out / "kernel-metadata.json", metadata)
    record = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "kind": args.kind,
              "notebook": relative(notebook_path), "notebook_sha256": sha256_file(notebook_path),
              "policy": f"{policy_module}.{policy_class} (rule controller {teacher})",
              "manifest": {"file": relative(manifest_path), "sha256": sha256_file(manifest_path)},
              "models": models, "checked_under": args.check_root, "checked_paths": checked,
              "from_build": args.from_build, "embedded": embedded}
    write_json(out / "embedded-files.json", record)
    print(json.dumps({k: record[k] for k in ("notebook", "notebook_sha256", "checked_paths")}
                     | {"kernel_sources": metadata["kernel_sources"], "dataset_sources": metadata["dataset_sources"]},
                     indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("manifest", help="the manifest from the gate's evaluation rows")
    p.add_argument("--kind", choices=sorted(KINDS), required=True)
    p.add_argument("--chain-job", required=True, help="the training chain the rows name (j9; students: j7b-v5s)")
    p.add_argument("--results", nargs="+", required=True, help="results.jsonl files with the S4 rows (repo-relative)")
    p.add_argument("--folds", nargs="+", default=FOLDS)
    p.add_argument("--mode", default="full")
    p.add_argument("--username", required=True)
    p.add_argument("--dataset", default="sb-j9-models", help="private dataset of the folds trained on the Mac")
    p.add_argument("--runs-root", default="data/runs/remote")
    p.add_argument("--builds-root", default="data/remote")
    p.add_argument("--state", default="data/remote/state.json", help="where fetched chain outputs are recorded")
    p.add_argument("--datasets-record", default="data/remote/datasets.json")
    p.add_argument("--out", required=True, help="manifest path (repo-relative)")
    p = sub.add_parser("stage", help="local copies of the manifest's files laid out as /kaggle/input (notebook test)")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True, help="folder to use as SB_ENSEMBLE_INPUT_ROOT (repo-relative, under .tmp/)")
    p = sub.add_parser("build", help="the notebook, kernel-metadata.json and embedded-files.json")
    p.add_argument("--kind", choices=sorted(KINDS), default="rl")
    p.add_argument("--manifest", required=True)
    p.add_argument("--username", required=True)
    p.add_argument("--title")
    p.add_argument("--slug")
    p.add_argument("--out")
    p.add_argument("--teacher-class", help=f"class in policies/c3_teachers.py or policies/g36.py (default: the "
                                           f"manifest's, else {TEACHER}; the cold3 chain: G36PolicyC3LaaOccrowDampfloorCold3)")
    p.add_argument("--from-build", help="kernel folder of the work package whose code is embedded (S4's)")
    p.add_argument("--check-root", help="a folder laid out as /kaggle/input: refuse unless every model is found there "
                                        "with its SHA-256")
    args = parser.parse_args(argv)
    return {"manifest": cmd_manifest, "stage": cmd_stage, "build": cmd_build}[args.command](parser, args)


if __name__ == "__main__":
    main()
