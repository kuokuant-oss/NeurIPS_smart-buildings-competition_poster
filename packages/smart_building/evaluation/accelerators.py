"""J0 accelerator-machine probes (remote plan §1, §6 J0 row, §12).

Entry point for the TPU and GPU machine probes; it wraps the unchanged shard
runner (smart_building.remote.runner.main):

1. compare the installed versions with job["expected_packages"]; on Kaggle,
   with internet enabled for this probe only, install the pinned versions of
   simulation packages that are missing or different (torch is only recorded);
   local dry runs only list what would be installed;
2. GPU check (job["gpu_check"]): torch version and whether torch sees a GPU;
3. hand over to the runner: environment record, precheck, parity probe and the
   job's throughput probes (for example 40 and 80 processes on the TPU machine).

accelerators.json in the output folder records all of it.
"""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from smart_building.remote import paths, runner

PIP_NAMES = {"numpy": "numpy", "numba": "numba", "llvmlite": "llvmlite", "scipy": "scipy",
             "pandas": "pandas", "gymnasium": "gymnasium", "gin-config": "gin-config",
             "holidays": "holidays", "protobuf": "protobuf", "PyYAML": "PyYAML",
             "matplotlib": "matplotlib", "opencv": "opencv-python"}


def installed(name):
    if name == "opencv":
        return runner.package_versions()["opencv"]
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def gpu_record():
    record = {"torch": installed("torch")}
    try:
        import torch
        record.update(cuda_available=bool(torch.cuda.is_available()),
                      device_count=int(torch.cuda.device_count()),
                      devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    except Exception as exc:  # recorded, the probe goes on
        record["error"] = repr(exc)
    try:
        record["nvidia_smi"] = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True,
                                              timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        record["nvidia_smi"] = repr(exc)
    return record


def main(job, shard, mode, package, environ=None):
    environ = dict(os.environ if environ is None else environ)
    locations = paths.resolve_locations(job["job_id"], shard, environ)
    out = Path(locations["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    record = {"started_utc": runner.utc_now(), "backend": locations["backend"],
              "logical_cpus": os.cpu_count(), "memory": runner.memory_record(),
              "python": sys.version, "cpu_model": runner.cpu_model()}
    before = {name: installed(name) for name in PIP_NAMES}
    wanted = [f"{PIP_NAMES[name]}=={job['expected_packages'][name]}" for name in PIP_NAMES
              if job["expected_packages"].get(name) and before[name] != job["expected_packages"][name]]
    record.update(versions_before=before, to_install=wanted)
    # Only shards that the job gives internet try to install (J0: the offline GPU shard retried pip
    # for about 203 s before failing).
    online = bool(job["shards"][shard].get("kernel_overrides", {}).get("enable_internet"))
    record["internet"] = online
    if wanted and job.get("install_missing") and online and locations["backend"] == "kaggle":
        started = time.time()
        result = subprocess.run([sys.executable, "-m", "pip", "install", "--no-input",
                                 "--disable-pip-version-check", *wanted], capture_output=True, text=True)
        record["pip"] = {"returncode": result.returncode, "seconds": time.time() - started,
                         "stdout_tail": result.stdout[-4000:], "stderr_tail": result.stderr[-4000:]}
        record["versions_after"] = {name: installed(name) for name in PIP_NAMES}
    elif wanted:
        record["pip"] = "not run (only on Kaggle, on a shard with internet, with install_missing)"
    if job.get("gpu_check"):
        record["gpu"] = gpu_record()
    with open(out / "accelerators.json", "w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"[accelerators] {json.dumps({k: record.get(k) for k in ('logical_cpus', 'to_install', 'gpu')})}",
          flush=True)
    return runner.main(job, shard, mode, package, environ)
