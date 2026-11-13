"""The shard runner (remote plan §3.4).

Order: 0 cache and temp variables; 1 environment.json; 2 inputs-tree.txt and
earlier outputs of the same shard; 3 scenario overlay (and generated pools);
4 precheck (12 steps per task kind; failure -> precheck.json, non-zero exit);
5 parity probe (pid_water x building_train_1/atlanta_2 x 288 steps);
6 task loop (workers processes, first launches staggered, every result
appended and flushed at once, failures recorded and skipped, progress.json
every 5 minutes); 7 time guard; 8 job-output.json and summary.json.

Test hooks for the local dry run (remote plan §4 DR2) are environment
variables, recorded in environment.json and job-output.json when set:
SB_REMOTE_TEST_BUDGET_SECONDS, SB_REMOTE_TEST_FAIL_TASK and
SB_REMOTE_TEST_MAX_PROCESSES. Kaggle never sets them.
"""
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import traceback

from smart_building.remote import chain_inputs, jobs, paths, scenarios, tasks as task_lib

TEST_VARIABLES = ("SB_REMOTE_TEST_BUDGET_SECONDS", "SB_REMOTE_TEST_FAIL_TASK",
                  "SB_REMOTE_TEST_MAX_PROCESSES", "SB_REMOTE_TEST_FORCE_VENDORED_CMA")
CMA_VERSION = "4.5.0"  # D-P03


def ensure_cma(code_root):
    """pycma for cmaes tasks (remote plan §12 item 6): the image's cma if it is 4.5.0, else the copy
    vendored in the code snapshot (<code>/vendor, with its BSD-3-Clause license). Returns the record
    written to precheck.json and cmaes.jsonl: source, version, file."""
    if not os.environ.get("SB_REMOTE_TEST_FORCE_VENDORED_CMA"):
        try:
            import cma
            if cma.__version__ == CMA_VERSION:
                return {"source": "image" if "vendor" not in Path(cma.__file__).parts else "vendored",
                        "version": cma.__version__, "file": cma.__file__}
        except ImportError:
            pass
    vendor = Path(code_root) / "vendor"
    if not (vendor / "cma" / "__init__.py").is_file():
        raise RuntimeError(f"no cma {CMA_VERSION} in the image and none vendored in the code snapshot")
    for name in [m for m in sys.modules if m == "cma" or m.startswith("cma.")]:
        del sys.modules[name]
    sys.path.insert(0, str(vendor))
    import cma
    if cma.__version__ != CMA_VERSION:
        raise RuntimeError(f"vendored cma is {cma.__version__}, not {CMA_VERSION}")
    return {"source": "vendored", "version": cma.__version__, "file": cma.__file__,
            "license": str(vendor / f"cma-{CMA_VERSION}.LICENSE")}
PACKAGES = ("numpy", "numba", "llvmlite", "scipy", "pandas", "gymnasium", "gin-config",
            "holidays", "protobuf", "PyYAML", "matplotlib", "torch", "cma")
OPENCV_DISTRIBUTIONS = ("opencv-python", "opencv-python-headless", "opencv-contrib-python",
                        "opencv-contrib-python-headless")
PRECHECK_STEPS = 12
# Test objectives for the cmaes kind (unit tests and local dry runs only; never in a real job).
TEST_OBJECTIVES = {"sphere": lambda x: float(sum((float(v) - 0.3) ** 2 for v in x))}
PRECHECK_TIMEOUT_SECONDS = 300
PROGRESS_SECONDS = 300
GUARD_MARGIN = 1.2


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


SEGMENT_MIN_SECONDS = 3 * 3600


def window_budget(started, budget, windows, min_seconds=SEGMENT_MIN_SECONDS):
    """(budget, record): a chain segment's budget cut so that it ends inside a daily window (job field
    "segment_end_windows_utc": [["HH:MM", "HH:MM"], ...], UTC; a window may cross midnight). Chains are relayed
    by a pipeline on a computer that is on only part of the day (user 2026-10-20: about 20:00-08:00 Taipei),
    so a segment that would end outside every window ends instead at the latest window close before its
    budget runs out, if that still leaves `min_seconds`; otherwise the budget is unchanged. Scheduling only:
    a segment checkpoints and the next one resumes, so the results do not depend on where segments end."""
    import datetime as dt

    def minutes(text):
        hours, mins = text.split(":")
        return int(hours) * 60 + int(mins)

    end = started + budget
    end_dt = dt.datetime.fromtimestamp(end, dt.timezone.utc)
    record = {"windows_utc": windows, "min_seconds": min_seconds, "full_budget_seconds": budget,
              "full_end_utc": end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")}
    of_day = end_dt.hour * 60 + end_dt.minute
    for start, stop in windows:
        a, b = minutes(start), minutes(stop)
        if (a <= of_day <= b) if a <= b else (of_day >= a or of_day <= b):
            return budget, dict(record, cut=False, reason="ends inside a window")
    midnight = end_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    closes = [(midnight + dt.timedelta(days=day, minutes=minutes(stop))).timestamp()
              for _, stop in windows for day in (-2, -1, 0)]
    usable = [c for c in closes if started + min_seconds <= c <= end]
    if not usable:
        return budget, dict(record, cut=False, reason="no window close leaves the minimum segment")
    close = max(usable)
    return close - started, dict(record, cut=True, end_utc=dt.datetime.fromtimestamp(
        close, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), budget_seconds=close - started)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Output:
    """Writes into the shard's output folder; JSON Lines rows are flushed at once."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def write_json(self, name, data):
        path = self.directory / name
        temporary = path.with_name(path.name + ".partial")
        with open(temporary, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(data, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)

    def write_text(self, name, text):
        with open(self.directory / name, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)

    def append(self, name, row):
        with open(self.directory / name, "a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def touch(self, name):
        (self.directory / name).touch()


def cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    if sys.platform == "darwin":
        try:
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                  text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return platform.processor() or platform.machine()


def memory_record():
    record = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            if key in ("MemTotal", "MemAvailable"):
                record[key] = int(value.split()[0]) * 1024
    except OSError:
        if sys.platform == "darwin":
            try:
                record["MemTotal"] = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                                                        text=True, check=True).stdout.strip())
            except (OSError, subprocess.CalledProcessError, ValueError):
                pass
    return record


def package_versions():
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    versions["opencv"] = None
    for name in OPENCV_DISTRIBUTIONS:
        try:
            versions["opencv"] = importlib.metadata.version(name)
            break
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def configure_environment(work_dir):
    """Step 0: the cache and temp variables of scripts/workspace.py local_environment()."""
    locations = {"TMPDIR": "temp", "TEMP": "temp", "TMP": "temp", "XDG_CACHE_HOME": "cache",
                 "MPLCONFIGDIR": "matplotlib", "NUMBA_CACHE_DIR": "numba", "IPYTHONDIR": "ipython",
                 "JUPYTER_RUNTIME_DIR": "jupyter"}
    values = {}
    for key, relative in locations.items():
        path = Path(work_dir) / "env" / relative
        path.mkdir(parents=True, exist_ok=True)
        values[key] = str(path)
    values.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1", MPLBACKEND="Agg")
    os.environ.update(values)
    sys.dont_write_bytecode = True
    return values


def collect_previous(input_root, job_id, shard, expected_inputs):
    """Earlier outputs of this shard: their finished task ids, after hash checks. Inputs of another
    job (entries with that "job_id", e.g. a finished DAgger chain for J7c) are checked the same way
    and listed with "source": True; their results are not this shard's."""
    report, done, problems = [], set(), []
    foreign = [e for e in expected_inputs or [] if e.get("job_id", job_id) != job_id]
    expected = {entry["attempt"]: entry["files"] for entry in expected_inputs or [] if entry not in foreign}
    for directory, marker in paths.find_markers(input_root, job_id):
        if marker.get("shard") != shard:
            continue
        entry = {"path": str(directory), "attempt": marker.get("attempt"),
                 "status": marker.get("status"), "package_sha256": marker.get("package_sha256")}
        listed = marker.get("files", {})
        bad = [name for name, digest in listed.items()
               if not (directory / name).is_file() or sha256_file(directory / name) != digest]
        wanted = expected.get(marker.get("attempt"), {})
        unexpected = [name for name, digest in wanted.items()
                      if not (directory / name).is_file() or sha256_file(directory / name) != digest]
        entry.update(files_checked=len(listed), files_bad=bad, expected_mismatch=unexpected)
        if bad or unexpected:
            problems.append(entry)
        results = directory / "results.jsonl"
        if results.is_file():
            for line in results.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    done.add(json.loads(line)["task_id"])
        report.append(entry)
    missing = sorted(set(expected) - {e["attempt"] for e in report})
    if missing:
        problems.append({"missing_expected_attempts": missing})
    for source in foreign:
        found = [(d, m) for d, m in paths.find_markers(input_root, source["job_id"])
                 if m.get("shard") == source["shard"] and m.get("attempt") == source["attempt"]]
        if len(found) != 1:
            problems.append({"missing_source": {k: source.get(k) for k in ("job_id", "shard", "attempt")},
                             "found": len(found)})
            continue
        directory, marker = found[0]
        unexpected = [name for name, digest in source["files"].items()
                      if not (directory / name).is_file() or sha256_file(directory / name) != digest]
        entry = {"path": str(directory), "job_id": source["job_id"], "shard": source["shard"],
                 "attempt": source["attempt"], "status": marker.get("status"), "source": True,
                 "files_checked": len(source["files"]), "expected_mismatch": unexpected}
        if unexpected:
            problems.append(entry)
        report.append(entry)
    return report, done, problems


class TimeGuard:
    """Step 7: start a task only if elapsed + 1.2 x P90 of its class fits the budget."""

    def __init__(self, job, start, budget):
        self.job, self.start, self.budget = job, start, budget
        self.durations = defaultdict(list)

    def expected(self, task):
        done = self.durations.get(task["class"])
        return jobs.p90(done) if done else jobs.estimate_seconds(self.job, task)

    def allows(self, task, now):
        return (now - self.start) + GUARD_MARGIN * self.expected(task) <= self.budget

    def record(self, task, seconds):
        self.durations[task["class"]].append(seconds)


def annotate(row, task, attempt_info, started, finished):
    row = dict(row)
    spec = task.get("scenario", {})
    row.update(task_id=task["task_id"], kind=task["kind"], task_class=task["class"],
               formal=task["formal"], steps_planned=task["steps"], scenario_spec=spec,
               started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
               task_seconds=finished - started, **attempt_info)
    return row


def spawn_pool(workers):
    return ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))


def run_isolated(task, context, timeout):
    """One task in a fresh single-worker process (precheck, parity)."""
    with spawn_pool(1) as pool:
        return pool.submit(task_lib.execute_task, task, context).result(timeout=timeout)


def precheck(planned, context):
    """Step 4: one task of each kind for 12 steps."""
    checks, ok = [], True
    seen = set()
    for task in planned:
        if task["kind"] in seen or task["kind"] == "fail":
            continue
        seen.add(task["kind"])
        trial = dict(task, steps=PRECHECK_STEPS)
        start = time.time()
        try:
            if trial["kind"] == "cmaes":
                # pycma importable at the D-P03 version (image or vendored), and the strategy can be set up.
                cma_record = ensure_cma(context.get("code_root", "."))
                from smart_building.evaluation import cmaes_loop
                cmaes_loop.Strategy(len(trial["parameters"]), trial["x0_seed"], trial["cma_seed"],
                                    trial.get("max_generations"))
                if trial.get("test_objective"):
                    result = {"ok": True}
                else:
                    # One candidate (x0) on the first evaluation scenario for 12 steps.
                    x0 = cmaes_loop.initial_point(len(trial["parameters"]), trial["x0_seed"])
                    episode = {"task_id": f"{trial['task_id']}/precheck", "kind": "episode",
                               "policy": trial.get("policy", "candidate"),
                               "policy_params": cmaes_loop.denormalize(x0, trial["parameters"]),
                               "policy_factory": trial["policy_factory"], "scenario": trial["evaluation"][0],
                               "path": trial.get("path", "gateway"), "steps": PRECHECK_STEPS}
                    result = run_isolated(episode, context, PRECHECK_TIMEOUT_SECONDS)
            elif trial["kind"] == "dagger":
                import inspect
                function = task_lib.load_entry(context, trial["entry"])
                names = list(inspect.signature(function).parameters)[:4]
                if names != ["config", "state_dir", "out_dir", "deadline_utc"]:
                    raise TypeError(f"{trial['entry']}: parameters {names}, expected config, state_dir, out_dir, deadline_utc")
                if chain_inputs.chain_specs(trial.get("config")):
                    # J9: the files another chain's chain.json names are mounted and match (remote/chain_inputs.py)
                    chain_inputs.resolve(trial["config"], context.get("sources", []))
                result = {"ok": True}
            elif trial["kind"] == "probe":
                trial["processes"] = 1
                task_lib.run_probe_task(trial, context)
                result = {"ok": True}
            else:
                result = run_isolated(trial, context, PRECHECK_TIMEOUT_SECONDS)
        except Exception as exc:
            result = {"ok": False, "error": repr(exc), "traceback": traceback.format_exc()}
        entry = {"kind": task["kind"], "task_id": task["task_id"], "ok": result["ok"],
                 "seconds": time.time() - start}
        if task["kind"] == "cmaes" and "cma_record" in locals():
            entry["cma"] = cma_record
        if not result["ok"]:
            entry.update(error=result.get("error"), traceback=result.get("traceback"))
            ok = False
        checks.append(entry)
    return ok, checks


def parity_probe(job, context):
    """Step 5: compared bit for bit with the local reference; a mismatch does not stop the shard."""
    probe = job["parity_probe"]
    task = {"task_id": "parity", "kind": "episode", "policy": probe["policy"],
            "scenario": {"set": "official", "name": probe["scenario"]}, "steps": probe["steps"]}
    result = run_isolated(task, context, PRECHECK_TIMEOUT_SECONDS)
    record = {"policy": probe["policy"], "scenario": probe["scenario"], "steps": probe["steps"],
              "reference_repr": probe.get("reference_total_reward_repr"),
              "reference_source": probe.get("reference_source"), "cpu_model": context["cpu_model"]}
    if result["ok"]:
        value = repr(result["row"]["total_reward"])
        record.update(ok=True, total_reward_repr=value, row=result["row"],
                      match=None if record["reference_repr"] is None else value == record["reference_repr"])
    else:
        record.update(ok=False, error=result["error"], traceback=result["traceback"], match=False)
    return record


class TaskLoop:
    """Steps 6 and 7."""

    def __init__(self, job, shard, mode, context, out, guard, workers, stagger, attempt_info):
        self.job, self.shard, self.mode = job, shard, mode
        self.context, self.out, self.guard = context, out, guard
        self.workers, self.stagger, self.attempt_info = workers, stagger, attempt_info
        self.done, self.failed, self.running = [], [], {}
        self.not_started = []
        self.last_progress = 0.0
        self.planned = 0

    def record(self, task, result):
        started, finished = result.get("started", time.time()), result.get("finished", time.time())
        if result["ok"]:
            self.out.append("results.jsonl", annotate(result["row"], task, self.attempt_info,
                                                      started, finished))
            self.guard.record(task, finished - started)
            self.done.append(task["task_id"])
            print(f"[runner] done {task['task_id']} ({finished - started:.0f}s)", flush=True)
        else:
            self.out.append("errors.jsonl", {"task_id": task["task_id"], "kind": task["kind"],
                                             "error": result.get("error"),
                                             "traceback": result.get("traceback"),
                                             "seconds": finished - started, **self.attempt_info})
            self.failed.append(task["task_id"])
            print(f"[runner] FAILED {task['task_id']}: {result.get('error')}", flush=True)

    def progress(self, force=False):
        now = time.time()
        if not force and now - self.last_progress < PROGRESS_SECONDS:
            return
        self.last_progress = now
        pending = [t for t, _ in self.running.values()] + self.not_started
        remaining = sum(self.guard.expected(t) for t in pending) / max(1, self.workers)
        self.out.write_json("progress.json", {
            "job_id": self.job["job_id"], "shard": self.shard, "mode": self.mode,
            "updated_utc": utc_now(), "elapsed_seconds": now - self.guard.start,
            "planned": self.planned, "done": len(self.done), "failed": len(self.failed),
            "running": [t["task_id"] for t, _ in self.running.values()],
            "not_started": len(self.not_started), "estimated_remaining_seconds": remaining,
            **self.attempt_info})

    def run(self, remaining):
        self.planned = len(remaining)
        # latency (D-K04 estimate) runs alone like the throughput probe, so other tasks do not slow it down
        exclusive = [t for t in remaining if t["kind"] in ("probe", "cmaes", "dagger", "latency")]
        pooled = [t for t in remaining if t["kind"] not in ("probe", "cmaes", "dagger", "latency")]
        self.not_started = list(remaining)
        self.progress(force=True)
        for index, task in enumerate(exclusive):
            if task["kind"] == "cmaes":
                # The time guard is applied per generation inside (remote plan §12 item 2).
                self.not_started.remove(task)
                self.run_cmaes(task)
                self.progress()
                continue
            if task["kind"] == "dagger":
                # A chain segment: the learned agent's function stops itself before the deadline.
                self.not_started.remove(task)
                self.run_dagger(task)
                self.progress()
                continue
            if not self.guard.allows(task, time.time()):
                print(f"[runner] time guard: not starting {task['task_id']}", flush=True)
                return
            self.not_started.remove(task)
            started = time.time()
            try:
                if task["kind"] == "latency":
                    row = task_lib.run_latency_task(task, self.context)
                else:
                    row = task_lib.run_probe_task(task, self.context)
                result = {"ok": True, "row": row, "started": started, "finished": time.time()}
            except Exception as exc:
                result = {"ok": False, "error": repr(exc), "traceback": traceback.format_exc(),
                          "started": started, "finished": time.time()}
            self.record(task, result)
            self.progress()
        self.run_pool(pooled)
        self.progress(force=True)

    def run_dagger(self, task):
        """A DAgger chain segment (stage45 §3, J7): call the learned agent's
        run_dagger_segment(config, state_dir, out_dir, deadline_utc). state_dir is the newest
        segment state among the mounted earlier outputs (<previous>/dagger), or None for the first
        segment; out_dir is <output>/dagger; deadline_utc is the time guard's end (ISO, UTC). The
        function runs complete iterations, checkpoints and returns a dict with "finished". The
        segment writes chain.json; an unfinished chain leaves the task for the next segment."""
        started = time.time()

        def attempt_rank(directory):
            marker = Path(directory) / "job-output.json"
            attempt = json.loads(marker.read_text(encoding="utf-8")).get("attempt", "a") if marker.is_file() else "a"
            return 0 if attempt == "a" else int(attempt.lstrip("c") or 0)

        state_dir, previous_chain = None, None
        for directory in sorted(self.context.get("previous_dirs", []), key=attempt_rank):
            if (Path(directory) / "dagger").is_dir() and (Path(directory) / "chain.json").is_file():
                state_dir = str(Path(directory) / "dagger")
                previous_chain = json.loads((Path(directory) / "chain.json").read_text(encoding="utf-8"))
        if state_dir:  # the state files must be the ones the previous segment recorded
            listed = previous_chain.get("state_files", {})
            bad = sorted(name for name, digest in listed.items()
                         if not (Path(state_dir) / name).is_file() or sha256_file(Path(state_dir) / name) != digest)
            if bad or not listed:
                self.record(task, {"ok": False, "error": f"previous DAgger state fails its SHA-256 check: {bad[:5]}",
                                   "started": started, "finished": time.time()})
                return
        out_dir = self.out.directory / "dagger"
        out_dir.mkdir(parents=True, exist_ok=True)
        deadline = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.guard.start + self.guard.budget))
        # Resolved for the learned agent: where the code and each listed scenario are in this shard.
        os.environ["SB_SCENARIO_ROOT"] = str(self.context.get("scenario_root", ""))
        # learned/dagger.py DaggerSettings: unknown keys raise, so only its own keys are added here.
        config = dict(task["config"], remote_mode=self.mode, steps_cap=task.get("steps"),
                      code_root=self.context.get("code_root"), scenario_root=self.context.get("scenario_root"))
        chain = {"task_id": task["task_id"], "state_from": state_dir, "deadline_utc": deadline,
                 "started_utc": utc_now(), **self.attempt_info}
        if self.mode == "smoke" and config.get("iteration_estimate_seconds") and task.get("full_steps"):
            # smoke rollouts are steps_cap long: scale the first-iteration estimate the same way
            config["iteration_estimate_seconds"] *= task["steps"] / task["full_steps"]
            chain["smoke_iteration_estimate_seconds"] = config["iteration_estimate_seconds"]
        fallback = self.mode == "smoke" or task.get("device_fallback") == "cpu"
        if fallback and str(config.get("device", "cpu")).startswith("cuda"):
            import torch
            if not torch.cuda.is_available():
                # local DR2 (smoke), or a chain that may continue on a CPU machine (a J9 fold, coordinator
                # 2026-10-20: device is not part of a PPO run's identity; recorded per segment here)
                config["device"] = "cpu"
                chain["device_fallback"] = "cpu (no CUDA on this machine)"
        chain["device"] = config.get("device")
        try:
            if chain_inputs.chain_specs(config):
                # J9: settings that name another (mounted) chain's files by rule, resolved from its chain.json the same
                # way on every segment and machine (remote/chain_inputs.py); recorded in this segment's chain.json
                config, chain["resolved_inputs"] = chain_inputs.resolve(config, self.context.get("sources", []))
            function = task_lib.load_entry(self.context, task["entry"])
            result = function(config, state_dir, str(out_dir), deadline) or {}
        except Exception as exc:
            chain.update(finished=False, error=repr(exc))
            self.out.write_json("chain.json", chain)
            self.record(task, {"ok": False, "error": repr(exc), "traceback": traceback.format_exc(),
                               "started": started, "finished": time.time()})
            return
        chain.update(finished=bool(result.get("finished")), result=result, ended_utc=utc_now(),
                     state_files={path.relative_to(out_dir).as_posix(): sha256_file(path)
                                  for path in sorted(out_dir.rglob("*")) if path.is_file()})
        self.out.write_json("chain.json", chain)
        if chain["finished"]:
            self.record(task, {"ok": True, "row": {"policy": task.get("policy"), "dagger": result},
                               "started": started, "finished": time.time()})
        else:
            print(f"[runner] dagger {task['task_id']}: segment ends unfinished ({result.get('iteration')})", flush=True)
            self.not_started.append(task)  # continued by the next chain segment

    def run_cmaes(self, task):
        """CMA-ES inside the shard (D-P03; remote plan §12 item 2).

        Each generation evaluates every candidate on the job's evaluation set as episodes in the
        worker pool; fitness is the mean official loss over the set, the worst is recorded (D-P04).
        After every generation the strategy (with its RNG state) is checkpointed and the generation
        is appended to cmaes.jsonl. When pycma stops (its own criteria, or "maxiter" = the job's
        max_generations, the budget approved under D-P05), the final mean is evaluated as in
        cma.fmin2 (cmaes_loop.run). The time guard stops between generations and before the final
        mean; a continuation shard resumes from the newest checkpoint among the mounted earlier
        outputs. Smoke mode (remote plan §4 DR2/DR4) stops after smoke_generations generations.
        """
        cma_record = ensure_cma(self.context.get("code_root", "."))
        from smart_building.evaluation import cmaes_loop
        name = "cmaes-" + re.sub(r"[^A-Za-z0-9]+", "-", task["task_id"]).strip("-") + ".pkl"
        strategy, source = None, None
        for directory in self.context.get("previous_dirs", []):
            candidate = Path(directory) / name
            if candidate.is_file():
                loaded = cmaes_loop.load_checkpoint(candidate)
                if strategy is None or (loaded.generation, loaded.finished) > (strategy.generation, strategy.finished):
                    strategy, source = loaded, str(candidate)
        if strategy is None:
            strategy = cmaes_loop.Strategy(len(task["parameters"]), task["x0_seed"], task["cma_seed"],
                                           task.get("max_generations"))
        first_generation = strategy.generation
        generation_task = {"task_id": task["task_id"], "kind": "cmaes", "class": "cmaes-generation",
                           "steps": task["steps"], "full_steps": task.get("full_steps", task["steps"])}
        self.out.append("cmaes.jsonl", {"event": "start", "task_id": task["task_id"], "resumed_from": source,
                                        "generation": strategy.generation, "finished": strategy.finished,
                                        "settings": strategy.settings, "cma": cma_record, **self.attempt_info})
        pool = None if task.get("test_objective") else spawn_pool(self.workers)
        started = time.time()
        worst = {"last": None}  # D-P04: worst evaluation scenario of each candidate, for the log

        def fitness_of(points, label):
            worst["last"] = None
            if task.get("test_objective"):
                return [TEST_OBJECTIVES[task["test_objective"]](x) for x in points]
            subtasks = []
            for i, x in enumerate(points):
                params = cmaes_loop.denormalize(x, task["parameters"])
                for j, spec in enumerate(task["evaluation"]):
                    subtasks.append({"task_id": f"{task['task_id']}/{label}/c{i}/e{j}", "kind": "episode",
                                     "policy": task.get("policy", "candidate"), "policy_params": params,
                                     "policy_factory": task["policy_factory"], "scenario": spec,
                                     "path": task.get("path", "gateway"), "steps": task["steps"],
                                     "candidate": i})
            futures = [pool.submit(task_lib.execute_task, sub, self.context) for sub in subtasks]
            losses = [[] for _ in points]
            scenarios = [[] for _ in points]
            for sub, future in zip(subtasks, futures):
                result = future.result()
                if not result["ok"]:
                    raise RuntimeError(f"{sub['task_id']} failed: {result['error']}")
                row = result["row"]
                # Official mean loss of the episode (stats.loss): gateway mean, or total / 8640 (D-E01).
                loss = -row["mean_reward"] if row.get("path") == "gateway" else -row["total_reward"] / 8640
                losses[sub["candidate"]].append(loss)
                scenarios[sub["candidate"]].append(sub["scenario"])
                self.out.append("cmaes.jsonl", {"event": "episode", "task_id": sub["task_id"], "label": label,
                                                "candidate": sub["candidate"],
                                                "parameters": sub["policy_params"], "loss": loss,
                                                "row": row, **self.attempt_info})
            # D-P04: the mean over the evaluation set is the fitness; the worst scenario is recorded.
            worst["last"] = [{"loss": max(values), "scenario": names[values.index(max(values))]}
                             for values, names in zip(losses, scenarios)]
            return [sum(values) / len(values) for values in losses]

        def evaluate(points):
            return fitness_of(points, f"g{strategy.generation + 1}")

        def evaluate_final(point):
            return fitness_of([point], "final")[0]

        smoke_limit = None if self.mode == "full" else int(task.get("smoke_generations", 2))

        def keep_going():
            if smoke_limit is not None and strategy.generation - first_generation >= smoke_limit:
                return False
            return self.guard.allows(generation_task, time.time())

        clock = {"last": time.time()}

        def after_generation(strategy_, points, fitness):
            digest = cmaes_loop.save_checkpoint(self.out.directory / name, strategy_)
            now = time.time()
            self.guard.record(generation_task, now - clock["last"])
            self.out.append("cmaes.jsonl", {"event": "generation", "task_id": task["task_id"],
                                            "generation": strategy_.generation,
                                            "points": [list(map(float, x)) for x in points],
                                            "parameters": [cmaes_loop.denormalize(x, task["parameters"])
                                                           for x in points],
                                            "fitness": list(map(float, fitness)), "worst": worst["last"],
                                            "sigma": float(strategy_.es.sigma),
                                            "best_fitness": float(strategy_.es.result.fbest),
                                            "seconds": now - clock["last"], "checkpoint": name,
                                            "checkpoint_sha256": digest, **self.attempt_info})
            clock["last"] = now
            print(f"[runner] cmaes {task['task_id']} generation {strategy_.generation}: "
                  f"best {strategy_.es.result.fbest:.6g}", flush=True)
            self.progress()

        def after_final(strategy_, point, value):
            digest = cmaes_loop.save_checkpoint(self.out.directory / name, strategy_)
            now = time.time()
            self.out.append("cmaes.jsonl", {"event": "final_mean", "task_id": task["task_id"],
                                            "generation": strategy_.generation,
                                            "point": list(map(float, point)),
                                            "parameters": cmaes_loop.denormalize(point, task["parameters"]),
                                            "fitness": float(value), "worst": worst["last"],
                                            "stop": strategy_.result()["stop"],
                                            "seconds": now - clock["last"], "checkpoint": name,
                                            "checkpoint_sha256": digest, **self.attempt_info})
            clock["last"] = now

        try:
            cmaes_loop.run(strategy, evaluate, keep_going, after_generation, evaluate_final, after_final)
        except Exception as exc:
            self.record(task, {"ok": False, "error": repr(exc), "traceback": traceback.format_exc(),
                               "started": started, "finished": time.time()})
            return
        finally:
            if pool is not None:
                pool.shutdown(wait=True)
        smoke_stop = (not strategy.finished and smoke_limit is not None
                      and strategy.generation - first_generation >= smoke_limit)
        if strategy.finished or smoke_stop:
            result = strategy.result()
            row = {"policy": task.get("policy"), "generations": strategy.generation,
                   "finished": strategy.finished, "stopped_by": "pycma" if strategy.finished else "smoke_generations",
                   "result": result, "best_parameters": cmaes_loop.denormalize(result["xbest"], task["parameters"]),
                   "settings": strategy.settings, "resumed_from": source}
            self.record(task, {"ok": True, "row": row, "started": started, "finished": time.time()})
        else:
            print(f"[runner] time guard: cmaes {task['task_id']} stops after generation {strategy.generation}",
                  flush=True)
            self.not_started.append(task)  # resumed from the checkpoint by a continuation shard

    def run_pool(self, queue):
        queue = list(queue)
        launched, last_launch, stopped = 0, 0.0, False
        pool = spawn_pool(self.workers)
        try:
            while True:
                while queue and not stopped and len(self.running) < self.workers:
                    now = time.time()
                    if 0 < launched < self.workers and now < last_launch + self.stagger:
                        break
                    task = queue[0]
                    if not self.guard.allows(task, now):
                        stopped = True
                        print(f"[runner] time guard: not starting {task['task_id']} or later tasks",
                              flush=True)
                        break
                    queue.pop(0)
                    self.not_started.remove(task)
                    future = pool.submit(task_lib.execute_task, task, self.context)
                    self.running[future] = (task, now)
                    launched, last_launch = launched + 1, now
                if not self.running and (not queue or stopped):
                    break
                if not self.running:
                    time.sleep(max(0.0, min(5.0, last_launch + self.stagger - time.time())))
                    continue
                timeout = 30.0
                if queue and not stopped and 0 < launched < self.workers:
                    timeout = max(0.1, min(timeout, last_launch + self.stagger - time.time()))
                finished, _ = wait(list(self.running), timeout=timeout, return_when=FIRST_COMPLETED)
                broken = False
                for future in finished:
                    task, started = self.running.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:  # a worker process died (for example out of memory)
                        broken = True
                        result = {"ok": False, "error": f"worker process failed: {exc!r}",
                                  "traceback": traceback.format_exc(), "started": started,
                                  "finished": time.time()}
                    self.record(task, result)
                if broken:
                    for future, (task, started) in list(self.running.items()):
                        self.record(task, {"ok": False, "error": "worker pool broke while running",
                                           "started": started, "finished": time.time()})
                    self.running.clear()
                    pool.shutdown(wait=False, cancel_futures=True)
                    pool = spawn_pool(self.workers)
                self.progress()
        finally:
            pool.shutdown(wait=True)


def summarize(out_dir):
    rows = []
    results = Path(out_dir) / "results.jsonl"
    if results.is_file():
        rows = [json.loads(line) for line in results.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_policy = defaultdict(list)
    for row in rows:
        if row["kind"] == "episode":
            by_policy[row["policy"]].append(row)
    return {"note": "reference only; statistics are recomputed locally (remote plan §5.4)",
            "episodes": {policy: {"count": len(items),
                                  "mean_of_mean_reward": sum(r["mean_reward"] for r in items) / len(items)}
                         for policy, items in sorted(by_policy.items())},
            "probes": [{k: r[k] for k in ("task_id", "processes", "max_step_seconds",
                                          "steps_per_second_total")}
                       for r in rows if r["kind"] == "probe"]}


def finalize(out, job, shard, mode, package, attempt, status, counts, started, extra):
    out.write_json("summary.json", summarize(out.directory))
    files = {}
    for path in sorted(out.directory.iterdir()):
        if path.is_file() and path.name != paths.MARKER and not path.name.endswith((".log", ".partial")) \
                and not path.name.startswith("._"):
            files[path.name] = sha256_file(path)
    marker = {"job_id": job["job_id"], "shard": shard, "attempt": attempt, "mode": mode,
              "status": status, "counts": counts, "package_sha256": package["package_sha256"],
              "code_sha256": package["code_sha256"], "job_sha256": package["job_sha256"],
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
              "finished_utc": utc_now(), "elapsed_seconds": time.time() - started,
              "files": files, **extra}
    out.write_json(paths.MARKER, marker)
    print(f"[runner] {job['job_id']} {shard} {attempt}: {status} {json.dumps(counts)}", flush=True)
    return marker


def main(job, shard, mode, package, environ=None):
    """Run one shard; returns the process exit code (0 complete or partial, 1 failed)."""
    environ = dict(os.environ if environ is None else environ)
    started = time.time()
    attempt = package.get("attempt", "a")
    tests = {name: environ[name] for name in TEST_VARIABLES if environ.get(name)}
    locations = paths.resolve_locations(job["job_id"], shard, environ)
    out = Output(locations["output_dir"])
    work_dir = Path(locations["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    variables = configure_environment(work_dir)
    model = cpu_model()
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        cpus = None
    environment = {"job_id": job["job_id"], "shard": shard, "attempt": attempt, "mode": mode,
                   "started_utc": utc_now(), "backend": locations["backend"], "cpu_model": model,
                   "logical_cpus": os.cpu_count(), "available_cpus": cpus, "memory": memory_record(),
                   "platform": platform.platform(), "python": platform.python_version(),
                   "packages": package_versions(), "expected_packages": job["expected_packages"],
                   "package_sha256": package["package_sha256"], "code_sha256": package["code_sha256"],
                   "job_sha256": package["job_sha256"],
                   "locations": {k: str(v) for k, v in locations.items()},
                   "test_overrides": tests, "kernel": package.get("kernel")}
    out.write_json("environment.json", environment)
    out.write_text("inputs-tree.txt", paths.inputs_tree(locations["input_root"]))
    for name in ("results.jsonl", "errors.jsonl"):
        out.touch(name)
    # Remote plan §5.2a item 4: a marker from the start, so a shard stopped by kaggle -t (which keeps
    # every file written before the stop, J0 s4) can still be found by its continuation shard.
    out.write_json(paths.MARKER, {"job_id": job["job_id"], "shard": shard, "attempt": attempt, "mode": mode,
                                  "status": "running", "counts": {}, "files": {},
                                  "package_sha256": package["package_sha256"],
                                  "code_sha256": package["code_sha256"], "job_sha256": package["job_sha256"],
                                  "started_utc": utc_now()})
    attempt_info = {"job_id": job["job_id"], "shard": shard, "attempt": attempt, "mode": mode,
                    "cpu_model": model}
    extra = {"test_overrides": tests, "cpu_model": model}
    planned = jobs.shard_tasks(job, shard, mode)
    if tests.get("SB_REMOTE_TEST_FAIL_TASK"):
        planned.append({"task_id": "test/injected-failure", "kind": "fail", "class": "fail/",
                        "formal": False, "steps": 0, "scenario": {}})
    counts = {"planned": len(planned), "skipped_previous": 0, "done": 0, "failed": 0,
              "not_started": len(planned)}
    try:
        previous, done_ids, problems = collect_previous(locations["input_root"], job["job_id"], shard,
                                                        package.get("inputs"))
        extra["inputs"] = previous
        if problems:
            raise RuntimeError(f"earlier outputs fail their SHA-256 checks: {problems}")
        competition_root = Path(locations["competition_root"])
        overlay_root, link_methods = scenarios.build_overlay(competition_root, work_dir)
        extra["overlay"] = {"root": str(overlay_root), "links": link_methods}
        code_root = Path(package["code_root"])
        sys_path = [str(code_root / "packages"), str(competition_root)]
        for path in reversed(sys_path):
            if path not in sys.path:
                sys.path.insert(0, path)
        context = {"previous_dirs": [entry["path"] for entry in previous if not entry.get("source")],
                   "sources": [entry for entry in previous if entry.get("source")],
                   "scenario_root": str(overlay_root), "code_root": str(code_root),
                   "competition_root": str(competition_root), "sys_path": sys_path,
                   "environ": variables, "cpu_model": model,
                   "policy_sources": job.get("policy_sources", {})}
        generated = [t for t in planned if t.get("scenario", {}).get("set") == "generated"]
        # dagger tasks list their rollout scenarios in config["scenarios"] (stage45 §2: TRAIN, purpose train)
        generated += [{"scenario": spec, "purpose": purpose} for t in planned if t["kind"] == "dagger"
                      for key, purpose in (("rollout_scenarios", "train"), ("eval_scenarios", "tune"))
                      for spec in t.get("config", {}).get(key, []) if spec.get("set") == "generated"]
        if generated:
            from smart_building.scenarios.factory import materialize_for_tasks
            extra["generated"] = materialize_for_tasks(code_root, overlay_root, generated)
    except Exception as exc:
        out.write_json("precheck.json", {"ok": False, "stage": "inputs and scenario preparation",
                                         "error": repr(exc), "traceback": traceback.format_exc()})
        finalize(out, job, shard, mode, package, attempt, "failed", counts, started, extra)
        return 1
    remaining = [t for t in planned if t["task_id"] not in done_ids]
    counts["skipped_previous"] = len(planned) - len(remaining)
    ok, checks = precheck(planned, context)
    out.write_json("precheck.json", {"ok": ok, "steps": PRECHECK_STEPS, "checks": checks})
    if not ok:
        counts["not_started"] = len(remaining)
        finalize(out, job, shard, mode, package, attempt, "failed", counts, started, extra)
        return 1
    out.write_json("parity.json", parity_probe(job, context))
    budget = float(tests.get("SB_REMOTE_TEST_BUDGET_SECONDS") or jobs.setting(job, shard, "budget_seconds", mode))
    first_end = job.get("first_segment_end_utc")
    if first_end and attempt == "a" and mode == "full":
        # Reviewer 2026-10-18: a first segment ends by a fixed time, so the local pipeline relays it
        # before the computer stops; later segments keep the full budget.
        import datetime as dt
        end = dt.datetime.fromisoformat(first_end.replace("Z", "+00:00")).timestamp()
        budget = min(budget, max(0.0, end - started))
        extra["first_segment_end"] = {"utc": first_end, "budget_seconds": budget}
    first_seconds = job["shards"][shard].get("first_segment_seconds")
    if first_seconds and attempt == "a" and mode == "full":
        # A short first segment of one chain (J9: a GPU-first fold's first segment fits the week's GPU hours, then the
        # chain continues from it); later segments keep the full budget.
        budget = min(budget, float(first_seconds))
        extra["first_segment"] = {"seconds": float(first_seconds), "budget_seconds": budget}
    windows = job.get("segment_end_windows_utc")
    if windows and mode == "full" and any(t["kind"] == "dagger" for t in remaining) \
            and not environ.get("SB_REMOTE_LOCAL_DEADLINE"):
        # Chain segments end while the relaying computer is on (user 2026-10-20); see window_budget.
        budget, extra["segment_end_window"] = window_budget(
            started, budget, windows, float(job.get("segment_min_seconds", SEGMENT_MIN_SECONDS)))
    workers = int(jobs.setting(job, shard, "workers", mode))
    if tests.get("SB_REMOTE_TEST_MAX_PROCESSES"):
        cap = int(tests["SB_REMOTE_TEST_MAX_PROCESSES"])
        workers = min(workers, cap)
        for task in remaining:
            if task["kind"] == "probe":
                task["processes"] = min(int(task["processes"]), cap)
    # Local backend (scripts/remote.py run-local; remote plan §5.2a as revised 2026-10-17): the worker
    # count and the deadline come from the launcher and are recorded, like Kaggle's machine settings.
    variables_now = environ if environ is not None else os.environ
    if variables_now.get("SB_REMOTE_LOCAL_WORKERS") or variables_now.get("SB_REMOTE_LOCAL_DEADLINE"):
        local = {}
        if variables_now.get("SB_REMOTE_LOCAL_WORKERS"):
            workers = int(variables_now["SB_REMOTE_LOCAL_WORKERS"])
            local["workers"] = workers
        if variables_now.get("SB_REMOTE_LOCAL_DEADLINE"):
            deadline = float(variables_now["SB_REMOTE_LOCAL_DEADLINE"])
            budget = min(budget, max(0.0, deadline - started - GUARD_MARGIN * 300))
            local["deadline_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(deadline))
        local["budget_seconds"] = budget
        extra["local_backend"] = local
    stagger = float(jobs.setting(job, shard, "stagger_seconds", mode))
    guard = TimeGuard(job, started, budget)
    loop = TaskLoop(job, shard, mode, context, out, guard, workers, stagger, attempt_info)
    extra.update(budget_seconds=budget, workers=workers, stagger_seconds=stagger)
    try:
        loop.run(remaining)
    except Exception as exc:
        out.write_json("fatal.json", {"error": repr(exc), "traceback": traceback.format_exc()})
        counts.update(done=len(loop.done), failed=len(loop.failed),
                      not_started=len(loop.not_started) + len(loop.running))
        finalize(out, job, shard, mode, package, attempt, "failed", counts, started, extra)
        return 1
    counts.update(done=len(loop.done), failed=len(loop.failed), not_started=len(loop.not_started))
    status = "partial" if loop.not_started else "complete"
    finalize(out, job, shard, mode, package, attempt, status, counts, started, extra)
    return 0
