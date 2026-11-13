"""Job definitions (remote plan §3.2): checks, shard task lists and smoke mode.

A job definition is configs/remote/<job_id>.json. Its SHA-256 (of the file
bytes) is the "job definition" half of the work package; the code snapshot is
the other half. Shard settings (workers, budget_seconds, push_timeout,
stagger_seconds) are job-wide with optional per-shard overrides, each keyed by
mode ("smoke" or "full"). These execution settings only decide when a shard
stops and continues; they never change a result (remote plan, header).
"""
import copy
import json
import math
from pathlib import Path

KNOWN_KINDS = {"episode", "probe", "fail", "cmaes", "dagger", "latency"}
# smart_buildings.evaluate.POLICY_MAP keys (starter kit evaluate.py:91-100); any other policy
# comes from the job's policy_sources (file and class inside the code snapshot).
OFFICIAL_POLICIES = {"transfer", "bang_bang", "fixed_setpoint", "pid", "pid_water", "all_off"}
# remote plan §4 DR4: precheck + probe + two tasks of each kind x 288 steps.
SMOKE_TASKS_PER_KIND = 2
SMOKE_STEPS = 288
MODES = ("smoke", "full")
# remote plan §3.2: full 39,600 / 41,400 s; smoke 1,200 / 1,800 s.
DEFAULT_SETTINGS = {"budget_seconds": {"full": 39600, "smoke": 1200},
                    "push_timeout": {"full": 41400, "smoke": 1800},
                    "stagger_seconds": {"full": 30, "smoke": 30}}
# validation-v1 §2, D-E07: LOCK buildings and minneapolis only in reviewer-built lockbox jobs.
LOCKBOX_POOLS = {"lock"}
LOCKBOX_STATIONS = {"minneapolis"}


def job_file(root, job_id):
    return Path(root) / "configs/remote" / f"{job_id}.json"


def dump_job(job):
    return (json.dumps(job, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def load_job(root, job_id):
    """(job dict, file bytes) of configs/remote/<job_id>.json."""
    data = job_file(root, job_id).read_bytes()
    return json.loads(data.decode("utf-8")), data


def station_of(weather):
    """'atlanta_3' -> 'atlanta'."""
    return weather.rsplit("_", 1)[0]


def task_uses_lockbox(task):
    spec = task.get("scenario") or {}
    if spec.get("set") == "generated" and spec.get("pool") in LOCKBOX_POOLS:
        return True
    name = spec.get("weather") or spec.get("name", "").split("/")[-1]
    return bool(name) and station_of(name) in LOCKBOX_STATIONS


def jobs_task_uses_lockbox(task):
    return task_uses_lockbox(task)


def is_lockbox_job(job):
    return (job.get("lockbox") or {}).get("approved_by") == "reviewer"


def check_job(job):
    """List of problems; empty when the job definition is usable."""
    problems = []
    for key in ("job_id", "stage", "registry_ids", "backend", "kaggle_user", "code_paths",
                "tasks", "shards", "workers", "parity_probe", "expected_packages"):
        if key not in job:
            problems.append(f"missing field {key}")
    if problems:
        return problems
    ids = [t["task_id"] for t in job["tasks"]]
    if len(ids) != len(set(ids)):
        problems.append("task ids are not unique")
    known = set(ids)
    seen = []
    for shard, entry in job["shards"].items():
        if not shard.replace("-", "").isalnum():
            problems.append(f"shard name {shard!r} must be alphanumeric")
        for task_id in entry["tasks"]:
            if task_id not in known:
                problems.append(f"shard {shard} lists unknown task {task_id}")
            seen.append(task_id)
    if sorted(seen) != sorted(ids):
        problems.append("every task must be in exactly one shard")
    sources = job.get("policy_sources", {})
    for task in job["tasks"]:
        policy = task.get("policy")
        if "policy_kwargs" in task and not isinstance(task["policy_kwargs"], dict):
            problems.append(f"{task['task_id']}: policy_kwargs must be a JSON object")
        if task.get("kind") == "dagger" and not (task.get("entry") or {}).get("function"):
            problems.append(f"{task['task_id']}: a dagger task needs an entry (file, function)")
        if task.get("kind") == "cmaes":
            if not task.get("policy_factory") and not task.get("test_objective"):
                problems.append(f"{task['task_id']}: a cmaes task needs a policy_factory")
            if len(task.get("parameters", [])) < 2:
                problems.append(f"{task['task_id']}: a cmaes task needs at least 2 parameters "
                                "(pycma 4.5.0 fails on one-parameter problems)")
            for spec in task.get("evaluation", []):
                if jobs_task_uses_lockbox({"scenario": spec}) and not is_lockbox_job(job):
                    problems.append(f"{task['task_id']}: LOCK pool or minneapolis in the evaluation set")
        elif task.get("student") is not None:
            if not isinstance(task["student"], dict) or not {"job", "shard"} <= set(task["student"]):
                problems.append(f"{task['task_id']}: student must name the chain's job and shard")
        elif task.get("rl") is not None:   # J9: a PPO agent of a finished chain (remote/tasks.py run_rl_task)
            rl = task["rl"]
            chain = rl.get("chain") if isinstance(rl, dict) else None
            select = rl.get("select") if isinstance(rl, dict) else None
            if not isinstance(chain, dict) or not {"job", "shard"} <= set(chain) \
                    or ("snapshot" in rl) == bool(select) \
                    or (select and not (select.get("job") and (select.get("shard") or select.get("shards"))
                                        and select.get("episodes"))) \
                    or task.get("kind") != "episode":
                problems.append(f"{task['task_id']}: rl must be an episode naming the chain (job, shard) and either a "
                                "snapshot target or the INNER selection (job, shards, episodes)")
        elif policy and task.get("kind") != "dagger" and policy not in OFFICIAL_POLICIES and policy not in sources:
            problems.append(f"{task['task_id']}: policy {policy} is neither official nor in policy_sources")
        if task.get("kind") not in KNOWN_KINDS:
            problems.append(f"{task['task_id']}: unknown kind {task.get('kind')!r}")
        if task.get("kind") == "fail":
            problems.append(f"{task['task_id']}: 'fail' tasks are only injected by the dry run")
        if task_uses_lockbox(task) and not is_lockbox_job(job):
            problems.append(f"{task['task_id']}: LOCK pool or minneapolis outside a reviewer "
                            "lockbox job (validation-v1 §2, D-E07)")
    return problems


def setting(job, shard, key, mode):
    """A shard setting: per-shard override, else job-wide, else the plan default."""
    for source in (job["shards"][shard], job, DEFAULT_SETTINGS):
        value = source.get(key)
        if value is None:
            continue
        return value[mode] if isinstance(value, dict) else value
    raise KeyError(key)


def shard_tasks(job, shard, mode):
    """The tasks a shard runs in this mode, in shard order (copies)."""
    by_id = {t["task_id"]: t for t in job["tasks"]}
    tasks = [copy.deepcopy(by_id[task_id]) for task_id in job["shards"][shard]["tasks"]]
    for task in tasks:
        task.setdefault("formal", task["kind"] == "episode" and task.get("steps") == 8640)
        task.setdefault("class", f"{task['kind']}/{task.get('policy', '')}")
    if mode == "full":
        return tasks
    kept, counts = [], {}
    for task in tasks:
        counts[task["kind"]] = counts.get(task["kind"], 0) + 1
        if counts[task["kind"]] > SMOKE_TASKS_PER_KIND:
            continue
        task["full_steps"] = task["steps"]
        task["steps"] = min(task["steps"], SMOKE_STEPS)
        task["formal"] = False
        kept.append(task)
    return kept


def estimate_seconds(job, task):
    """Planned duration for the time guard before any task of the class has finished."""
    if task["kind"] == "fail":
        return 1.0  # dry-run test task: raises at once
    estimates = job.get("estimates", {})
    full = estimates.get(task["class"], estimates.get(task["kind"], 3600.0))
    full_steps = task.get("full_steps", task["steps"])
    if task["steps"] >= full_steps:
        return float(full)
    return 60.0 + float(full) * task["steps"] / full_steps


def p90(values):
    """Nearest-rank 90th percentile."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)]
