"""What one remote task does (remote plan §3.1).

Task kinds:
- "episode": one policy x one scenario x steps, on the scoring path shared with
  scripts/batch_evaluate.py (smart_building.evaluation.episode.run_episode).
- "probe": a throughput probe; `processes` copies of one policy x scenario run
  side by side and the per-step time of the slowest is reported, the method of
  scripts/benchmark_throughput.py (notes/00 §H3). It runs alone, in the runner
  process, because it starts its own processes.
- "fail": always raises; only the local dry run injects it, to test that one
  failed task does not stop the others (remote plan §4 DR2 (c)).
Training kinds (DAgger, PPO, SAC) are added with their work packages.

Worker processes are started with "spawn" (remote plan §3.4 step 0), so
everything a worker needs travels in the `context` dict.
"""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
import sys
import time
import traceback


def prepare_worker(context):
    """Environment, import path and working directory of a task process."""
    os.environ.update(context["environ"])
    sys.dont_write_bytecode = True
    for path in reversed(context["sys_path"]):
        if path not in sys.path:
            sys.path.insert(0, path)
    os.chdir(context["scenario_root"])


def load_policy_class(path, class_name, key):
    """A policy class from its source file, without importing smart_building.policies.

    The job names the file and class (job["policy_sources"]), as the submission
    notebook embeds one module's source; the package registry, which other work
    edits, never has to be part of the code snapshot.
    """
    import importlib.util
    name = f"sb_remote_policy_{key}"
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return getattr(module, class_name)


def load_entry(context, source):
    """A function from a file of the code snapshot (job task "entry": {file, function})."""
    import importlib.util
    name = "sb_remote_entry_" + Path(source["file"]).stem
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, Path(context["code_root"]) / source["file"])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return getattr(module, source["function"])


def load_factory(context, source):
    """A policy factory function(params) -> policy class, loaded from its file like load_policy_class."""
    import importlib.util
    name = "sb_remote_factory_" + Path(source["file"]).stem
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, Path(context["code_root"]) / source["file"])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return getattr(module, source["function"])


def policy_map(context):
    """Official POLICY_MAP plus the job's own policies (same keys as scripts/batch_evaluate.py)."""
    from smart_buildings import evaluate
    policies = dict(evaluate.POLICY_MAP)
    for key, source in context.get("policy_sources", {}).items():
        policies[key] = load_policy_class(Path(context["code_root"]) / source["file"], source["class"], key)
    return policies


def scenario_dir(spec, context):
    root = Path(context["scenario_root"])
    if spec["set"] == "official":
        from smart_building.remote.scenarios import official_scenario_dir
        return official_scenario_dir(root, spec["name"])
    if spec["set"] == "generated":
        from smart_building.scenarios.factory import generated_scenario_dir, load_split
        # The folder follows the split file's "version" (v1-inner.json registers INNER under split-v1/inner).
        version = spec["split"]
        if context.get("code_root"):
            version = load_split(context["code_root"], spec["split"]).get("version", version)
        return generated_scenario_dir(root, version, spec["pool"], spec["building"], spec["weather"])
    raise ValueError(f"unknown scenario set {spec['set']!r}")


def scenario_label(spec):
    if spec["set"] == "official":
        return spec["name"]
    return f"{spec['building']}/{spec['weather']}"


def run_episode_task(task, context):
    """task["path"]: "tracker" (starter kit evaluator; S1 reproduction, parity probes) or "gateway"
    (Kaggle gateway loop; selection and S4 from J1b on, reviewer decision 2026-10-16)."""
    if task.get("student"):
        return run_student_task(task, context)
    if task.get("rl"):
        return run_rl_task(task, context)
    spec = task["scenario"]
    if task.get("policy_params") is not None:
        # A CMA-ES candidate: the job's factory builds the policy class for these parameter values.
        policy_class = load_factory(context, task["policy_factory"])(task["policy_params"])
    else:
        policy_class = policy_map(context)[task["policy"]]
    if task.get("policy_kwargs"):
        policy_class = with_kwargs(policy_class, task["policy_kwargs"])
    if task.get("path", "tracker") == "gateway":
        from smart_building.evaluation.gateway_episode import run_gateway_episode
        row = run_gateway_episode(scenario_dir(spec, context), policy_class, policy_key=task["policy"],
                                  label=scenario_label(spec), max_steps=task["steps"])
    else:
        from smart_building.evaluation.episode import run_episode
        row = run_episode(scenario_dir(spec, context), policy_class, task["steps"],
                          policy_key=task["policy"], label=scenario_label(spec))
        row["path"] = "tracker"
    if task.get("policy_kwargs"):
        row["policy_kwargs"] = task["policy_kwargs"]
    return row


def run_student_task(task, context):
    """J7c (stage45 §2, D-L11): the student of a finished DAgger chain on one episode, as S4 scores it.
    task["student"] = {"job", "shard"} names the chain; its last segment is mounted as a source input
    (checked against the build's file hashes). The model is the one its chain.json result names (the
    best checkpoint by the registered rule in learned/dagger.py: lowest mean INNER cost, ties to the
    earlier iteration), checked against its SHA-256; the episode is learned/rollouts.py run_evaluation
    (student mean action, the chain's teacher for on/off), unchanged."""
    from smart_building.learned import rollouts
    wanted = task["student"]
    model, result, attempt = mounted_student(wanted, context)
    spec = task["scenario"]
    # A chain whose result names its own evaluation (j7b-v5, learned/dagger_v5.py: the student reads the
    # B1-R state) is scored by that function; earlier chains name none and keep rollouts.run_evaluation.
    evaluation = load_entry(context, result["evaluation_entry"]) if result.get("evaluation_entry") else rollouts.run_evaluation
    row = evaluation({"scenario": spec, "label": scenario_label(spec),
                                   "scenario_dir": str(scenario_dir(spec, context)),
                                   "scenario_root": context["scenario_root"], "code_root": context["code_root"],
                                   "teacher": result["teacher"], "model_path": str(model),
                                   "iteration": result["best_iteration"], "steps_cap": task["steps"], "threads": 1})
    row.update(policy=task["policy"], student={"job": wanted["job"], "shard": wanted["shard"],
                                               "attempt": attempt, "model": result["model"],
                                               "best_iteration": result["best_iteration"],
                                               "mean_cost_by_iteration": result.get("mean_cost_by_iteration"),
                                               "teacher": result["teacher"]})
    return row


def mounted_student(wanted, context):
    """(model path, chain.json result, attempt) of the finished chain {"job", "shard"} mounted as a source;
    the model file is checked against the SHA-256 its chain.json result records."""
    import hashlib
    import json
    matches = [s for s in context.get("sources", []) if s["job_id"] == wanted["job"] and s["shard"] == wanted["shard"]]
    if len(matches) != 1:
        raise RuntimeError(f"{wanted}: {len(matches)} mounted chain outputs (expected 1)")
    root = Path(matches[0]["path"])
    result = json.loads((root / "chain.json").read_text(encoding="utf-8"))["result"]
    if not result.get("finished") or not result.get("model"):
        raise RuntimeError(f"{wanted}: the chain has not finished ({result.get('stopped')})")
    model = root / "dagger" / result["model"]["file"]
    if hashlib.sha256(model.read_bytes()).hexdigest() != result["model"]["sha256"]:
        raise RuntimeError(f"{model}: SHA-256 differs from chain.json")
    return model, result, matches[0]["attempt"]


RL_POLICY_ENTRY = {"file": "packages/smart_building/learned/ppo.py", "function": "policy_class"}


def finished_chain(wanted, context):
    """(output folder, chain.json, attempt) of the finished chain {"job", "shard"} mounted as a source: a Kaggle chain's
    last segment, or a chain run on this computer by scripts/local_chain.py (checked by the runner against the build)."""
    import json
    matches = [s for s in context.get("sources", []) if s["job_id"] == wanted["job"] and s["shard"] == wanted["shard"]]
    if len(matches) != 1:
        raise RuntimeError(f"{wanted}: {len(matches)} mounted chain outputs (expected 1)")
    root = Path(matches[0]["path"])
    chain = json.loads((root / "chain.json").read_text(encoding="utf-8"))
    if not chain.get("finished") or not (chain.get("result") or {}).get("finished"):
        raise RuntimeError(f"{wanted}: the chain has not finished")
    return root, chain, matches[0]["attempt"]


def checkpoint_selection(select, chain, context):
    """B1 stage 3 checkpoint choice (plan-direction-20261020 §13; the students' rule, stage45 §2): the agent snapshot with
    the lowest mean cost over the fold's INNER episodes, ties to the earlier target. select = {"job", "shards" (or
    "shard"), "episodes"} names the mounted INNER evaluation of this chain, one output per shard; every snapshot target
    needs exactly `episodes` rows over those outputs, each made with the snapshot file this chain records (row["rl"]),
    and no task twice. Returns (target, {target: mean cost}, rows used)."""
    import json
    rows, seen = [], set()
    for shard in select.get("shards") or [select["shard"]]:
        matches = [s for s in context.get("sources", []) if s["job_id"] == select["job"] and s["shard"] == shard]
        if len(matches) != 1:
            raise RuntimeError(f"{select['job']} {shard}: {len(matches)} mounted INNER outputs (expected 1)")
        for line in (Path(matches[0]["path"]) / "results.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row["task_id"] in seen:
                    raise RuntimeError(f"INNER task {row['task_id']} appears in more than one mounted output")
                seen.add(row["task_id"])
                rows.append(row)
    snapshots = {int(t): record for record in chain["result"]["agent_snapshots"] for t in record["targets"]}
    costs = {}
    for row in rows:
        agent = row.get("rl") or {}
        if row.get("kind") != "episode" or "target" not in agent:
            continue
        target = int(agent["target"])
        if target not in snapshots or agent.get("sha256") != snapshots[target]["sha256"]:
            raise RuntimeError(f"INNER row {row['task_id']} was not made with this chain's snapshot {target}")
        costs.setdefault(target, []).append(-float(row["mean_reward"]))
    missing = {t: len(costs.get(t, [])) for t in snapshots if len(costs.get(t, [])) != int(select["episodes"])}
    if missing:
        raise RuntimeError(f"INNER rows per snapshot target differ from {select['episodes']}: {missing}")
    means = {t: sum(v) / len(v) for t, v in costs.items()}
    chosen = min(sorted(means), key=lambda t: (means[t], t))
    return chosen, means, sum(len(v) for v in costs.values())


def run_rl_task(task, context):
    """J9 evaluation (plan-direction-20261020 §13: B1 stage 3, PPO started from the fold's B1-L v5 student): one gateway
    episode of a PPO agent of a finished chain mounted as a source, acting with its mean action through the agent's own
    competition policy. task["rl"] = {"chain": {"job", "shard"}, "policy_entry" (optional; default learned/ppo.py
    policy_class(path, base_class)), and either "snapshot": the agent snapshot target (the fold's INNER checkpoint
    evaluation) or "select": the mounted INNER evaluation that picks it (S4: lowest mean INNER cost, ties to the
    earlier; checkpoint_selection)}. The base controller class (on/off, fallback) is the one the agent file names
    (learned/ppo.py load_agent settings["rule_controller"]). The agent file is checked against the SHA-256 its chain
    recorded; the row records the agent, the selection, fallbacks and the per-step policy time."""
    import hashlib
    import time
    from smart_building.evaluation.gateway_episode import run_gateway_episode
    from smart_building.learned import ppo, ppo_env
    spec = task["rl"]
    root, chain, attempt = finished_chain(spec["chain"], context)
    result = chain["result"]
    selection = None
    if spec.get("select"):
        target, means, used = checkpoint_selection(spec["select"], chain, context)
        selection = {"rule": "lowest mean INNER cost, ties to the earlier", "means": means, "rows": used,
                     "inner": spec["select"]}
    else:
        target = int(spec["snapshot"])
    records = [r for r in result["agent_snapshots"] if target in [int(t) for t in r["targets"]]]
    if len(records) != 1:
        raise RuntimeError(f"{spec['chain']}: {len(records)} agent snapshots for target {target}")
    record = records[0]
    path = root / "dagger" / record["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise RuntimeError(f"{path}: SHA-256 differs from chain.json")
    _, settings, _ = ppo.load_agent(path)
    base_class = ppo_env.load_rule_controller(context["code_root"], settings["rule_controller"])
    policy_class = load_entry(context, spec.get("policy_entry") or RL_POLICY_ENTRY)(path, base_class)
    built = []

    class Timed(policy_class):
        def __init__(self, building_meta_data, action_spec):
            super().__init__(building_meta_data, action_spec)
            self.durations = []
            built.append(self)

        def get_action(self, obs):
            started = time.perf_counter()
            action = super().get_action(obs)
            self.durations.append(time.perf_counter() - started)
            return action

    scenario = task["scenario"]
    row = run_gateway_episode(scenario_dir(scenario, context), Timed, policy_key=task["policy"],
                              label=scenario_label(scenario), max_steps=task["steps"])
    policy = built[-1]
    durations = sorted(1000.0 * d for d in policy.durations)
    row.update(rl={"chain": spec["chain"], "attempt": attempt, "target": target, "iteration": record["iteration"],
                   "global_step": record["global_step"], "file": record["file"], "sha256": record["sha256"],
                   "selection": selection, "rule_controller": settings.get("rule_controller"),
                   "anchor": settings.get("anchor"), "switches": settings.get("switches")},
               fallback_steps=getattr(policy, "fallback_steps", None),
               get_action_ms={"mean": sum(durations) / len(durations), "p99": durations[int(0.99 * (len(durations) - 1))],
                              "max": durations[-1]} if durations else None)
    return row


def run_latency_task(task, context):
    """D-K04 runtime estimate (reviewer 2026-10-18): per-step time of the B1-L fold ensemble
    (task["copies"] copies of a mounted chain's student; policies/b1l_ensemble.py; one torch thread)
    against the chain's teacher alone, on the gateway loop of one scenario for task["steps"] steps;
    environment step times are recorded too. The weights do not change the cost, only the network."""
    import time
    import numpy as np
    import torch
    from smart_building.learned import rollouts
    from smart_building.policies.b1l_ensemble import ensemble_policy_class
    torch.set_num_threads(1)
    model, result, attempt = mounted_student(task["student"], context)
    if result.get("ensemble_entry"):     # j7b-v5 students (policies/b1l_ensemble_v5.py), as in run_student_task
        ensemble_policy_class = load_entry(context, result["ensemble_entry"])
    teacher_class = rollouts.load_class(context["code_root"], result["teacher"])
    spec = task["scenario"]
    env, metadata = rollouts.load_environment(scenario_dir(spec, context), context["scenario_root"])
    action_spec = env.action_spec()
    copies = int(task["copies"])
    measured = {}
    for label, build in (("teacher", lambda: teacher_class(metadata, action_spec)),
                         (f"ensemble{copies}", lambda: ensemble_policy_class([model] * copies, teacher_class)(metadata, action_spec))):
        observation, _ = env.reset()
        started = time.perf_counter()
        policy = build()
        construct = time.perf_counter() - started
        act, sim = [], []
        for _ in range(int(task["steps"])):
            t0 = time.perf_counter()
            action = policy.get_action(observation)
            t1 = time.perf_counter()
            observation, _, terminated, truncated, _ = env.step(action)
            act.append(t1 - t0)
            sim.append(time.perf_counter() - t1)
            if terminated or truncated:
                break
        a, s = np.asarray(act[10:]) * 1000.0, np.asarray(sim[10:]) * 1000.0
        measured[label] = {"steps": len(act), "construct_s": construct, "get_action_ms_mean": float(a.mean()),
                           "get_action_ms_p99": float(np.percentile(a, 99)), "get_action_ms_max": float(a.max()),
                           "env_step_ms_mean": float(s.mean()), "fallback_steps": getattr(policy, "fallback_steps", None)}
    env.close()
    return {"scenario": scenario_label(spec), "policy": task["policy"], "steps": int(task["steps"]),
            "copies": copies, "torch_threads": torch.get_num_threads(), "latency": measured,
            "student": {"job": task["student"]["job"], "shard": task["student"]["shard"], "attempt": attempt,
                        "model": result["model"], "teacher": result["teacher"]}}


def with_kwargs(policy_class, kwargs):
    """The policy class with constructor keyword arguments from the job (task "policy_kwargs"), e.g.
    G36PolicyC3Configured(overrides=...) for the per-fold WP3 combinations; recorded in the result row."""
    class Configured(policy_class):
        def __init__(self, building_meta_data, action_spec):
            super().__init__(building_meta_data, action_spec, **kwargs)
    Configured.__name__ = policy_class.__name__
    Configured.__qualname__ = policy_class.__qualname__
    return Configured


def throughput_worker(context, spec, policy_key, steps):
    """One probe process: load, then time `steps` steps (as scripts/benchmark_throughput.py)."""
    prepare_worker(context)
    from smart_buildings import evaluate
    start = time.perf_counter()
    directory = scenario_dir(spec, context)
    env = evaluate.load_hybrid_environment(directory / "config.gin")
    env.metrics_path = None
    env._occupancy_normalization_constant = 125.0
    metadata = evaluate.create_building_meta_data(env, directory)
    action_spec = env.action_spec()
    policy = evaluate.initialize_policy(policy_map(context)[policy_key], metadata, action_spec)
    obs, _ = env.reset()
    loaded = time.perf_counter()
    for _ in range(steps):
        obs, *_ = env.step(evaluate.validate_action(policy.get_action(obs), action_spec))
    done = time.perf_counter()
    return {"load_seconds": loaded - start, "step_seconds": (done - loaded) / steps}


def run_probe_task(task, context):
    """Throughput with `processes` simultaneous copies; run in the runner process."""
    n = int(task["processes"])
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n, mp_context=multiprocessing.get_context("spawn")) as pool:
        rows = list(pool.map(throughput_worker, [context] * n, [task["scenario"]] * n,
                             [task["policy"]] * n, [task["steps"]] * n))
    wall = time.perf_counter() - start
    slowest = max(r["step_seconds"] for r in rows)
    return {"scenario": scenario_label(task["scenario"]), "policy": task["policy"],
            "steps": task["steps"], "processes": n, "wall_seconds": wall,
            "max_step_seconds": slowest,
            "mean_step_seconds": sum(r["step_seconds"] for r in rows) / n,
            "steps_per_second_total": n / slowest,
            "mean_load_seconds": sum(r["load_seconds"] for r in rows) / n,
            "per_process": rows}


def execute_task(task, context):
    """Pool entry point: never raises; returns the row or the error with its traceback."""
    started = time.time()
    try:
        prepare_worker(context)
        if task["kind"] == "episode":
            row = run_episode_task(task, context)
        elif task["kind"] == "latency":
            row = run_latency_task(task, context)
        elif task["kind"] == "fail":
            raise RuntimeError(f"injected failure for the dry-run test ({task['task_id']})")
        else:
            raise ValueError(f"task kind {task['kind']!r} cannot run in a worker pool")
        return {"ok": True, "row": row, "started": started, "finished": time.time()}
    except Exception as exc:  # recorded per task; the shard goes on (remote plan §3.4 step 6)
        return {"ok": False, "error": repr(exc), "traceback": traceback.format_exc(),
                "started": started, "finished": time.time()}
