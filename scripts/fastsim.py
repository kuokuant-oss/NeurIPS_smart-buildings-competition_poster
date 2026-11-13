"""Fast simulator command line (docs/plan/fastsim.md): an opt-in local substitute for the official simulator in
training and validation. Submissions never use it.

  fastsim.py episodes --mode rom|exact|official --policies P1,P2 (--set NAME | --specs S1 S2 ... | --pool P)
                      [--steps N] [--workers W]
      gateway episodes (the loop of .tmp/reviewer/rules/diagnose.py), one per policy and scenario, written to
      data/runs/fastsim/episodes/<mode>/<policy>__<building>__<weather>.npz (+ .json); finished ones are skipped.
  fastsim.py report --mode rom --policies P1,P2 (--set NAME | --specs ...) [--pairs A:B,...] --out <file>.json
      fast versus official per scenario (cost and its comfort, energy and carbon parts) and the paired
      differences of policy pairs: same sign, better/worse counts. Official rows: .tmp/reviewer/rules/diag, then
      data/runs/fastsim/episodes/official.
  fastsim.py build (--set NAME | --specs ... | --pool train|inner|val) [--workers W]
      the reduced models of the scenarios into the cache (data/runs/fastsim/rom-cache), one process each
      (about 1 minute and 1-3 GB per scenario; cached ones are skipped).
  fastsim.py cache [--groups train inner s4_first s4_rest] [--workers W] [--reserve-gb G] [--threads T]
      the same for whole groups with a memory-aware scheduler (one fresh process per scenario; progress in
      data/runs/fastsim/cache-status.json and logs/cache.log; rom-cache/index/ records each scenario's model).
  fastsim.py bench --mode rom|exact|official --spec S [--steps N] [--procs P]
      ms per 5-minute step (after setup) of P concurrent processes on one scenario with the final rule.
  fastsim.py run --mode rom|exact -- <command> [args ...]
      runs a command with SB_FASTSIM set and a sitecustomize hook on PYTHONPATH, so the command and every
      process it starts (spawn workers included) build their simulators on the fast simulator. Write its outputs
      under data/runs/fastsim/: rows from the fast simulator are not official rows.

Policies are rule-screen names of .tmp/reviewer/rules/diagnose.py (a1_cold3+a1_econ, a1_cold3, v4_next120, ...)
or POLICIES names (G36PolicyB2R2). Sets are those of configs/b2rl/eval-sets.json. Start with the host python3 or
the project .venv; it re-executes itself in the .venv with the repository-local caches of scripts/workspace.py.
"""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"
PYTHON = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _environment():
    sys.path.insert(0, str(ROOT / "scripts"))
    from workspace import local_environment  # noqa: E402
    return local_environment()


if __name__ == "__main__" and (Path(sys.prefix).resolve() != VENV.resolve() or "NUMBA_CACHE_DIR" not in os.environ):
    environment = _environment()
    command = [str(PYTHON), "-B", str(Path(__file__).resolve()), *sys.argv[1:]]
    raise SystemExit(subprocess.run(command, cwd=ROOT, env=environment).returncode)

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing  # noqa: E402
import platform  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402

sys.path[:0] = [str(ROOT / "packages")]
EVAL_SETS = ROOT / "configs/b2rl/eval-sets.json"
OUT = ROOT / "data/runs/fastsim"
DIAG = ROOT / ".tmp/reviewer/rules/diag"
DIAGNOSE = ROOT / ".tmp/reviewer/rules/diagnose.py"
FINAL_RULE = "a1_cold3+a1_econ"   # G36PolicyB2R2 since 2026-10-27 21:03
# paired differences within 1e-6 relative are the same result: the official rows come from three machines (Mac,
# this host, Kaggle), whose floating point differs by about that much (handoff/RUNBOOK-windows-b2rl.md §2.5.6)
TIE = 1e-6
HOOK = ROOT / "packages/smart_building/fastsim/hook"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=1) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def specs_of(args):
    if getattr(args, "specs", None):
        return list(args.specs)
    if getattr(args, "pool", None):
        from smart_building.learned.b2rl_env import SCENARIOS, TRAIN_POOL, scenario_dirs
        folder = TRAIN_POOL if args.pool == "train" else SCENARIOS / "split-v1" / args.pool
        return [f"{args.pool}/{Path(d).parent.name}/{Path(d).name}" for d in scenario_dirs(folder)]
    sets = json.loads(EVAL_SETS.read_text(encoding="utf-8"))
    return list(sets[args.set])


def resolve_policy(name):
    """The policy class of a rule-screen name, exactly as diagnose.py builds it (read from its file, not changed)."""
    if DIAGNOSE.is_file():
        spec = importlib.util.spec_from_file_location("_fastsim_diagnose", DIAGNOSE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.resolve(name)
    from smart_building.learned.b2rl_env import rule_policy_class
    return rule_policy_class(name)


def episode_path(mode, policy, spec, steps=None):
    """Full episodes under episodes/<mode>/; prefixes of N steps under episodes/<mode>-s<N>/."""
    _, building, weather = spec.split("/")
    return OUT / "episodes" / (f"{mode}-s{steps}" if steps else mode) / f"{policy}__{building}__{weather}.npz"


def _init_worker(mode):
    sys.dont_write_bytecode = True
    from smart_building.learned.b2rl_env import setup_simulator
    setup_simulator(ROOT)
    if mode != "official":
        from smart_building import fastsim
        fastsim.install(mode)


def _episode(job):
    import numpy as np
    from smart_building.fastsim import episode
    from smart_building.learned.b2rl_env import spec_dir
    mode, policy, spec, steps = job
    started = time.time()
    arrays, info = episode.run(spec_dir(spec), resolve_policy(policy), steps)
    path = episode_path(mode, policy, spec, steps)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + f".{os.getpid()}.tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)
    info.update(mode=mode, policy=policy, spec=spec, wall_s=time.time() - started, **episode.summary(arrays))
    write_json(path.with_suffix(".json"), info)
    return info


def cmd_episodes(args):
    policies = args.policies.split(",")
    jobs = [(args.mode, p, s, args.steps) for s in specs_of(args) for p in policies
            if args.force or not episode_path(args.mode, p, s, args.steps).is_file()]
    print(f"{len(jobs)} episodes to run ({args.mode}, {args.workers} workers)", flush=True)
    if not jobs:
        return
    context = multiprocessing.get_context("spawn")
    with context.Pool(min(args.workers, len(jobs)), initializer=_init_worker, initargs=(args.mode,),
                      maxtasksperchild=args.tasks_per_worker) as pool:
        for info in pool.imap_unordered(_episode, jobs):
            print(f"{info['mode']} {info['policy']} {info['spec']} cost {info['cost']:.6f} "
                  f"steps {info['steps']} setup {info['setup_s']:.0f}s run {info['run_s']:.0f}s "
                  f"({info['ms_per_step']:.1f} ms/step)", flush=True)


def load_rows(path):
    import numpy as np
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def official_row(policy, spec):
    """(summary, source) of the official episode of policy on spec, or (None, None)."""
    from smart_building.fastsim import episode
    _, building, weather = spec.split("/")
    for path in (DIAG / f"{policy}__{building}__{weather}.npz", episode_path("official", policy, spec)):
        if path.is_file():
            return episode.summary(load_rows(path)), str(path.relative_to(ROOT))
    return None, None


def fast_row(mode, policy, spec):
    from smart_building.fastsim import episode
    path = episode_path(mode, policy, spec)
    if not path.is_file():
        return None, None
    info = json.loads(path.with_suffix(".json").read_text(encoding="utf-8")) if path.with_suffix(".json").is_file() \
        else {}
    return episode.summary(load_rows(path)), info


def _mean_se(values):
    if not values:
        return None, None
    mean = statistics.fmean(values)
    se = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else float("nan")
    return mean, se


def cmd_report(args):
    policies = args.policies.split(",")
    specs = specs_of(args)
    table, worst = [], 0.0
    for policy in policies:
        for spec in specs:
            official, source = official_row(policy, spec)
            fast, info = fast_row(args.mode, policy, spec)
            row = {"policy": policy, "spec": spec, "official": official, "official_source": source, "fast": fast}
            if info:
                row["fast_ms_per_step"] = info.get("ms_per_step")
                row["fast_setup_s"] = info.get("setup_s")
            if official and fast:
                row["cost_rel_err"] = fast["cost"] / official["cost"] - 1.0
                worst = max(worst, abs(row["cost_rel_err"]))
                for part in ("comfort", "energy", "carbon"):
                    if part in official and part in fast:
                        row[f"{part}_abs_err"] = fast[part] - official[part]
                        row[f"{part}_rel_err"] = (fast[part] / official[part] - 1.0) if official[part] else None
            table.append(row)
    pairs = [tuple(p.split(":")) for p in args.pairs.split(",")] if args.pairs else \
        [(a, b) for i, a in enumerate(policies) for b in policies[i + 1:]]
    paired = []
    for a, b in pairs:
        diffs = []
        for spec in specs:
            oa, _ = official_row(a, spec)
            ob, _ = official_row(b, spec)
            fa, _ = fast_row(args.mode, a, spec)
            fb, _ = fast_row(args.mode, b, spec)
            if oa and ob and fa and fb:
                diffs.append({"spec": spec, "official": oa["cost"] - ob["cost"], "fast": fa["cost"] - fb["cost"],
                              "official_b": ob["cost"], "fast_b": fb["cost"],
                              "official_rel": oa["cost"] / ob["cost"] - 1.0, "fast_rel": fa["cost"] / fb["cost"] - 1.0})
        if not diffs:
            continue
        summary = {"a": a, "b": b, "n": len(diffs), "rows": diffs, "tie": TIE}
        for d in diffs:   # better (-1), same (0) or worse (+1); |relative difference| <= TIE counts as the same
            for key in ("official", "fast"):
                d[f"{key}_class"] = 0 if abs(d[f"{key}_rel"]) <= TIE else (1 if d[key] > 0 else -1)
        for key in ("official", "fast"):
            values = [d[key] for d in diffs]
            classes = [d[f"{key}_class"] for d in diffs]
            mean, se = _mean_se(values)
            # rel: mean difference over the mean cost of b (the rule screen's convention); mean_rel: mean of the
            # per-scenario relative differences
            summary[key] = {"mean": mean, "se": se, "better": classes.count(-1), "worse": classes.count(1),
                            "same": classes.count(0), "rel": mean / statistics.fmean(d[f"{key}_b"] for d in diffs),
                            "mean_rel": statistics.fmean(d[f"{key}_rel"] for d in diffs)}
        summary["same_sign"] = sum(d["official_class"] == d["fast_class"] for d in diffs)
        summary["opposite"] = sum(d["official_class"] * d["fast_class"] < 0 for d in diffs)
        paired.append(summary)
    result = {"mode": args.mode, "policies": policies, "specs": specs, "worst_cost_rel_err": worst,
              "rows": table, "paired": paired, "written": time.strftime("%Y-%m-%d %H:%M")}
    if args.out:
        write_json(args.out, result)
    print(f"fast ({args.mode}) vs official: worst |cost error| {100 * worst:.3f}%")
    for row in table:
        if row.get("cost_rel_err") is None:
            print(f"  {row['policy']:28s} {row['spec']:38s} missing ({'official' if not row['official'] else 'fast'})")
            continue
        parts = " ".join(f"{p} {100 * row[p + '_rel_err']:+.2f}%" for p in ("comfort", "energy", "carbon")
                         if row.get(p + "_rel_err") is not None)
        print(f"  {row['policy']:28s} {row['spec']:38s} official {row['official']['cost']:.6f} fast "
              f"{row['fast']['cost']:.6f} ({100 * row['cost_rel_err']:+.3f}%) {parts}")
    for s in paired:
        o, f = s["official"], s["fast"]
        print(f"  pair {s['a']} - {s['b']} (n {s['n']}): official {o['mean']:+.6f} ({100 * o['rel']:+.2f}%, SE "
              f"{o['se']:.6f}) better/worse/same {o['better']}/{o['worse']}/{o['same']} | fast {f['mean']:+.6f} "
              f"({100 * f['rel']:+.2f}%, SE {f['se']:.6f}) {f['better']}/{f['worse']}/{f['same']} | same class "
              f"{s['same_sign']}/{s['n']}, opposite {s['opposite']}")


def _build(spec):
    from smart_buildings import evaluate
    from smart_building.learned.b2rl_env import spec_dir
    started = time.time()
    env = evaluate.load_hybrid_environment(spec_dir(spec) / "config.gin")
    core = env.building.simulator._fastsim
    info = {"spec": spec, "seconds": round(time.time() - started, 1), **core.model.info}
    env.close()
    return info


def cmd_build(args):
    context = multiprocessing.get_context("spawn")
    with context.Pool(args.workers, initializer=_init_worker, initargs=("rom",), maxtasksperchild=1) as pool:
        for info in pool.imap_unordered(_build, specs_of(args)):
            print(json.dumps(info, default=str), flush=True)


def index_path(spec):
    """rom-cache/index/<pool>__<building>__<weather>.json: which cached model a scenario uses (written by build-one)."""
    return OUT / "rom-cache" / "index" / (spec.replace("/", "__") + ".json")


def _status_mb(field, pid="self"):
    """VmRSS / VmHWM of a process in MB (Linux; None elsewhere)."""
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
            if line.startswith(field + ":"):
                return int(line.split()[1]) / 1024
    except (OSError, ValueError):
        return None
    return None


def cmd_build_one(args):
    """One scenario's reduced model into the cache, then a second load from the cache timed in the same process."""
    import gc
    import logging
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    logging.getLogger("smart_building.fastsim").setLevel(logging.INFO)   # the build's phase timings in its log
    _init_worker("rom")
    from smart_buildings import evaluate
    from smart_building.learned.b2rl_env import spec_dir
    started = time.time()
    env = evaluate.load_hybrid_environment(spec_dir(args.spec) / "config.gin")
    core = env.building.simulator._fastsim
    info = {"spec": args.spec, "first_s": round(time.time() - started, 1),
            **{k: v for k, v in core.model.info.items() if k != "plan"}}
    if "raw" in core.model.info and args.time_load:   # built here: time a load from the cache as well
        env.close()
        del env, core
        gc.collect()
        started = time.time()
        env = evaluate.load_hybrid_environment(spec_dir(args.spec) / "config.gin")
        core = env.building.simulator._fastsim
    info.update(cached_load_s=round(time.time() - started, 1) if "raw" not in core.model.info else None,
                model=Path(core.model.info["cached"]).name,
                zones=len(core.data.zones), n=core.data.n, cells=core.data.n_cells,
                peak_rss_mb=_status_mb("VmHWM"), when=time.strftime("%Y-%m-%d %H:%M:%S"))
    env.close()
    write_json(index_path(args.spec), info)
    print(json.dumps(info, default=str), flush=True)


def _cached(spec):
    path = index_path(spec)
    if not path.is_file():
        return False
    model = json.loads(path.read_text(encoding="utf-8")).get("model")
    return bool(model) and (OUT / "rom-cache" / model).is_file()


def _cells(spec):
    import numpy as np
    from smart_building.learned.b2rl_env import spec_dir
    shape = np.load(spec_dir(spec) / "floor_plan.npy", mmap_mode="r").shape
    return int(shape[0] * shape[1])


def _zones(spec):
    import re
    from smart_building.learned.b2rl_env import spec_dir
    text = (spec_dir(spec) / "config.gin").read_text(encoding="utf-8")
    return max(1, sum(len(re.findall(r"room_\d+", rooms))
                      for rooms in re.findall(r"^ahu_\d+_rooms = \[(.*)\]", text, re.M)))


def _build_cost(spec):
    """Relative build time of a scenario's reduced model: about n x (raw vectors)^2, raw = 12 zones + 11."""
    return _cells(spec) * (12 * _zones(spec) + 11) ** 2


def _basis_mb(spec):
    """Upper bound of the build's basis storage (rom.py: float32 above 2 GB in float64)."""
    raw = 12 * _zones(spec) + 11
    item = 8 if _cells(spec) * raw * 8 <= 2.0e9 else 4
    return _cells(spec) * raw * item / 2**20


def _anon_mb(spec, in_memory_bytes=1.0e9):
    """Memory a builder needs besides a basis kept in a scratch file: the simulator, a factorization, work arrays
    (and the basis itself when it stays in memory, below rom.IN_MEMORY_BYTES)."""
    basis = _basis_mb(spec)
    return 800.0 + 2.5e-3 * _cells(spec) + (basis if basis * 2**20 <= in_memory_bytes else 0.0)


def _available_mb():
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return float("inf")


def cmd_cache(args):
    """Every scenario of the groups into the reduced-model cache: one fresh builder process per scenario, at most
    --workers at once (a number in data/runs/fastsim/cache-workers overrides it while running), and a new one only
    when the available memory minus what the running builders may still grow by stays above --reserve-gb."""
    sets = json.loads(EVAL_SETS.read_text(encoding="utf-8"))
    groups = {}
    for name in args.groups:
        if name == "train":
            from smart_building.learned.b2rl_env import scenario_dirs
            groups[name] = [f"train/{Path(d).parent.name}/{Path(d).name}" for d in scenario_dirs()]
        else:
            groups[name] = list(sets[name])
    order = [spec for name in args.groups for spec in groups[name]]
    # the scenarios of one building at one weather station usually share a model (same geometry, materials and
    # convection coefficient): build one per such family first, and its siblings (then mostly cache hits) after it
    family = {spec: spec.rsplit("_", 1)[0] for spec in order}
    group_of = {spec: name for name in reversed(args.groups) for spec in groups[name]}
    pending, seen = [], set()
    for name in args.groups:   # each group complete before the next: its first scenario per family, then the rest
        firsts, rest = [], []
        for spec in dict.fromkeys(groups[name]):
            if group_of[spec] != name or _cached(spec):
                continue
            (rest if family[spec] in seen else firsts).append(spec)
            seen.add(family[spec])
        if args.largest_first:
            # longest builds first, so the largest floor plans do not form the tail alone
            firsts.sort(key=lambda s: -_build_cost(s))
        pending += firsts + rest
    status_path, log_path = OUT / "cache-status.json", OUT / "logs" / "cache.log"
    build_logs = OUT / "logs" / "build"
    build_logs.mkdir(parents=True, exist_ok=True)
    environment = _environment()
    environment.update(OPENBLAS_NUM_THREADS=str(args.threads), OMP_NUM_THREADS=str(args.threads),
                       MKL_NUM_THREADS=str(args.threads))
    running, failures = {}, {}

    def note(text):
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {text}\n")

    def write_status():
        write_json(status_path, {
            "groups": {name: [sum(_cached(s) for s in specs), len(specs)] for name, specs in groups.items()},
            "running": sorted(r["spec"] for r in running.values()), "pending": len(pending),
            "failures": failures, "updated": time.strftime("%Y-%m-%d %H:%M:%S")})

    note(f"start: {len(order)} scenarios in {args.groups}, {len(pending)} to build, workers {args.workers}, "
         f"reserve {args.reserve_gb} GB, {args.threads} BLAS threads")
    while pending or running:
        for pid, r in list(running.items()):
            code = r["process"].poll()
            if code is None:
                continue
            r["handle"].close()
            del running[pid]
            if code == 0 and _cached(r["spec"]):
                info = json.loads(index_path(r["spec"]).read_text(encoding="utf-8"))
                note(f"done {r['spec']} in {time.time() - r['started']:.0f}s (build {info.get('seconds', 'cached')}, "
                     f"peak {info.get('peak_rss_mb', 0):.0f} MB, cached load {info.get('cached_load_s')}s)")
            else:
                tail = (build_logs / (r["spec"].replace("/", "__") + ".log")).read_text(
                    encoding="utf-8", errors="replace").splitlines()[-5:]
                failures[r["spec"]] = {"code": code, "tail": tail}
                note(f"FAILED {r['spec']} (exit {code}): {' | '.join(tail)}")
        override = OUT / "cache-workers"
        limit = int(override.read_text(encoding="utf-8").strip()) if override.is_file() else args.workers
        budget_file = OUT / "cache-basis-gb"
        if budget_file.is_file():
            args.basis_budget_gb = float(budget_file.read_text(encoding="utf-8").strip())
        while pending and len(running) < limit:
            busy = {family[r["spec"]] for r in running.values()}
            # what a running builder may still add to its anonymous memory (the basis scratch files are budgeted
            # apart: they live in the page cache, and the bases of the running builds must fit there together or
            # every pass over them goes to the disk)
            growing = sum(max(0.0, r["need"] - (_status_mb("RssAnon", r["process"].pid) or 0.0))
                          for r in running.values())
            room = _available_mb() - growing - 1024 * args.reserve_gb
            in_memory = args.in_memory_gb * 1e9
            scratch = sum(r["basis"] for r in running.values() if r["basis"] * 2**20 > in_memory)
            large = sum(r["basis"] > 1024 * args.large_gb for r in running.values())

            def fits(s):
                basis = _basis_mb(s)
                if _anon_mb(s, in_memory) > room:
                    return False
                if basis * 2**20 > in_memory and scratch and scratch + basis > 1024 * args.basis_budget_gb:
                    return False
                return not (basis > 1024 * args.large_gb and large)

            # the first pending scenario (in order) whose family is not building and that fits
            spec = next((s for s in pending if family[s] not in busy and fits(s)), None)
            if spec is None:
                break   # wait for a builder to finish or for memory to come back (the other jobs of the host)
            need, basis = _anon_mb(spec, in_memory), _basis_mb(spec)
            threads = str(args.large_threads if basis > 1024 * args.large_gb else args.threads)
            handle = (build_logs / (spec.replace("/", "__") + ".log")).open("w", encoding="utf-8")
            process = subprocess.Popen([str(PYTHON), "-B", str(Path(__file__).resolve()), "build-one", "--spec", spec],
                                       cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                       env={**environment, "OPENBLAS_NUM_THREADS": threads,
                                            "OMP_NUM_THREADS": threads, "MKL_NUM_THREADS": threads,
                                            "SB_FASTSIM_IN_MEMORY_BYTES": str(in_memory)})
            running[process.pid] = {"process": process, "spec": spec, "need": need, "basis": basis,
                                    "started": time.time(), "handle": handle}
            pending.remove(spec)
            note(f"launch {spec} (memory {need:.0f} MB, basis {basis:.0f} MB; available {_available_mb():.0f} MB)")
        write_status()
        time.sleep(5)
    write_status()
    note(f"end: {len(failures)} failures")


def _bench(job):
    from smart_buildings import evaluate
    from smart_building.learned.b2rl_env import spec_dir
    mode, spec, steps, start_at = job
    while time.time() < start_at:   # the processes step together after their own setup
        time.sleep(0.05)
    t = time.perf_counter()
    scenario = spec_dir(spec)
    env = evaluate.load_hybrid_environment(scenario / "config.gin")
    env.metrics_path = None
    env._occupancy_normalization_constant = 125.0
    metadata = evaluate.create_building_meta_data(env, scenario)
    observation, _ = env.reset()
    policy = resolve_policy(FINAL_RULE)(metadata, env.action_spec())
    setup = time.perf_counter() - t
    for _ in range(5):
        observation, *_ = env.step(policy.get_action(observation))
        env.get_reward_info_and_response()
    t = time.perf_counter()
    for _ in range(steps):
        observation, *_ = env.step(policy.get_action(observation))
        env.get_reward_info_and_response()
    per_step = (time.perf_counter() - t) / steps
    env.close()
    return {"setup_s": setup, "ms_per_step": 1000 * per_step}


def cmd_bench(args):
    context = multiprocessing.get_context("spawn")
    start_at = time.time() + 5
    jobs = [(args.mode, args.spec, args.steps, start_at)] * args.procs
    with context.Pool(args.procs, initializer=_init_worker, initargs=(args.mode,)) as pool:
        results = pool.map(_bench, jobs)
    per_step = statistics.fmean(r["ms_per_step"] for r in results)
    record = {"mode": args.mode, "spec": args.spec, "procs": args.procs, "steps": args.steps,
              "ms_per_step": per_step, "ms_per_step_each": [round(r["ms_per_step"], 2) for r in results],
              "setup_s": statistics.fmean(r["setup_s"] for r in results),
              "episode_s": statistics.fmean(r["setup_s"] for r in results) + 8640 * per_step / 1000,
              "steps_per_s_total": args.procs * 1000 / per_step, "host": platform.node(),
              "when": time.strftime("%Y-%m-%d %H:%M"),
              "load": os.getloadavg() if hasattr(os, "getloadavg") else None}
    with (OUT / "bench.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    print(json.dumps(record))


def cmd_ready(args):
    """data/runs/fastsim/READY.json for the training pipeline: cache counts from rom-cache/index, the worst cost error
    and the paired verdicts from the given reports; written to a temporary name, then renamed."""
    sets = json.loads(EVAL_SETS.read_text(encoding="utf-8"))
    from smart_building.learned.b2rl_env import scenario_dirs
    groups = {"train": [f"train/{Path(d).parent.name}/{Path(d).name}" for d in scenario_dirs()],
              **{name: list(sets[name]) for name in ("inner", "s4_first", "s4_rest")}}
    cache = {name: [sum(_cached(s) for s in specs), len(specs)] for name, specs in groups.items()}
    reports = [json.loads((ROOT / path).read_text(encoding="utf-8")) for path in args.reports]
    worst = max(r["worst_cost_rel_err"] for r in reports)
    opposite = sum(p["opposite"] for r in reports for p in r["paired"])
    pairs = sum(p["n"] for r in reports for p in r["paired"])
    rows = sum(1 for r in reports for row in r["rows"] if row.get("cost_rel_err") is not None)
    value = {"mode": args.mode, "validated": args.validated == "true", "verdicts_agree": opposite == 0,
             "worst_cost_rel_err": round(worst, 6), "cache": cache, "worker_memory_mb": args.worker_memory_mb,
             "excluded": args.excluded, "summary": args.summary, "report": "docs/plan/fastsim.md",
             "evidence": {"reports": args.reports, "episodes_compared": rows, "paired_scenarios": pairs,
                          "opposite_verdicts": opposite, "cached_load_s": args.cached_load_s},
             "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    write_json(OUT / "READY.json", value)
    print(json.dumps(value, ensure_ascii=False, indent=1))


def cmd_run(args):
    from smart_building import fastsim
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise SystemExit("fastsim.py run: give the command after --")
    if args.mode not in fastsim.MODES:
        raise SystemExit(f"--mode must be one of {fastsim.MODES}")
    environment = _environment()
    environment[fastsim.ENV] = args.mode
    environment["PYTHONPATH"] = os.pathsep.join([str(HOOK), str(ROOT / "packages")])
    if Path(command[0]).name in ("python", "python3", "python.exe"):
        # a host python would make the project scripts re-execute themselves through scripts/workspace.py, whose
        # environment drops PYTHONPATH and with it the hook: run them in the project .venv directly
        command = [str(PYTHON), *command[1:]]
    print(f"fastsim run ({args.mode}): {' '.join(command)}", flush=True)
    raise SystemExit(subprocess.run(command, cwd=ROOT, env=environment).returncode)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command_name", required=True)

    def add_specs(p):
        group = p.add_mutually_exclusive_group(required=True)
        group.add_argument("--set")
        group.add_argument("--specs", nargs="+")
        group.add_argument("--pool", choices=("train", "inner", "val"))

    p = sub.add_parser("episodes")
    p.add_argument("--mode", choices=("rom", "exact", "official"), required=True)
    p.add_argument("--policies", default=FINAL_RULE)
    add_specs(p)
    p.add_argument("--steps", type=int)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--tasks-per-worker", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_episodes)
    p = sub.add_parser("report")
    p.add_argument("--mode", choices=("rom", "exact"), required=True)
    p.add_argument("--policies", default=FINAL_RULE)
    add_specs(p)
    p.add_argument("--pairs")
    p.add_argument("--out")
    p.set_defaults(func=cmd_report)
    p = sub.add_parser("build")
    add_specs(p)
    p.add_argument("--workers", type=int, default=1)
    p.set_defaults(func=cmd_build)
    p = sub.add_parser("build-one")
    p.add_argument("--spec", required=True)
    p.add_argument("--time-load", action="store_true", help="after a build, also time a load from the cache")
    p.set_defaults(func=cmd_build_one)
    p = sub.add_parser("cache")
    p.add_argument("--groups", nargs="+", default=["train", "inner", "s4_first", "s4_rest"],
                   help="train (the TRAIN pool) and/or sets of configs/b2rl/eval-sets.json, built in this order")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--reserve-gb", type=float, default=4.0)
    p.add_argument("--threads", type=int, default=3, help="BLAS threads per builder (the Gram products)")
    p.add_argument("--largest-first", action="store_true", help="the longest builds of each group first")
    p.add_argument("--basis-budget-gb", type=float, default=16.0,
                   help="the running builds' basis scratch files together (one alone may exceed it)")
    p.add_argument("--large-gb", type=float, default=8.0, help="at most one build with a basis above this at a time")
    p.add_argument("--large-threads", type=int, default=8, help="BLAS threads of that one large build")
    p.add_argument("--in-memory-gb", type=float, default=1.0,
                   help="bases up to this size stay in memory (counted in the builder's memory); larger ones go to "
                        "a scratch file (whose writeback slows the builds)")
    p.set_defaults(func=cmd_cache)
    p = sub.add_parser("bench")
    p.add_argument("--mode", choices=("rom", "exact", "official"), required=True)
    p.add_argument("--spec", default="inner/building_29410/atlanta_1")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--procs", type=int, default=1)
    p.set_defaults(func=cmd_bench)
    p = sub.add_parser("ready")
    p.add_argument("--mode", default="rom")
    p.add_argument("--validated", choices=("true", "false"), required=True)
    p.add_argument("--reports", nargs="+", required=True)
    p.add_argument("--worker-memory-mb", type=int, required=True)
    p.add_argument("--cached-load-s", type=float)
    p.add_argument("--excluded", nargs="*", default=[])
    p.add_argument("--summary", required=True)
    p.set_defaults(func=cmd_ready)
    p = sub.add_parser("run")
    p.add_argument("--mode", default="rom")
    p.add_argument("command", nargs=argparse.REMAINDER)
    p.set_defaults(func=cmd_run)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
