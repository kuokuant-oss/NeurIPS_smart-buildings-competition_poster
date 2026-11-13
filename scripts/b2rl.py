"""B2-RL command line (handoff/RUNBOOK-windows-b2rl.md §4.3, §7).

  b2rl.py train --seed S --envs N --out data/runs/b2rl/<name> [--total-decisions D] [--threads T] [--rule NAME]
          [--batch official|module:Class [--controllers K] [--sim-device cuda]] [--device cpu|cuda]
      PPO on the hourly rule parameters (learned/b2rl_ppo.py); the same --out continues the run (on the simulator
      it started with, backend.json). --batch: all environments advanced together by a batch simulator, the rule
      controllers in K processes (learned/b2rl_batch.py); --device: the learner's network.
          [--layout v1|v2] [--gamma G] [--gae-lambda L] [--reward-baseline DIR] [--anchor PARAMETERS.json]
      --layout v2: + the static pressure cap, each side of a default scaled on its own (policies/b2rl.py LAYOUTS).
      --anchor: a JSON of rule parameter values ({"parameters": {...}} or the dict itself) that a = 0 gives.
  b2rl.py validate --run data/runs/b2rl/<name> [--set train_val10] [--max-steps 2016] [--every 25] [--fast rom]
      beside a running train: H3C's deterministic validation with early stopping; writes STOP on a plateau.
  b2rl.py bench --batch official|held|module:Class --slots B [--controllers K] [--decisions D] [--out rows.jsonl]
      simulator steps per second of the batched backend with the rule; held = the controllers alone (the ceiling).
  b2rl.py export --checkpoint <run folder | ckpt-*.pt> --out <model>.json
      the actor as a policies/b2rl.py model (numpy, no torch).
  b2rl.py eval --policy rule|<model>.json --set inner|s4_first|s4_rest|screen8|screen_inner8 --workers P --out <rows>.json
      gateway episodes of configs/b2rl/eval-sets.json (evaluation/gateway_episode.py); resumable: finished rows stay.
  b2rl.py from-diag --name <diag policy name> --set <set> --out <rows>.json
      the rule screen's own episodes of this host (.tmp/reviewer/rules/diag, gateway loop) as rows.
  b2rl.py compare --policy <rows>.json --baseline <rows>.json [--gate]
      paired differences of cost = -mean reward, with the comfort, energy and carbon parts (§7 gate: mean < 0,
      >= 60% of the scenarios better, |mean| > 2 standard errors).
  b2rl.py status [run folders]

Start it with the host python3: it re-executes itself in the project's .venv with the repository-local caches and
temporary folders of scripts/workspace.py (as pipeline/launch/launch.py does).
"""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"


def _in_project_environment():
    return Path(sys.prefix).resolve() == VENV.resolve()


if __name__ == "__main__" and not _in_project_environment():
    sys.path.insert(0, str(ROOT / "scripts"))
    from workspace import local_environment  # noqa: E402
    environment = local_environment()
    python = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    command = [str(python), "-B", str(Path(__file__).resolve()), *sys.argv[1:]]
    os.chdir(ROOT)
    if os.name == "posix":   # the trainer keeps this process id (signals, nohup)
        os.execve(str(python), command, environment)
    raise SystemExit(subprocess.run(command, cwd=ROOT, env=environment).returncode)

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import statistics as st  # noqa: E402
import time  # noqa: E402

sys.path[:0] = [str(ROOT / "packages")]
EVAL_SETS = ROOT / "configs/b2rl/eval-sets.json"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=1) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def use_fast(args):
    """--fast rom|exact: the opt-in fast simulator (packages/smart_building/fastsim) in this process and in every
    worker it starts (they inherit SB_FASTSIM; b2rl_env.setup_simulator installs it). Returns the simulator's name."""
    if not getattr(args, "fast", None):
        return "official"
    from smart_building.fastsim import ENV, requested
    os.environ[ENV] = args.fast
    return f"fastsim-{requested()}"


def cmd_train(args):
    from smart_building.learned.b2rl_ppo import Settings, train
    fast = use_fast(args)
    settings = Settings(seed=args.seed, num_envs=args.envs, total_decisions=args.total_decisions, rule=args.rule,
                        threads=args.threads, steps_cap=args.steps_cap, layout=args.layout)
    if args.steps_per_env:
        settings.steps_per_env = args.steps_per_env
    if args.gamma is not None:
        settings.gamma = args.gamma
    if args.gae_lambda is not None:
        settings.gae_lambda = args.gae_lambda
    if args.reward_baseline:
        baseline = json.loads((ROOT / args.reward_baseline / "simulator.json").read_text(encoding="utf-8"))
        if baseline["simulator"] != fast:
            raise SystemExit(f"{args.reward_baseline} holds {baseline['simulator']} rewards; this run is on {fast}")
        settings.reward_baseline = args.reward_baseline
    if args.anchor:
        from smart_building.policies.b2rl import check_anchor
        anchor = json.loads((ROOT / args.anchor).read_text(encoding="utf-8"))
        settings.anchor = check_anchor(dict(anchor.get("parameters", anchor)), settings.layout)

    def log(text):
        print(time.strftime("%m-%d %H:%M:%S ") + text, flush=True)

    backend = None
    if args.batch:
        import functools
        from smart_building.learned.b2rl_batch import BatchedHourlyEnvs
        backend = functools.partial(BatchedHourlyEnvs, simulator=args.batch, controllers=args.controllers,
                                    device=args.sim_device)
    # the simulator is part of a run's identity: a run continues only on the simulator it started with (the official
    # one through per-environment workers or through OfficialBatch is the same simulator)
    simulator = args.batch if args.batch and args.batch != "official" else fast
    out_dir = ROOT / args.out
    record = out_dir / "backend.json"
    previous = json.loads(record.read_text(encoding="utf-8"))["simulator"] if record.is_file() else (
        "official" if (out_dir / "latest.json").is_file() else None)
    if previous not in (None, simulator):
        raise SystemExit(f"{args.out} was trained on the {previous} simulator; use a new folder")
    write_json(record, {"simulator": simulator, "batch": bool(args.batch), "controllers": args.controllers,
                        "sim_device": args.sim_device if args.batch else None, "device": args.device})
    train(out_dir, settings, max_updates=args.max_updates, log=log, backend=backend, device=args.device)


def cmd_export(args):
    from smart_building.learned.b2rl_ppo import export, latest_checkpoint
    source = ROOT / args.checkpoint
    checkpoint = latest_checkpoint(source) if source.is_dir() else source
    if checkpoint is None:
        raise SystemExit(f"{args.checkpoint}: no checkpoint")
    model = export(checkpoint, ROOT / args.out)
    print(json.dumps(model["source"]))


def _episode(job):
    """One gateway episode in a spawn worker: (spec, row)."""
    policy, spec, rule, max_steps = job
    from smart_building.learned.b2rl_env import rule_policy_class, setup_simulator, spec_dir
    setup_simulator(ROOT)
    from smart_building.evaluation.gateway_episode import run_gateway_episode
    base = rule_policy_class(rule) if rule else None
    if policy == "rule":
        cls, key = rule_policy_class(rule), rule or "G36PolicyB2R2"
    elif json.loads(Path(policy).read_text(encoding="utf-8")).get("guard") is not None:  # tree + per-building guard
        from smart_building.policies.b2rx_guard import guarded_policy_class
        cls, key = guarded_policy_class(Path(policy), rule_class=base), "b2rx-guard"
    elif json.loads(Path(policy).read_text(encoding="utf-8")).get("tree") is not None:   # oracle_rollout.py tree
        from smart_building.policies.b2rl import b2rx_policy_class
        cls, key = b2rx_policy_class(Path(policy), rule_class=base), "b2rx"
    else:
        from smart_building.policies.b2rl import b2rl_policy_class
        cls, key = b2rl_policy_class(Path(policy), rule_class=base), "b2rl"
    row = run_gateway_episode(spec_dir(spec), cls, policy_key=key, label=spec, max_steps=max_steps)
    row["cost"] = -row["mean_reward"]
    return spec, row


def cmd_eval(args):
    import multiprocessing as mp
    specs = args.specs or json.loads(EVAL_SETS.read_text(encoding="utf-8"))[args.set]
    out = ROOT / args.out
    policy = args.policy if args.policy == "rule" else str((ROOT / args.policy).resolve())
    record = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else {}
    identity = {"policy": args.policy, "rule": args.rule, "set": args.set, "max_steps": args.max_steps}
    if use_fast(args) != "official":   # official records carry no "simulator" key, so they stay valid as they are
        identity["simulator"] = use_fast(args)
    if args.policy != "rule":
        model = json.loads(Path(policy).read_text(encoding="utf-8"))
        identity["model_source"] = model.get("source") or model.get("runs")
        from smart_building.policies.g36 import C3_COMBINATIONS
        if model.get("tree") is None and not args.rule \
                and model.get("rule", {}).get("b2r2_combination") != C3_COMBINATIONS["laa_occrow_dampfloor_b2r2"]:
            raise SystemExit("the model was trained on another G36PolicyB2R2 configuration than policies/g36.py's")
    if record and {k: record.get(k) for k in identity} != identity:
        raise SystemExit(f"{args.out} holds another evaluation ({ {k: record.get(k) for k in identity} })")
    from smart_building.policies.g36 import C3_COMBINATIONS
    record = {**identity, "b2r2_combination": C3_COMBINATIONS["laa_occrow_dampfloor_b2r2"],
              "host": os.uname().nodename if hasattr(os, "uname") else None, "rows": record.get("rows", {})}
    todo = [s for s in specs if s not in record["rows"]]
    print(f"{len(specs) - len(todo)} of {len(specs)} done; {len(todo)} to run on {args.workers} workers", flush=True)
    if todo:
        with mp.get_context("spawn").Pool(min(args.workers, len(todo)), maxtasksperchild=1) as pool:
            for spec, row in pool.imap_unordered(_episode, [(policy, s, args.rule, args.max_steps) for s in todo]):
                record["rows"][spec] = row
                write_json(out, record)
                print(f"{spec} cost {row['cost']:.5f} ({row['elapsed_seconds']:.0f} s)", flush=True)
    write_json(out, record)
    rows = [record["rows"][s] for s in specs if s in record["rows"]]
    print(f"{args.set}: {len(rows)} scenarios, mean cost {st.mean(r['cost'] for r in rows):.6f}")


def cmd_from_diag(args):
    """Rows of the rule screen's episodes (.tmp/reviewer/rules/diag/<name>__<building>__<weather>.npz, written by
    diagnose.py on the gateway loop) in the eval format: the rule baseline without running the episodes again."""
    import numpy as np
    specs = args.specs or json.loads(EVAL_SETS.read_text(encoding="utf-8"))[args.set]
    since = time.mktime(time.strptime(args.since, "%Y-%m-%d %H:%M")) if args.since else None
    rows = {}
    for spec in specs:
        _, building, weather = spec.split("/")
        path = ROOT / ".tmp/reviewer/rules/diag" / f"{args.name}__{building}__{weather}.npz"
        if not path.is_file() or (since is not None and path.stat().st_mtime < since):
            continue
        data = np.load(path)
        reward = data["reward"]
        row = {"scenario": spec, "policy": args.name, "steps": int(reward.size), "mean_reward": float(reward.mean()),
               "cost": float(-reward.mean()), "source": path.name}
        if "regret" not in data.files:      # a Kaggle row (kaggle_screen.py): the episode's mean reward only
            rows[spec] = {**row, "path": "kaggle"}
            continue
        w = data["weights"]
        scale = float(w.sum())
        # parts as evaluation/gateway_episode.py writes them: episode totals of the weighted reward terms
        rows[spec] = {**row, "path": "diag", "comfort_part": float(w[0] * data["regret"].sum() / scale),
                      "energy_part": float(-w[1] * data["energy"].sum() / scale),
                      "carbon_part": float(-w[2] * data["carbon"].sum() / scale)}
    missing = [s for s in specs if s not in rows]
    record = {"policy": f"diag:{args.name}", "rule": None, "set": args.set, "rows": rows, "missing": missing}
    if args.as_rule_eval:
        # the seed of `eval --policy rule --set <set>` (same identity), which then runs only the missing scenarios;
        # only for the screen name whose configuration G36PolicyB2R2 now builds (autorun2.py R4 checked it)
        from smart_building.policies.g36 import C3_COMBINATIONS
        if any(row["path"] == "kaggle" for row in rows.values()):
            raise SystemExit("--as-rule-eval takes rows of this host only (use --since to leave the others out)")
        record = {"policy": "rule", "rule": None, "set": args.set, "max_steps": None,
                  "b2r2_combination": C3_COMBINATIONS["laa_occrow_dampfloor_b2r2"], "seeded_from": f"diag:{args.name}",
                  "rows": rows}
    write_json(ROOT / args.out, record)
    print(f"{len(rows)} of {len(specs)} rows from diag/{args.name}; missing {missing}")


def paired(policy_rows, baseline_rows):
    common = sorted(set(policy_rows) & set(baseline_rows))
    if not common:
        return None
    d = [policy_rows[s]["cost"] - baseline_rows[s]["cost"] for s in common]
    base = st.mean(baseline_rows[s]["cost"] for s in common)
    out = {"n": len(d), "mean_diff": st.mean(d), "se": st.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else float("nan"),
           "better": sum(x < 0 for x in d), "worse": sum(x > 0 for x in d),
           "policy_mean": st.mean(policy_rows[s]["cost"] for s in common), "baseline_mean": base,
           "rel": st.mean(d) / base}
    # the cost split per step (gateway_episode parts are episode totals of the weighted reward terms, all <= 0)
    for part in ("comfort_part", "energy_part", "carbon_part"):
        if all(part in policy_rows[s] and part in baseline_rows[s] for s in common):
            out[part.replace("_part", "_cost")] = {
                "policy": -st.mean(policy_rows[s][part] / policy_rows[s]["steps"] for s in common),
                "baseline": -st.mean(baseline_rows[s][part] / baseline_rows[s]["steps"] for s in common)}
    out["gate"] = bool(out["mean_diff"] < 0 and out["better"] >= 0.6 * out["n"]
                       and abs(out["mean_diff"]) > 2 * out["se"])
    return out


def cmd_compare(args):
    records = [json.loads((ROOT / path).read_text(encoding="utf-8")) for path in (args.policy, args.baseline)]
    simulators = [r.get("simulator", "official") for r in records]
    if simulators[0] != simulators[1]:
        raise SystemExit(f"the two evaluations ran on different simulators ({simulators[0]} vs {simulators[1]})")
    policy, baseline = records[0]["rows"], records[1]["rows"]
    result = paired(policy, baseline)
    print(json.dumps(result, indent=1))
    if args.gate and not (result and result["gate"]):
        raise SystemExit(1)


def cmd_bench(args):
    """Simulator steps per second of the batched backend with the rule (zero actions): simulator plus controllers."""
    import numpy as np
    from smart_building.learned.b2rl_batch import BatchedHourlyEnvs
    from smart_building.learned.b2rl_env import TRAIN_POOL, scenario_dirs
    from smart_building.policies.b2rl import PARAMETERS
    fast = use_fast(args)
    started = time.time()
    envs = BatchedHourlyEnvs(args.slots, args.seed, scenario_dirs(ROOT / args.pool if args.pool else TRAIN_POOL),
                             controllers=args.controllers, simulator=args.batch, device=args.sim_device)
    try:
        envs.call([(i, "start", (0,)) for i in range(args.slots)])
        loaded = time.time()
        zero, steps = np.zeros(len(PARAMETERS)), 0
        for _ in range(args.decisions):
            steps += sum(r["steps"] for r in envs.call([(i, "decide", (zero,)) for i in range(args.slots)]))
        seconds = time.time() - loaded
    finally:
        envs.close()
    row = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "simulator": args.batch if args.batch != "official" else fast, "slots": args.slots,
           "controllers": envs.controllers.processes, "decisions": args.decisions, "steps": steps,
           "load_seconds": round(loaded - started, 1), "seconds": round(seconds, 2),
           "steps_per_second": round(steps / seconds, 1)}
    print(json.dumps(row))
    if args.out:
        out = ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(row) + "\n")


def cmd_status(args):
    from smart_building.learned.b2rl_ppo import latest_checkpoint
    folders = [ROOT / f for f in args.runs] or sorted((ROOT / "data/runs/b2rl").glob("*/"))
    for folder in folders:
        rows = []
        if (folder / "iterations.jsonl").is_file():
            rows = [json.loads(line) for line in (folder / "iterations.jsonl").read_text(encoding="utf-8").splitlines()
                    if line.strip()]
        checkpoint = latest_checkpoint(folder)
        if not rows:
            print(f"{folder.relative_to(ROOT).as_posix()}: no update yet")
            continue
        last = rows[-1]
        trend = " ".join(f"{r['reward_12_mean']:+.4f}" for r in rows[-5:])
        print(f"{folder.relative_to(ROOT).as_posix()}: update {last['update']}, {last['decisions']} decisions, "
              f"{last['episodes']} episodes, sigma {st.mean(last['sigma']):.3f}, kl {last['approx_kl']:.4f}, "
              f"{last['decisions_per_second']:.2f}/s, 12-step reward (last 5) {trend}; "
              f"checkpoint {checkpoint.name if checkpoint else '-'}; STOP {'yes' if (folder / 'STOP').exists() else 'no'}")


def _rule_decisions(job):
    """The rule's reward sum of every hourly decision of one scenario (spawn worker; zero actions = the rule)."""
    scenario, out = job
    import numpy as np
    from smart_building.learned.b2rl_env import HourlyEnv, setup_simulator
    from smart_building.policies.b2rl import PARAMETERS
    setup_simulator(ROOT)
    env = HourlyEnv([scenario], 0, 0)
    env.start(0)
    rewards, done, zero = [], False, np.zeros(len(PARAMETERS))
    while not done:
        result = env.decide(zero)
        rewards.append(result["reward"])
        done = result["done"]
    env.close()
    temporary = Path(out).with_name(Path(out).name + ".partial.npy")
    np.save(temporary, np.array(rewards, dtype=np.float64))
    os.replace(temporary, out)
    return scenario, len(rewards), -sum(rewards) / result["episode"]["steps"]


def cmd_rule_baseline(args):
    """The rule-relative reward's baseline (learned/b2rl_env.py HourlyEnv reward_baseline): for every scenario of the
    pool, the rule's reward sum of each hourly decision, <out>/<building>__<weather>.npy (resumable; the simulator
    is part of the folder: <out>/simulator.json)."""
    import multiprocessing as mp
    from smart_building.learned.b2rl_env import TRAIN_POOL, scenario_dirs
    simulator = use_fast(args)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    record = out / "simulator.json"
    if record.is_file() and json.loads(record.read_text(encoding="utf-8"))["simulator"] != simulator:
        raise SystemExit(f"{args.out} holds another simulator's rewards")
    write_json(record, {"simulator": simulator, "pool": str((ROOT / args.pool if args.pool else TRAIN_POOL)
                                                            .relative_to(ROOT))})
    scenarios = scenario_dirs(ROOT / args.pool if args.pool else TRAIN_POOL)
    todo = [(s, str(out / f"{Path(s).parent.name}__{Path(s).name}.npy")) for s in scenarios
            if not (out / f"{Path(s).parent.name}__{Path(s).name}.npy").is_file()]
    print(f"{len(scenarios) - len(todo)} of {len(scenarios)} done; {len(todo)} to run on {args.workers} workers",
          flush=True)
    if todo:
        with mp.get_context("spawn").Pool(min(args.workers, len(todo)), maxtasksperchild=1) as pool:
            for scenario, n, cost in pool.imap_unordered(_rule_decisions, todo):
                print(f"{Path(scenario).parent.name}/{Path(scenario).name}: {n} decisions, cost {cost:.5f}",
                      flush=True)


def plateau(costs, min_update=100, patience=3, min_delta=0.01):
    """H3C's early stopping (h3c-drl-training EXPERIMENT_PROTOCOL.md, "Early stopping and selection") on the
    validation costs [(update, cost)] in order, lower is better. The first validation is the anchor; a cost below
    anchor x (1 - min_delta) is a significant improvement: it becomes the anchor and the misses start again (H3C
    compares returns with 1% of max(|A|, 1); our costs are about 0.01, so the 1% is relative). A validation after
    min_update that is not one is a miss; patience misses stop (H3C: every 25 epochs, misses after epoch 100, 3 of
    them, so 175 is the earliest stop). The best is the lowest cost, also when it is no significant improvement."""
    anchor, misses, best = None, 0, None
    for update, cost in costs:
        if best is None or cost < best[1]:
            best = (update, cost)
        if anchor is None or cost < anchor * (1.0 - min_delta):
            anchor, misses = cost, 0
        elif update > min_update:
            misses += 1
    return {"anchor": anchor, "misses": misses, "stop": misses >= patience,
            "best_update": best[0] if best else None, "best_cost": best[1] if best else None}


def plateau_smoothed(rows, min_update=300, patience=4, min_gain=0.005, smooth=3):
    """The validation rule of 2026-10-28 (docs/reference/notes/19-auto-rl-hyperparameter-tuning.md §5): rows carry
    score = the mean over the scenarios of (policy cost - rule cost) / rule cost (every scenario weighs the same;
    lower is better) and se_score, its paired standard error. The score is smoothed over the last `smooth`
    validations; a smoothed score below anchor - max(min_gain, se_score) is an improvement (the first one is the
    anchor); misses count only after min_update (the learning rate's floor, so a lucky early peak cannot end the run);
    patience misses stop. ranked: the updates of the lowest smoothed scores, the candidates for the final pick."""
    anchor, misses, smoothed = None, 0, []
    for i, row in enumerate(rows):
        window = rows[max(0, i - smooth + 1):i + 1]
        score = sum(r["score"] for r in window) / len(window)
        smoothed.append((row["update"], score))
        if anchor is None or score < anchor - max(min_gain, row["se_score"]):
            anchor, misses = score, 0
        elif row["update"] > min_update:
            misses += 1
    ranked = [update for update, _ in sorted(smoothed, key=lambda x: (x[1], x[0]))]
    return {"anchor": anchor, "misses": misses, "stop": misses >= patience, "smoothed": smoothed[-1][1] if smoothed
            else None, "ranked": ranked[:3], "best_update": ranked[0] if ranked else None}


def relative_scores(policy_rows, rule_rows):
    """Per-scenario (policy - rule) / rule over the common scenarios: (mean, paired standard error)."""
    common = sorted(set(policy_rows) & set(rule_rows))
    rel = [(policy_rows[s]["cost"] - rule_rows[s]["cost"]) / rule_rows[s]["cost"] for s in common]
    return st.mean(rel), (st.stdev(rel) / math.sqrt(len(rel)) if len(rel) > 1 else float("nan"))


def cmd_validate(args):
    """Deterministic validation with early stopping beside a running `train` (H3C protocol; the training window is a
    fixed set of TRAIN scenarios, configs/b2rl/eval-sets.json, their first --max-steps steps): every --every updates
    the checkpoint's mean action runs those episodes (the fast simulator with --fast) and is paired with the rule on
    the same episodes; plateau() decides (--protocol h3c), or plateau_smoothed() on the per-scenario relative score
    (--protocol smoothed). On a stop, or at --cap updates, it writes STOP into the run folder (the trainer ends after
    its current update); with --select-top K (smoothed) the K best smoothed checkpoints then run whole episodes of
    --select-set against the rule and the lowest mean relative difference is the pick. Rows <run>/validation.jsonl;
    the best checkpoint <run>/best.json and <run>/validation/model-best.json (with --tag T: validation-T.jsonl,
    validation-T/, best-T.json, so a second protocol on the same run keeps its own record; --start: the first
    update it validates). Resumable: validated updates are not run again."""
    import shutil
    from smart_building.learned.b2rl_ppo import export
    run = ROOT / args.run
    suffix = f"-{args.tag}" if args.tag else ""
    folder = run / f"validation{suffix}"
    folder.mkdir(parents=True, exist_ok=True)
    rows_path = run / f"validation{suffix}.jsonl"
    best_path = run / f"best{suffix}.json"
    common = ["--set", args.set, "--workers", str(args.workers), "--max-steps", str(args.max_steps)]
    common += ["--fast", args.fast] if args.fast else []
    trainer = rf"^[^ ]*python[^ ]* -B [^ ]*scripts/b2rl\.py train .*--out {args.run}( |$)"

    def log(text):
        print(time.strftime("%m-%d %H:%M:%S ") + text, flush=True)

    def evaluate(policy, out):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/b2rl.py"), "eval", "--policy", policy,
                                 *common, "--out", str(out.relative_to(ROOT))], cwd=ROOT, capture_output=True, text=True)
        if result.returncode:
            raise SystemExit(f"eval {policy} failed: {(result.stdout + result.stderr).strip()[-600:]}")
        return json.loads(out.read_text(encoding="utf-8"))["rows"]

    def latest_update():
        try:
            return int(json.loads((run / "latest.json").read_text(encoding="utf-8"))["update"])
        except (OSError, ValueError, KeyError):
            return 0

    rule_rows = evaluate("rule", folder / "rule.json")
    log(f"rule on {args.set} ({len(rule_rows)} episodes, {args.max_steps} steps): "
        f"mean cost {st.mean(r['cost'] for r in rule_rows.values()):.6f}")
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()] \
        if rows_path.is_file() else []
    update = rows[-1]["update"] + args.every if rows else (args.start or args.every)
    while update <= args.cap:
        while latest_update() < update:
            alive = subprocess.run(["pgrep", "-f", trainer], capture_output=True, text=True).stdout.split()
            if not alive and latest_update() < update:
                log(f"the trainer is not running and its last update is {latest_update()}; validation ends")
                return
            time.sleep(20)
        started = time.time()
        model = folder / f"model-u{update}.json"
        export(run / f"ckpt-{update:05d}.pt", model)
        policy_rows = evaluate(str(model.relative_to(ROOT)), folder / f"u{update}.json")
        result = paired(policy_rows, rule_rows)
        score, se_score = relative_scores(policy_rows, rule_rows)
        row = {"update": update, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "n": result["n"],
               "cost": result["policy_mean"], "rule_cost": result["baseline_mean"], "rel": result["rel"],
               "score": score, "se_score": se_score, "better": result["better"], "worse": result["worse"],
               "seconds": round(time.time() - started, 1)}
        rows.append(row)
        if args.protocol == "smoothed":
            decision = plateau_smoothed(rows, args.min_update, args.patience, args.min_gain, args.smooth)
        else:
            decision = plateau([(r["update"], r["cost"]) for r in rows], args.min_update, args.patience, args.min_delta)
        row.update(decision)
        with rows_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(row) + "\n")
        if decision["best_update"] == update:
            shutil.copyfile(model, folder / "model-best.json")
            write_json(best_path, {"update": update, "cost": row["cost"], "rule_cost": row["rule_cost"],
                                           "rel": row["rel"], "score": row["score"],
                                           "model": str((folder / "model-best.json").relative_to(ROOT)),
                                           "checkpoint": f"ckpt-{update:05d}.pt"})
        if args.protocol == "smoothed":
            log(f"update {update}: score {score:+.2%} (SE {se_score:.2%}, {row['better']} better, {row['worse']} "
                f"worse), smoothed {decision['smoothed']:+.2%}; anchor {decision['anchor']:+.2%}, misses "
                f"{decision['misses']}, ranked {decision['ranked']} ({row['seconds']:.0f} s)")
        else:
            log(f"update {update}: cost {row['cost']:.6f} vs rule {row['rule_cost']:.6f} ({row['rel']:+.2%}, "
                f"{row['better']} better, {row['worse']} worse); anchor {decision['anchor']:.6f}, misses "
                f"{decision['misses']}, best update {decision['best_update']} ({row['seconds']:.0f} s)")
        if decision["stop"] or update >= args.cap:
            if decision["stop"]:
                reason = (f"plateau: {decision['misses']} validations without an improvement after update "
                          f"{args.min_update}")
            else:
                reason = f"cap of {args.cap} updates"
            (run / "STOP").write_text(f"b2rl.py validate {time.strftime('%Y-%m-%d %H:%M:%S')}: {reason}\n",
                                      encoding="utf-8")
            log(f"STOP written ({reason}); best update {decision['best_update']}")
            if args.protocol == "smoothed" and args.select_top:
                select(args, best_path, folder, decision["ranked"][:args.select_top], log)
            return
        update += args.every


def select(args, best_path, folder, updates, log):
    """The final pick of the smoothed protocol: whole episodes of --select-set for the candidate checkpoints and the
    rule (the fast simulator with --fast), the lowest mean per-scenario relative difference wins (best.json)."""
    import shutil
    options = ["--set", args.select_set, "--workers", str(args.select_workers)]
    options += ["--fast", args.fast] if args.fast else []

    def evaluate(policy, out):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/b2rl.py"), "eval", "--policy", policy,
                                 *options, "--out", str(out.relative_to(ROOT))], cwd=ROOT, capture_output=True, text=True)
        if result.returncode:
            raise SystemExit(f"select eval {policy} failed: {(result.stdout + result.stderr).strip()[-600:]}")
        return json.loads(out.read_text(encoding="utf-8"))["rows"]

    rule_rows = evaluate("rule", folder / f"select-rule-{args.select_set}.json")
    results = []
    for update in updates:
        model = folder / f"model-u{update}.json"
        rows = evaluate(str(model.relative_to(ROOT)), folder / f"select-u{update}-{args.select_set}.json")
        score, se_score = relative_scores(rows, rule_rows)
        result = paired(rows, rule_rows)
        results.append({"update": update, "score": score, "se_score": se_score, "better": result["better"],
                        "worse": result["worse"], "n": result["n"], "gate": result["gate"]})
        log(f"select {args.select_set} update {update}: {score:+.2%} (SE {se_score:.2%}, {result['better']} better, "
            f"{result['worse']} worse, gate {result['gate']})")
    best = min(results, key=lambda r: r["score"])
    shutil.copyfile(folder / f"model-u{best['update']}.json", folder / "model-best.json")
    write_json(best_path, {"update": best["update"], "selected_on": args.select_set, "selection": results,
                                   "model": str((folder / "model-best.json").relative_to(ROOT)),
                                   "checkpoint": f"ckpt-{best['update']:05d}.pt"})
    log(f"selected update {best['update']} ({best['score']:+.2%} on {args.select_set})")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--envs", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--total-decisions", type=int, default=400_000)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--rule", help="rule class (default G36PolicyB2R2; a diagnose.py preset name for pilot runs)")
    p.add_argument("--max-updates", type=int)
    p.add_argument("--steps-cap", type=int, help="smoke runs: episodes end after this many steps")
    p.add_argument("--steps-per-env", type=int, help="smoke runs: decisions per environment per update")
    p.add_argument("--batch", help="batch simulator (learned/b2rl_batch.py): official, or module:Class of a fast one")
    p.add_argument("--controllers", type=int, help="with --batch: controller processes (default CPU count - 2)")
    p.add_argument("--sim-device", default="cuda", help="with --batch: the batch simulator's device")
    p.add_argument("--device", default="cpu", help="the learner's network: cpu or cuda")
    p.add_argument("--fast", help="rom or exact: the opt-in fast simulator (packages/smart_building/fastsim)")
    p.add_argument("--layout", default="v1", choices=["v1", "v2"],
                   help="policies/b2rl.py LAYOUTS: v2 = + static pressure cap, each side of a default scaled alone")
    p.add_argument("--gamma", type=float, help="discount per decision (default 0.99)")
    p.add_argument("--gae-lambda", type=float, help="GAE lambda per decision (default 0.95)")
    p.add_argument("--reward-baseline", help="a rule-baseline folder: train on reward minus the rule's reward")
    p.add_argument("--anchor", help="a JSON of rule parameter values that the zero action gives (the new base)")
    p = sub.add_parser("rule-baseline")
    p.add_argument("--out", required=True, help="e.g. data/processed/b2rl/rule-decisions-rom")
    p.add_argument("--pool", help="scenario pool (default the TRAIN pool)")
    p.add_argument("--workers", type=int, required=True)
    p.add_argument("--fast", help="rom or exact: the fast simulator (the training's own simulator)")
    p = sub.add_parser("validate")
    p.add_argument("--run", required=True, help="the run folder of a train that runs or will run")
    p.add_argument("--set", default="train_val10", help="the validation set of configs/b2rl/eval-sets.json")
    p.add_argument("--max-steps", type=int, default=2016, help="steps of every validation episode (2016 = 7 days)")
    p.add_argument("--every", type=int, default=25)
    p.add_argument("--min-update", type=int, default=100)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--min-delta", type=float, default=0.01)
    p.add_argument("--cap", type=int, default=700)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--fast", help="rom or exact: validate on the fast simulator")
    p.add_argument("--protocol", default="h3c", choices=["h3c", "smoothed"],
                   help="h3c: plateau() on the mean cost; smoothed: plateau_smoothed() on the per-scenario score")
    p.add_argument("--smooth", type=int, default=3, help="smoothed: validations averaged")
    p.add_argument("--min-gain", type=float, default=0.005, help="smoothed: the least improvement, relative units")
    p.add_argument("--select-top", type=int, default=0, help="smoothed: candidates run on --select-set at the end")
    p.add_argument("--select-set", default="inner")
    p.add_argument("--select-workers", type=int, default=4)
    p.add_argument("--tag", help="a separate record of this protocol: validation-TAG.jsonl, validation-TAG/, best-TAG.json")
    p.add_argument("--start", type=int, help="the first update to validate (default --every)")
    p = sub.add_parser("export")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("eval")
    p.add_argument("--policy", required=True)
    p.add_argument("--set", required=True, help="a set of configs/b2rl/eval-sets.json (or a name for --specs)")
    p.add_argument("--specs", nargs="+", help="pool/building/weather specs instead of the set's (train = split-v1-r1)")
    p.add_argument("--workers", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rule", help="rule class under the policy (default G36PolicyB2R2)")
    p.add_argument("--max-steps", type=int, help="smoke runs only: stop every episode after this many steps")
    p.add_argument("--fast", help="rom or exact: evaluate on the fast simulator (its rows compare only with its own)")
    p = sub.add_parser("from-diag")
    p.add_argument("--name", required=True, help="the diag policy name, e.g. a1_cold3+a1_econ")
    p.add_argument("--set", required=True)
    p.add_argument("--specs", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--since", help="only diag files written at or after this local time (YYYY-MM-DD HH:MM)")
    p.add_argument("--as-rule-eval", action="store_true", help="write the seed of eval --policy rule --set <set>")
    p = sub.add_parser("compare")
    p.add_argument("--policy", required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument("--gate", action="store_true")
    p = sub.add_parser("status")
    p.add_argument("runs", nargs="*")
    p = sub.add_parser("bench")
    p.add_argument("--batch", default="official", help="official, held (the controllers alone) or module:Class")
    p.add_argument("--slots", type=int, required=True)
    p.add_argument("--controllers", type=int, help="controller processes (default CPU count - 2)")
    p.add_argument("--decisions", type=int, default=5, help="hourly decisions per slot")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pool", help="scenario pool (default the TRAIN pool)")
    p.add_argument("--sim-device", default="cuda")
    p.add_argument("--out", help="append the row to this JSON-lines file")
    p.add_argument("--fast", help="rom or exact: the opt-in fast simulator in every slot")
    args = parser.parse_args(argv)
    {"train": cmd_train, "export": cmd_export, "eval": cmd_eval, "from-diag": cmd_from_diag, "compare": cmd_compare,
     "status": cmd_status, "bench": cmd_bench, "validate": cmd_validate,
     "rule-baseline": cmd_rule_baseline}[args.command](args)


if __name__ == "__main__":
    main()
