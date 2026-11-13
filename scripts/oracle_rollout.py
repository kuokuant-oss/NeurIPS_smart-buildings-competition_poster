"""Lookahead search for the best hourly decision on the final rule (handoff/RUNBOOK-windows-b2rl.md §2.6).

Bertsekas' rollout: at every decision point (one per hour, from the first step) each candidate k is applied for one
hour from the same saved state, then the final rule G36PolicyB2R2 runs to the lookahead end (H hours; the episode's
end at the latest); the candidate with the largest reward sum over those 1 + H hours is applied for the hour and the
episode moves on to the next decision point. With the simulator's own random generators the search knows the future
occupants and weather: the result is an optimistic bound of what choosing among these candidates can reach.

Candidates are rule parameters or rule variants already implemented (no actuator values), each for one hour, after
which the rule's own configuration comes back: keep the rule; occupied margin 0.5 K; economizer by supply air; static
pressure held at 200 Pa (the reset loops' maximum); AHUs off this hour (only during occupied hours with every zone
more than 0.5 K inside the comfort band; the rule's discrete AHU commands are replaced by off); pre-cooling before the
price peak (precool_until_hour 12); evening charge (evening_charge_min 120); optimal start cap 6 h. A candidate equal
to the rule's own configuration is skipped.

Workers: every worker loads the scenario, keeps the decision state as a local snapshot (smart_building.sim.snapshot)
and evaluates its share of the candidates; then every worker advances its own copy with the chosen candidate, so
nothing large crosses the pipes and the copies are checked to agree (digest of the observation). The real
trajectory's features are policies/b2rl.py's (the RL features), recorded with the choice for the decision tree.
decisions.jsonl is appended per decision; a restart replays the recorded choices and continues.

  oracle_rollout.py run --scenario train/<building>/<weather> --out data/runs/oracle/<name> [--horizon 8] [--workers 8]
  oracle_rollout.py tree --runs data/runs/oracle/<name> ... --out <tree>.json [--depth 4]
Start with the host python3 (re-executes in the project's .venv, as scripts/b2rl.py).
"""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"

if __name__ == "__main__" and Path(sys.prefix).resolve() != VENV.resolve():
    sys.path.insert(0, str(ROOT / "scripts"))
    from workspace import local_environment  # noqa: E402
    environment = local_environment()
    python = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    command = [str(python), "-B", str(Path(__file__).resolve()), *sys.argv[1:]]
    os.chdir(ROOT)
    if os.name == "posix":
        os.execve(str(python), command, environment)
    raise SystemExit(subprocess.run(command, cwd=ROOT, env=environment).returncode)

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import multiprocessing  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

sys.path[:0] = [str(ROOT / "packages")]
from smart_building.policies.b2rl import CANDIDATE_NAMES as NAMES, CANDIDATES  # noqa: E402
from smart_building.policies.b2rl import ahu_off, apply_candidate, candidate_eligible  # noqa: E402

HOUR = 12


class Trajectory:
    """One worker's copy of the scenario: the decision state as a snapshot, candidate evaluation, advancing."""

    def __init__(self, scenario, rule=None):
        from smart_building.learned.b2rl_env import rule_policy_class, setup_simulator, spec_dir
        from smart_building.policies.b2rl import HourlyRule
        setup_simulator(ROOT)                    # the simulator's import path first
        from smart_buildings import evaluate
        folder = spec_dir(scenario)
        env = evaluate.load_hybrid_environment(folder / "config.gin")
        env.metrics_path = None
        env._occupancy_normalization_constant = 125.0
        metadata = evaluate.create_building_meta_data(env, folder)
        obs, _ = env.reset()
        agent = HourlyRule(rule_policy_class(rule)(metadata, env.action_spec()))
        self.scenario, self.rule = scenario, rule
        agent.tracker.observe(obs)
        self.n_ahu = len(metadata["ahu_zone_assignments"])
        self.state = self._take((env, agent, obs, 0, 0.0, False))
        self.current = self._info(agent, 0, False)

    @staticmethod
    def _take(objects):
        from smart_building.sim.snapshot import take
        return take(objects)

    def _restore(self):
        from smart_building.sim.snapshot import restore
        return list(restore(self.state))

    def _run(self, env, agent, obs, steps, spec=None, track=False):
        total, done, n = 0.0, False, 0
        for _ in range(steps):
            action = agent.rule.get_action(obs)
            if spec and spec.get("ahu_off"):
                action = ahu_off(action, self.n_ahu)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            n += 1
            if track:
                agent.tracker.observe(obs)
            if terminated or truncated:
                done = True
                break
        return obs, total, n, done

    @staticmethod
    def eligible(agent, spec):
        return candidate_eligible(agent.rule._controller, agent.tracker, spec)

    def _info(self, agent, steps, done):
        o = agent.tracker.current
        return {"steps": steps, "done": done, "minute_of_day": o.minute_of_day, "weekday": o.weekday,
                "features": agent.features().tolist(),
                "eligible": [self.eligible(agent, spec) for _, spec in CANDIDATES]}

    def info(self):
        """The decision point's time, features and which candidates can be tried."""
        return self.current

    def evaluate(self, indices, horizon):
        """Reward sums over 1 + horizon hours of the candidates `indices` (None where not eligible)."""
        out = {}
        for i in indices:
            env, agent, obs, steps, total, done = self._restore()
            spec = CANDIDATES[i][1]
            if done or not self.eligible(agent, spec):
                out[i] = None
                env.close()
                continue
            undo = apply_candidate(agent.rule._controller, spec)
            obs, first, n, ended = self._run(env, agent, obs, HOUR, spec)
            undo()
            rest = 0.0
            if not ended:
                obs, rest, _, _ = self._run(env, agent, obs, HOUR * horizon)
            out[i] = first + rest
            env.close()
        return out

    def baseline(self, scenario, steps):
        """The plain rule's reward sum over the first `steps` steps of a fresh copy (the bound's comparison)."""
        fresh = Trajectory(scenario, self.rule)
        from smart_building.sim.snapshot import restore
        env, agent, obs, _, _, _ = restore(fresh.state)
        _, total, n, _ = self._run(env, agent, obs, steps)
        env.close()
        return {"reward": total, "steps": n}

    def advance(self, index):
        """Apply candidate `index` for the hour on the real trajectory; the new decision state replaces the old."""
        env, agent, obs, steps, total, done = self._restore()
        spec = CANDIDATES[index][1]
        undo = apply_candidate(agent.rule._controller, spec)
        obs, reward, n, done = self._run(env, agent, obs, HOUR, spec, track=True)
        undo()
        self.state = self._take((env, agent, obs, steps + n, total + reward, done))
        self.current = self._info(agent, steps + n, done)
        digest = hashlib.sha256(np.asarray(obs, dtype=np.float32).tobytes()).hexdigest()
        env.close()
        return {"reward": reward, "steps": steps + n, "total": total + reward, "done": done, "digest": digest}


def worker_main(connection, scenario, rule):
    sys.dont_write_bytecode = True
    try:
        trajectory = Trajectory(scenario, rule)
        connection.send(("ok", None))
    except Exception as exc:  # noqa: BLE001
        connection.send(("error", f"{exc!r}\n{traceback.format_exc()}"))
        return
    while True:
        command, args = connection.recv()
        if command == "close":
            connection.send(("ok", None))
            return
        try:
            connection.send(("ok", getattr(trajectory, command)(*args)))
        except Exception as exc:  # noqa: BLE001
            connection.send(("error", f"{exc!r}\n{traceback.format_exc()}"))


class Workers:
    def __init__(self, count, scenario, rule=None):
        context = multiprocessing.get_context("spawn")
        self.pipes, self.processes = [], []
        for _ in range(count):
            parent, child = context.Pipe()
            process = context.Process(target=worker_main, args=(child, scenario, rule), daemon=True)
            process.start()
            child.close()
            self.pipes.append(parent)
            self.processes.append(process)
        for pipe in self.pipes:
            self._result(pipe)

    @staticmethod
    def _result(pipe):
        status, value = pipe.recv()
        if status != "ok":
            raise RuntimeError(f"oracle worker failed: {value}")
        return value

    def call(self, commands):
        for pipe, (command, args) in zip(self.pipes, commands):
            pipe.send((command, args))
        return [self._result(pipe) for pipe in self.pipes[:len(commands)]]

    def close(self):
        for pipe, process in zip(self.pipes, self.processes):
            try:
                pipe.send(("close", ()))
                pipe.recv()
            except (EOFError, OSError):
                pass
            process.join(timeout=60)
            if process.is_alive():
                process.terminate()


def cmd_run(args):
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    meta = {"scenario": args.scenario, "horizon": args.horizon, "candidates": list(NAMES), "hours": args.hours,
            "rule": args.rule or "G36PolicyB2R2"}
    meta_path = out / "meta.json"
    if meta_path.is_file() and json.loads(meta_path.read_text(encoding="utf-8")) != meta:
        raise SystemExit(f"{args.out} holds another search")
    meta_path.write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    log_path = out / "decisions.jsonl"
    done_rows = [json.loads(x) for x in log_path.read_text(encoding="utf-8").splitlines() if x.strip()] \
        if log_path.is_file() else []
    workers = Workers(args.workers, args.scenario, args.rule)
    started = time.time()
    try:
        state = None
        for row in done_rows:                        # replay the recorded choices
            results = workers.call([("advance", (NAMES.index(row["chosen"]),))] * args.workers)
            if len({r["digest"] for r in results}) != 1 or results[0]["digest"] != row["digest"]:
                raise SystemExit(f"replay of decision {row['decision']} does not reproduce the recorded state")
            state = results[0]
        decision = len(done_rows)
        while not (state and state["done"]) and (args.hours is None or decision < args.hours):
            info = workers.call([("info", ())])[0]
            todo = [i for i, ok in enumerate(info["eligible"]) if ok or i == 0]
            shares = [todo[w::args.workers] for w in range(args.workers)]
            sums = {}
            for part in workers.call([("evaluate", (share, args.horizon)) for share in shares]):
                sums.update(part)
            valid = {i: v for i, v in sums.items() if v is not None}
            best = max(valid, key=lambda i: (valid[i], i == 0))        # ties: the rule
            results = workers.call([("advance", (best,))] * args.workers)
            if len({r["digest"] for r in results}) != 1:
                raise SystemExit(f"decision {decision}: the workers' copies disagree")
            state = results[0]
            row = {"decision": decision, "step": info["steps"], "minute_of_day": info["minute_of_day"],
                   "weekday": info["weekday"], "features": info["features"],
                   "sums": {NAMES[i]: sums.get(i) for i in range(len(NAMES))}, "chosen": NAMES[best],
                   "hour_reward": state["reward"], "digest": state["digest"], "done": state["done"]}
            with open(log_path, "a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(row) + "\n")
            decision += 1
            if decision % 24 == 0:
                print(f"{time.strftime('%H:%M:%S')} {args.scenario}: decision {decision}, step {state['steps']}, "
                      f"mean reward so far {state['total'] / state['steps']:+.6f} "
                      f"({(time.time() - started) / max(1, decision - len(done_rows)):.1f} s per decision)", flush=True)
        rows = [json.loads(x) for x in log_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        counts = {name: sum(r["chosen"] == name for r in rows) for name in NAMES}
        rule = workers.call([("baseline", (args.scenario, state["steps"]))])[0]
        oracle_cost, rule_cost = -state["total"] / state["steps"], -rule["reward"] / rule["steps"]
        summary = {**meta, "decisions": len(rows), "steps": state["steps"], "oracle_cost": oracle_cost,
                   "rule_cost": rule_cost, "relative": oracle_cost / rule_cost - 1.0,
                   "chosen": counts, "seconds": round(time.time() - started)}
        (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
        print(json.dumps(summary))
    finally:
        workers.close()


def cmd_tree(args):
    """A depth-limited decision tree of "which candidate" from the features of the recorded decisions."""
    from sklearn.tree import DecisionTreeClassifier, export_text
    from smart_building.policies.b2rl import FEATURES
    x, y = [], []
    for run in args.runs:
        for line in (ROOT / run / "decisions.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                x.append(row["features"])
                y.append(row["chosen"])
    tree = DecisionTreeClassifier(max_depth=args.depth, min_samples_leaf=args.min_leaf, random_state=0).fit(x, y)
    text = export_text(tree, feature_names=list(FEATURES))
    counts = {name: y.count(name) for name in sorted(set(y))}
    t = tree.tree_
    arrays = {"feature": t.feature.tolist(), "threshold": t.threshold.tolist(), "left": t.children_left.tolist(),
              "right": t.children_right.tolist(),
              "label": [str(tree.classes_[int(np.argmax(v))]) for v in t.value[:, 0, :]]}   # policies/b2rl.CandidateTree
    result = {"runs": args.runs, "depth": args.depth, "samples": len(y), "classes": counts,
              "train_accuracy": float(tree.score(x, y)), "rule_share": counts.get("rule", 0) / len(y),
              "nodes": int(t.node_count), "features": list(FEATURES), "tree": arrays, "text": text}
    (ROOT / args.out).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(text)
    print(json.dumps({k: v for k, v in result.items() if k != "text"}))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    p.add_argument("--scenario", required=True, help="train/<building>/<weather> (split-v1-r1) or an eval-set spec")
    p.add_argument("--out", required=True)
    p.add_argument("--horizon", type=int, default=8, help="lookahead hours after the candidate's hour")
    p.add_argument("--workers", type=int, default=len(CANDIDATES))
    p.add_argument("--hours", type=int, help="stop after this many decisions (default: the whole episode)")
    p.add_argument("--rule", help="the base rule (default G36PolicyB2R2; a b2rl_env preset before the rule is final)")
    p = sub.add_parser("tree")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--min-leaf", type=int, default=20)
    args = parser.parse_args()
    {"run": cmd_run, "tree": cmd_tree}[args.command](args)


if __name__ == "__main__":
    main()
