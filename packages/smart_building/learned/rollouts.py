"""DAgger rollouts and checkpoint evaluations of the student B1-L (WP4 4.4), one per call, run in a
worker process (spawn) or in-process; importable by its package name so a pool can pickle them.

Rollout (D-L08, D-L12, D-L14; stage45 §2), on the Kaggle gateway's loop
(smart_building.evaluation.gateway_episode): the scenario is loaded with the working directory
at its scenario root, env.reset(), then a new teacher instance is built (D-L14) and called once
per step with the observation the student receives; its continuous action is the label and its
on/off actions are executed (D-L07). In the first iteration the teacher acts (beta_1 = 1); from
the second the student's mean action is executed (no perturbation, D-L12), with the teacher's
action on a step where the network fails (D-T02). Every 10th step (t % 10 == 0) is sampled.

Evaluation (stage45 §2, checkpoint iterations): one episode of run_gateway_episode with the
student (mean action, teacher on/off), as S4 scores it.
"""
import contextlib
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import re
import sys
import time

import numpy as np

REJECTED = -math.inf


@contextlib.contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def load_class(code_root, source):
    """A class from a file of the code snapshot ({"file", "class"}), without the policy registry."""
    path = Path(code_root) / source["file"]
    name = "sb_dagger_" + re.sub(r"\W", "_", source["file"]) + "_" + source["class"]
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return getattr(module, source["class"])


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_environment(scenario_dir, scenario_root):
    """(env, metadata) as the gateway loads them (gateway.py:308-340), parsing config.gin from the root."""
    from smart_buildings import evaluate
    scenario_dir = Path(scenario_dir)
    with working_directory(scenario_root):
        env = evaluate.load_hybrid_environment(scenario_dir / "config.gin")
        env.metrics_path = None
        env._occupancy_normalization_constant = 125.0
        metadata = evaluate.create_building_meta_data(env, scenario_dir)
    return env, metadata


def _latency(durations):
    if not durations:
        return None
    d = np.asarray(durations) * 1000.0
    return {"mean_ms": float(d.mean()), "p99_ms": float(np.percentile(d, 99)), "max_ms": float(d.max())}


def run_rollout(task):
    """One DAgger rollout; writes task["out_path"] and returns its record (no arrays)."""
    import torch
    from smart_building.learned.dataset import write_rollout
    from smart_building.learned.model import StudentConfig, StudentNetwork, load_state_bytes
    from smart_building.learned.tokens import TokenLayout
    torch.set_num_threads(task.get("threads", 1))
    started = time.perf_counter()
    env, metadata = load_environment(task["scenario_dir"], task["scenario_root"])
    spec = env.action_spec()
    continuous = spec["continuous_action"]
    low, high = np.asarray(continuous.low, np.float32), np.asarray(continuous.high, np.float32)
    observation, _ = env.reset()
    teacher = load_class(task["code_root"], task["teacher"])(metadata, spec)          # D-L14: new instance
    student = None
    if task.get("student") is not None:
        config = StudentConfig(**task["student"]["config"])
        student = StudentNetwork(metadata, spec, config, load_state_bytes(task["student"]["state"]))
    layout = student.layout if student is not None else TokenLayout(metadata, spec, task["history"])
    rows, steps, labels, durations = [np.asarray(observation, np.float32)], [], [], []
    total, step, student_fallbacks = 0.0, 0, 0
    terminated = truncated = False
    while True:
        # D-L14: same observation. The teacher's own-command state follows its unexecuted commands
        # (D-R25 as corrected); REQ-018 (9) is decided at K1 (J7a: C3; J7b: the readback variant
        # if WP3 finds it no worse than C3 by more than 1 SE), chosen through config["teacher"].
        label = teacher.get_action(observation)
        if step % task["sample_every"] == 0:
            steps.append(step)
            labels.append(np.asarray(label["continuous_action"], np.float32))
        if student is None:
            action = label                                               # beta_1 = 1 (D-L08)
        else:
            t0 = time.perf_counter()
            try:
                mean = student.mean_action(observation)
                if not np.all(np.isfinite(mean)):
                    raise FloatingPointError("non-finite student action")
                action = {"continuous_action": np.clip(mean, low, high).astype(np.float32),
                          "discrete_action": label["discrete_action"]}  # D-L07
            except Exception:  # noqa: BLE001 - D-T02
                student_fallbacks += 1
                action = label
            durations.append(time.perf_counter() - t0)
        observation, reward, terminated, truncated, _ = env.step(action)
        reward = float(reward)
        if reward == REJECTED:
            raise RuntimeError(f"{task['label']} step {step}: the building refused the action")
        total += reward
        step += 1
        if terminated or truncated or (task.get("steps_cap") and step >= task["steps_cap"]):
            break
        rows.append(np.asarray(observation, np.float32))
    env.close()
    meta = {"label": task["label"], "scenario": task["scenario"], "iteration": task["iteration"],
            "rollout": task["rollout"], "actor": "teacher" if student is None else "student",
            "teacher": task["teacher"], "steps": step, "total_reward": total, "mean_reward": total / step,
            "samples": len(steps), "student_fallback_steps": student_fallbacks,
            "teacher_fallback_steps": int(teacher.diagnostics.get("fallback_steps", 0))
            if hasattr(teacher, "diagnostics") else None}
    write_rollout(task["out_path"], np.stack(rows), steps, np.stack(labels), layout.spec(), meta)
    return {**meta, "file": task["file"], "sha256": sha256_file(task["out_path"]),
            "student_latency": _latency(durations), "seconds": time.perf_counter() - started}


def run_evaluation(task):
    """One checkpoint-evaluation episode (gateway loop) of the saved student in task["model_path"]."""
    import torch
    from smart_building.evaluation.gateway_episode import run_gateway_episode
    from smart_building.learned.model import StudentPolicy, load_student
    torch.set_num_threads(task.get("threads", 1))
    config, state_dict, _ = load_student(task["model_path"])
    teacher_class = load_class(task["code_root"], task["teacher"])
    built = []

    class EvaluatedStudent(StudentPolicy):
        def __init__(self, building_meta_data, action_spec):
            super().__init__(building_meta_data, action_spec, config, state_dict, teacher_class)
            self.durations = []
            built.append(self)

        def get_action(self, obs):
            t0 = time.perf_counter()
            action = super().get_action(obs)
            self.durations.append(time.perf_counter() - t0)
            return action

    with working_directory(task["scenario_root"]):
        row = run_gateway_episode(task["scenario_dir"], EvaluatedStudent, policy_key="b1_l",
                                  label=task["label"], max_steps=task.get("steps_cap"))
    policy = built[-1]
    row.update(iteration=task["iteration"], scenario=task["label"], cost=-row["mean_reward"],
               student_fallback_steps=policy.fallback_steps, get_action=_latency(policy.durations))
    return row
