"""j7b-v5 DAgger rollouts and checkpoint evaluations (docs/plan/plan-direction-20261020.md §10): learned/rollouts.py
with the B1-R internal state (learned/privileged.py) as extra student inputs; everything else unchanged.

Rollout: as rollouts.run_rollout. Before each call of the shadow teacher (the B1-R object, D-L14) its state is
read (privileged.policy_features); at every sampled step it is stored next to the label (arrays priv_zone,
priv_ahu, priv_hws of the rollout file, aligned with sample_steps), and from the second iteration the student
acts on the observation plus this state (StudentNetworkV5). Importable by package name so a spawn pool can
pickle the functions.

Evaluation: as rollouts.run_evaluation with privileged.StudentPolicyV5 (the switches' own B1-R state).
"""
import json
import os
from pathlib import Path
import time

import numpy as np

from smart_building.learned import rollouts as base
from smart_building.learned.privileged import (COUNTS, FEATURES, PRIVILEGED_TYPES, StudentNetworkV5, StudentPolicyV5,
                                               load_student_v5, policy_features)

REJECTED = base.REJECTED
ROLLOUT_FORMAT = "b1-l-rollout/1"        # dataset.Rollout reads these files; the state arrays are extra entries


def write_rollout(path, rows, sample_steps, labels, spec, meta, privileged):
    """dataset.write_rollout plus the B1-R state arrays priv_<type> (samples, count, features)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial.npz")
    arrays = {"rows": np.asarray(rows, dtype=np.float32), "sample_steps": np.asarray(sample_steps, dtype=np.int64),
              "labels": np.asarray(labels, dtype=np.float32),
              "meta": np.array(json.dumps({"format": ROLLOUT_FORMAT, **meta}, sort_keys=True))}
    arrays.update({f"spec_{k}": np.asarray(v) for k, v in spec.items()})
    arrays.update({f"priv_{k}": np.asarray(privileged[k], dtype=np.float32) for k in PRIVILEGED_TYPES})
    with open(temporary, "wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, path)


def run_rollout(task):
    """One j7b-v5 DAgger rollout; writes task["out_path"] and returns its record (no arrays)."""
    import torch
    from smart_building.learned.model import StudentConfig, load_state_bytes
    from smart_building.learned.tokens import TokenLayout
    torch.set_num_threads(task.get("threads", 1))
    started = time.perf_counter()
    env, metadata = base.load_environment(task["scenario_dir"], task["scenario_root"])
    spec = env.action_spec()
    continuous = spec["continuous_action"]
    low, high = np.asarray(continuous.low, np.float32), np.asarray(continuous.high, np.float32)
    observation, _ = env.reset()
    teacher = base.load_class(task["code_root"], task["teacher"])(metadata, spec)          # D-L14: new instance
    student = None
    if task.get("student") is not None:
        config = StudentConfig(**task["student"]["config"])
        student = StudentNetworkV5(metadata, spec, config, load_state_bytes(task["student"]["state"]),
                                   task["student"]["input_dims"])
    layout = TokenLayout(metadata, spec, task["history"])
    rows, steps, labels, durations = [np.asarray(observation, np.float32)], [], [], []
    privileged = {k: [] for k in PRIVILEGED_TYPES}
    total, step, student_fallbacks = 0.0, 0, 0
    terminated = truncated = False
    while True:
        features = policy_features(teacher)                       # §10 item 3: the state before this call
        label = teacher.get_action(observation)
        if step % task["sample_every"] == 0:
            steps.append(step)
            labels.append(np.asarray(label["continuous_action"], np.float32))
            for k in PRIVILEGED_TYPES:
                privileged[k].append(features[k])
        if student is None:
            action = label                                        # beta_1 = 1 (D-L08)
        else:
            t0 = time.perf_counter()
            try:
                mean = student.mean_action(observation, features)
                if not np.all(np.isfinite(mean)):
                    raise FloatingPointError("non-finite student action")
                action = {"continuous_action": np.clip(mean, low, high).astype(np.float32),
                          "discrete_action": label["discrete_action"]}                       # D-L07
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
            if hasattr(teacher, "diagnostics") else None,
            "privileged": {k: list(FEATURES[k]) for k in PRIVILEGED_TYPES}}
    write_rollout(task["out_path"], np.stack(rows), steps, np.stack(labels), layout.spec(), meta,
                  {k: np.stack(privileged[k]).reshape(len(steps), -1, COUNTS[k]) for k in PRIVILEGED_TYPES})
    return {**meta, "file": task["file"], "sha256": base.sha256_file(task["out_path"]),
            "student_latency": base._latency(durations), "seconds": time.perf_counter() - started}


def run_evaluation(task):
    """One checkpoint-evaluation episode (gateway loop) of the saved j7b-v5 student in task["model_path"]."""
    import torch
    from smart_building.evaluation.gateway_episode import run_gateway_episode
    torch.set_num_threads(task.get("threads", 1))
    config, state_dict, input_dims, _ = load_student_v5(task["model_path"])
    teacher_class = base.load_class(task["code_root"], task["teacher"])
    built = []

    class EvaluatedStudent(StudentPolicyV5):
        def __init__(self, building_meta_data, action_spec):
            super().__init__(building_meta_data, action_spec, config, state_dict, teacher_class, input_dims)
            self.durations = []
            built.append(self)

        def get_action(self, obs):
            t0 = time.perf_counter()
            action = super().get_action(obs)
            self.durations.append(time.perf_counter() - t0)
            return action

    with base.working_directory(task["scenario_root"]):
        row = run_gateway_episode(task["scenario_dir"], EvaluatedStudent, policy_key="b1_l",
                                  label=task["label"], max_steps=task.get("steps_cap"))
    policy = built[-1]
    row.update(iteration=task["iteration"], scenario=task["label"], cost=-row["mean_reward"],
               student_fallback_steps=policy.fallback_steps, get_action=base._latency(policy.durations),
               student_format="v5")
    return row
