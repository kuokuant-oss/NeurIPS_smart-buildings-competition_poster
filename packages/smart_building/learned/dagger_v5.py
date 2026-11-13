"""DAgger for the second student version j7b-v5 (docs/plan/plan-direction-20261020.md §10, registered 2026-10-20 16:44).

Identical to learned/dagger.py (j7b-v4, whose files stay byte for byte as pinned) except three points:
  1. every iteration trains from a fresh random initialisation with a fresh Adam, on the aggregated data
     D_1..D_i (Ross et al. 2011 Algorithm 3.1; Ash & Adams 2020); replaces REQ-018 item 3. The frozen input
     statistics and log_std of the first iteration are copied into every fresh model. The initialisation seed
     of iteration i is init_seed(seed, i); torch's global RNG is not touched (fork_rng), so a resumed chain is
     still bit for bit an uninterrupted one;
  2. the BC learning rate is the settings' lr, 1e-4 in the j7b-v5 job (robomimic App. D.2; the V1a / V1b screen);
     the rest of D-L09 is unchanged (MSE, batch 32 with replacement, 5 epochs, log_std of the demonstrations);
  3. the B1-R internal state (learned/privileged.py, the WP4 diagnosis probe's list) of the shadow teacher before
     each call is appended to the zone, AHU and hot-water tokens (learned/rollouts_v5.py); its normalisation is the
     existing rule: first-iteration demonstration statistics, positions with std 0 left out (REQ-014, REQ-018 (6)).
Everything else (E, sampling, H, architecture, checkpoints, evaluation iterations, selection rule, resume and
the SHA-256 manifests) is dagger.py's own code, imported, not copied, wherever it does not depend on these points.

Entry point (remote task kind "dagger", entry {"file": "packages/smart_building/learned/dagger_v5.py",
"function": "run_dagger_segment"}): run_dagger_segment(config, state_dir, out_dir, deadline_utc) -> dict, as dagger.py.
The settings identity carries VARIANT, so a j7b-v4 checkpoint is never resumed by this code (and vice versa).
final.json and the result add "student_format": "v5" and the evaluation / ensemble entries that load such a student.
"""
from dataclasses import asdict
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from smart_building.learned import dagger as base
from smart_building.learned.dataset import Rollout
from smart_building.learned.privileged import COUNTS, PRIVILEGED_TYPES

VARIANT = "j7b-v5: fresh init and Adam every iteration, BC lr from settings, B1-R state inputs (plan-direction-20261020 §10)"
STUDENT_FORMAT = "v5"
EVALUATION_ENTRY = {"file": "packages/smart_building/learned/rollouts_v5.py", "function": "run_evaluation"}
ENSEMBLE_ENTRY = {"file": "packages/smart_building/policies/b1l_ensemble_v5.py", "function": "ensemble_policy_class"}
BUFFER_PREFIXES = ("input_index_", "input_mean_", "input_std_", "log_std_")


class DaggerSettingsV5(base.DaggerSettings):
    """dagger.DaggerSettings; the identity also names the variant (no other new setting)."""

    def identity(self):
        return {**super().identity(), "variant": VARIANT}


def init_seed(seed, iteration):
    """The torch seed of the fresh model of iteration `iteration` (1-based)."""
    return int(seed) * 1_000_003 + int(iteration)


def parameters_sha256(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class RolloutV5(Rollout):
    """dataset.Rollout of a j7b-v5 rollout file: the token inputs carry the stored B1-R state."""

    def __init__(self, path):
        super().__init__(path)
        with np.load(path, allow_pickle=False) as data:
            self.privileged = {k: data[f"priv_{k}"] for k in PRIVILEGED_TYPES}
        for k in PRIVILEGED_TYPES:
            count = self.spec[f"dynamic_{k}"].shape[0]
            if self.privileged[k].shape != (len(self.steps), count, COUNTS[k]):
                raise ValueError(f"{path}: priv_{k} shape {self.privileged[k].shape}")

    def feature_counts(self):
        # The state features count as extra static features of the token (input = H x dynamic + static).
        return {k: (d, s + COUNTS[k]) for k, (d, s) in super().feature_counts().items()}

    def inputs(self, positions, history):
        out = super().inputs(positions, history)
        index = np.asarray(list(positions))
        for k in PRIVILEGED_TYPES:
            out[k] = np.concatenate([out[k], self.privileged[k][index].astype(np.float32)], axis=-1)
        return out


class ChainV5(base.Chain):
    def dataset(self):
        from smart_building.learned.dataset import Dataset
        loaded = len(self.data.rollouts) if self.data is not None else 0
        if self.data is None or loaded < len(self.rollouts):
            previous = self.data.rollouts if self.data is not None else []
            new = [RolloutV5(self.out_dir / r["file"]) for r in self.rollouts[loaded:]]
            self.data = Dataset(previous + new)
        return self.data


def collect(chain, workers):
    """Step 1: the iteration's E rollouts (rollouts_v5.run_rollout)."""
    from smart_building.learned import rollouts_v5
    from smart_building.learned.model import state_bytes
    s, i = chain.settings, chain.iteration + 1
    specs = s.rollout_scenarios[(i - 1) * s.rollouts_per_iteration:i * s.rollouts_per_iteration]
    student = None
    if i > 1:
        cpu_model = chain.model.to("cpu")
        student = {"config": asdict(chain.student_config()), "state": state_bytes(cpu_model),
                   "input_dims": dict(cpu_model.input_dims)}
        chain.model.to(s.device)
    tasks = []
    for r, spec in enumerate(specs, start=1):
        name = f"data/rollout-{i:04d}-{r:02d}.npz"
        tasks.append({"scenario": spec, "label": base.scenario_label(spec),
                      "scenario_dir": str(base.resolve_scenario(spec, s, "train")), "scenario_root": s.scenario_root,
                      "code_root": s.code_root, "teacher": base.select_teacher(s, spec), "student": student,
                      "history": chain.student_config().history, "sample_every": s.sample_every,
                      "steps_cap": s.steps_cap, "iteration": i, "rollout": r, "file": name,
                      "out_path": str(chain.out_dir / name), "threads": 1})
    return [base.plain(r) for r in workers.map(rollouts_v5.run_rollout, tasks)]


def first_statistics(chain, records):
    """Step 2 (iteration 1): statistics of the augmented inputs over the demonstrations (dagger.first_statistics)."""
    from smart_building.learned.dataset import action_statistics, input_statistics
    from smart_building.learned.privileged import FEATURES, PRIVILEGED_TYPES
    demonstrations = [RolloutV5(chain.out_dir / r["file"]) for r in records]
    counts = demonstrations[0].feature_counts()
    if any(d.feature_counts() != counts for d in demonstrations):
        raise ValueError("the rollouts' token feature counts differ between buildings")
    history = chain.student_config().history
    mean, std, zero = input_statistics(demonstrations, history)
    keep = {k: np.flatnonzero(std[k] > 0.0) for k in std}                       # REQ-018 (6)
    chain.build_model(counts, keep)
    chain.model.set_normalization({k: mean[k][keep[k]] for k in keep},
                                  {k: std[k][keep[k]] for k in keep})            # REQ-014 (1, 2)
    chain.model.set_log_std(action_statistics(demonstrations))                   # D-L09, REQ-018 (7)
    privileged = {}
    for k in PRIVILEGED_TYPES:
        first = len(std[k]) - len(FEATURES[k])
        privileged[k] = [{"name": name, "mean": float(mean[k][first + j]), "std": float(std[k][first + j]),
                          "kept": bool(std[k][first + j] > 0.0)} for j, name in enumerate(FEATURES[k])]
    chain.statistics = {"input_dims_excluded": zero, "input_dims_kept": {k: int(len(v)) for k, v in keep.items()},
                        "samples": int(sum(len(d) for d in demonstrations)), "privileged": privileged}


def fresh_model(chain, iteration):
    """§10 item 1: a new randomly initialised model (seed init_seed(seed, iteration)) carrying the frozen
    first-iteration statistics and log_std of the chain's model, and a new Adam at the settings' lr."""
    import torch
    from smart_building.learned.dataset import input_dims
    from smart_building.learned.model import EntityTransformer, keep_from_state
    buffers = {name: value.detach().cpu().clone() for name, value in chain.model.state_dict().items()
               if name.startswith(BUFFER_PREFIXES)}
    config = chain.student_config()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(init_seed(chain.settings.seed, iteration))
        model = EntityTransformer(input_dims(chain.feature_counts, config.history), config, keep_from_state(buffers))
    missing = model.load_state_dict(buffers, strict=False)
    if missing.unexpected_keys or any(name.startswith(BUFFER_PREFIXES) for name in missing.missing_keys):
        raise RuntimeError(f"frozen statistics not carried over: {missing}")
    model = model.to(chain.settings.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=chain.settings.lr)      # D-L09 with §10 item 2
    return model, optimizer


def train(chain):
    """Step 3: BC on D_1..D_i from a fresh model and Adam (§10 items 1, 2); the minibatch rule is dagger.train's."""
    import torch
    from smart_building.learned.dataset import minibatch_loss
    s = chain.settings
    iteration = chain.iteration + 1
    chain.model, chain.optimizer = fresh_model(chain, iteration)
    init = {"seed": init_seed(s.seed, iteration), "parameters_sha256": parameters_sha256(chain.model), "lr": s.lr}
    torch.set_num_threads(s.train_threads)
    data = chain.dataset()
    n = data.num_samples
    history = chain.student_config().history
    chain.model.train()
    epoch_losses = []
    for _ in range(s.epochs):
        losses = []
        for _ in range(n // s.batch_size):
            indices = chain.generator.choice(n, size=s.batch_size)              # with replacement (mjrl)
            chain.optimizer.zero_grad()
            loss = minibatch_loss(chain.model, data, indices, history, s.device)
            loss.backward()
            chain.optimizer.step()
            losses.append(float(loss.detach()))
            chain.counters["minibatches"] += 1
        epoch_losses.append(float(np.mean(losses)) if losses else None)
    chain.model.eval()
    return {"samples": n, "minibatches_per_epoch": n // s.batch_size, "epoch_mean_loss": epoch_losses, "init": init}


def evaluate(chain, workers):
    """Step 4: save the student (privileged.save_student_v5) and score it on the INNER episodes (rollouts_v5)."""
    from smart_building.learned import rollouts_v5
    from smart_building.learned.privileged import save_student_v5
    s, i = chain.settings, chain.iteration
    name = f"models/student-{i:04d}.pt"
    path = chain.out_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    save_student_v5(path, chain.model.to("cpu"), teacher=base.select_teacher(s, None),
                    extra={"iteration": i, "statistics": chain.statistics, "variant": VARIANT})
    chain.model.to(s.device)
    chain.models[i] = {"file": name, "sha256": base.sha256_file(path)}
    tasks = [{"scenario": spec, "label": base.scenario_label(spec),
              "scenario_dir": str(base.resolve_scenario(spec, s, s.eval_purpose)),
              "scenario_root": s.scenario_root, "code_root": s.code_root, "teacher": base.select_teacher(s, spec),
              "model_path": str(path), "iteration": i, "steps_cap": s.steps_cap, "threads": 1}
             for spec in s.eval_scenarios]
    rows = [base.plain(r) for r in workers.map(rollouts_v5.run_evaluation, tasks)]
    chain.evaluations.extend(rows)
    return rows


def run_iteration(chain, workers):
    """dagger.run_iteration with the v5 steps."""
    import torch
    timing = {}
    started = time.time()
    if chain.iteration == 0:
        torch.manual_seed(chain.settings.seed)
    records = collect(chain, workers)
    timing["rollouts_s"] = time.time() - started
    if chain.iteration == 0:
        t0 = time.time()
        first_statistics(chain, records)
        timing["statistics_s"] = time.time() - t0
    chain.rollouts.extend(records)
    chain.counters["rollouts"] += len(records)
    chain.counters["samples"] += sum(r["samples"] for r in records)
    chain.counters["env_steps"] += sum(r["steps"] for r in records)
    chain.counters["student_fallback_steps"] += sum(r["student_fallback_steps"] for r in records)
    t0 = time.time()
    bc = train(chain)
    timing["training_s"] = time.time() - t0
    chain.iteration += 1
    evaluation = None
    if chain.iteration in chain.settings.eval_iterations:
        t0 = time.time()
        rows = evaluate(chain, workers)
        timing["evaluation_s"] = time.time() - t0
        evaluation = {"mean_cost": float(np.mean([r["cost"] for r in rows])) if rows else None,
                      "episodes": len(rows)}
    timing["iteration_s"] = time.time() - started
    log = {"iteration": chain.iteration, "variant": VARIANT,
           "rollouts": [{k: r[k] for k in ("label", "actor", "steps", "mean_reward", "samples",
                                           "student_fallback_steps", "student_latency", "seconds")} for r in records],
           "bc": bc, "evaluation": evaluation, "timing": timing, "counters": dict(chain.counters)}
    log = base.plain(log)
    chain.history.append(log)
    return log


def run_dagger_segment(config, state_dir, out_dir, deadline_utc):
    """One segment of a j7b-v5 DAgger chain (dagger.run_dagger_segment with the v5 steps)."""
    import os
    import torch
    settings = DaggerSettingsV5.from_config(config)
    if settings.device.startswith("cuda"):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    previous = (torch.are_deterministic_algorithms_enabled(), torch.get_num_threads())
    torch.use_deterministic_algorithms(True)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    deadline = base.parse_deadline(deadline_utc)
    chain = ChainV5(settings, out_dir)
    if state_dir:
        chain.load(state_dir)
    workers = base.Workers(settings.workers)
    done_here, stopped = 0, None
    try:
        while chain.iteration < settings.iterations:
            if settings.max_iterations_per_segment is not None and done_here >= settings.max_iterations_per_segment:
                stopped = "max_iterations_per_segment"
                break
            durations = [h["timing"]["iteration_s"] for h in chain.history]
            estimate = max(durations) if durations else settings.iteration_estimate_seconds
            if deadline is not None and estimate is not None and time.time() + settings.deadline_margin * estimate > deadline:
                stopped = "deadline"
                break
            log = run_iteration(chain, workers)
            chain.save()
            base.append_jsonl(out_dir / "iterations.jsonl", log)
            for row in chain.evaluations[len(chain.evaluations) - (log["evaluation"] or {}).get("episodes", 0):]:
                base.append_jsonl(out_dir / "evaluations.jsonl", row)
            done_here += 1
    finally:
        workers.close()
        torch.use_deterministic_algorithms(previous[0])
        torch.set_num_threads(previous[1])
    finished = chain.iteration >= settings.iterations
    result = {"finished": finished, "iteration": chain.iteration, "iterations_this_segment": done_here,
              "stopped": stopped if not finished else "finished",
              "checkpoint": base.checkpoint_name(chain.iteration) if chain.iteration else None,
              "counters": chain.counters, "variant": VARIANT, "student_format": STUDENT_FORMAT}
    if finished:
        best, means = base.best_iteration(chain)
        final = {"best_iteration": best, "mean_cost_by_iteration": means, "model": chain.models.get(best),
                 "teacher": base.select_teacher(settings, None), "student_format": STUDENT_FORMAT,
                 "evaluation_entry": EVALUATION_ENTRY, "ensemble_entry": ENSEMBLE_ENTRY}
        base.write_json(out_dir / "final.json", final)
        result.update(final)
    return result
