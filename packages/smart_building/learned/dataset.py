"""DAgger data of the student B1-L (WP4 4.4: D-L08, D-L09; stage45 §2): stored rollouts, the
first-iteration statistics and the behaviour-cloning minibatches.

A rollout file keeps the observation of every step (rows), the sampled steps (every 10th,
stage45 §2) with the teacher's continuous action as the label, and the layout tables
(tokens.TokenLayout.spec) that rebuild token inputs and targets without the metadata, so
any history length H can be cut from the same rows exactly as the student sees them.

Statistics (REQ-014 (1, 2) as settled): per token type and input feature, the mean and the
population standard deviation (np.std, mjrl `behavior_cloning.py:52-61`) over every token of
the type in every sample of the first-iteration demonstrations; the action standard
deviation for log_std likewise per type and output (mjrl `set_variance_with_data`).

BC (D-L09, mjrl `behavior_cloning.py` fit/mse_loss at 3871d93): each epoch runs
floor(N / batch) minibatches whose indices are drawn with replacement. The loss weights
every sample (one step of one building) equally (REQ-018 (4), 2026-10-17): the mean squared
error over the sample's own action dimensions, then the mean over the minibatch's samples.
"""
import json
import os
from pathlib import Path

import numpy as np

from smart_building.learned.tokens import OUTPUT_DIMS, TOKEN_TYPES, gather_targets, token_inputs

ROLLOUT_FORMAT = "b1-l-rollout/1"


def write_rollout(path, rows, sample_steps, labels, spec, meta):
    """Write one rollout atomically (np.savez, uncompressed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial.npz")
    arrays = {"rows": np.asarray(rows, dtype=np.float32), "sample_steps": np.asarray(sample_steps, dtype=np.int64),
              "labels": np.asarray(labels, dtype=np.float32),
              "meta": np.array(json.dumps({"format": ROLLOUT_FORMAT, **meta}, sort_keys=True))}
    arrays.update({f"spec_{k}": np.asarray(v) for k, v in spec.items()})
    with open(temporary, "wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, path)


class Rollout:
    """One stored rollout, loaded into memory."""

    def __init__(self, path):
        with np.load(path, allow_pickle=False) as data:
            self.rows = data["rows"]
            self.steps = data["sample_steps"]
            self.labels = data["labels"]
            self.spec = {k[len("spec_"):]: data[k] for k in data.files if k.startswith("spec_")}
            self.meta = json.loads(str(data["meta"]))
        if self.meta.get("format") != ROLLOUT_FORMAT:
            raise ValueError(f"{path}: not a {ROLLOUT_FORMAT} file")
        self.targets_all = gather_targets(self.spec, self.labels)          # {type: (samples, count, out)}

    def __len__(self):
        return len(self.steps)

    def feature_counts(self):
        """(dynamic features, static features) per token type."""
        return {k: (self.spec[f"dynamic_{k}"].shape[1], self.spec[f"static_{k}"].shape[1]) for k in TOKEN_TYPES}

    def inputs(self, positions, history):
        """{type: (len(positions), count, input_dim)} raw token inputs of the sampled steps."""
        per_sample = [token_inputs(self.spec, self.rows, int(self.steps[p]), history) for p in positions]
        return {k: np.stack([s[k] for s in per_sample]) for k in TOKEN_TYPES}

    def targets(self, positions):
        return {k: self.targets_all[k][np.asarray(positions)] for k in self.targets_all}


class Dataset:
    """The aggregated DAgger data (D-L08): every rollout so far, indexed sample by sample."""

    def __init__(self, rollouts):
        self.rollouts = list(rollouts)
        self.offsets = np.cumsum([0] + [len(r) for r in self.rollouts])

    @property
    def num_samples(self):
        return int(self.offsets[-1])

    def locate(self, index):
        r = int(np.searchsorted(self.offsets, index, side="right") - 1)
        return r, int(index - self.offsets[r])


def input_dims(feature_counts, history):
    return {k: history * dynamic + static for k, (dynamic, static) in feature_counts.items()}


def input_statistics(rollouts, history, chunk=64):
    """REQ-014 (1, 2): {type: mean}, {type: std} over every token of the type in every sample
    (two passes in float64; population standard deviation). Also returns, per type, the number
    of input features whose standard deviation is 0 (they are divided by 1e-8 later)."""
    sums, counts = {}, {}
    for rollout in rollouts:
        for start in range(0, len(rollout), chunk):
            inputs = rollout.inputs(range(start, min(start + chunk, len(rollout))), history)
            for k, x in inputs.items():
                flat = x.reshape(-1, x.shape[-1]).astype(np.float64)
                sums[k] = sums.get(k, 0.0) + flat.sum(axis=0)
                counts[k] = counts.get(k, 0) + flat.shape[0]
    mean = {k: sums[k] / counts[k] for k in sums}
    squares = {}
    for rollout in rollouts:
        for start in range(0, len(rollout), chunk):
            inputs = rollout.inputs(range(start, min(start + chunk, len(rollout))), history)
            for k, x in inputs.items():
                flat = x.reshape(-1, x.shape[-1]).astype(np.float64) - mean[k]
                squares[k] = squares.get(k, 0.0) + (flat * flat).sum(axis=0)
    std = {k: np.sqrt(squares[k] / counts[k]) for k in squares}
    zero = {k: int(np.sum(std[k] == 0.0)) for k in std}
    return mean, std, zero


def action_statistics(rollouts):
    """Per type and output, the population standard deviation of the demonstrations' actions."""
    out = {}
    for k in (k for k in TOKEN_TYPES if OUTPUT_DIMS[k]):
        values = np.concatenate([r.targets_all[k].reshape(-1, OUTPUT_DIMS[k]) for r in rollouts]).astype(np.float64)
        out[k] = values.std(axis=0)
    return out


def minibatch_loss(model, dataset, indices, history, device):
    """REQ-018 (4): mean over the minibatch's samples of each sample's MSE over its own action
    dimensions (every building step weighs the same).

    Samples are grouped by rollout (one building each), so each group is a dense batch
    without padding; groups run in rollout order, so the loss is deterministic.
    """
    import torch
    groups = {}
    for index in indices:
        r, position = dataset.locate(int(index))
        groups.setdefault(r, []).append(position)
    per_sample = []
    for r in sorted(groups):
        rollout, positions = dataset.rollouts[r], groups[r]
        inputs = {k: torch.from_numpy(v).to(device) for k, v in rollout.inputs(positions, history).items()}
        targets = {k: torch.from_numpy(v).to(device) for k, v in rollout.targets(positions).items()}
        outputs = model(inputs)
        squared, elements = None, 0
        for k in sorted(outputs):
            diff = outputs[k] - targets[k]
            term = (diff * diff).flatten(1).sum(dim=1)                      # (samples,)
            squared = term if squared is None else squared + term
            elements += diff[0].numel()                                     # the sample's own dimensions
        per_sample.append(squared / elements)
    return torch.cat(per_sample).mean()
