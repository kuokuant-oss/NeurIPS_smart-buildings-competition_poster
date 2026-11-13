"""DAgger for the student B1-L (WP4 4.4: D-L08-D-L10, D-L12, D-L14, D-L15; docs/plan/baseline-v1-stage45.md §2).

Entry point (remote task kind "dagger", remote/runner.py run_dagger):

    run_dagger_segment(config, state_dir, out_dir, deadline_utc) -> dict

runs as many complete iterations as fit before deadline_utc (ISO UTC; None = no limit),
checkpointing at every iteration boundary into out_dir, and returns
{"finished", "iteration", "stopped", "checkpoint", ...}. state_dir is the out_dir of the
previous segment (read only) or None for a new chain; its newest checkpoint is checked
against its SHA-256 manifest, and the files it refers to are copied into out_dir, so every
segment's out_dir holds the whole state of the chain.

Iteration i (D-L08: pi_i = beta_i pi* + (1 - beta_i) pi_hat_i, beta_i = 1(i = 1)):
  1. E rollouts (stage45 §2: E = 4) on the next E scenarios of config["rollout_scenarios"]
     (never reused, D-L15); i = 1: the teacher acts (demonstrations); i >= 2: the student's
     mean action (D-L08, D-L12) labelled by a shadow teacher (D-L14); on/off from the teacher
     (D-L07). Every 10th step is kept (stage45 §2). Rollouts run in config["workers"] processes.
  2. i = 1: the input statistics and log_std from these demonstrations, frozen in the model
     (REQ-014 as settled, model.py); input features whose standard deviation is 0 in them are
     left out of the inputs and counted per token type (REQ-018 (6)).
  3. BC on all data so far, continuing the same model and Adam state (REQ-018 (3)) (D-L09: MSE
     with every sample weighted equally, REQ-018 (4); Adam lr 1e-3, batch 32, 5 epochs; dataset.py).
  4. i in eval_iterations (stage45 §2: 5, 10, 15, 20): the student just trained is saved as
     models/student-<i>.pt and scored on config["eval_scenarios"] (the fold's INNER episodes).
  5. Checkpoint ckpt-<i>: model, optimiser, every RNG state, the statistics (model buffers),
     the dataset (rollout files with SHA-256), counters and logs; the last 2 are kept.
After the last iteration, final.json names the evaluated iteration with the lowest mean cost.

Teacher (D-L13, user decision 2026-10-17): the B1-R champion only (C3 = G36PolicyV3Ti900 now);
config["teacher"] = {"selection": "single", "candidates": [{"file", "class"}]}. The selection
hook takes other rules, but "single" is the only one implemented.

Determinism: the simulator, the teacher and CPU inference are deterministic; model init, the
minibatch draws (numpy PCG64) and torch use seeded generators whose states are checkpointed;
torch.use_deterministic_algorithms is on. A resumed chain therefore equals an uninterrupted
one bit for bit (tests/test_learned_dagger.py) for the same thread count and device; on
CUDA, CUBLAS_WORKSPACE_CONFIG is set before the first CUDA call (to be confirmed in J7a).

REQ-018 (2026-10-17): H = 64; tanh output against the normalised teacher action; model and Adam
carried across iterations; per-sample equal loss weight; "iteration i" = trained on D_1..D_i;
zero-std inputs left out; log_std from the first iteration; static features as defined; the
shadow teacher is C3 for J7a (item 9 is decided at K1).
"""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, fields
import datetime as dt
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np

STATE_FORMAT = "b1-l-dagger/1"
DEFAULT_TEACHER = {"selection": "single",
                   "candidates": [{"file": "packages/smart_building/policies/g36.py", "class": "G36PolicyV3Ti900"}]}
# Settings that define a chain: a resumed segment must have the same values.
IDENTITY = ("iterations", "rollouts_per_iteration", "sample_every", "epochs", "batch_size", "lr", "eval_iterations",
            "seed", "student", "teacher", "rollout_scenarios", "eval_scenarios", "steps_cap", "train_threads",
            "device")


@dataclass
class DaggerSettings:
    iterations: int = 20                       # D-L08, D-L15
    rollouts_per_iteration: int = 4            # stage45 §2: E = 4
    sample_every: int = 10                     # stage45 §2
    epochs: int = 5                            # D-L09
    batch_size: int = 32                       # D-L09
    lr: float = 1e-3                           # D-L09
    eval_iterations: tuple = (5, 10, 15, 20)   # stage45 §2
    seed: int = 0
    student: dict = field(default_factory=dict)        # StudentConfig fields (model.py defaults otherwise)
    teacher: dict = field(default_factory=lambda: json.loads(json.dumps(DEFAULT_TEACHER)))
    rollout_scenarios: list = field(default_factory=list)   # remote-style specs, E per iteration, in order
    eval_scenarios: list = field(default_factory=list)      # the fold's INNER episodes (stage45 §2)
    steps_cap: int | None = None               # cap on every rollout and evaluation episode (smoke runs, tests)
    workers: int = 1                           # rollout / evaluation processes (E = 4 on Kaggle)
    train_threads: int = 1                     # torch threads while training
    device: str = "cpu"                        # "cuda" on a GPU kernel (J7a)
    keep_checkpoints: int = 2
    deadline_margin: float = 1.25              # start an iteration only if margin x the longest one still fits
    iteration_estimate_seconds: float | None = None   # used before the chain has a measured iteration
    max_iterations_per_segment: int | None = None     # tests and manual stops
    scenario_root: str | None = None           # working directory for config.gin (overlay or starter-kit root)
    code_root: str | None = None               # code snapshot root (teacher file, split files)
    remote_mode: str | None = None             # passed by the remote runner
    eval_purpose: str = "tune"                 # scenarios.factory purpose of the INNER episodes
    meta: dict = field(default_factory=dict)   # free-form job information (fold, job id); not used

    @classmethod
    def from_config(cls, config):
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(config) - known)
        if unknown:
            raise KeyError(f"unknown DAgger settings {unknown}")
        values = dict(config)
        if "eval_iterations" in values:
            values["eval_iterations"] = tuple(int(i) for i in values["eval_iterations"])
        settings = cls(**values)
        root = Path(__file__).resolve().parents[3]
        settings.code_root = str(Path(settings.code_root) if settings.code_root else root)
        if not settings.scenario_root:
            starter = Path(settings.code_root) / "reference/starter-kit"
            settings.scenario_root = os.environ.get("SB_SCENARIO_ROOT") or (str(starter) if starter.is_dir()
                                                                             else os.getcwd())
        needed = settings.iterations * settings.rollouts_per_iteration
        if len(settings.rollout_scenarios) < needed:
            raise ValueError(f"{needed} rollout scenarios needed (iterations x E), {len(settings.rollout_scenarios)} given")
        labels = [scenario_label(s) for s in settings.rollout_scenarios[:needed]]
        if len(set(labels)) != len(labels):
            raise ValueError("a rollout scenario repeats; D-L15 uses a new scenario for every rollout")
        if any(i < 1 or i > settings.iterations for i in settings.eval_iterations):
            raise ValueError("eval_iterations outside 1..iterations")
        return settings

    def identity(self):
        values = asdict(self)
        values["eval_iterations"] = list(self.eval_iterations)
        return {k: values[k] for k in IDENTITY}


def scenario_label(spec):
    if spec["set"] == "official":
        return spec["name"]
    return f"{spec['set']}/{spec.get('split')}/{spec.get('pool')}/{spec['building']}/{spec['weather']}"


def resolve_scenario(spec, settings, purpose):
    """The scenario directory of a remote-style spec; a generated one is materialised if missing."""
    root = Path(settings.scenario_root)
    if spec["set"] == "official":
        from smart_building.remote.scenarios import official_scenario_dir
        return official_scenario_dir(root, spec["name"])
    if spec["set"] == "generated":
        from smart_building.scenarios.factory import generated_scenario_dir, load_split, materialize_for_tasks
        version = load_split(settings.code_root, spec["split"]).get("version", spec["split"])
        directory = generated_scenario_dir(root, version, spec["pool"], spec["building"], spec["weather"])
        if not (directory / "config.gin").is_file():
            materialize_for_tasks(settings.code_root, root, [{"scenario": spec, "purpose": purpose}])
        return directory
    raise ValueError(f"unknown scenario set {spec['set']!r}")


def select_teacher(settings, spec):
    """D-L13 hook. "single": the B1-R champion for every scenario (user decision 2026-10-17)."""
    teacher = settings.teacher
    if teacher.get("selection", "single") == "single":
        return teacher["candidates"][0]
    raise NotImplementedError(f"teacher selection {teacher.get('selection')!r}: only 'single' is implemented "
                              "(D-L13: the user chose the B1-R champion only, 2026-10-17)")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def parse_deadline(deadline_utc):
    if deadline_utc is None:
        return None
    text = deadline_utc.replace("Z", "+00:00")
    return dt.datetime.fromisoformat(text).astimezone(dt.timezone.utc).timestamp()


def plain(value):
    """A JSON round trip: only plain types reach the checkpoint (torch.load weights_only=True)."""
    return json.loads(json.dumps(value, sort_keys=True, default=float))


def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def write_json(path, value):
    temporary = Path(str(path) + ".partial")
    with open(temporary, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=1, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


# ------------------------------------------------------------------ checkpoints

def checkpoint_name(iteration):
    return f"ckpt-{iteration:04d}"


def newest_checkpoint(directory):
    directory = Path(directory)
    latest = directory / "latest.json"
    if latest.is_file():
        return directory / json.loads(latest.read_text(encoding="utf-8"))["checkpoint"]
    found = sorted(p for p in directory.glob("ckpt-*") if (p / "manifest.json").is_file())
    return found[-1] if found else None


def verify_checkpoint(directory, checkpoint):
    """Every file of the checkpoint's manifest exists with its SHA-256; returns the manifest."""
    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    bad = sorted(name for name, digest in manifest["files"].items()
                 if not (Path(directory) / name).is_file() or sha256_file(Path(directory) / name) != digest)
    if bad:
        raise RuntimeError(f"{checkpoint}: files fail their SHA-256 check: {bad[:5]}")
    return manifest


def rng_states(generator):
    import torch
    state = {"numpy_generator": generator.bit_generator.state, "python": random.getstate(),
             "numpy_global": _numpy_global_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _numpy_global_state():
    name, keys, pos, has_gauss, cached = np.random.get_state()
    return [name, keys.tolist(), int(pos), int(has_gauss), float(cached)]


def restore_rng(state, generator):
    import torch
    generator.bit_generator.state = state["numpy_generator"]
    version, internal, gauss = state["python"]
    random.setstate((version, tuple(internal), gauss))
    name, keys, pos, has_gauss, cached = state["numpy_global"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(state["torch"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


class Chain:
    """The state of one DAgger chain inside a segment."""

    def __init__(self, settings, out_dir):
        self.settings = settings
        self.out_dir = Path(out_dir)
        self.iteration = 0
        self.model = self.optimizer = None
        self.feature_counts = None
        self.rollouts = []            # [{"file", "sha256", ...meta}] in collection order
        self.models = {}              # {iteration: {"file", "sha256"}}
        self.evaluations = []         # rows
        self.history = []             # per-iteration logs
        self.statistics = None        # {"zero_std_features": {...}}
        self.counters = {"rollouts": 0, "samples": 0, "env_steps": 0, "minibatches": 0, "student_fallback_steps": 0}
        self.generator = np.random.Generator(np.random.PCG64(settings.seed))
        self.data = None              # dataset.Dataset (loaded lazily)

    # -- model ---------------------------------------------------------

    def student_config(self):
        from smart_building.learned.model import StudentConfig
        return StudentConfig(**self.settings.student)

    def build_model(self, feature_counts, keep=None):
        import torch
        from smart_building.learned.dataset import input_dims
        from smart_building.learned.model import EntityTransformer
        self.feature_counts = {k: tuple(v) for k, v in feature_counts.items()}
        config = self.student_config()
        self.model = EntityTransformer(input_dims(self.feature_counts, config.history), config,
                                       keep).to(self.settings.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.settings.lr)   # D-L09; kept (REQ-018 (3))

    # -- state ---------------------------------------------------------

    def save(self):
        import torch
        name = checkpoint_name(self.iteration)
        directory = self.out_dir / name
        directory.mkdir(parents=True, exist_ok=True)
        state = {"format": STATE_FORMAT, "identity": self.settings.identity(),
                 "identity_sha256": sha256_json(self.settings.identity()), "iteration": self.iteration,
                 "feature_counts": {k: list(v) for k, v in self.feature_counts.items()},
                 "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                 "rng": rng_states(self.generator), "rollouts": self.rollouts,
                 "models": {str(k): v for k, v in self.models.items()}, "evaluations": self.evaluations,
                 "history": self.history, "statistics": self.statistics, "counters": self.counters}
        torch.save(state, directory / "state.pt")
        files = {f"{name}/state.pt": sha256_file(directory / "state.pt")}
        files.update({r["file"]: r["sha256"] for r in self.rollouts})
        files.update({m["file"]: m["sha256"] for m in self.models.values()})
        write_json(directory / "manifest.json", {"format": STATE_FORMAT, "iteration": self.iteration, "files": files})
        write_json(self.out_dir / "latest.json", {"checkpoint": name, "iteration": self.iteration})
        kept = sorted(p for p in self.out_dir.glob("ckpt-*") if p.is_dir())
        for old in kept[:-self.settings.keep_checkpoints]:
            shutil.rmtree(old)
        return name

    def load(self, state_dir):
        """Resume from the newest checkpoint of state_dir (checked), copying its files into out_dir."""
        import torch
        state_dir = Path(state_dir)
        checkpoint = newest_checkpoint(state_dir)
        if checkpoint is None:
            raise FileNotFoundError(f"no checkpoint in {state_dir}")
        manifest = verify_checkpoint(state_dir, checkpoint)
        if state_dir.resolve() != self.out_dir.resolve():
            for name in manifest["files"]:
                target = self.out_dir / name
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.is_file() or sha256_file(target) != manifest["files"][name]:
                    shutil.copy2(state_dir / name, target)
            write_json(self.out_dir / "latest.json", {"checkpoint": checkpoint.name, "iteration": manifest["iteration"]})
            shutil.copy2(checkpoint / "manifest.json", self.out_dir / checkpoint.name / "manifest.json")
        state = torch.load(self.out_dir / checkpoint.name / "state.pt", map_location="cpu", weights_only=True)
        if state.get("format") != STATE_FORMAT:
            raise ValueError(f"{checkpoint}: not a {STATE_FORMAT} state")
        if state["identity"] != self.settings.identity():
            changed = sorted(k for k in IDENTITY if state["identity"].get(k) != self.settings.identity().get(k))
            raise ValueError(f"the chain's settings changed since it started: {changed}")
        self.iteration = state["iteration"]
        from smart_building.learned.model import keep_from_state
        self.build_model(state["feature_counts"], keep_from_state(state["model"]))
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"], self.generator)
        self.rollouts, self.evaluations, self.history = state["rollouts"], state["evaluations"], state["history"]
        self.models = {int(k): v for k, v in state["models"].items()}
        self.statistics, self.counters = state["statistics"], state["counters"]
        # This segment's logs start with the chain's history (the earlier segments' rows).
        for name, rows in (("iterations.jsonl", self.history), ("evaluations.jsonl", self.evaluations)):
            (self.out_dir / name).write_text("", encoding="utf-8")
            for row in rows:
                append_jsonl(self.out_dir / name, row)

    def dataset(self):
        from smart_building.learned.dataset import Dataset, Rollout
        loaded = len(self.data.rollouts) if self.data is not None else 0
        if self.data is None or loaded < len(self.rollouts):
            previous = self.data.rollouts if self.data is not None else []
            new = [Rollout(self.out_dir / r["file"]) for r in self.rollouts[loaded:]]
            self.data = Dataset(previous + new)
        return self.data


# ------------------------------------------------------------------ one iteration

class Workers:
    """Rollout and evaluation calls: in-process for one worker, else a spawn pool."""

    def __init__(self, count):
        self.count = int(count)
        self.pool = None

    def map(self, function, tasks):
        if self.count <= 1:
            import torch
            with torch.random.fork_rng(devices=[]):      # in-process calls leave this process's RNG as a pool would
                return [function(task) for task in tasks]
        if self.pool is None:
            self.pool = ProcessPoolExecutor(max_workers=self.count, mp_context=multiprocessing.get_context("spawn"))
        return list(self.pool.map(function, tasks))

    def close(self):
        if self.pool is not None:
            self.pool.shutdown()


def collect(chain, workers):
    """Step 1: the iteration's E rollouts."""
    from dataclasses import asdict as as_dict
    from smart_building.learned import rollouts as rollout_lib
    from smart_building.learned.model import state_bytes
    s, i = chain.settings, chain.iteration + 1
    specs = s.rollout_scenarios[(i - 1) * s.rollouts_per_iteration:i * s.rollouts_per_iteration]
    student = None
    if i > 1:
        cpu_model = chain.model.to("cpu")
        student = {"config": as_dict(chain.student_config()), "state": state_bytes(cpu_model)}
        chain.model.to(s.device)
    tasks = []
    for r, spec in enumerate(specs, start=1):
        name = f"data/rollout-{i:04d}-{r:02d}.npz"
        tasks.append({"scenario": spec, "label": scenario_label(spec),
                      "scenario_dir": str(resolve_scenario(spec, s, "train")), "scenario_root": s.scenario_root,
                      "code_root": s.code_root, "teacher": select_teacher(s, spec), "student": student,
                      "history": chain.student_config().history, "sample_every": s.sample_every,
                      "steps_cap": s.steps_cap, "iteration": i, "rollout": r, "file": name,
                      "out_path": str(chain.out_dir / name), "threads": 1})
    return [plain(r) for r in workers.map(rollout_lib.run_rollout, tasks)]


def first_statistics(chain, records):
    """Step 2 (iteration 1): input statistics and log_std from the demonstrations."""
    from smart_building.learned.dataset import Rollout, action_statistics, input_statistics
    demonstrations = [Rollout(chain.out_dir / r["file"]) for r in records]
    counts = demonstrations[0].feature_counts()
    if any(d.feature_counts() != counts for d in demonstrations):
        raise ValueError("the rollouts' token feature counts differ between buildings")
    mean, std, zero = input_statistics(demonstrations, chain.student_config().history)
    keep = {k: np.flatnonzero(std[k] > 0.0) for k in std}                       # REQ-018 (6)
    chain.build_model(counts, keep)
    chain.model.set_normalization({k: mean[k][keep[k]] for k in keep},
                                  {k: std[k][keep[k]] for k in keep})            # REQ-014 (1, 2)
    chain.model.set_log_std(action_statistics(demonstrations))   # D-L09; first iteration only (REQ-018 (7))
    chain.statistics = {"input_dims_excluded": zero, "input_dims_kept": {k: int(len(v)) for k, v in keep.items()},
                        "samples": int(sum(len(d) for d in demonstrations))}


def train(chain):
    """Step 3: D-L09 BC on the aggregated data (dataset.py for the minibatch rule)."""
    import torch
    from smart_building.learned.dataset import minibatch_loss
    s = chain.settings
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
    return {"samples": n, "minibatches_per_epoch": n // s.batch_size, "epoch_mean_loss": epoch_losses}


def evaluate(chain, workers):
    """Step 4: save the student of this iteration (trained on D_1..D_i, REQ-018 (5)) and score it on the
    INNER episodes."""
    from smart_building.learned import rollouts as rollout_lib
    from smart_building.learned.model import save_student
    s, i = chain.settings, chain.iteration
    name = f"models/student-{i:04d}.pt"
    path = chain.out_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    save_student(path, chain.model.to("cpu"), teacher=select_teacher(s, None),
                 extra={"iteration": i, "statistics": chain.statistics})
    chain.model.to(s.device)
    chain.models[i] = {"file": name, "sha256": sha256_file(path)}
    tasks = [{"scenario": spec, "label": scenario_label(spec), "scenario_dir": str(resolve_scenario(spec, s, s.eval_purpose)),
              "scenario_root": s.scenario_root, "code_root": s.code_root, "teacher": select_teacher(s, spec),
              "model_path": str(path), "iteration": i, "steps_cap": s.steps_cap, "threads": 1}
             for spec in s.eval_scenarios]
    rows = [plain(r) for r in workers.map(rollout_lib.run_evaluation, tasks)]
    chain.evaluations.extend(rows)
    return rows


def run_iteration(chain, workers):
    import torch
    timing = {}
    started = time.time()
    if chain.iteration == 0:
        torch.manual_seed(chain.settings.seed)            # model initialisation of a new chain
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
    log = {"iteration": chain.iteration, "rollouts": [{k: r[k] for k in ("label", "actor", "steps", "mean_reward",
                                                                          "samples", "student_fallback_steps",
                                                                          "student_latency", "seconds")}
                                                       for r in records],
           "bc": bc, "evaluation": evaluation, "timing": timing, "counters": dict(chain.counters)}
    log = plain(log)
    chain.history.append(log)
    return log


def best_iteration(chain):
    """stage45 §2: the evaluated iteration with the lowest mean cost (ties: the earlier one)."""
    means = {}
    for i in chain.settings.eval_iterations:
        rows = [r for r in chain.evaluations if r["iteration"] == i]
        if rows:
            means[i] = float(np.mean([r["cost"] for r in rows]))
    if not means:
        return None, means
    return min(means, key=lambda i: (means[i], i)), means


# ------------------------------------------------------------------ entry point

def run_dagger_segment(config, state_dir, out_dir, deadline_utc):
    """One segment of a DAgger chain (see the module docstring); returns a JSON-able dict."""
    import torch
    settings = DaggerSettings.from_config(config)
    if settings.device.startswith("cuda"):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    previous = (torch.are_deterministic_algorithms_enabled(), torch.get_num_threads())
    torch.use_deterministic_algorithms(True)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    deadline = parse_deadline(deadline_utc)
    chain = Chain(settings, out_dir)
    if state_dir:
        chain.load(state_dir)
    workers = Workers(settings.workers)
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
            append_jsonl(out_dir / "iterations.jsonl", log)
            for row in chain.evaluations[len(chain.evaluations) - (log["evaluation"] or {}).get("episodes", 0):]:
                append_jsonl(out_dir / "evaluations.jsonl", row)
            done_here += 1
    finally:
        workers.close()
        torch.use_deterministic_algorithms(previous[0])
        torch.set_num_threads(previous[1])
    finished = chain.iteration >= settings.iterations
    result = {"finished": finished, "iteration": chain.iteration, "iterations_this_segment": done_here,
              "stopped": stopped if not finished else "finished",
              "checkpoint": checkpoint_name(chain.iteration) if chain.iteration else None,
              "counters": chain.counters}
    if finished:
        best, means = best_iteration(chain)
        final = {"best_iteration": best, "mean_cost_by_iteration": means,
                 "model": chain.models.get(best), "teacher": select_teacher(settings, None)}
        write_json(out_dir / "final.json", final)
        result.update(final)
    return result
