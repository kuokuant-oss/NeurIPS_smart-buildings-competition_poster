"""PPO fine-tuning B1-RL (WP5 5.1: D-RL01-D-RL04, D-RL07; docs/plan/baseline-v1-stage45.md §4).

Entry point (remote task kind "dagger", remote/runner.py run_dagger, the same interface as
learned/dagger.py so the existing runner relays its segments through kernel_sources):

    run_ppo_segment(config, state_dir, out_dir, deadline_utc) -> dict

runs as many complete PPO iterations as fit before deadline_utc (ISO UTC; None = no limit),
checkpointing at every iteration boundary into out_dir (model, optimiser, every RNG state, the
reward normalisers, the history rows and a snapshot of every running episode), and returns
{"finished", "iteration", "stopped", "checkpoint", ...}. state_dir is the out_dir of the previous
segment (read only; it may be out_dir itself) or None; its newest checkpoint is checked against its
SHA-256 manifest and copied into out_dir, so every segment's out_dir holds the whole state of the run.

D-RL01: the program follows CleanRL `cleanrl/ppo_continuous_action.py` @fe8d8a0 (MIT; 353 lines;
fetched 2026-10-18 with WebFetch, line map checked twice and against notes/05 §B). Lines followed:
  L17-84    Args                     -> PPOSettings (defaults: CleanRL unless D-RL02/D-RL07/REQ-021 say otherwise)
  L87-103   make_env wrappers        -> ppo_env.EnvSlot (episodes, auto reset), ClipAction (L96) as the
                                        clip of the executed action, NormalizeObservation + clip (L97-98)
                                        as frozen statistics + clip (REQ-021 (2)),
                                        NormalizeReward(gamma) + clip (L99-100) = ppo_env.RewardNormalizer
  L106-109  layer_init               -> layer_init
  L112-141  Agent (separate critic and actor, state-independent actor_logstd, Normal, log-prob and
            entropy summed over the action dimensions)  -> Agent (entity networks, D-RL02)
  L146-148  batch_size, minibatch_size, num_iterations  -> Run.__init__
  L168-172  seeding; L183 Adam eps 1e-5 (actor and critic parameter groups, REQ-021 (3)); L185-191 storage
  L200-205  iteration loop and learning-rate schedule (REQ-021 (3) replaces the linear anneal)
  L207-230  rollout (stored action unclipped, L216; next_done = terminated or truncated, L221)
  L232-246  GAE (gae(); REQ-021 (12): bootstrap at time limits); L248-254 flatten
  L256-308  update: shuffle per epoch (L260), ratio and KL (L266-273), advantage normalisation
            (L276-277), clipped policy loss (L280-282), clipped value loss (L285-297), entropy
            (L299), loss (L300), zero_grad / backward / clip_grad_norm_ / step (L302-305; the backward runs
            once per chunk of at most max_chunk_samples samples of one episode group with the gradients
            accumulated, then one clip and one step per minibatch: REQ-021 (1)), target_kl (L307-308)
  L310-312  explained variance

D-RL03 (a) anchor "residual": the plan's other anchoring option (its values are provisional until its ablation is
registered). pi = pi_B1-R + action_scale x f_theta: the rule controller is the base, called once per step with that
step's observation; f has its own randomly initialised trunk and a zero last layer, so it starts at exactly 0; the
observations are normalised with frozen statistics read from a save_student file (`normalization`, statistics only)
and clipped; f's log std is a constant or learned (log_std_init, learn_log_std); the critic may be warmed up first
until its mean value loss is below a threshold (critic_warmup_loss, D-RL03 (a)); the Gaussian, its log-prob and the
PPO ratio are in f's space, the sample is scaled only when it is executed; evaluation with base + action_scale x the
mean of f, the rule controller's action on a network failure (D-T02).
D-RL03 (b) anchor "student" (baseline-v1 stage 3: B1-R v4 -> B1-L j7b-v5s -> PPO initialised from that student,
D-RL02 "由 B1-L 初始化"; reviewer settings 2026-10-21; job specification docs/plan/b1-rl-stage3.md):
  actor  the fold's saved B1-L student itself: same weights, same frozen first-iteration statistics, decoders tanh as
         trained (D-L04), action_scale 1, no observation clip (REQ-021 (2) (b), stage45 §2), so at iteration 0 the
         policy mean is the student's action bit for bit. A j7b-v5 / v5s student (learned/privileged.py format) also
         reads the B1-R internal state: the env slots read it from the rule controller before each of its calls
         (ppo_env.py) and the trainer appends it to the zone, AHU and hot-water tokens as privileged.append_features;
  rule controller  v4 + readback, G36PolicyC3LaaOccrowDampfloorReadback (REQ-021 (11) (c): the student chain's own
         teacher; a privileged student must name it), for the on/off (D-RL04 (a)), the D-T02 fallback and the state;
  critic separate, random initialisation with the student's frozen statistics, value = mean over all tokens (REQ-021
         (5) (b); MetaMorph §4.2);
  demonstrations  DAPG (mjrl `dapg.py` @3871d93): -w_k x sum log pi(a_demo | s_demo) / minibatch size added to every
         minibatch loss, w_k = 0.1 x 0.95^k, k = actor-update iterations so far from 0, no max-advantage factor (REQ-021
         (8) text); data: the student chain's first-iteration teacher rollouts (REQ-021 (8) data (a); checked: teacher
         rollouts of iteration 1 with the student's teacher, and, at a new run, their input statistics against the
         student's frozen ones);
  (1) 4 envs x 2560 steps, minibatch 5120, 8 epochs, chunks of at most 64; (3) MetaMorph's schedule over all planned
  iterations (critic base 3e-4, D-RL02; actor base below); (4) dropout 0; (9) uniform draws; (12) bootstrap at time
  limits; (14) the mean action with the rule fallback.
  Reviewer decisions 2026-10-21 01:30 (sources checked by the reviewer; docs/reference/notes/15-student-anchor-sources.md):
    gamma 0.995 (user decision; D-RL05's registered alternative; hand_dapg dapg.txt @9a1de23 rl_gamma 0.995), GAE
      lambda 0.95 (D-RL02);
    target_kl 0.025, checked as CleanRL L307-308 (mjrl dapg.py @3871d93 kl_dist = 0.5 x normalized_step_size;
      hand_dapg dapg.txt rl_step_size 0.05);
    critic warm-up: 2 fixed critic-only iterations at the start (critic_warmup_iterations): the actor is not stepped,
      the demonstration term is not applied and k does not advance, the data come from the student's stochastic
      actions and count toward total_timesteps (DPPO @cc7234a train_ppo_gaussian_agent.py n_critic_warmup_itr, its
      robomimic can config 2); no loss threshold; the learning-rate curve unchanged;
    exploration: the parameter starts at the student's D-L09 log_std per output and is learned, floor -3 after every
      update (hand_dapg job_script.py set_variance_with_data; mjrl gaussian_mlp.py min_log_std -3); entropy
      coefficient 0 (DAPG's update has no entropy term; DPPO train_ppo_agent.py ent_coef 0); no KL penalty toward
      the student.
  User decisions 2026-10-21 02:05:
    actor learning rate 3e-6 (actor_learning_rate), the critic's 3e-4, both on the MetaMorph curve: VPT (Baker et al.
      2022), whose RL fine-tuning without a KL loss used 3e-6; the stage-3 measurement (docs/plan/b1-rl-stage3.md §6;
      .tmp/stage3-student/kl-measure): with 3e-6 the epoch-1 approx_kl stays under target_kl 0.025, the per-iteration
      KL of DAPG's step size (delta / 2);
    sigma at most 0.2 (max_std), clamped in the forward pass as DPPO does (its Gaussian transformer clamps the variance
      between std_min^2 and std_max^2 when forming the distribution; std_max 0.2 in its fine-tuning configs); the
      mjrl floor -3 stays a clamp of the parameter.
  Not settled: critic_lr_warmup (True: MetaMorph's single curve for both groups, as implemented).
  Round 2 (J10; user 2026-10-24: train longer, the learning rate not decayed to zero; reviewer 2026-10-24 with the
  user's deadline of 10/27: docs/reference/notes/16-rl-tuning-after-baseline.md §C5 (b), continuation): init_from
  continues a finished run from its final state.pt (Run.continue_from: agent, Adam state, iteration, k, counters, RNG
  states and reward normalisers kept, new episodes; its identity must equal this run's except CONTINUATION_CHANGES);
  lr_schedule "warmup_constant": MetaMorph's 5-iteration warm-up counted from schedule_origin (the continued run's
  iteration), then the bases kept (actor 3e-6, critic 3e-4), so the curve does not depend on total_timesteps.
D-RL04 (a) switches "backbone": on/off from the rule controller (D-L07); (b) "bernoulli": a Bernoulli
head on the AHU and hot-water tokens (notes/06 candidate table, §AL), its own clipped objective with the
shared advantage (H-PPO Eq. 9-10).
D-RL07: total_timesteps 1e7 environment steps by default (the registry's upper limit); stage 3 runs 1e6 per fold
(user decision 2026-10-20).

Determinism: the simulator, the rule controller and CPU inference are deterministic; the network
initialisation, the action noise (a torch.Generator), dropout (torch's global generator), the
minibatch shuffles and the scenario draws (numpy PCG64) are seeded, and their states are in every
checkpoint; torch.use_deterministic_algorithms is on. A resumed run therefore equals an
uninterrupted one bit for bit on CPU for the same thread count (tests/test_learned_ppo.py).
"""
import copy
from dataclasses import asdict, dataclass, field, fields
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np

try:
    import fcntl
except ImportError:          # Windows
    fcntl = None

STATE_FORMAT = "b1-rl-ppo/2"
AGENT_FORMAT = "b1-rl-ppo-agent/2"
ANCHORS = ("residual", "student")           # D-RL03 (a), (b)
SWITCHES = ("backbone", "bernoulli")        # D-RL04 (a), (b)
SWITCH_PRIORS = ("none", "rule")           # D-RL04 (b): the learned switches alone, or anchored on B1-R (2026-10-25)
LR_SCHEDULES = ("metamorph", "warmup_cosine", "warmup_constant")
MEAN_ACTIVATIONS = ("tanh", "linear")
BY_ANCHOR = "by_anchor"                     # a setting that takes its anchor's value from ANCHOR_DEFAULTS
METAMORPH_WARMUP_FACTOR = 0.1               # MetaMorph config.py PPO.WARMUP_FACTOR (REQ-021 (3) (b))
METAMORPH_MIN_LR = 0.0                      # MetaMorph config.py PPO.MIN_LR
# The D-RL03 (a) option's base and on/off source (provisional): v4 + readback (notes/09 §F, §M; REQ-018 (9)).
RESIDUAL_RULE_CONTROLLER = {"file": "packages/smart_building/policies/c3_teachers.py",
                            "class": "G36PolicyC3LaaOccrowDampfloorReadback"}
# REQ-021 (11) (c) for the student anchor (reviewer 2026-10-21): the student chain's own teacher, v4 + readback for the
# j7b-v5s students (configs/remote/j7b-v5s.json teacher), which is also the source of their B1-R state inputs.
STUDENT_RULE_CONTROLLER = {"file": "packages/smart_building/policies/c3_teachers.py",
                           "class": "G36PolicyC3LaaOccrowDampfloorReadback"}
# Values of the settings left at BY_ANCHOR. "student": stage 3, PPO initialised from the B1-L student (the reviewer's
# settings and decisions of 2026-10-21 and the user's of 02:05; sources in the module docstring). "residual": the
# D-RL03 (a) option, values provisional until its ablation is registered.
ANCHOR_DEFAULTS = {
    "residual": {"rule_controller": RESIDUAL_RULE_CONTROLLER,
                 "dropout": 0.0, "target_kl": 0.1,
                 "log_std_init": -1.0, "learn_log_std": False, "min_log_std": None, "max_std": None,
                 "mean_activation": "linear", "action_scale": 0.1,
                 "lr_schedule": "metamorph", "critic_lr_warmup": False, "actor_learning_rate": None,
                 "bootstrap_time_limit": True,
                 "max_chunk_samples": 64,
                 "observation_clip": 10.0,                                              # D-RL02 +-10
                 "critic_warmup_loss": 1.0, "critic_warmup_iterations": None,           # D-RL03 (a) beta
                 "gamma": 0.99, "ent_coef": 0.01},                                      # D-RL02
    "student": {"rule_controller": STUDENT_RULE_CONTROLLER,                             # (11) (c)
                "dropout": 0.0,                                                         # (4) (b)
                "gamma": 0.995,                                                         # D-RL05 alternative; dapg.txt
                "target_kl": 0.025,                                                     # mjrl dapg.py kl_dist; dapg.txt
                "log_std_init": "student", "learn_log_std": True, "min_log_std": -3.0,  # D-L09 log_std, mjrl floor
                "max_std": 0.2,                                                         # DPPO std_max, forward clamp
                "ent_coef": 0.0,                                                        # DAPG; DPPO ent_coef 0
                "mean_activation": "tanh", "action_scale": 1.0,                         # D-L04 tanh as trained; the action
                "lr_schedule": "metamorph",                                             # (3) MetaMorph's schedule
                "actor_learning_rate": 3e-6,                                            # VPT; kl-measure (user 02:05)
                "critic_lr_warmup": True,                                               # not settled: MetaMorph's curve
                "bootstrap_time_limit": True,                                           # (12) (c), Pardo et al. Eq. 6
                "max_chunk_samples": 64,                                                # (1): gradient accumulation, M 64
                "observation_clip": None,                                               # (2) (b): the student exactly
                "critic_warmup_loss": None, "critic_warmup_iterations": 2},             # DPPO n_critic_warmup_itr 2
}
# Settings that define a run: a resumed segment must have the same values.
IDENTITY = ("anchor", "switches", "student_model", "normalization", "rule_controller", "demonstrations",
            "rollout_scenarios", "total_timesteps", "learning_rate", "lr_schedule", "critic_lr_warmup", "num_envs",
            "num_steps", "warmup_iterations", "gamma", "gae_lambda", "minibatch_size", "max_chunk_samples",
            "update_epochs", "norm_adv", "clip_coef", "clip_vloss", "ent_coef", "vf_coef", "max_grad_norm", "target_kl",
            "adam_eps", "dropout", "observation_clip", "reward_clip", "reward_epsilon", "critic_warmup_loss",
            "critic_warmup_iterations",
            "demo_lambda0", "demo_lambda1", "switch_init_gain", "log_std_init", "learn_log_std", "min_log_std",
            "max_std", "actor_learning_rate",
            "mean_activation", "action_scale", "bootstrap_time_limit", "seed", "steps_cap",
            "init_from", "schedule_origin", "switch_prior", "switch_prior_gain", "switch_learning_rate")
# IDENTITY keys added after runs were saved without them (J10): a saved identity that lacks one has its default, so
# a checkpoint written before them resumes as it did.
IDENTITY_DEFAULTS = {"init_from": None, "schedule_origin": 0,
                     # reviewer 2026-10-25: the switch prior and the switch heads' own learning rate
                     "switch_prior": "none", "switch_prior_gain": 4.0, "switch_learning_rate": None}
# What a continuation (init_from) may change against the finished run it continues (J10, notes/16 §C5 (b); the agent
# snapshot targets are not part of IDENTITY): every other IDENTITY value must be the continued run's.
CONTINUATION_CHANGES = ("total_timesteps", "lr_schedule", "init_from", "schedule_origin")
# Settings of the machine a segment runs on, not of the run (reviewer 2026-10-20): recorded per segment (iterations.jsonl
# rows, checkpoint manifests) and free to change at a resume (a GPU kernel, a Kaggle CPU kernel or the Mac continue the
# same run). An unchanged resume stays bit for bit; a changed one continues correctly, with the change recorded.
RUNTIME = ("device", "train_threads")


@dataclass
class PPOSettings:
    """CleanRL Args (L17-84) with the registry's values; comments give the source of each. A value of
    BY_ANCHOR is replaced by its anchor's entry in ANCHOR_DEFAULTS when the settings are made."""
    anchor: str = "residual"                   # D-RL03: "residual" (a) | "student" (b)
    switches: str = "backbone"                 # D-RL04: "backbone" (a) | "bernoulli" (b)
    student_model: dict = field(default_factory=dict)   # anchor "student": the saved B1-L student ({"path"} or
    #                                                     {"source", "file"}; "sha256")
    normalization: dict = field(default_factory=dict)   # anchor "residual": the observation statistics' source, a
    #                                                     save_student file whose frozen statistics are read (REQ-021 (2))
    rule_controller: dict | str = BY_ANCHOR    # REQ-021 (11)
    demonstrations: list = field(default_factory=list)  # D-RL03 (b): stored rollout files (REQ-021 (8))
    rollout_scenarios: list = field(default_factory=list)  # the fold's TRAIN scenarios (v1-train-r1), remote-style specs
    total_timesteps: int = 10_000_000          # D-RL07 1e7 (CleanRL L45: 1e6); critic warm-up counted: REQ-021 (12)
    learning_rate: float = 3e-4                # D-RL02 (= CleanRL L47; MetaMorph BASE_LR); the critic group's base
    actor_learning_rate: float | None | str = BY_ANCHOR   # the actor group's base (None: learning_rate); same curve
    lr_schedule: str = BY_ANCHOR               # REQ-021 (3): "metamorph" | "warmup_cosine" | "warmup_constant" (J10)
    critic_lr_warmup: bool | str = BY_ANCHOR   # REQ-021 (3): whether the critic group has the warm-up too
    num_envs: int = 4                          # REQ-021 (1) (CleanRL L49: 1; MetaMorph 32)
    num_steps: int = 2560                      # REQ-021 (1) (MetaMorph timesteps per rollout; CleanRL L51: 2048)
    warmup_iterations: int = 5                 # D-RL02 via notes/05 §R (MetaMorph Table 1, WARMUP_ITERS)
    gamma: float | str = BY_ANCHOR             # D-RL02, D-RL05 main case 0.99; the student anchor 0.995 (D-RL05 alt.)
    gae_lambda: float = 0.95                   # D-RL02
    minibatch_size: int = 5120                 # D-RL02 "batch 5120" read as the minibatch (REQ-021 (1))
    max_chunk_samples: int | None | str = BY_ANCHOR   # REQ-021 (1): samples per backward (None: a whole episode group)
    update_epochs: int = 8                     # D-RL02
    norm_adv: bool = True                      # CleanRL L63
    clip_coef: float = 0.2                     # D-RL02
    clip_vloss: bool = True                    # CleanRL L67; MetaMorph Table 1 "clipped value function"
    ent_coef: float | str = BY_ANCHOR          # D-RL02 0.01; the student anchor 0 (no entropy term in DAPG)
    vf_coef: float = 0.5                       # D-RL02
    max_grad_norm: float = 0.5                 # D-RL02
    target_kl: float | None | str = BY_ANCHOR  # CleanRL L75, L307-308 (None: no early stop)
    adam_eps: float = 1e-5                     # CleanRL L183
    dropout: float | str = BY_ANCHOR           # REQ-021 (4)
    observation_clip: float | None | str = BY_ANCHOR   # D-RL02 +-10 (residual); REQ-021 (2) (b) None (student)
    reward_clip: float = 10.0                  # D-RL02 (CleanRL L100)
    reward_epsilon: float = 1e-8               # gymnasium NormalizeReward default (CleanRL L99 passes only gamma)
    critic_warmup_loss: float | None | str = BY_ANCHOR  # critic-only iterations until the mean value loss is below it
    #                                                    (D-RL03 (a) beta 1.0; REQ-021 (6) (a)); None: no warm-up
    critic_warmup_iterations: int | None | str = BY_ANCHOR   # or a fixed number of critic-only iterations first (the
    #                                                student anchor: 2, DPPO n_critic_warmup_itr); None: none; at most one
    demo_lambda0: float = 0.1                  # D-RL03 (b)
    demo_lambda1: float = 0.95                 # D-RL03 (b)
    switch_init_gain: float = 0.01             # D-RL04 (b) head initialisation: REQ-021 (10)
    switch_prior: str = "none"                 # D-RL04 (b), reviewer 2026-10-25: "none" | "rule": each switch's
    #                                            Bernoulli logit = its head + switch_prior_gain x (2 x the rule
    #                                            controller's decision of this step - 1): the switches start at B1-R's
    #                                            decisions and the heads learn when to depart (the residual idea of
    #                                            D-RL03 (a), Silver et al. 2018, applied to the discrete actions)
    switch_prior_gain: float = 4.0             # the prior logit: with the head at 0, p(B1-R's decision) = sigmoid(4)
    switch_learning_rate: float | None = None  # the switch heads' own Adam group on the actor's curve (None: actor)
    log_std_init: float | str = BY_ANCHOR      # REQ-021 (7): a number, or "student" (the student's log_std, D-L09)
    learn_log_std: bool | str = BY_ANCHOR      # REQ-021 (7)
    max_std: float | None | str = BY_ANCHOR    # ceiling on the sigma the Gaussian uses, forward only (None: none)
    min_log_std: float | None | str = BY_ANCHOR    # REQ-021 (7): a lower clamp on a learned log_std (mjrl), or None
    mean_activation: str = BY_ANCHOR           # REQ-021 (13): Gaussian mean = "tanh"(decoder) (D-L04) or "linear"
    action_scale: float | str = BY_ANCHOR      # REQ-021 (13): executed residual = action_scale x f (f's own space)
    bootstrap_time_limit: bool | str = BY_ANCHOR   # REQ-021 (12): bootstrap from v(s_last) at every episode end
    seed: int = 0                              # D-E11: one per fold in screening, three for final candidates
    steps_cap: int | None = None               # cap on every episode (smoke runs, tests; the runner passes task steps)
    init_from: dict | None = None              # J10: a finished run's final state.pt to continue (a resolve_input
    #                                            spec with its "sha256"; Run.continue_from); None: a new run
    schedule_origin: int = 0                   # J10: the iteration "warmup_constant" counts its warm-up from (a
    #                                            continuation: the continued run's iteration); 0 for a new run
    in_process: bool = False                   # env slots in this process (tests) instead of worker processes
    train_threads: int = 1                     # torch threads for the update only (RUNTIME: may change at a resume)
    device: str = "cpu"                        # the networks and the update; env workers stay on the CPU (RUNTIME)
    keep_checkpoints: int = 2
    deadline_margin: float = 1.25              # start an iteration only if margin x the longest one still fits
    iteration_estimate_seconds: float | None = None
    max_iterations_per_segment: int | None = None
    agent_snapshot_steps: list = field(default_factory=list)   # save the agent at the first iteration at or after each
    stop_file: str | None = None               # if this file exists, stop before the next iteration (a clean pause)
    scenario_root: str | None = None
    code_root: str | None = None
    remote_mode: str | None = None             # passed by the remote runner
    meta: dict = field(default_factory=dict)   # free-form job information; not used

    def __post_init__(self):
        for key, value in ANCHOR_DEFAULTS.get(self.anchor, {}).items():
            if isinstance(getattr(self, key), str) and getattr(self, key) == BY_ANCHOR:
                setattr(self, key, copy.deepcopy(value))

    @classmethod
    def from_config(cls, config):
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(config) - known)
        if unknown:
            raise KeyError(f"unknown PPO settings {unknown}")
        settings = cls(**copy.deepcopy(dict(config)))
        root = Path(__file__).resolve().parents[3]
        settings.code_root = str(Path(settings.code_root) if settings.code_root else root)
        if not settings.scenario_root:
            starter = Path(settings.code_root) / "reference/starter-kit"
            settings.scenario_root = os.environ.get("SB_SCENARIO_ROOT") or (str(starter) if starter.is_dir()
                                                                             else os.getcwd())
        if settings.anchor not in ANCHORS or settings.switches not in SWITCHES:
            raise ValueError(f"anchor must be one of {ANCHORS}, switches one of {SWITCHES}")
        if settings.anchor == "student":
            if not settings.student_model:
                raise ValueError("anchor 'student' needs student_model (D-RL03 (b): the network starts from B1-L)")
            if settings.normalization:
                raise ValueError("anchor 'student' takes its statistics from the student; normalization must be empty")
            if settings.demo_lambda0 > 0 and not settings.demonstrations:
                raise ValueError("anchor 'student' needs demonstrations for the D-RL03 (b) term (or demo_lambda0 = 0)")
        else:
            if not settings.normalization:
                raise ValueError("anchor 'residual' needs normalization (REQ-021 (2): the B1-R statistics' source)")
            if settings.student_model:
                raise ValueError("anchor 'residual' does not use a student (REQ-021 (5)); student_model must be empty")
            if settings.log_std_init == "student":
                raise ValueError("anchor 'residual' has no student log_std; log_std_init must be a number")
        if not settings.rollout_scenarios:
            raise ValueError("rollout_scenarios is empty")
        if settings.num_envs * settings.num_steps < settings.minibatch_size:
            raise ValueError("minibatch_size is larger than num_envs x num_steps")
        if settings.max_chunk_samples is not None and int(settings.max_chunk_samples) < 1:
            raise ValueError("max_chunk_samples must be None or at least 1")
        if settings.lr_schedule not in LR_SCHEDULES:
            raise ValueError(f"lr_schedule must be one of {LR_SCHEDULES}")
        if settings.mean_activation not in MEAN_ACTIVATIONS:
            raise ValueError(f"mean_activation must be one of {MEAN_ACTIVATIONS}")
        if settings.critic_warmup_iterations is not None and settings.critic_warmup_loss is not None:
            raise ValueError("set at most one critic warm-up: critic_warmup_loss or critic_warmup_iterations")
        if settings.critic_warmup_iterations is not None and int(settings.critic_warmup_iterations) < 1:
            raise ValueError("critic_warmup_iterations must be None or at least 1")
        if settings.max_std is not None and not float(settings.max_std) > 0:
            raise ValueError("max_std must be None or positive")
        if settings.actor_learning_rate is not None and not float(settings.actor_learning_rate) > 0:
            raise ValueError("actor_learning_rate must be None or positive")
        if not settings.action_scale > 0:
            raise ValueError("action_scale must be positive")
        if settings.anchor == "student" and settings.action_scale != 1.0:
            raise ValueError("anchor 'student' acts with the network's output itself; action_scale must be 1")
        if settings.switch_prior not in SWITCH_PRIORS:
            raise ValueError(f"unknown switch_prior {settings.switch_prior!r}")
        if settings.switch_prior != "none" and settings.switches != "bernoulli":
            raise ValueError("switch_prior needs learned switches (switches 'bernoulli')")
        if not float(settings.switch_prior_gain) > 0:
            raise ValueError("switch_prior_gain must be positive")
        if settings.switch_learning_rate is not None and (settings.switches != "bernoulli"
                                                          or not float(settings.switch_learning_rate) > 0):
            raise ValueError("switch_learning_rate needs switches 'bernoulli' and a positive value")
        if settings.init_from is not None:
            digest = str((settings.init_from or {}).get("sha256", "")) if isinstance(settings.init_from, dict) else ""
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("init_from must be a file spec with the SHA-256 of the finished run's state.pt")
        if not isinstance(settings.schedule_origin, int) or settings.schedule_origin < 0:
            raise ValueError("schedule_origin must be a non-negative integer")
        if settings.schedule_origin and (settings.init_from is None or settings.lr_schedule != "warmup_constant"):
            raise ValueError("schedule_origin is a continuation's start (init_from) for lr_schedule warmup_constant")
        return settings

    def identity(self):
        values = asdict(self)
        return {k: values[k] for k in IDENTITY}

    def runtime(self):
        return {k: getattr(self, k) for k in RUNTIME}


def identity_changes(saved, current):
    """The keys (sorted) whose values differ between a saved identity and the current one; a key of IDENTITY_DEFAULTS
    that the saved identity lacks (saved before J10) counts as its default."""
    keys = set(IDENTITY) | set(saved)
    return sorted(k for k in keys if saved.get(k, IDENTITY_DEFAULTS.get(k)) != current.get(k))


# ------------------------------------------------------------------ small helpers (dagger.py's, imported)

def _dagger():
    from smart_building.learned import dagger
    return dagger


def sha256_file(path):
    return _dagger().sha256_file(path)


def code_module(code_root, relative):
    """A module of the code snapshot loaded from its file (the policies package registry is not in snapshots)."""
    import importlib.util
    path = Path(code_root) / relative
    spec = importlib.util.spec_from_file_location("sb_code_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_input(spec, settings):
    """A file the run reads: {"path"} (absolute or relative to code_root) or, on a remote shard,
    {"source": {"job", "shard"[, "attempt"]}, "file"}: the file inside that earlier output, found by
    its job-output.json marker under the input root (remote/paths.py), or {"dataset", "file"}: a file of a mounted
    private dataset (J10). "sha256" is checked when given."""
    if "path" in spec:
        path = Path(spec["path"])
        if not path.is_absolute():
            path = Path(settings.code_root) / path
    elif "dataset" in spec:
        # J10: a file of a private Kaggle dataset ({"dataset": "<user>/<slug>", "file", "sha256"}, e.g. a continued
        # run's state in <user>/sb-j9-states), mounted under the input root and found by name and SHA-256 the way the
        # submission notebooks find their model files (policies/mounted_models.py locate_models, loaded from the code)
        from smart_building.remote import paths
        mounted = code_module(settings.code_root, "packages/smart_building/policies/mounted_models.py")
        root = Path(os.environ.get(paths.ENV_INPUT_ROOT, paths.KAGGLE_INPUT))
        path = mounted.locate_models([{"fold": spec["file"], "file": spec["file"], "sha256": spec.get("sha256", ""),
                                       "source": {"type": "dataset", "ref": spec["dataset"]}}], root)[0]
    else:
        from smart_building.remote import paths
        source = spec["source"]
        root = Path(os.environ.get(paths.ENV_INPUT_ROOT, paths.KAGGLE_INPUT))
        found = [d for d, m in paths.find_markers(root, source["job"]) if m.get("shard") == source["shard"]
                 and (source.get("attempt") is None or m.get("attempt") == source["attempt"])]
        if len(found) != 1:
            raise FileNotFoundError(f"{source}: {len(found)} mounted outputs under {root} (expected 1)")
        path = found[0] / spec["file"]
    if not path.is_file():
        raise FileNotFoundError(path)
    if spec.get("sha256") and sha256_file(path) != spec["sha256"]:
        raise RuntimeError(f"{path}: SHA-256 differs from the configuration")
    return path


def metamorph_warmup(c, w):
    """MetaMorph's warm-up factor of iteration index c (from 0) with W = w warm-up iterations: WARMUP_FACTOR (1 - alpha)
    + alpha, alpha = c / W, for c < W (0.1, 0.28, 0.46, 0.64, 0.82 for W = 5), else 1."""
    warmup = 1.0
    if c < w:
        alpha = c / w
        warmup = METAMORPH_WARMUP_FACTOR * (1.0 - alpha) + alpha
    return warmup


def learning_rates(settings, iteration, num_iterations):
    """The groups' learning rates of iteration `iteration`: _group_learning_rates, plus "switch" (the switch heads'
    own group, reviewer 2026-10-25) = switch_learning_rate on the actor group's curve."""
    rates = _group_learning_rates(settings, iteration, num_iterations)
    if getattr(settings, "switch_learning_rate", None) is not None:
        actor_base = settings.learning_rate if settings.actor_learning_rate is None else float(settings.actor_learning_rate)
        rates["switch"] = float(settings.switch_learning_rate) * (rates["actor"] / actor_base if actor_base else 1.0)
    return rates


def _group_learning_rates(settings, iteration, num_iterations):
    """{"actor": lr, "critic": lr} of iteration `iteration` (1-based; every iteration counts, the critic-only
    warm-up too: REQ-021 (3)).

    "metamorph" (REQ-021 (3) (b)): MetaMorph `metamorph/utils/optimizer.py` (d498264): lr = BASE_LR x cos(c) x
    warmup(c), c = iteration - 1 counted from 0, cos(c) = (1 - MIN_LR) x 0.5 (1 + cos(pi c / MAX_ITERS)) + MIN_LR with
    MAX_ITERS the planned iterations (the cosine covers the warm-up) and MIN_LR 0; for c < WARMUP_ITERS (5): alpha =
    c / WARMUP_ITERS, warmup = WARMUP_FACTOR (1 - alpha) + alpha (0.1, 0.28, 0.46, 0.64, 0.82), else 1. Without
    critic_lr_warmup the critic group takes cos(c) only.
    "warmup_cosine" (REQ-021 (3) candidate (a)): lr = base x i / W for i <= W, then a cosine from base towards 0 over
    the remaining iterations with CleanRL's convention of never reaching 0 (L203-204); both groups the same curve.
    "warmup_constant" (J10, round 2 of stage 3; docs/reference/notes/16-rl-tuning-after-baseline.md §A4, §C5, §F1): lr =
    BASE_LR x warmup(c), MetaMorph's warm-up above (0.1, 0.28, 0.46, 0.64, 0.82 at c = 0-4) with c = iteration - 1 -
    schedule_origin (a continuation re-warms from the continued run's iteration), then BASE_LR for every later
    iteration; no cosine, so the curve does not depend on num_iterations (a shorter run is the start of a longer
    one). The critic group takes the warm-up only with critic_lr_warmup, as in "metamorph". Source: the PPO fine-tuning
    of a behaviour-cloned Gaussian policy in DPPO (irom-princeton/dppo @cc7234ad, cfg/furniture/finetune/one_leg_low/
    ft_ppo_gaussian_mlp.yaml L48-59): actor 1e-5 on a cosine cycle of 10,000 steps of which only 1,000 iterations run
    (in effect constant), critic 1e-3 constant; actor and critic bases kept apart (here 3e-6 and 3e-4).
    BASE_LR is learning_rate for the critic group and actor_learning_rate (when set) for the actor group, the same curve
    on both (the student anchor: actor 3e-6, critic 3e-4; user decision 2026-10-21 02:05)."""
    base, w = settings.learning_rate, settings.warmup_iterations
    actor_base = base if settings.actor_learning_rate is None else float(settings.actor_learning_rate)
    if settings.lr_schedule == "metamorph":
        c = iteration - 1
        cosine = (1.0 - METAMORPH_MIN_LR) * (0.5 * (1.0 + math.cos(math.pi * c / num_iterations))) + METAMORPH_MIN_LR
        warmup = metamorph_warmup(c, w)
        return {"actor": cosine * actor_base * warmup,
                "critic": cosine * base * (warmup if settings.critic_lr_warmup else 1.0)}
    if settings.lr_schedule == "warmup_constant":
        c = iteration - 1 - int(settings.schedule_origin)
        if c < 0:
            raise ValueError(f"iteration {iteration} is not after the schedule origin {settings.schedule_origin}")
        warmup = metamorph_warmup(c, w)
        return {"actor": actor_base * warmup, "critic": base * (warmup if settings.critic_lr_warmup else 1.0)}
    if iteration <= w:
        factor = iteration / w
    else:
        progress = (iteration - w - 1) / max(1, num_iterations - w)
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return {"actor": actor_base * factor, "critic": base * factor}


def gae(rewards, values, dones, next_value, next_done, gamma, gae_lambda, end_values=None):
    """CleanRL L232-246: (advantages, returns), tensors (T, N).

    dones[t] = 1 when the observation of step t starts a new episode (the previous step ended one); next_value and
    next_done are those of the observation after the last step. end_values[t] (REQ-021 (12), Pardo et al. 2018
    Eq. 6: y = r + gamma v(s') at time limits) is v(s_last) of the episode that ended at step t (its last
    observation, which the next episode's first observation replaces), 0 elsewhere; None = no bootstrap (CleanRL
    L221: every end is terminal). The GAE recursion still stops at every episode boundary."""
    import torch
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(T)):
        if t == T - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
        if end_values is None:
            delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        else:
            delta = rewards[t] + gamma * (nextvalues * nextnonterminal + end_values[t]) - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
    return advantages, advantages + values


def window_inputs(spec, rows, offset, step, history):
    """Token inputs at episode step `step` from the episode's rows (rows[0] is step `offset`), with
    the oldest-row padding of REQ-014 (3): tokens.token_inputs, which is the offset-0 case."""
    from smart_building.learned.tokens import TOKEN_TYPES
    index = np.maximum(np.arange(step - history + 1, step + 1), 0) - offset
    if index.min() < 0:
        raise IndexError(f"step {step}: rows before {offset} were dropped")
    window = rows[index]
    return window_tokens(spec, window, TOKEN_TYPES)


def window_tokens(spec, window, kinds):
    out = {}
    for kind in kinds:
        idx = spec[f"dynamic_{kind}"]
        stacked = window[:, idx]
        dyn = np.transpose(stacked, (1, 0, 2)).reshape(idx.shape[0], -1)
        out[kind] = np.concatenate([dyn, spec[f"static_{kind}"]], axis=1).astype(np.float32)
    return out


def scatter(spec, outputs, n_continuous):
    """Continuous action vector from per-type arrays (tokens.TokenLayout.scatter on the stored tables)."""
    action = np.zeros(n_continuous, dtype=np.float32)
    action[spec["damper"]] = outputs["zone"][:, 0]
    action[spec["valve"]] = outputs["zone"][:, 1]
    action[spec["sat"]] = outputs["ahu"][:, 0]
    action[spec["sp"]] = outputs["ahu"][:, 1]
    action[spec["recirculation"]] = outputs["ahu"][:, 2]
    action[spec["water_temp"]] = outputs["hws"][:, 0]
    action[spec["water_dp"]] = outputs["hws"][:, 1]
    return action


def _features(base):
    """The B1-R state an env slot sent with its base action (privileged slots, ppo_env.EnvSlot._base), or None."""
    return base.get("features") if isinstance(base, dict) else None


def combine(base, residual):
    """pi_B1-R + f (D-RL03 (a)). Adding an exact zero keeps the base value itself, including the sign
    of a zero (IEEE -0.0 + 0.0 = +0.0), so a zero residual returns the rule controller's bits."""
    base = np.asarray(base, dtype=np.float32)
    residual = np.asarray(residual, dtype=np.float32)
    return np.where(residual == 0, base, base + residual).astype(np.float32)


def _flush(stream):
    """An open file's data on stable storage: flush, fsync, and on macOS F_FULLFSYNC (also the drive's write cache)."""
    stream.flush()
    os.fsync(stream.fileno())
    if fcntl is not None and hasattr(fcntl, "F_FULLFSYNC"):
        try:
            fcntl.fcntl(stream.fileno(), fcntl.F_FULLFSYNC)
        except OSError:
            pass


def _fsync_directory(path):
    """A folder's entries (a created, renamed or replaced file) on stable storage, where the system allows it."""
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _write_bytes(path, data):
    with open(path, "wb") as stream:
        stream.write(data)
        _flush(stream)


def _write_json(path, value):
    """JSON written to a temporary file, flushed, then renamed over `path` (atomic replacement)."""
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    with open(temporary, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=1, sort_keys=True, default=str) + "\n")
        _flush(stream)
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def checkpoint_candidates(directory):
    """The checkpoints of a run folder in the order a resume tries them: latest.json's, then every other complete
    ckpt-<i> folder (ckpt-<i>.partial is an interrupted save) from the newest iteration down."""
    directory = Path(directory)
    names = []
    try:
        names.append(json.loads((directory / "latest.json").read_text(encoding="utf-8"))["checkpoint"])
    except (OSError, ValueError, KeyError, TypeError):
        pass
    names += sorted((p.name for p in directory.glob("ckpt-*") if p.is_dir() and p.name[len("ckpt-"):].isdigit()),
                    reverse=True)
    return [directory / name for name in dict.fromkeys(names)]


# ------------------------------------------------------------------ networks (CleanRL L106-141, D-RL02)

def layer_init(layer, std=math.sqrt(2), bias_const=0.0):
    """CleanRL L106-109."""
    import torch
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def encode(model, tokens, clip):
    """EntityTransformer.forward up to the decoder input (model.py, D-L02, D-L04): normalised inputs
    (frozen statistics), then the +-clip of D-RL02 (None: none), typed tokenizers x sqrt(d), the
    encoder, and per type the encoder output concatenated with the token's own normalised features.
    Without clipping, the continuous decoders applied to this equal EntityTransformer.forward bit for bit."""
    import torch
    x = model.normalized(tokens)
    if clip is not None:
        x = {k: v.clamp(-clip, clip) for k, v in x.items()}
    counts = [x[k].shape[1] for k in model.kinds]
    embedded = torch.cat([model.tokenizers[k](x[k]) * model.scale for k in model.kinds], dim=1)
    encoded = model.encoder(embedded)
    parts = torch.split(encoded, counts, dim=1)
    return {k: torch.cat([part, x[k]], dim=-1) for k, part in zip(model.kinds, parts)}


STATISTICS_KEYS = ("input_index_", "input_mean_", "input_std_")


def load_normalization(path):
    """REQ-021 (2): the frozen observation statistics of a save_student file (model.py: the per-type, per-feature
    mean and population std of the first-iteration B1-R demonstrations and the kept input positions, REQ-014 (1, 2),
    REQ-018 (6)), its StudentConfig (the history H the statistics are laid out for, the D-L05 sizes) and input
    sizes. The network weights in the file are not read. Returns a dict with the statistics as numpy arrays and
    "sha256", a digest of exactly those arrays (key, dtype, shape, bytes)."""
    import torch
    from smart_building.learned.model import MODEL_FORMAT
    from smart_building.learned.tokens import TOKEN_TYPES
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != MODEL_FORMAT:
        raise ValueError(f"{path}: not a {MODEL_FORMAT} file")
    state = payload["state_dict"]
    digest = hashlib.sha256()
    arrays = {}
    for kind in TOKEN_TYPES:
        for prefix in STATISTICS_KEYS:
            name = prefix + kind
            if name not in state:
                raise KeyError(f"{path}: no {name} (statistics are buffers of the student model)")
            value = state[name].detach().cpu().numpy()
            arrays[name] = value
            digest.update(name.encode() + str(value.dtype).encode() + str(value.shape).encode() + value.tobytes())
    return {"config": dict(payload["config"]), "input_dims": dict(payload["input_dims"]),
            "keep": {k: arrays[f"input_index_{k}"].tolist() for k in TOKEN_TYPES},
            "mean": {k: arrays[f"input_mean_{k}"] for k in TOKEN_TYPES},
            "std": {k: arrays[f"input_std_{k}"] for k in TOKEN_TYPES},
            "sha256": digest.hexdigest(), "teacher": payload.get("teacher"),
            "iteration": payload.get("iteration"), "statistics": payload.get("statistics")}


def build_agent(config, input_dims, settings, actor_state=None, critic_state=None, normalization=None):
    """The Agent of a run: actor and critic entity transformers (D-RL02) with the dropout of the settings.

    Anchor "student", new run: the actor is the student (its state: weights, frozen statistics, kept positions and
    log_std); the critic has the same architecture with a random initialisation (torch's global generator; the caller
    seeds it) and the student's frozen statistics and kept positions (REQ-021 (5) (b), reviewer 2026-10-21: DAPG's and
    MetaMorph's critics are trained from scratch).
    Anchor "residual", new run (REQ-021 (5)): both with a random initialisation (torch's global generator; the
    caller seeds it) and the frozen statistics and kept positions of `normalization` (load_normalization).
    A saved agent (resume, evaluation) passes its actor and critic states."""
    import torch
    from smart_building.learned.model import EntityTransformer, StudentConfig, keep_from_state
    from smart_building.learned.tokens import TOKEN_TYPES
    config = StudentConfig(**{**config, "dropout": settings.dropout})
    if actor_state is not None:
        keep = keep_from_state(actor_state)
        actor = EntityTransformer(input_dims, config, keep)
        actor.load_state_dict(actor_state)
        critic = EntityTransformer(input_dims, config, keep)
        if critic_state is None:
            if settings.anchor != "student":
                raise ValueError("a residual agent needs its saved critic state")
            with torch.no_grad():               # REQ-021 (5) (b): random trunk, the student's frozen statistics
                for kind in TOKEN_TYPES:
                    for prefix in ("input_mean_", "input_std_"):
                        getattr(critic, prefix + kind).copy_(actor_state[prefix + kind])
        else:
            critic.load_state_dict(critic_state)
    else:
        if settings.anchor != "residual" or normalization is None:
            raise ValueError("a new agent without saved states is the residual anchor's, with its normalization")
        actor = EntityTransformer(input_dims, config, normalization["keep"])
        actor.set_normalization(normalization["mean"], normalization["std"])
        critic = EntityTransformer(input_dims, config, normalization["keep"])
        critic.set_normalization(normalization["mean"], normalization["std"])
    return Agent(actor, critic, anchor=settings.anchor, switches=settings.switches,
                 observation_clip=settings.observation_clip, switch_init_gain=settings.switch_init_gain,
                 min_log_std=settings.min_log_std, log_std_init=settings.log_std_init,
                 learn_log_std=settings.learn_log_std, mean_activation=settings.mean_activation,
                 max_std=settings.max_std, switch_own_group=settings.switch_learning_rate is not None,
                 switch_zero_init=settings.switch_prior == "rule")


def draw_noise(mean, generator):
    """Standard normal noise shaped like `mean`, drawn from the run's CPU generator and moved to mean's device: a
    device "cuda" run (the update on a GPU, the env workers on the CPU) draws the same sequence as a CPU run, and
    the generator state in the checkpoints stays a CPU state. On the CPU the move returns the tensor itself."""
    import torch
    return torch.randn(mean.shape, generator=generator, dtype=mean.dtype).to(mean.device)


def _agent_class():
    import torch
    from torch import nn
    from torch.distributions import Bernoulli, Normal
    from smart_building.learned.tokens import AHU, HWS, OUTPUT_DIMS, TOKEN_TYPES

    class Agent(nn.Module):
        """CleanRL Agent (L112-141) on the entity networks of D-RL02.

        actor: an EntityTransformer; Gaussian mean per token = tanh(decoder) (D-L04; the student, D-RL03 (b)) or the
          decoder's linear output with zero-initialised decoders (the residual f of D-RL03 (a)). The distribution is
          in the network's own space; Run.act and PPOPolicy execute action_scale x f for the residual.
        actor_logstd: CleanRL's state-independent log-std (L129), one per token type and output (D-L04 type
          sharing). REQ-021 (7): a constant or the student's log_std (D-L09); learned or not (requires_grad);
          min_log_std is mjrl's lower clamp of the parameter (`policies/gaussian_mlp.py` @3871d93 MLP, after every
          update), None for none. max_std caps the sigma the Gaussian uses, in the forward pass only (DPPO: the
          variance clamped between std_min^2 and std_max^2 when the distribution is formed, std_max 0.2): sigma =
          exp(min(log_std, log max_std)); the parameter keeps its value (the student's at the start) and gets no
          gradient while above the cap. None for no cap.
        critic: a separate EntityTransformer with one scalar head per token type (CleanRL L121 output std 1.0);
          value = mean over all tokens (MetaMorph §4.2).
        switch_heads (D-RL04 (b)): one Bernoulli logit per AHU token and for the hot-water token.
        """

        def __init__(self, actor, critic, *, anchor, switches, observation_clip, switch_init_gain, min_log_std,
                     log_std_init, learn_log_std, mean_activation, max_std=None, switch_own_group=False,
                     switch_zero_init=False):
            super().__init__()
            self.actor, self.critic = actor, critic
            self.anchor, self.switches, self.observation_clip = anchor, switches, observation_clip
            self.switch_own_group = bool(switch_own_group)       # reviewer 2026-10-25: switch_learning_rate
            self.min_log_std, self.mean_activation = min_log_std, mean_activation
            self.max_log_std = None if max_std is None else math.log(float(max_std))
            d = actor.config.d_model
            if isinstance(log_std_init, str):
                if log_std_init != "student":
                    raise ValueError(f"log_std_init {log_std_init!r}: a number or 'student'")
                initial = {k: getattr(actor, f"log_std_{k}").detach().clone() for k in actor.decoders}
            else:
                initial = {k: torch.full((OUTPUT_DIMS[k],), float(log_std_init)) for k in actor.decoders}
            self.actor_logstd = nn.ParameterDict({k: nn.Parameter(v, requires_grad=bool(learn_log_std))
                                                  for k, v in initial.items()})
            self.clamp_log_std()
            self.value_heads = nn.ModuleDict({k: layer_init(nn.Linear(d + critic.kept_dims[k], 1), std=1.0)
                                              for k in TOKEN_TYPES})
            if switches == "bernoulli":
                # REQ-021 (10) placeholder: as the student's last layers (REQ-014 (4): orthogonal, gain 0.01, bias 0).
                self.switch_heads = nn.ModuleDict({k: layer_init(nn.Linear(d + actor.kept_dims[k], 1),
                                                                 std=switch_init_gain) for k in (AHU, HWS)})
                if switch_zero_init:
                    # reviewer 2026-10-25 (switch_prior rule): the switch heads are the residual on B1-R's decisions,
                    # so they start at exactly 0 (D-RL03 (a): the residual's last layer zero-initialised; Silver et al.
                    # 2018): the first decisions are B1-R's whatever the inputs (the heads read raw kept inputs too)
                    for head in self.switch_heads.values():
                        nn.init.zeros_(head.weight)
                        nn.init.zeros_(head.bias)
            if anchor == "residual":
                for decoder in actor.decoders.values():                    # D-RL03 (a): f starts at 0
                    nn.init.zeros_(decoder.weight)
                    nn.init.zeros_(decoder.bias)

        def clamp_log_std(self):
            """mjrl set_param_values: log_std = max(log_std, min_log_std) (REQ-021 (7)); only a learned log_std."""
            if self.min_log_std is not None:
                with torch.no_grad():
                    for p in self.actor_logstd.values():
                        if p.requires_grad:
                            p.clamp_(min=self.min_log_std)

        def parameter_groups(self):
            """Adam parameter groups (REQ-021 (3)): "actor" (actor, learned log std, switch heads) and "critic"
            (critic and value heads); parameters that are not learned are left out. Adam's state is per
            parameter, so the split changes only which learning rate each parameter gets."""
            critic = list(self.critic.parameters()) + list(self.value_heads.parameters())
            ids = {id(p) for p in critic}
            switch = []
            if self.switch_own_group:           # reviewer 2026-10-25: the switch heads on their own learning rate
                switch = [p for p in self.switch_heads.parameters() if p.requires_grad]
                ids |= {id(p) for p in switch}
            actor = [p for p in self.parameters() if id(p) not in ids and p.requires_grad]
            groups = [{"params": actor, "name": "actor"},
                      {"params": [p for p in critic if p.requires_grad], "name": "critic"}]
            if switch:
                groups.append({"params": switch, "name": "switch"})
            return groups

        def _mean(self, decoded):
            if self.mean_activation == "linear":                                     # the residual f (D-RL03 (a))
                return decoded
            return torch.tanh(decoded)                                                # D-L04

        def get_value(self, tokens):
            """CleanRL L131-132: (batch,) value of each sample."""
            f = encode(self.critic, tokens, self.observation_clip)
            per_token = torch.cat([self.value_heads[k](f[k]).squeeze(-1) for k in TOKEN_TYPES], dim=1)
            return per_token.mean(dim=1)

        def policy(self, tokens, switch_prior=None):
            """({type: Normal}, {type: Bernoulli} or None) of each sample. switch_prior ({AHU, HWS: logits shaped
            as the heads'}) is added to the switch heads' logits (switch_prior "rule", reviewer 2026-10-25)."""
            f = encode(self.actor, tokens, self.observation_clip)
            normals = {}
            for k in self.actor.decoders:
                mean = self._mean(self.actor.decoders[k](f[k]))
                log_std = self.actor_logstd[k]
                if self.max_log_std is not None:                       # DPPO: the ceiling in the forward pass
                    log_std = torch.clamp(log_std, max=self.max_log_std)
                normals[k] = Normal(mean, torch.exp(log_std).expand_as(mean))
            bernoullis = None
            if self.switches == "bernoulli":
                bernoullis = {}
                for k in (AHU, HWS):
                    logits = self.switch_heads[k](f[k]).squeeze(-1)
                    if switch_prior is not None:
                        logits = logits + switch_prior[k]
                    bernoullis[k] = Bernoulli(logits=logits)
            return normals, bernoullis

        def mean_action(self, tokens):
            f = encode(self.actor, tokens, self.observation_clip)
            return {k: self._mean(self.actor.decoders[k](f[k])) for k in self.actor.decoders}

        @staticmethod
        def _sum(values):
            total = None
            for v in values:
                v = v.flatten(1).sum(dim=1)
                total = v if total is None else total + v
            return total

        def get_action_and_value(self, tokens, action=None, switch=None, generator=None, with_actor=True,
                                 switch_prior=None):
            """CleanRL L134-141: sample (or score the given) action; log-prob and entropy summed over
            all action dimensions of the building. Returns a dict."""
            out = {"value": self.get_value(tokens)}
            if not with_actor:
                return out
            normals, bernoullis = self.policy(tokens) if switch_prior is None else self.policy(tokens, switch_prior)
            if action is None:
                action = {k: n.mean + n.stddev * draw_noise(n.mean, generator) for k, n in normals.items()}
            out.update(action=action, logprob=self._sum(normals[k].log_prob(action[k]) for k in normals),
                       entropy=self._sum(normals[k].entropy() for k in normals))
            if bernoullis is not None:
                if switch is None:
                    switch = {k: torch.bernoulli(b.probs.cpu(), generator=generator).to(b.probs.device)
                              for k, b in bernoullis.items()}
                out.update(switch=switch, switch_logprob=self._sum(bernoullis[k].log_prob(switch[k]) for k in bernoullis),
                           switch_entropy=self._sum(bernoullis[k].entropy() for k in bernoullis))
            return out

        def log_prob(self, tokens, action):
            normals, _ = self.policy(tokens)
            return self._sum(normals[k].log_prob(action[k]) for k in normals)

    return Agent


_AGENT = None


def Agent(*args, **kwargs):  # noqa: N802 - the class is built lazily so this module imports without torch
    global _AGENT
    if _AGENT is None:
        _AGENT = _agent_class()
    return _AGENT(*args, **kwargs)


def privileged_entry(payload, path=None):
    """None for a student without B1-R state inputs (j7b-v4); for a j7b-v5 / v5s student, its learned/privileged.py
    entry {"format", "features"}, checked as privileged.load_student_v5 checks it (format and feature list)."""
    entry = payload.get("privileged")
    if entry is None:
        return None
    from smart_building.learned.privileged import FEATURES, PRIVILEGED_FORMAT
    if (not isinstance(entry, dict) or entry.get("format") != PRIVILEGED_FORMAT
            or {k: tuple(v) for k, v in entry.get("features", {}).items()} != FEATURES):
        raise ValueError(f"{path}: privileged entry {entry!r} is not {PRIVILEGED_FORMAT} with privileged.FEATURES")
    return {"format": entry["format"], "features": {k: list(v) for k, v in entry["features"].items()}}


def load_student_file(path, with_privileged=False):
    """(student config dict, input_dims, state dict, teacher spec) of a save_student file; with_privileged adds its
    privileged_entry (None for a j7b-v4 student)."""
    import torch
    from smart_building.learned.model import MODEL_FORMAT
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != MODEL_FORMAT:
        raise ValueError(f"{path}: not a {MODEL_FORMAT} file")
    out = (payload["config"], payload["input_dims"], payload["state_dict"], payload.get("teacher"))
    return out + (privileged_entry(payload, path),) if with_privileged else out


# ------------------------------------------------------------------ episodes and samples

class Episode:
    """Observation rows of one episode on the trainer's side (rows[0] is episode step `offset`). With privileged
    (a j7b-v5 / v5s student), each row also keeps the B1-R state read before the rule controller's call on that
    observation (ppo_env.EnvSlot), appended to the zone, AHU and hot-water tokens as privileged.append_features does
    for the v5s students (StudentNetworkV5.mean_action; dagger_v5.RolloutV5.inputs)."""

    def __init__(self, key, label, tables, history, privileged=False):
        from smart_building.learned.tokens import HistoryBuffer
        self.key, self.label, self.tables = key, label, tables
        self.rows, self.offset, self.steps = [], 0, 0
        self.privileged = bool(privileged)
        self.features = []             # privileged: per row, {type: float32 (count, features)}
        self.buffer = HistoryBuffer(history, int(tables["n_fields"]))
        self.n_continuous = int(len(tables["low"]))
        self.n_ahu = int(tables["dynamic_ahu"].shape[0])
        self._array = None

    def _with_features(self, tokens, features):
        from smart_building.learned.privileged import append_features
        return append_features(tokens, features) if self.privileged else tokens

    def push(self, obs, features=None):
        """The observation of episode step len(rows) + offset (and, privileged, the B1-R state before the rule
        controller's call on it); returns its token inputs (REQ-014 (3) padding)."""
        from smart_building.learned.privileged import PRIVILEGED_TYPES
        from smart_building.learned.tokens import TOKEN_TYPES
        if self.privileged:
            if features is None:
                raise RuntimeError(f"{self.label}: the B1-R state of the step is missing")
            self.features.append({k: np.asarray(features[k], dtype=np.float32) for k in PRIVILEGED_TYPES})
        self.rows.append(np.asarray(obs, dtype=np.float32))
        self._array = None
        window = self.buffer.push(obs)
        return self._with_features(window_tokens(self.tables, window, TOKEN_TYPES),
                                   self.features[-1] if self.privileged else None)

    def current_tokens(self):
        """The token inputs of the last pushed row (the history window as it stands, and its B1-R state)."""
        from smart_building.learned.tokens import TOKEN_TYPES
        return self._with_features(window_tokens(self.tables, self.buffer.buffer, TOKEN_TYPES),
                                   self.features[-1] if self.privileged else None)

    def inputs(self, steps, history):
        from smart_building.learned.tokens import TOKEN_TYPES
        if self._array is None:
            self._array = np.stack(self.rows)
        per = [self._with_features(window_inputs(self.tables, self._array, self.offset, int(s), history),
                                   self.features[int(s) - self.offset] if self.privileged else None) for s in steps]
        return {k: np.stack([p[k] for p in per]) for k in TOKEN_TYPES}

    def trim(self, keep):
        drop = max(0, len(self.rows) - keep)
        if drop:
            self.rows = self.rows[drop:]
            if self.privileged:
                self.features = self.features[drop:]
            self.offset += drop
            self._array = None

    def state(self):
        import torch
        state = {"key": list(self.key), "label": self.label, "offset": self.offset,
                 "rows": torch.from_numpy(np.stack(self.rows)), "buffer": torch.from_numpy(self.buffer.buffer.copy()),
                 "filled": bool(self.buffer.filled),
                 "tables": {k: torch.from_numpy(np.asarray(v).copy()) for k, v in self.tables.items()}}
        if self.privileged:
            state["features"] = {k: torch.from_numpy(np.stack([f[k] for f in self.features]))
                                 for k in self.features[0]}
        return state

    @classmethod
    def from_state(cls, state, history):
        tables = {k: v.numpy() for k, v in state["tables"].items()}
        episode = cls(tuple(state["key"]), state["label"], tables, history, privileged="features" in state)
        episode.rows = list(state["rows"].numpy())
        if episode.privileged:
            stacked = {k: v.numpy() for k, v in state["features"].items()}
            episode.features = [{k: stacked[k][j] for k in stacked} for j in range(len(episode.rows))]
        episode.offset = int(state["offset"])
        episode.buffer.buffer[:] = state["buffer"].numpy()
        episode.buffer.filled = bool(state["filled"])
        return episode


class Demonstrations:
    """D-RL03 (b) demonstration pairs (state, teacher action): stored rollout files (dataset.Rollout:
    the observation rows, every 10th step labelled with the teacher's continuous action). For a privileged (j7b-v5 /
    v5s) student they are read as dagger_v5.RolloutV5, the reader of that student's own training: the B1-R state the
    rollout stored next to each label is appended to the token inputs.

    REQ-021 (8) data (a): the student chain's first-iteration teacher rollouts, so every file must be a teacher rollout
    of iteration 1 (its meta) by the student's teacher when the student file names one."""

    def __init__(self, paths, privileged=False, teacher=None):
        if privileged:
            from smart_building.learned.dagger_v5 import RolloutV5 as Rollout
        else:
            from smart_building.learned.dataset import Rollout
        self.rollouts = [Rollout(p) for p in paths]
        for path, rollout in zip(paths, self.rollouts):
            meta = rollout.meta
            if meta.get("actor") != "teacher" or int(meta.get("iteration", -1)) != 1:
                raise ValueError(f"{path}: not a first-iteration teacher rollout (actor {meta.get('actor')!r}, "
                                 f"iteration {meta.get('iteration')!r}; REQ-021 (8) data (a))")
            if teacher is not None and meta.get("teacher") not in (None, teacher):
                raise ValueError(f"{path}: rollout teacher {meta.get('teacher')} is not the student's teacher {teacher}")
        self.index = [(r, p) for r, rollout in enumerate(self.rollouts) for p in range(len(rollout))]

    def records(self, paths):
        """What identifies the data: per file its SHA-256, label, samples and teacher."""
        return [{"file": str(p), "sha256": sha256_file(p), "label": r.meta.get("label"), "samples": len(r),
                 "teacher": r.meta.get("teacher")} for p, r in zip(paths, self.rollouts)]

    def statistics_check(self, actor_state, history):
        """REQ-021 (8) data (a), checked: the student's frozen input statistics and log_std were computed from its
        chain's first-iteration demonstrations (dagger.first_statistics / dagger_v5.first_statistics), so recomputing
        them from these files (dataset.input_statistics, action_statistics) gives the same kept positions and, after
        the float32 cast of set_normalization / set_log_std, the same values. Returns the comparison; raises when the
        kept positions differ or a value differs beyond float32 rounding (other data)."""
        from smart_building.learned.dataset import action_statistics, input_statistics
        from smart_building.learned.model import LOG_STD_EPS
        from smart_building.learned.tokens import TOKEN_TYPES
        mean, std, _ = input_statistics(self.rollouts, history)
        report, bitwise, problems = {}, True, []
        for k in TOKEN_TYPES:
            keep = np.flatnonzero(std[k] > 0.0)
            saved_keep = actor_state[f"input_index_{k}"].numpy()
            if not np.array_equal(keep, saved_keep):
                problems.append(f"{k}: kept positions {len(keep)} vs the student's {len(saved_keep)}")
                continue
            pairs = ((np.float32(mean[k][keep]), actor_state[f"input_mean_{k}"].numpy()),
                     (np.float32(std[k][keep]), actor_state[f"input_std_{k}"].numpy()))
            for name, (got, saved) in zip(("mean", "std"), pairs):
                bitwise &= bool(np.array_equal(got.view(np.uint32), saved.view(np.uint32)))
                diff = float(np.max(np.abs(got.astype(np.float64) - saved))) if got.size else 0.0
                report[f"{name}_{k}_max_abs_diff"] = diff
                if not np.allclose(got, saved, rtol=1e-5, atol=0.0):
                    problems.append(f"{name}_{k}: max |difference| {diff:.3g}")
        actions = action_statistics(self.rollouts)
        for k, value in actions.items():
            got = np.float32(np.log(np.asarray(value, dtype=np.float64) + LOG_STD_EPS))
            saved = actor_state[f"log_std_{k}"].numpy()
            bitwise &= bool(np.array_equal(got.view(np.uint32), saved.view(np.uint32)))
            if not np.allclose(got, saved, rtol=1e-5, atol=1e-6):
                problems.append(f"log_std_{k}: {got.tolist()} vs the student's {saved.tolist()}")
        if problems:
            raise ValueError("the demonstrations are not the data of the student's frozen statistics (REQ-021 (8) "
                             f"data (a): its chain's first-iteration teacher rollouts): {problems}")
        return {"match": True, "bitwise": bitwise, "samples": len(self.index), **report}

    def __len__(self):
        return len(self.index)


# ------------------------------------------------------------------ the run

class Run:
    """The state of one PPO run inside a segment."""

    def __init__(self, settings, out_dir):
        import torch
        self.settings = s = settings
        self.out_dir = Path(out_dir)
        self.batch_size = s.num_envs * s.num_steps                       # CleanRL L146
        self.num_iterations = s.total_timesteps // self.batch_size       # CleanRL L148
        self.iteration, self.global_step = 0, 0
        self.actor_training = s.critic_warmup_loss is None and not s.critic_warmup_iterations   # no critic warm-up
        self.actor_updates = 0                                           # DAPG k (mjrl iter_count)
        self.history_logs, self.counters = [], {"episodes": 0, "env_steps": 0, "minibatches": 0}
        self.snapshots = []            # agent files saved at agent_snapshot_steps: [{"targets", "iteration", "file", ...}]
        sequence = np.random.SeedSequence(s.seed)
        shuffle_seed, scenario_seed = sequence.spawn(2)
        self.shuffle_rng = np.random.Generator(np.random.PCG64(shuffle_seed))   # CleanRL L260
        self.scenario_rng = np.random.Generator(np.random.PCG64(scenario_seed))  # REQ-021 (9)
        self.sampler = torch.Generator().manual_seed(s.seed)              # action noise
        self.agent = self.optimizer = None
        self.network = None            # {"kind", "sha256", "config", "input_dims", ...}: where the networks came from
        self.envs = None
        self.streams = []              # per env slot: {"episode", "tokens", "base", "normalizer", "next_done", "serial"}
        self.episodes = {}             # key -> Episode, every episode with samples in this iteration
        self.demonstrations = None
        self.clock = {}                # the last rollout's time split (inference, environment steps, scenario starts)
        self.checkpoint_seconds = None  # how long the last save() took (logged with the next iteration)
        self.resume = None             # {"checkpoint", "skipped"}: where this segment resumed from
        self.continued = None          # J10: what this run continues (init_from; Run.continue_from), kept in checkpoints

    # -- construction --------------------------------------------------

    def history_length(self):
        return int(self.network["config"]["history"])

    def privileged(self):
        """Whether the networks read the B1-R internal state (a j7b-v5 / v5s student, learned/privileged.py)."""
        return bool((self.network or {}).get("privileged"))

    def need_base(self):
        return self.settings.anchor == "residual" or self.settings.switches == "backbone" or self.privileged()

    def build(self, agent_state=None, optimizer_state=None):
        import torch
        s = self.settings
        student_state = None
        if agent_state is None:
            if s.anchor == "student":
                path = resolve_input(s.student_model, s)
                config, input_dims, student_state, teacher, privileged = load_student_file(path, with_privileged=True)
                if privileged is not None and teacher != s.rule_controller:
                    # REQ-021 (11) (c): the rule controller is the student chain's own teacher, the source of its
                    # B1-R state inputs (dagger_v5 reads them from that teacher before each of its calls).
                    raise ValueError(f"the student reads the B1-R state of its teacher {teacher}; rule_controller "
                                     f"{s.rule_controller} must be that class")
                self.network = {"kind": "student", "sha256": sha256_file(path), "config": config,
                                "input_dims": input_dims, "teacher": teacher, "privileged": privileged}
                torch.manual_seed(s.seed)                                 # CleanRL L171; REQ-021 (5) (b) critic
                self.agent = build_agent(config, input_dims, s, actor_state=student_state).to(s.device)
            else:
                path = resolve_input(s.normalization, s)
                stats = load_normalization(path)
                self.network = {"kind": "normalization", "sha256": sha256_file(path),
                                "statistics_sha256": stats["sha256"], "config": stats["config"],
                                "input_dims": stats["input_dims"], "teacher": stats["teacher"],
                                "source_iteration": stats["iteration"], "source_statistics": stats["statistics"]}
                torch.manual_seed(s.seed)                                 # CleanRL L171; REQ-021 (5) random init
                self.agent = build_agent(stats["config"], stats["input_dims"], s, normalization=stats).to(s.device)
        else:
            actor = {k[len("actor."):]: v for k, v in agent_state.items() if k.startswith("actor.")}
            critic = {k[len("critic."):]: v for k, v in agent_state.items() if k.startswith("critic.")}
            self.agent = build_agent(self.network["config"], self.network["input_dims"], s, actor_state=actor,
                                     critic_state=critic).to(s.device)
            self.agent.load_state_dict(agent_state)
        self.optimizer = torch.optim.Adam(self.agent.parameter_groups(), lr=s.learning_rate, eps=s.adam_eps)   # L183
        if optimizer_state is not None:
            # The checkpoint is read with map_location "cpu" (Run.load); the agent was moved to the device above, and
            # Adam's moment buffers follow their parameters (load_state_dict casts them; made explicit here so a run
            # saved on "cuda" continues on "cpu" and the reverse). Adam's step counts stay CPU tensors.
            self.optimizer.load_state_dict(optimizer_state)
            for group in self.optimizer.param_groups:
                for parameter in group["params"]:
                    for key, value in self.optimizer.state.get(parameter, {}).items():
                        if key != "step" and torch.is_tensor(value) and value.device != parameter.device:
                            self.optimizer.state[parameter][key] = value.to(parameter.device)
        if s.anchor == "student" and s.demo_lambda0 > 0:
            paths = [resolve_input(d, s) for d in s.demonstrations]
            self.demonstrations = Demonstrations(paths, privileged=self.privileged(), teacher=self.network.get("teacher"))
            records = self.demonstrations.records(paths)
            if student_state is not None:
                # A new run: record the data and check it is the student chain's first iteration (REQ-021 (8) (a)).
                check = self.demonstrations.statistics_check(student_state, self.history_length())
                self.network["demonstrations"] = {
                    "files": [{"spec": d, **{k: r[k] for k in ("sha256", "label", "samples", "teacher")}}
                              for d, r in zip(s.demonstrations, records)],
                    "statistics_check": check}
                self.network = _dagger().plain(self.network)
            else:
                saved = (self.network.get("demonstrations") or {}).get("files")
                if saved is not None and [r["sha256"] for r in saved] != [r["sha256"] for r in records]:
                    raise ValueError("the demonstration files differ from those the run started with")

    def slot_arguments(self):
        s = self.settings
        return {"scenario_root": s.scenario_root, "code_root": s.code_root, "rule_controller": s.rule_controller,
                "history": self.history_length(), "steps_cap": s.steps_cap, "need_base": self.need_base(),
                "privileged": self.privileged()}

    def open_envs(self):
        from smart_building.learned.ppo_env import EnvSlots
        self.envs = EnvSlots(self.settings.num_envs, self.slot_arguments(), self.settings.in_process)

    def draw_scenario(self):
        """REQ-021 (9) (a): uniform with replacement over the fold's TRAIN
        scenarios (MetaMorph's length-balanced draw reduces to this for equal-length episodes)."""
        specs = self.settings.rollout_scenarios
        return specs[int(self.scenario_rng.integers(len(specs)))]

    def start_commands(self, slots):
        dagger = _dagger()
        commands = []
        for i in slots:
            spec = self.draw_scenario()
            label = dagger.scenario_label(spec)
            directory = dagger.resolve_scenario(spec, self.settings, "train")
            commands.append((i, "start", (label, str(directory))))
        return commands

    def begin_episode(self, i, result):
        stream = self.streams[i]
        stream["serial"] += 1
        tables = {k: np.asarray(v) for k, v in result["layout"].items()}
        episode = Episode((i, stream["serial"]), result["label"], tables, self.history_length(),
                          privileged=self.privileged())
        stream["episode"] = episode
        stream["tokens"] = episode.push(result["obs"], _features(result.get("base")))
        stream["base"] = result["base"]
        self.episodes[episode.key] = episode
        if self.settings.anchor == "student":
            # The student's inputs on this building must have the sizes it was trained with (a missing or extra B1-R
            # state block would otherwise be cut or misread by the kept positions).
            dims = {k: int(v.shape[-1]) for k, v in stream["tokens"].items()}
            if dims != {k: int(v) for k, v in self.network["input_dims"].items()}:
                raise ValueError(f"{result['label']}: token input sizes {dims} differ from the student's "
                                 f"{self.network['input_dims']}")

    def start(self):
        """A new run: the agent (from the student, or new on the frozen statistics), the env slots on their first
        scenarios; with init_from, the continuation of a finished run (continue_from)."""
        from smart_building.learned.ppo_env import RewardNormalizer
        s = self.settings
        if s.init_from is not None:
            return self.continue_from()
        self.build()
        self.open_envs()
        self.streams = [{"serial": 0, "episode": None, "tokens": None, "base": None, "next_done": 0.0,
                         "normalizer": RewardNormalizer(s.gamma, s.reward_clip, s.reward_epsilon)}
                        for _ in range(s.num_envs)]
        for i, result in zip(range(s.num_envs), self.envs.call(self.start_commands(range(s.num_envs)))):
            self.begin_episode(i, result)

    def continue_from(self):
        """J10 (round 2 of stage 3; docs/reference/notes/16-rl-tuning-after-baseline.md §C5 (b)): this run continues a
        finished run from its final state.pt (init_from, SHA-256 checked by resolve_input). Kept: the agent (actor,
        critic, log std), the Adam state, the iteration and step counts, the critic warm-up state, the demonstration
        count k (actor_updates), the counters, every RNG state and each env slot's reward normaliser. The env slots start
        new episodes (the environment snapshots are not read; scenarios drawn by the restored RNG). Refused: another
        state format, an unfinished source, an identity that differs in anything but CONTINUATION_CHANGES, a
        schedule_origin other than the source's iteration, no iteration left. This run's own history and agent
        snapshots start empty; targets at or before the start step are saved at once (the source's final agent, a
        candidate of the INNER choice). Its checkpoints then resume as any run's (Run.load)."""
        import torch
        from smart_building.learned.ppo_env import RewardNormalizer
        s = self.settings
        path = resolve_input(s.init_from, s)
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("format") != STATE_FORMAT:
            raise ValueError(f"init_from {path}: not a {STATE_FORMAT} state")
        changes = identity_changes(state["identity"], s.identity())
        refused = [k for k in changes if k not in CONTINUATION_CHANGES]
        if refused:
            raise ValueError(f"init_from: the continued run's settings differ in {refused} (only "
                             f"{list(CONTINUATION_CHANGES)} may change)")
        if state["iteration"] != state.get("num_iterations"):
            raise ValueError(f"init_from: the continued run has not finished (iteration {state['iteration']} of "
                             f"{state.get('num_iterations')})")
        if s.schedule_origin != state["iteration"]:
            raise ValueError(f"schedule_origin {s.schedule_origin} must be the continued run's iteration {state['iteration']}")
        if self.num_iterations <= state["iteration"]:
            raise ValueError(f"total_timesteps {s.total_timesteps} leaves no iteration after the continued run's "
                             f"{state['iteration']}")
        self.iteration, self.global_step = state["iteration"], state["global_step"]
        self.actor_training, self.actor_updates = state["actor_training"], state["actor_updates"]
        self.network, self.counters = state["network"], dict(state["counters"])
        self.history_logs, self.snapshots = [], []
        self.build(state["agent"], state["optimizer"])
        torch.set_rng_state(state["rng"]["torch"])
        if "torch_cuda" in state["rng"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["rng"]["torch_cuda"])
        self.sampler.set_state(state["rng"]["sampler"])
        self.shuffle_rng.bit_generator.state = state["rng"]["shuffle"]
        self.scenario_rng.bit_generator.state = state["rng"]["scenario"]
        self.open_envs()
        self.streams = []
        for saved in state["streams"]:
            normalizer = RewardNormalizer(s.gamma, s.reward_clip, s.reward_epsilon)
            normalizer.load(saved["normalizer"])
            self.streams.append({"serial": saved["serial"], "episode": None, "tokens": None, "base": None,
                                 "next_done": 0.0, "normalizer": normalizer})
        for i, result in zip(range(s.num_envs), self.envs.call(self.start_commands(range(s.num_envs)))):
            self.begin_episode(i, result)
        self.continued = _dagger().plain({
            "init_from": s.init_from, "iteration": state["iteration"], "global_step": state["global_step"],
            "actor_updates": state["actor_updates"], "identity_sha256": state.get("identity_sha256"),
            "changed": {k: [state["identity"].get(k, IDENTITY_DEFAULTS.get(k)), s.identity().get(k)] for k in changes}})
        self.save_snapshots()

    # -- checkpoints ---------------------------------------------------

    def save(self):
        """Checkpoint ckpt-<iteration> after every iteration: the whole state of the run (agent, optimiser, every RNG
        state, reward normalisers, running episodes and their environment snapshots, history and counters).

        Crash safety (the Mac may stop at any moment): the files are written into ckpt-<i>.partial/ and flushed to
        stable storage (fsync, and F_FULLFSYNC on macOS), the folder is renamed to ckpt-<i>, then latest.json is
        replaced atomically; until that replacement the previous checkpoint is the run's state, and the previous one
        is kept after it (keep_checkpoints = 2). A resume checks every file against the manifest's SHA-256 and falls
        back to the older checkpoint when the newest is incomplete (Run.load)."""
        import torch
        dagger = _dagger()
        started = time.perf_counter()
        name = dagger.checkpoint_name(self.iteration)
        final, partial = self.out_dir / name, self.out_dir / f"{name}.partial"
        for leftover in self.out_dir.glob("ckpt-*.partial"):                # an interrupted earlier save
            shutil.rmtree(leftover, ignore_errors=True)
        partial.mkdir(parents=True)
        snapshots = self.envs.call([(i, "snapshot", ()) for i in range(len(self.envs))])
        files = {}
        for i, payload in enumerate(snapshots):
            _write_bytes(partial / f"env-{i:02d}.pkl", payload)
            files[f"{name}/env-{i:02d}.pkl"] = hashlib.sha256(payload).hexdigest()
        streams = [{"serial": st["serial"], "next_done": st["next_done"], "normalizer": st["normalizer"].state(),
                    "episode": st["episode"].state()} for st in self.streams]
        state = {"format": STATE_FORMAT, "identity": self.settings.identity(),
                 "identity_sha256": dagger.sha256_json(self.settings.identity()), "iteration": self.iteration,
                 "global_step": self.global_step, "num_iterations": self.num_iterations,
                 "actor_training": self.actor_training, "actor_updates": self.actor_updates,
                 "network": self.network, "snapshots": self.snapshots,
                 "agent": self.agent.state_dict(), "optimizer": self.optimizer.state_dict(),
                 "rng": {"torch": torch.get_rng_state(), "sampler": self.sampler.get_state(),
                         "shuffle": self.shuffle_rng.bit_generator.state, "scenario": self.scenario_rng.bit_generator.state},
                 "streams": streams, "history": self.history_logs, "counters": self.counters,
                 "continuation": self.continued}
        state["runtime"] = self.settings.runtime()
        if torch.cuda.is_available() and torch.cuda.is_initialized():      # device "cuda": its generators too
            state["rng"]["torch_cuda"] = torch.cuda.get_rng_state_all()
        with open(partial / "state.pt", "wb") as stream:
            torch.save(state, stream)
            _flush(stream)
        files[f"{name}/state.pt"] = sha256_file(partial / "state.pt")
        files.update({record["file"]: record["sha256"] for record in self.snapshots})   # the agent snapshots go along
        _write_json(partial / "manifest.json", {"format": STATE_FORMAT, "iteration": self.iteration, "files": files,
                                                "runtime": self.settings.runtime()})
        _fsync_directory(partial)
        if final.exists():                       # left by an interrupted earlier attempt at this iteration (not latest)
            shutil.rmtree(final)
        os.replace(partial, final)
        _fsync_directory(self.out_dir)
        _write_json(self.out_dir / "latest.json", {"checkpoint": name, "iteration": self.iteration})
        kept = sorted(p for p in self.out_dir.glob("ckpt-*") if p.is_dir() and p.name[len("ckpt-"):].isdigit())
        for old in kept[:-self.settings.keep_checkpoints]:
            shutil.rmtree(old, ignore_errors=True)
        self.checkpoint_seconds = time.perf_counter() - started
        return name

    def load(self, state_dir):
        """Resume from the newest complete checkpoint of state_dir (every file checked against its manifest's SHA-256;
        an incomplete newest checkpoint, e.g. one cut by a shutdown, falls back to the previous one), copying its files
        into out_dir when state_dir is another folder. iterations.jsonl and episodes.jsonl are rewritten from the
        run's history, so they equal those of an uninterrupted run."""
        import torch
        from smart_building.learned.ppo_env import RewardNormalizer
        dagger = _dagger()
        state_dir = Path(state_dir)
        problems, state = [], None
        for checkpoint in checkpoint_candidates(state_dir):
            try:
                manifest = dagger.verify_checkpoint(state_dir, checkpoint)
                if state_dir.resolve() != self.out_dir.resolve():
                    for name in manifest["files"]:
                        target = self.out_dir / name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        if not target.is_file() or sha256_file(target) != manifest["files"][name]:
                            shutil.copy2(state_dir / name, target)
                    shutil.copy2(checkpoint / "manifest.json", self.out_dir / checkpoint.name / "manifest.json")
                    _write_json(self.out_dir / "latest.json", {"checkpoint": checkpoint.name, "iteration": manifest["iteration"]})
                state = torch.load(self.out_dir / checkpoint.name / "state.pt", map_location="cpu", weights_only=True)
                break
            except (OSError, ValueError, KeyError, RuntimeError, EOFError) as exc:
                problems.append(f"{checkpoint.name}: {exc!r}"[:500])
        if state is None:
            raise FileNotFoundError(f"no complete checkpoint in {state_dir}: {problems}")
        saved_runtime = state.get("runtime") or {}
        self.resume = {"checkpoint": checkpoint.name, "skipped": problems,
                       "changed": {k: [saved_runtime.get(k), v] for k, v in self.settings.runtime().items()
                                   if saved_runtime.get(k) != v}}
        if state.get("format") != STATE_FORMAT:
            raise ValueError(f"{checkpoint}: not a {STATE_FORMAT} state")
        changed = identity_changes(state["identity"], self.settings.identity())
        if changed:
            raise ValueError(f"the run's settings changed since it started: {changed}")
        self.iteration, self.global_step = state["iteration"], state["global_step"]
        self.actor_training, self.actor_updates = state["actor_training"], state["actor_updates"]
        self.network, self.history_logs, self.counters = state["network"], state["history"], state["counters"]
        self.snapshots = state["snapshots"]
        self.continued = state.get("continuation")
        self.build(state["agent"], state["optimizer"])
        torch.set_rng_state(state["rng"]["torch"])
        if "torch_cuda" in state["rng"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["rng"]["torch_cuda"])
        self.sampler.set_state(state["rng"]["sampler"])
        self.shuffle_rng.bit_generator.state = state["rng"]["shuffle"]
        self.scenario_rng.bit_generator.state = state["rng"]["scenario"]
        s = self.settings
        self.open_envs()
        payloads = [(self.out_dir / checkpoint.name / f"env-{i:02d}.pkl").read_bytes() for i in range(s.num_envs)]
        restored = self.envs.call([(i, "restore", (payload,)) for i, payload in enumerate(payloads)])
        self.streams, self.episodes = [], {}
        for i, (saved, result) in enumerate(zip(state["streams"], restored)):
            normalizer = RewardNormalizer(s.gamma, s.reward_clip, s.reward_epsilon)
            normalizer.load(saved["normalizer"])
            episode = Episode.from_state(saved["episode"], self.history_length())
            if result["label"] != episode.label or result["steps"] != episode.offset + len(episode.rows) - 1:
                raise RuntimeError(f"env {i}: the snapshot ({result['label']}, step {result['steps']}) does not match "
                                   f"the saved episode ({episode.label}, {episode.offset + len(episode.rows) - 1})")
            last = episode.rows[-1]
            if not np.array_equal(np.asarray(result["obs"], np.float32).view(np.uint32), last.view(np.uint32)):
                raise RuntimeError(f"env {i}: the snapshot's observation differs from the saved rows")
            if episode.privileged:
                state_now = _features(result["base"])
                if state_now is None or any(not np.array_equal(np.asarray(state_now[k], np.float32).view(np.uint32),
                                                               episode.features[-1][k].view(np.uint32))
                                            for k in episode.features[-1]):
                    raise RuntimeError(f"env {i}: the snapshot's B1-R state differs from the saved rows")
            self.streams.append({"serial": saved["serial"], "episode": episode, "next_done": saved["next_done"],
                                 "tokens": episode.current_tokens(), "base": result["base"], "normalizer": normalizer})
            self.episodes[episode.key] = episode
        # This segment's logs start with the run's history (the rows an uninterrupted run has written so far).
        for name in ("iterations.jsonl", "episodes.jsonl"):
            (self.out_dir / name).write_text("", encoding="utf-8")
        for row in self.history_logs:
            dagger.append_jsonl(self.out_dir / "iterations.jsonl", row)
            for episode_row in row["episodes_finished"]:
                dagger.append_jsonl(self.out_dir / "episodes.jsonl", episode_row)

    # -- one iteration (CleanRL L200-322) --------------------------------

    def _switch_prior(self, discrete, n_ahu):
        """Reviewer 2026-10-25 (switch_prior "rule"): the prior logits switch_prior_gain x (2 d - 1) of the rule
        controller's switch decisions d (0 or 1; the AHUs, then the hot water), shaped (batch, tokens) as the switch
        heads' logits."""
        import torch
        s = self.settings
        d = np.atleast_2d(np.asarray(discrete, dtype=np.float32))
        prior = np.float32(s.switch_prior_gain) * (np.float32(2.0) * d - np.float32(1.0))
        return {"ahu": torch.from_numpy(np.ascontiguousarray(prior[:, :n_ahu])).to(s.device),
                "hws": torch.from_numpy(np.ascontiguousarray(prior[:, n_ahu:])).to(s.device)}

    def act(self, i):
        """One env's action for its current observation: (stored sample, executed action, log-probs, value)."""
        import torch
        s, stream = self.settings, self.streams[i]
        episode = stream["episode"]
        tokens = {k: torch.from_numpy(v).unsqueeze(0).to(s.device) for k, v in stream["tokens"].items()}
        with torch.no_grad():
            prior = (self._switch_prior(stream["base"]["discrete"], episode.n_ahu) if s.switch_prior == "rule"
                     else None)
            out = self.agent.get_action_and_value(tokens, generator=self.sampler, switch_prior=prior)
        sample = scatter(episode.tables, {k: v[0].cpu().numpy() for k, v in out["action"].items()}, episode.n_continuous)
        low, high = episode.tables["low"], episode.tables["high"]
        base = stream["base"]
        if s.anchor == "residual":                                               # D-RL03 (a); CleanRL L96
            residual = sample if s.action_scale == 1.0 else np.float32(s.action_scale) * sample   # REQ-021 (13)
            continuous = np.clip(base["continuous"] + residual, low, high)
        else:
            continuous = np.clip(sample, low, high)                              # CleanRL L96 (ClipAction)
        switch = None
        if s.switches == "bernoulli":
            switch = np.concatenate([out["switch"]["ahu"][0].cpu().numpy(), out["switch"]["hws"][0].cpu().numpy()])
            discrete = switch.astype(np.int32)                                   # AHU 1..K, hot water (D-R01)
        else:
            discrete = base["discrete"]                                          # D-RL04 (a), D-L07
        if not (np.all(np.isfinite(continuous)) and np.all(np.isfinite(sample))):
            raise FloatingPointError(f"env {i}: non-finite action")
        record = {"episode": episode.key, "step": episode.offset + len(episode.rows) - 1, "sample": sample,
                  "switch": switch, "logprob": float(out["logprob"][0]), "value": float(out["value"][0]),
                  "switch_logprob": float(out["switch_logprob"][0]) if switch is not None else 0.0,
                  "switch_rule": (np.asarray(base["discrete"], dtype=np.int32).copy() if s.switch_prior == "rule"
                                  else None)}
        return record, {"continuous_action": continuous.astype(np.float32), "discrete_action": discrete}

    def final_value(self, i, observation, features=None):
        """REQ-021 (12): v(s_last), the critic's value of an ended episode's last observation (the one the official
        environment returns with its end; environment.py: "We still need to get the final observation to add to the
        transition"), with the episode's history window (and, privileged, the B1-R state a call on it would start
        from: ppo_env "final_features")."""
        import torch
        tokens = self.streams[i]["episode"].push(observation, features)
        inputs = {k: torch.from_numpy(v).unsqueeze(0).to(self.settings.device) for k, v in tokens.items()}
        with torch.no_grad():
            return float(self.agent.get_value(inputs)[0])

    def rollout(self):
        """CleanRL L207-230 for num_steps steps of every env slot. Returns (samples, rewards, dones, end_values,
        finished episodes); end_values (REQ-021 (12)) is None without bootstrap_time_limit."""
        s = self.settings
        T, N = s.num_steps, s.num_envs
        samples = [[None] * N for _ in range(T)]
        rewards = np.zeros((T, N), dtype=np.float32)
        dones = np.zeros((T, N), dtype=np.float32)
        end_values = np.zeros((T, N), dtype=np.float32) if s.bootstrap_time_limit else None
        finished = []
        clock = {"inference_s": 0.0, "env_step_s": 0.0, "env_start_s": 0.0}
        self.agent.eval()
        for t in range(T):
            self.global_step += N                                                # L208
            actions = []
            t0 = time.perf_counter()
            for i in range(N):
                dones[t, i] = self.streams[i]["next_done"]                       # L210
                record, action = self.act(i)                                     # L213-217
                samples[t][i] = record
                actions.append(action)
            t1 = time.perf_counter()
            results = self.envs.call([(i, "step", (actions[i],)) for i in range(N)])   # L220
            clock["inference_s"] += t1 - t0
            clock["env_step_s"] += time.perf_counter() - t1
            restart = []
            for i, result in enumerate(results):
                stream = self.streams[i]
                done = result["terminated"] or result["truncated"]               # L221
                stream["next_done"] = float(done)
                rewards[t, i] = stream["normalizer"](result["reward"], result["terminated"])   # L222, L99-100
                if done:
                    row = {**result["episode"], "env": i, "iteration": self.iteration + 1, "global_step": self.global_step}
                    if end_values is not None:                                   # REQ-021 (12): a time limit
                        t2 = time.perf_counter()
                        end_values[t, i] = self.final_value(i, result["final_obs"], result.get("final_features"))
                        clock["inference_s"] += time.perf_counter() - t2
                        row["bootstrap_value"] = float(end_values[t, i])
                    restart.append(i)
                    finished.append(row)
                else:
                    stream["tokens"] = stream["episode"].push(result["obs"], _features(result["base"]))
                    stream["base"] = result["base"]
            if restart:                                                         # same-step reset (L220-223)
                t2 = time.perf_counter()
                for i, result in zip(restart, self.envs.call(self.start_commands(restart))):
                    self.begin_episode(i, result)
                clock["env_start_s"] += time.perf_counter() - t2
        self.counters["env_steps"] += T * N
        self.counters["episodes"] += len(finished)
        self.clock = clock
        return samples, rewards, dones, end_values, finished

    def advantages(self, samples, rewards, dones, end_values=None):
        """CleanRL L232-246 (gae()), with v(s_last) at time limits when end_values is given (REQ-021 (12))."""
        import torch
        s = self.settings
        T, N = rewards.shape
        values = torch.tensor([[samples[t][i]["value"] for i in range(N)] for t in range(T)], dtype=torch.float32)
        with torch.no_grad():
            next_value = torch.stack([self.agent.get_value(
                {k: torch.from_numpy(v).unsqueeze(0).to(s.device) for k, v in self.streams[i]["tokens"].items()})[0].cpu()
                for i in range(N)]).reshape(1, -1)                              # L234
        next_done = torch.tensor([st["next_done"] for st in self.streams], dtype=torch.float32)
        advantages, returns = gae(torch.from_numpy(rewards), values, torch.from_numpy(dones), next_value, next_done,
                                  s.gamma, s.gae_lambda,
                                  None if end_values is None else torch.from_numpy(end_values))
        return advantages, returns, values

    def _groups(self, positions, flat, with_actor):
        """For each episode group of the flat-batch positions (one building each), in sorted episode order, and
        each chunk of at most max_chunk_samples of the group's samples (REQ-021 (1); None: the whole group), in
        order: (indices into positions, the outputs of the chunk's forward pass)."""
        import torch
        from smart_building.learned.tokens import gather_targets
        s = self.settings
        groups = {}
        for j, p in enumerate(positions):
            groups.setdefault(flat[p]["episode"], []).append(j)
        history = self.history_length()
        for key in sorted(groups):
            members = groups[key]
            episode = self.episodes[key]
            size = int(s.max_chunk_samples) if s.max_chunk_samples is not None else len(members)
            for start in range(0, len(members), size):
                js = members[start:start + size]
                records = [flat[positions[j]] for j in js]
                tokens = {k: torch.from_numpy(v).to(s.device)
                          for k, v in episode.inputs([r["step"] for r in records], history).items()}
                action = switch = prior = None
                if with_actor:
                    per_type = gather_targets(episode.tables, np.stack([r["sample"] for r in records]))
                    action = {k: torch.from_numpy(np.ascontiguousarray(v)).to(s.device) for k, v in per_type.items()}
                    if s.switches == "bernoulli":
                        sw = np.stack([r["switch"] for r in records])
                        switch = {"ahu": torch.from_numpy(np.ascontiguousarray(sw[:, :episode.n_ahu])).to(s.device),
                                  "hws": torch.from_numpy(np.ascontiguousarray(sw[:, episode.n_ahu:])).to(s.device)}
                        if s.switch_prior == "rule":
                            prior = self._switch_prior(np.stack([r["switch_rule"] for r in records]), episode.n_ahu)
                yield js, self.agent.get_action_and_value(tokens, action=action, switch=switch, with_actor=with_actor,
                                                          switch_prior=prior)

    def _minibatch_backward(self, mb_inds, flat, batch, with_actor, demo_chunk=None, demo_weight=None):
        """The gradients of one minibatch's loss (CleanRL L266-300; every term a mean over the minibatch),
        accumulated into .grad with one backward per chunk: REQ-021 (1) (reviewer 2026-10-20: option (b) of
        docs/plan/reviews/WP5-throughput.md §5.1, then chunks of at most M samples of an episode group). Each chunk's
        terms are summed and divided by the minibatch size, so the accumulated gradient is the gradient of CleanRL's
        loss; only one chunk's graph is alive at a time. With dropout 0 the forward values do not depend on the
        chunking, so the update equals the single backward up to the order of the float sums (whole groups with
        dropout: the forward passes run in the same order as before, so dropout draws the same masks). Advantages
        are normalised over the whole minibatch first. Returns the logged values (means over the minibatch)."""
        import torch
        s = self.settings
        n = len(mb_inds)
        positions = torch.as_tensor(mb_inds, device=s.device)
        mb_advantages = batch["advantages"][positions]
        if s.norm_adv:                                                               # L276-277
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
        logged = dict.fromkeys(("v_loss", "pg_loss", "entropy", "approx_kl", "old_approx_kl", "clipfrac",
                                "switch_pg_loss"), 0.0)
        for js, out in self._groups(mb_inds, flat, with_actor):
            local = torch.as_tensor(js, device=s.device)
            idx = positions[local]
            advantage = mb_advantages[local]
            newvalue = out["value"].view(-1)                                         # L284-297
            if s.clip_vloss:
                v_unclipped = (newvalue - batch["returns"][idx]) ** 2
                v_clipped = batch["values"][idx] + torch.clamp(newvalue - batch["values"][idx], -s.clip_coef, s.clip_coef)
                v_terms = 0.5 * torch.max(v_unclipped, (v_clipped - batch["returns"][idx]) ** 2)
            else:
                v_terms = 0.5 * (newvalue - batch["returns"][idx]) ** 2
            v_part = v_terms.sum() / n
            loss = v_part * s.vf_coef
            logged["v_loss"] += float(v_part.detach())
            if with_actor:
                logratio = out["logprob"] - batch["logprobs"][idx]                   # L266-273
                ratio = logratio.exp()
                with torch.no_grad():
                    logged["old_approx_kl"] += float((-logratio).sum()) / n
                    logged["approx_kl"] += float(((ratio - 1) - logratio).sum()) / n
                    logged["clipfrac"] += float(((ratio - 1.0).abs() > s.clip_coef).float().sum()) / n
                pg_part = torch.max(-advantage * ratio, -advantage * torch.clamp(
                    ratio, 1 - s.clip_coef, 1 + s.clip_coef)).sum() / n                 # L279-282
                entropy_part = out["entropy"].sum() / n                                # L299
                if s.switches == "bernoulli":                                        # D-RL04 (b): own clipped objective
                    sw_ratio = (out["switch_logprob"] - batch["switch_logprobs"][idx]).exp()
                    sw_part = torch.max(-advantage * sw_ratio, -advantage * torch.clamp(
                        sw_ratio, 1 - s.clip_coef, 1 + s.clip_coef)).sum() / n
                    entropy_part = entropy_part + out["switch_entropy"].sum() / n    # REQ-021 (10)
                    loss = loss + sw_part
                    logged["switch_pg_loss"] += float(sw_part.detach())
                loss = loss + pg_part - s.ent_coef * entropy_part                    # L300
                logged["pg_loss"] += float(pg_part.detach())
                logged["entropy"] += float(entropy_part.detach())
            loss.backward()
            del loss, out
        if with_actor and demo_weight is not None:
            logged["demo_loss"] = self._demo_backward(demo_chunk, demo_weight, n)
        return logged

    def _demo_backward(self, chunk, weight, minibatch):
        """D-RL03 (b) term (REQ-021 (8) placeholder): -w * sum over the chunk's demonstration pairs of
        log pi(a_demo | s_demo) / minibatch size (DAPG Eq. 6 gradient; mjrl `dapg.py` @3871d93 weights
        demonstration samples per on-policy sample), backpropagated per demonstration rollout (one building
        each) like the on-policy groups, and within a rollout in chunks of at most max_chunk_samples pairs (REQ-021 (1),
        memory: the same accumulated gradient; None: the whole group). Returns its value."""
        import torch
        s = self.settings
        history = self.history_length()
        groups = {}
        for d in chunk:
            r, p = self.demonstrations.index[int(d)]
            groups.setdefault(r, []).append(p)
        total = 0.0
        for r in sorted(groups):
            rollout = self.demonstrations.rollouts[r]
            positions = groups[r]
            size = int(s.max_chunk_samples) if s.max_chunk_samples is not None else len(positions)
            for start in range(0, len(positions), size):
                part_positions = positions[start:start + size]
                tokens = {k: torch.from_numpy(v).to(s.device) for k, v in rollout.inputs(part_positions, history).items()}
                action = {k: torch.from_numpy(np.ascontiguousarray(v)).to(s.device)
                          for k, v in rollout.targets(part_positions).items()}
                part = -weight * self.agent.log_prob(tokens, action).sum() / minibatch
                part.backward()
                total += float(part.detach())
        return total

    def update(self, samples, advantages, returns, values):
        """CleanRL L248-312 (the backward of L302-305 per chunk: _minibatch_backward)."""
        import torch
        s = self.settings
        T, N = advantages.shape
        flat = [samples[t][i] for t in range(T) for i in range(N)]                       # L248-254
        b_advantages, b_returns, b_values = (x.reshape(-1).to(s.device) for x in (advantages, returns, values))
        batch = {"logprobs": torch.tensor([r["logprob"] for r in flat], dtype=torch.float32).to(s.device),
                 "switch_logprobs": torch.tensor([r["switch_logprob"] for r in flat], dtype=torch.float32).to(s.device),
                 "advantages": b_advantages, "returns": b_returns, "values": b_values}
        with_actor = self.actor_training
        demo_weight = None
        if with_actor and self.demonstrations is not None:
            demo_weight = s.demo_lambda0 * s.demo_lambda1 ** self.actor_updates     # D-RL03 (b), k from 0
        b_inds = np.arange(self.batch_size)
        n_minibatches = math.ceil(self.batch_size / s.minibatch_size)
        epochs_run, early_stop_epoch = 0, None
        clipfracs, logs = [], {"v_loss": [], "pg_loss": [], "entropy": [], "approx_kl": [], "old_approx_kl": [],
                               "switch_pg_loss": [], "demo_loss": []}
        self.agent.train()                                                               # dropout (REQ-021 (4))
        for epoch in range(s.update_epochs):                                             # L259
            self.shuffle_rng.shuffle(b_inds)                                             # L260
            demo_chunks = (np.array_split(self.shuffle_rng.permutation(len(self.demonstrations)), n_minibatches)
                           if demo_weight is not None else None)
            for j, start in enumerate(range(0, self.batch_size, s.minibatch_size)):       # L261-263
                mb_inds = b_inds[start:start + s.minibatch_size]
                self.optimizer.zero_grad()                                               # L302
                logged = self._minibatch_backward(mb_inds, flat, batch, with_actor,      # L264-300, L303
                                                  demo_chunks[j] if demo_chunks is not None else None, demo_weight)
                if with_actor:
                    clipfracs.append(logged["clipfrac"])
                    if s.switches == "bernoulli":
                        logs["switch_pg_loss"].append(logged["switch_pg_loss"])
                    if demo_weight is not None:
                        logs["demo_loss"].append(logged["demo_loss"])
                    logs["pg_loss"].append(logged["pg_loss"])
                    logs["entropy"].append(logged["entropy"])
                    logs["approx_kl"].append(logged["approx_kl"])
                    logs["old_approx_kl"].append(logged["old_approx_kl"])
                logs["v_loss"].append(logged["v_loss"])
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), s.max_grad_norm)   # L304-305
                self.optimizer.step()
                self.agent.clamp_log_std()                                               # REQ-021 (7)
                self.counters["minibatches"] += 1
            epochs_run = epoch + 1
            if s.target_kl is not None and with_actor and logs["approx_kl"] and logs["approx_kl"][-1] > s.target_kl:
                early_stop_epoch = epoch + 1       # L307-308: the last minibatch's approx_kl
                break
        self.agent.eval()
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()                # L310-312
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        summary = {k: (float(np.mean(v)) if v else None) for k, v in logs.items()}
        summary.update({f"last_{k}": (v[-1] if v else None) for k, v in logs.items()})
        summary.update(clipfrac=float(np.mean(clipfracs)) if clipfracs else None, explained_variance=float(explained_var),
                       actor_trained=with_actor, demo_weight=demo_weight, epochs=epochs_run,
                       early_stop_epoch=early_stop_epoch)
        if with_actor:
            self.actor_updates += 1
        elif s.critic_warmup_iterations and self.iteration + 1 >= int(s.critic_warmup_iterations):
            self.actor_training = True       # a fixed number of critic-only iterations, then actor updates (one way)
        elif s.critic_warmup_loss is not None and summary["v_loss"] is not None and summary["v_loss"] < s.critic_warmup_loss:
            self.actor_training = True       # D-RL03 (a): actor updates from the next iteration (REQ-021 (6) (a), one way)
        return summary

    def save_snapshots(self):
        """The agent at the first iteration boundary at or after each agent_snapshot_steps target, or at the last
        iteration for a target beyond the run (checkpoint evaluation). Returns the new record or None."""
        done = {t for record in self.snapshots for t in record["targets"]}
        due = [int(t) for t in self.settings.agent_snapshot_steps
               if int(t) not in done and (self.global_step >= int(t) or self.iteration >= self.num_iterations)]
        if not due:
            return None
        record = {"targets": due, "iteration": self.iteration, "global_step": self.global_step,
                  **self.save_agent(self.out_dir / "models" / f"agent-{self.iteration:04d}.pt")}
        self.snapshots.append(record)
        return record

    def iterate(self):
        import torch
        from smart_building.learned.ppo_env import process_memory
        s = self.settings
        started = time.time()
        rates = learning_rates(s, self.iteration + 1, self.num_iterations)          # L201-205; REQ-021 (3)
        for group in self.optimizer.param_groups:
            group["lr"] = rates[group["name"]]
        # torch threads: one for the rollout's batch-1 inference and the values (as the competition policy, REQ-014
        # (7)); train_threads only for the update (docs/plan/reviews/WP5-throughput.md §6 U1: the result depends on
        # the count, so it is part of the run's identity). The env workers are other processes and never load torch.
        torch.set_num_threads(1)
        samples, rewards, dones, end_values, finished = self.rollout()
        rollout_s = time.time() - started
        t0 = time.time()
        advantages, returns, values = self.advantages(samples, rewards, dones, end_values)
        torch.set_num_threads(s.train_threads)
        try:
            update = self.update(samples, advantages, returns, values)
        finally:
            torch.set_num_threads(1)
        update_s = time.time() - t0
        self.iteration += 1
        # Rows needed later: the current episodes' last H rows (the next windows reach back H - 1 steps).
        current = {st["episode"].key for st in self.streams}
        self.episodes = {k: e for k, e in self.episodes.items() if k in current}
        for episode in self.episodes.values():
            episode.trim(self.history_length())
        memory = {"trainer": process_memory(),
                  "workers": None if s.in_process else self.envs.call([(i, "memory", ()) for i in range(len(self.envs))])}
        if s.device.startswith("cuda"):
            memory["cuda_peak_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
        snapshot = self.save_snapshots()
        log = {"iteration": self.iteration, "global_step": self.global_step, "learning_rate": rates,
               "update": update, "episodes_finished": finished, "agent_snapshot": snapshot,
               "bootstrap": None if end_values is None else {
                   "ends": len(finished),
                   "mean_value": float(np.mean([r["bootstrap_value"] for r in finished])) if finished else None},
               "timing": {"rollout_s": rollout_s, "update_s": update_s, "iteration_s": time.time() - started,
                          "previous_checkpoint_s": self.checkpoint_seconds, **self.clock,
                          "env_steps_per_second_rollout": s.num_steps * s.num_envs / rollout_s if rollout_s else None,
                          "env_steps_per_second_iteration": s.num_steps * s.num_envs / (time.time() - started)},
               "memory": memory, "counters": dict(self.counters),
               "runtime": {**s.runtime(), "torch_threads": {"rollout": 1, "update": s.train_threads}}}
        log = _dagger().plain(log)
        self.history_logs.append(log)
        return log

    def save_agent(self, path):
        """The agent and what rebuilds it (AGENT_FORMAT), for evaluation (PPOPolicy)."""
        import torch
        s = self.settings
        payload = {"format": AGENT_FORMAT, "network": self.network, "agent": self.agent.state_dict(),
                   "settings": {k: getattr(s, k) for k in ("anchor", "switches", "observation_clip", "dropout",
                                                          "switch_init_gain", "min_log_std", "log_std_init",
                                                          "learn_log_std", "mean_activation", "action_scale",
                                                          "max_std", "rule_controller",
                                                          "switch_prior", "switch_prior_gain")},   # 2026-10-25: PPOPolicy
                   "iteration": self.iteration, "global_step": self.global_step}
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".partial")
        with open(temporary, "wb") as stream:                   # written whole, then renamed (crash safety)
            torch.save(payload, stream)
            _flush(stream)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
        return {"file": path.relative_to(self.out_dir).as_posix(), "sha256": sha256_file(path)}

    def close(self):
        if self.envs is not None:
            self.envs.close()
            self.envs = None


# ------------------------------------------------------------------ entry point

def run_ppo_segment(config, state_dir, out_dir, deadline_utc):
    """One segment of a PPO run (see the module docstring); returns a JSON-able dict."""
    import torch
    from smart_building.learned.ppo_env import absolute_cache_paths
    absolute_cache_paths(os.getcwd())                 # before any slot changes the working directory
    dagger = _dagger()
    settings = PPOSettings.from_config(config)
    if settings.device.startswith("cuda"):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    previous = (torch.are_deterministic_algorithms_enabled(), torch.get_num_threads())
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)                          # Run.iterate sets train_threads for the update only
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    deadline = dagger.parse_deadline(deadline_utc)
    run = Run(settings, out_dir)
    done_here, stopped = 0, None
    try:
        if state_dir:
            run.load(state_dir)
        while run.iteration < run.num_iterations:
            if settings.max_iterations_per_segment is not None and done_here >= settings.max_iterations_per_segment:
                stopped = "max_iterations_per_segment"
                break
            if settings.stop_file and Path(settings.stop_file).exists():
                stopped = "stop_file"
                break
            durations = [h["timing"]["iteration_s"] for h in run.history_logs]
            estimate = max(durations) if durations else settings.iteration_estimate_seconds
            if deadline is not None and estimate is not None and time.time() + settings.deadline_margin * estimate > deadline:
                stopped = "deadline"
                break
            if run.envs is None:
                run.start()                      # a new run: the agent and the first scenarios
            log = run.iterate()
            run.save()
            dagger.append_jsonl(out_dir / "iterations.jsonl", log)
            for row in log["episodes_finished"]:
                dagger.append_jsonl(out_dir / "episodes.jsonl", row)
            done_here += 1
    finally:
        run.close()
        torch.use_deterministic_algorithms(previous[0])
        torch.set_num_threads(previous[1])
    finished = run.iteration >= run.num_iterations
    result = {"finished": finished, "iteration": run.iteration, "iterations_this_segment": done_here,
              "num_iterations": run.num_iterations, "global_step": run.global_step,
              "stopped": stopped if not finished else "finished",
              "checkpoint": dagger.checkpoint_name(run.iteration) if run.iteration else None,
              "actor_training": run.actor_training, "counters": run.counters, "agent_snapshots": run.snapshots,
              "resumed": run.resume, "last_checkpoint_s": run.checkpoint_seconds, "continued": run.continued}
    if finished and run.agent is not None:
        last = [r for r in run.snapshots if r["iteration"] == run.iteration]
        model = ({"file": last[0]["file"], "sha256": last[0]["sha256"]} if last
                 else run.save_agent(out_dir / "models" / f"agent-{run.iteration:04d}.pt"))
        final = {"model": model, "iteration": run.iteration, "global_step": run.global_step, "network": run.network,
                 "rule_controller": settings.rule_controller, "agent_snapshots": run.snapshots}
        dagger.write_json(out_dir / "final.json", final)
        result.update(final)
    return result


# ------------------------------------------------------------------ competition interface (evaluation)

class PPOPolicy:
    """B1-RL with the competition interface, acting with the policy's mean (the deterministic policy;
    D-L08's rule for the student; REQ-021 (14) (a)). D-RL03 (a): the rule controller's action plus action_scale x f's
    mean, clipped to the action spec; (b): the actor's mean. D-RL04 (a): the rule controller's on/off; (b): switch on
    when p >= 0.5 (REQ-021 (10)). D-T02: on a network failure the step uses the rule controller's
    action; D-T03: finite values, clipped to the action spec.

    An agent from a j7b-v5 / v5s student (settings "privileged", added by load_agent) also reads the rule controller's
    internal state before its call on each observation and appends it to the tokens, in the order of the v5s
    evaluation policy privileged.StudentPolicyV5 (state, rule controller's call, history push, network; an unreadable
    state makes the step fall back, D-T02), so an untrained student-anchored agent acts as that policy bit for bit."""

    name = "b1_rl"

    def __init__(self, building_meta_data, action_spec, agent, settings, history, base_class=None):
        import torch
        from smart_building.learned.tokens import HistoryBuffer, TokenLayout
        torch.set_num_threads(1)                                                   # REQ-014 (7)
        self.agent, self.settings = agent.eval(), settings
        self.privileged = bool(settings.get("privileged"))
        if self.privileged and base_class is None:
            raise ValueError("an agent that reads the B1-R state needs its rule controller class")
        if settings.get("switch_prior", "none") == "rule" and base_class is None:
            raise ValueError("an agent whose switches start at the rule's decisions needs its rule controller class")
        self.layout = TokenLayout(building_meta_data, action_spec, history)
        self.tables = self.layout.spec()
        self.buffer = HistoryBuffer(history, self.layout.n_fields)
        self.base = base_class(building_meta_data, action_spec) if base_class is not None else None
        continuous = action_spec["continuous_action"]
        self._low = np.asarray(continuous.low, dtype=np.float32)
        self._high = np.asarray(continuous.high, dtype=np.float32)
        self.fallback_steps = 0

    def get_action(self, obs):
        import torch
        from smart_building.learned.tokens import TOKEN_TYPES
        features = None
        if self.privileged:
            from smart_building.learned.privileged import policy_features
            try:
                features = policy_features(self.base)                            # before this step's B1-R call
            except Exception:  # noqa: BLE001 - D-T02 below
                features = None
        base = self.base.get_action(obs) if self.base is not None else None
        try:
            tokens = window_tokens(self.tables, self.buffer.push(obs), TOKEN_TYPES)
            if self.privileged:
                from smart_building.learned.privileged import append_features
                if features is None:
                    raise RuntimeError("B1-R state unavailable")
                tokens = append_features(tokens, features)
            inputs = {k: torch.from_numpy(v).unsqueeze(0) for k, v in tokens.items()}
            with torch.no_grad():
                mean = self.agent.mean_action(inputs)
                logits = None
                if self.settings["switches"] == "bernoulli":
                    f = encode(self.agent.actor, inputs, self.agent.observation_clip)
                    logits = {k: self.agent.switch_heads[k](f[k]).squeeze(-1)[0].numpy() for k in ("ahu", "hws")}
                    if self.settings.get("switch_prior", "none") == "rule":     # reviewer 2026-10-25
                        d = np.asarray(base["discrete_action"], dtype=np.float32)
                        n_ahu = logits["ahu"].shape[0]
                        gain = np.float32(self.settings.get("switch_prior_gain", 4.0))
                        logits = {"ahu": logits["ahu"] + gain * (np.float32(2.0) * d[:n_ahu] - np.float32(1.0)),
                                  "hws": logits["hws"] + gain * (np.float32(2.0) * d[n_ahu:] - np.float32(1.0))}
            f_mean = scatter(self.tables, {k: v[0].numpy() for k, v in mean.items()}, self.layout.index.n_continuous)
            if self.settings["anchor"] == "residual":                               # action_scale x f (D-RL03 (a))
                scale = float(self.settings.get("action_scale", 1.0))
                continuous = combine(base["continuous_action"], f_mean if scale == 1.0 else np.float32(scale) * f_mean)
            else:
                continuous = f_mean
            if not np.all(np.isfinite(continuous)):
                raise FloatingPointError("non-finite action")                   # D-T03
            if logits is not None:
                discrete = (np.concatenate([logits["ahu"], logits["hws"]]) >= 0.0).astype(np.int32)
            else:
                discrete = np.asarray(base["discrete_action"], dtype=np.int32)
        except Exception:  # noqa: BLE001 - D-T02: the rule controller's action for this step
            if base is None:
                raise
            self.fallback_steps += 1
            return base
        return {"continuous_action": np.clip(continuous, self._low, self._high).astype(np.float32),
                "discrete_action": discrete}


def load_agent(path, settings_overrides=None):
    """(agent, settings dict, history) of a save_agent file. For an agent from a j7b-v5 / v5s student the settings dict
    also carries "privileged" (the student file's privileged_entry), which PPOPolicy reads."""
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != AGENT_FORMAT:
        raise ValueError(f"{path}: not a {AGENT_FORMAT} file")
    stored = {**payload["settings"], **(settings_overrides or {})}
    settings = PPOSettings(**{k: v for k, v in stored.items() if k in {f.name for f in fields(PPOSettings)}})
    state = payload["agent"]
    actor = {k[len("actor."):]: v for k, v in state.items() if k.startswith("actor.")}
    critic = {k[len("critic."):]: v for k, v in state.items() if k.startswith("critic.")}
    network = payload["network"]
    agent = build_agent(network["config"], network["input_dims"], settings, actor_state=actor, critic_state=critic)
    agent.load_state_dict(state)
    if network.get("privileged"):
        stored.setdefault("privileged", network["privileged"])
    return agent, stored, int(network["config"]["history"])


def policy_class(path, base_class=None):
    """A competition policy class (building_meta_data, action_spec) bound to a save_agent file."""
    agent, settings, history = load_agent(path)

    class BoundPPOPolicy(PPOPolicy):
        def __init__(self, building_meta_data, action_spec):
            super().__init__(building_meta_data, action_spec, agent, settings, history, base_class)

    return BoundPPOPolicy
