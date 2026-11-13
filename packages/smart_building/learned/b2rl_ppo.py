"""B2-RL trainer: PPO on the hourly rule parameters (handoff/RUNBOOK-windows-b2rl.md §4.2).

CleanRL ppo_continuous_action (the source learned/ppo.py follows, D-RL01) with the H3C h3c-drl-training settings of
the runbook: actor and critic each two ReLU layers of 128 (Wu, W. Zhang et al. 2024), the actor's output layer
initialised with standard deviation 0.01 so the first mean action is about 0 = the rule itself; a diagonal Gaussian
with a state-independent log sigma starting at -1.0; Adam 3e-4 decayed linearly to 3e-5 over the planned updates;
168 decisions (one week) per environment per update, handed out to the environments as each finishes its last one
(a small building does not wait for a large one, so it may contribute more), 10 epochs of 4 minibatches; gamma 0.99, GAE lambda 0.95, clip
0.2, value coefficient 0.5 (clipped value loss), entropy 0.02, gradient norm 0.5; features normalised by running mean
and variance (gymnasium NormalizeObservation arithmetic, clipped to +-10), rewards by the running return standard
deviation and clipped to +-10 (learned/ppo_env.RewardNormalizer). An episode's end is terminal (CleanRL).
The switches stay with the rule; nothing is imitated. CPU only (the simulator is the bottleneck).

A run folder (data/runs/b2rl/<name>) holds settings.json (the run's identity), iterations.jsonl (one row per update),
episodes.jsonl (one row per finished episode), ckpt-<update>.pt (written to a temporary file and renamed; the last 3
and every 10th are kept) and latest.json. A STOP file in the folder, SIGINT or SIGTERM end the run after the current
update's checkpoint. A new start with the same folder continues from the latest checkpoint; episodes that were
running restart from their first step (the runbook accepts this).
"""
from dataclasses import asdict, dataclass, fields
import io
import json
import math
import os
from pathlib import Path
import signal
import time

import numpy as np

from smart_building.learned.b2rl_env import ROOT, TRAIN_POOL, HourlyEnvs, scenario_dirs

IDENTITY_EXEMPT = ("threads",)   # runtime only: may change between segments of a run


@dataclass
class Settings:
    seed: int = 42
    num_envs: int = 4
    steps_per_env: int = 168           # decisions per environment per update (one week)
    total_decisions: int = 400_000     # the planned length: it sets the learning-rate decay, not a stop
    learning_rate: float = 3e-4
    final_learning_rate: float = 3e-5
    update_epochs: int = 10
    num_minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    clip_vloss: bool = True
    norm_adv: bool = True
    vf_coef: float = 0.5
    ent_coef: float = 0.02
    max_grad_norm: float = 0.5
    hidden: int = 128
    log_std_init: float = -1.0
    actor_out_std: float = 0.01
    obs_clip: float = 10.0
    reward_clip: float = 10.0
    obs_epsilon: float = 1e-8
    rule: str | None = None            # None: G36PolicyB2R2 (a diagnose.py preset name only for pilot runs)
    layout: str = "v1"                 # policies/b2rl.py LAYOUTS: the parameters and their mapping (v2: + sp_max_pa)
    reward_baseline: str | None = None  # a folder of the rule's per-decision rewards: train on reward - rule reward
    anchor: dict | None = None         # rule parameters a = 0 gives (policies/b2rl.py HourlyRule; None: the rule's own)
    pool: str = TRAIN_POOL.relative_to(ROOT).as_posix()
    steps_cap: int | None = None       # tests and smoke runs: episodes end after this many steps
    threads: int = 1

    @property
    def num_updates(self):
        return max(1, math.ceil(self.total_decisions / (self.num_envs * self.steps_per_env)))


def identity(settings):
    return {k: v for k, v in asdict(settings).items() if k not in IDENTITY_EXEMPT}


def make_agent(n_obs, n_act, settings):
    import torch
    from torch import nn
    from smart_building.learned.ppo import layer_init

    class Agent(nn.Module):
        """CleanRL Agent (L111-141) with ReLU 128 x 2 and the 0.01 output layer of the actor."""

        def __init__(self):
            super().__init__()
            h = settings.hidden
            self.critic = nn.Sequential(layer_init(nn.Linear(n_obs, h)), nn.ReLU(), layer_init(nn.Linear(h, h)),
                                        nn.ReLU(), layer_init(nn.Linear(h, 1), std=1.0))
            self.actor_mean = nn.Sequential(layer_init(nn.Linear(n_obs, h)), nn.ReLU(), layer_init(nn.Linear(h, h)),
                                            nn.ReLU(), layer_init(nn.Linear(h, n_act), std=settings.actor_out_std))
            self.actor_logstd = nn.Parameter(torch.full((1, n_act), float(settings.log_std_init)))

        def get_value(self, x):
            return self.critic(x)

        def get_action_and_value(self, x, action=None):
            mean = self.actor_mean(x)
            std = torch.exp(self.actor_logstd.expand_as(mean))
            distribution = torch.distributions.Normal(mean, std)
            if action is None:
                action = distribution.sample()
            return action, distribution.log_prob(action).sum(1), distribution.entropy().sum(1), self.critic(x)

    return Agent()


class ObservationNormalizer:
    """gymnasium NormalizeObservation: update the running statistics with the batch, then (x - mean) / sqrt(var + eps),
    clipped to +-clip (CleanRL L95-98)."""

    def __init__(self, shape, epsilon, clip):
        from gymnasium.wrappers.utils import RunningMeanStd
        self.rms = RunningMeanStd(shape=shape)
        self.epsilon, self.clip = float(epsilon), float(clip)

    def __call__(self, batch, update=True):
        batch = np.asarray(batch, dtype=np.float64)
        if update:
            self.rms.update(batch)
        return np.clip((batch - self.rms.mean) / np.sqrt(self.rms.var + self.epsilon), -self.clip, self.clip)

    def state(self):
        return {"mean": self.rms.mean.tolist(), "var": self.rms.var.tolist(), "count": float(self.rms.count)}

    def load(self, state):
        self.rms.mean = np.asarray(state["mean"], dtype=np.float64)
        self.rms.var = np.asarray(state["var"], dtype=np.float64)
        self.rms.count = float(state["count"])


def _append_jsonl(path, row):
    with open(path, "a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, default=float) + "\n")


def save_checkpoint(folder, update, payload, keep_last=3, keep_every=10):
    """ckpt-<update>.pt written to a temporary file and renamed, then latest.json; older checkpoints thinned out."""
    import torch
    from smart_building.learned.ppo import _fsync_directory, _write_bytes, _write_json
    folder = Path(folder)
    name = f"ckpt-{update:05d}.pt"
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    temporary = folder / (name + ".partial")
    _write_bytes(temporary, buffer.getvalue())
    os.replace(temporary, folder / name)
    _fsync_directory(folder)
    _write_json(folder / "latest.json", {"checkpoint": name, "update": update, "decisions": payload["decisions"]})
    for old in folder.glob("ckpt-*.pt"):
        number = int(old.stem.split("-")[1])
        # every 25th too: the validation points of `b2rl.py validate` (H3C's 25-epoch interval)
        if number <= update - keep_last and number % keep_every and number % 25:
            old.unlink(missing_ok=True)
    return folder / name


def latest_checkpoint(folder):
    folder = Path(folder)
    try:
        name = json.loads((folder / "latest.json").read_text(encoding="utf-8"))["checkpoint"]
    except (OSError, ValueError, KeyError):
        return None
    path = folder / name
    return path if path.is_file() else None


def load_checkpoint(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def train(out, settings, *, in_process=False, max_updates=None, log=print, backend=None, device="cpu"):
    """Train (or continue) the run in `out`; returns the last update's row. backend: a factory with the arguments of
    b2rl_env.HourlyEnvs (default HourlyEnvs; learned/b2rl_batch.BatchedHourlyEnvs for a batch simulator). device: where
    the learner's network runs ("cuda" pays off only with many environments per update)."""
    import torch
    from smart_building.learned.ppo import gae
    from smart_building.learned.ppo_env import RewardNormalizer, process_memory
    from smart_building.policies.b2rl import LAYOUTS, layout_features
    FEATURES, PARAMETERS = layout_features(settings.layout), LAYOUTS[settings.layout]["parameters"]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    settings_path = out / "settings.json"
    if settings_path.is_file():
        recorded = json.loads(settings_path.read_text(encoding="utf-8"))
        # a run older than a settings field has that field's default (layout: v1)
        recorded = {**{k: v for k, v in asdict(Settings()).items() if k not in recorded}, **recorded}
        if {k: v for k, v in recorded.items() if k not in IDENTITY_EXEMPT} != identity(settings):
            raise SystemExit(f"{out}: the settings differ from the run's settings.json; use a new folder")
    else:
        settings_path.write_text(json.dumps(asdict(settings), indent=1) + "\n", encoding="utf-8")
    torch.set_num_threads(int(settings.threads))
    torch.manual_seed(settings.seed)
    n_obs, n_act = len(FEATURES), len(PARAMETERS)
    agent = make_agent(n_obs, n_act, settings).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=settings.learning_rate, eps=1e-5)
    obs_norm = ObservationNormalizer((n_obs,), settings.obs_epsilon, settings.obs_clip)
    reward_norms = [RewardNormalizer(settings.gamma, settings.reward_clip) for _ in range(settings.num_envs)]
    rng = np.random.default_rng(settings.seed)
    update, decisions, episodes = 0, 0, 0
    # episodes finished per worker = the index, in the worker's own scenario sequence, of the episode it plays
    episode_counter = [0] * settings.num_envs
    state = None
    checkpoint = latest_checkpoint(out)
    if checkpoint is not None:
        state = load_checkpoint(checkpoint)
        agent.load_state_dict(state["agent"])
        optimizer.load_state_dict(state["optimizer"])
        obs_norm.load(state["obs_norm"])
        for normalizer, s in zip(reward_norms, state["reward_norms"]):
            normalizer.load(s)
        rng.bit_generator.state = state["rng"]["numpy"]
        torch.set_rng_state(state["rng"]["torch"])
        if str(device).startswith("cuda") and state["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state_all(state["rng"]["cuda"])
        update, decisions, episodes = state["update"], state["decisions"], state["episodes"]
        episode_counter = list(state["episode_counter"])
        log(f"continuing {out} from {checkpoint.name}: update {update}, {decisions} decisions, {episodes} episodes")
    scenarios = scenario_dirs(ROOT / settings.pool)
    if not scenarios:
        raise SystemExit(f"no scenarios in {settings.pool}")
    stop = {"flag": False}

    def request_stop(signum, frame):  # noqa: ARG001
        stop["flag"] = True
        log(f"signal {signum}: stopping after this update's checkpoint")

    previous = {s: signal.signal(s, request_stop) for s in (signal.SIGINT, signal.SIGTERM)}
    if backend is not None and (settings.layout != "v1" or settings.reward_baseline or settings.anchor):
        raise SystemExit("layout v2, the rule-relative reward and the anchor run on the per-environment workers only "
                         "(no --batch)")
    extra = {"layout": settings.layout} if settings.layout != "v1" else {}
    if settings.reward_baseline:
        extra["reward_baseline"] = ROOT / settings.reward_baseline
    if settings.anchor:
        extra["anchor"] = settings.anchor
    envs = (backend or HourlyEnvs)(settings.num_envs, settings.seed, scenarios, rule=settings.rule,
                                   steps_cap=settings.steps_cap, in_process=in_process, **extra)
    row = None
    try:
        # every worker (re)starts the episode of its own sequence it was playing (a new run: its first one). A
        # worker whose episode had just started at the checkpoint ("fresh") continues exactly: its first features are
        # the checkpoint's (not normalised twice); one that was in the middle restarts it as a new episode.
        T, N = settings.steps_per_env, settings.num_envs
        starts = envs.call([(i, "start", (episode_counter[i],)) for i in range(N)])
        if state is not None and state.get("next_obs") is not None:
            next_obs, next_done = np.array(state["next_obs"]), np.array(state["next_done"])
            stale = [i for i in range(N) if not state["fresh"][i]]
            if stale:
                next_obs[stale] = obs_norm(np.stack([starts[i]["features"] for i in stale]))
                next_done[stale] = 1.0
                for i in stale:                  # the restarted episode's discounted return starts again
                    reward_norms[i].discounted_reward = np.array([0.0])
        else:
            next_obs, next_done = obs_norm(np.stack([s["features"] for s in starts])), np.zeros(N)
        fresh = [True] * N
        while (max_updates is None or update < max_updates) and not stop["flag"] \
                and not (out / "STOP").exists():
            started = time.time()
            update += 1
            frac = 1.0 - (update - 1) / settings.num_updates
            lr = max(settings.final_learning_rate,
                     settings.final_learning_rate + (settings.learning_rate - settings.final_learning_rate) * frac)
            for group in optimizer.param_groups:
                group["lr"] = lr
            # every environment goes at its own pace (2026-10-27: in lock step the six waited for the largest building
            # at every decision and were busy only 45-99% of the time): an environment gets its next action as soon as
            # its last one is done, until the update's T * N decisions are handed out, so smaller buildings contribute
            # more of them. In-process (tests) the order is round-robin: exactly T per environment.
            seq = [{"obs": [], "act": [], "logp": [], "val": [], "done": [], "rew": []} for _ in range(N)]
            raw_rewards, parts, applied, finished = [], {"regret": [], "energy": [], "carbon": []}, [], []
            official = []            # the official 12-step rewards (= raw_rewards without a rule-relative reward)
            obs_now, done_now = [np.asarray(x) for x in next_obs], [float(x) for x in next_done]
            issued, deciding, starting = 0, set(), set()

            def hand_out(i):
                nonlocal issued
                o = torch.as_tensor(obs_now[i], dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(o)
                for key, item in (("obs", o[0]), ("act", action[0]), ("logp", logprob[0]), ("val", value.flatten()[0]),
                                  ("done", done_now[i])):
                    seq[i][key].append(item)
                envs.send(i, "decide", (action[0].cpu().numpy().astype(np.float64),))
                deciding.add(i)
                issued += 1

            for i in range(N):
                hand_out(i)
            while deciding or starting:
                for i in envs.ready(deciding | starting):
                    r = envs.receive(i)
                    if i in starting:            # its next episode is loaded: the first features of that episode
                        starting.discard(i)
                        obs_now[i], done_now[i] = obs_norm(np.asarray([r["features"]]))[0], 1.0
                    else:
                        deciding.discard(i)
                        raw_rewards.append(r["reward"])
                        official.append(r.get("official_reward", r["reward"]))
                        for key in parts:
                            parts[key].append(r["parts"][key])
                        applied.append([r["parameters"][k] for k in PARAMETERS])
                        seq[i]["rew"].append(float(np.ravel(reward_norms[i](r["reward"], float(r["done"])))[0]))
                        decisions += 1
                        if r["done"]:
                            episodes += 1
                            episode_counter[i] += 1
                            finished.append({**r["episode"], "worker": i, "update": update})
                            envs.send(i, "start", (episode_counter[i],))
                            starting.add(i)
                            continue
                        obs_now[i], done_now[i] = obs_norm(np.asarray([r["features"]]))[0], 0.0
                    if issued < T * N:
                        hand_out(i)
            next_obs, next_done = np.stack(obs_now), np.array(done_now)
            fresh = [bool(x) for x in done_now]
            batch = {k: [] for k in ("obs", "act", "logp", "adv", "ret", "val")}
            with torch.no_grad():
                for i in range(N):                # GAE along each environment's own sequence
                    if not seq[i]["rew"]:
                        continue
                    val = torch.stack(seq[i]["val"]).reshape(-1, 1)
                    last = agent.get_value(torch.as_tensor(obs_now[i], dtype=torch.float32, device=device)
                                           .unsqueeze(0)).reshape(1, 1)
                    adv, ret = gae(torch.tensor(seq[i]["rew"], dtype=torch.float32, device=device).reshape(-1, 1), val,
                                   torch.tensor(seq[i]["done"], dtype=torch.float32, device=device).reshape(-1, 1), last,
                                   torch.tensor([done_now[i]], dtype=torch.float32, device=device), settings.gamma,
                                   settings.gae_lambda)
                    for key, item in (("obs", torch.stack(seq[i]["obs"])), ("act", torch.stack(seq[i]["act"])),
                                      ("logp", torch.stack(seq[i]["logp"])), ("adv", adv.reshape(-1)),
                                      ("ret", ret.reshape(-1)), ("val", val.reshape(-1))):
                        batch[key].append(item)
            stats = _ppo_update(agent, optimizer, settings, rng, *(torch.cat(batch[k]) for k in
                                                                   ("obs", "act", "logp", "adv", "ret", "val")))
            for episode in finished:
                _append_jsonl(out / "episodes.jsonl", episode)
            applied = np.array(applied)
            seconds = time.time() - started
            row = {"update": update, "decisions": decisions, "episodes": episodes,
                   "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": round(seconds, 1),
                   "decisions_per_second": T * N / seconds, "env_decisions": [len(s["rew"]) for s in seq],
                   "learning_rate": lr,
                   "reward_12_mean": float(np.mean(raw_rewards)),
                   **({"official_reward_12_mean": float(np.mean(official))} if settings.reward_baseline else {}),
                   "parts": {k: float(np.mean(v)) for k, v in parts.items()},
                   "parameters": {k: {"mean": float(applied[:, j].mean()), "std": float(applied[:, j].std())}
                                  for j, k in enumerate(PARAMETERS)},
                   "sigma": torch.exp(agent.actor_logstd.detach()).flatten().tolist(),
                   "finished_episodes": [{k: e[k] for k in ("label", "cost", "worker")} for e in finished],
                   **stats,
                   "memory": {"trainer": process_memory(), "workers": envs.call([(i, "memory", ()) for i in range(N)])}}
            _append_jsonl(out / "iterations.jsonl", row)
            save_checkpoint(out, update, {
                "update": update, "decisions": decisions, "episodes": episodes, "episode_counter": episode_counter,
                "agent": agent.state_dict(), "optimizer": optimizer.state_dict(), "obs_norm": obs_norm.state(),
                "reward_norms": [n.state() for n in reward_norms],
                "rng": {"numpy": rng.bit_generator.state, "torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all() if str(device).startswith("cuda") else None},
                "next_obs": next_obs.tolist(), "next_done": next_done.tolist(), "fresh": fresh,
                "settings": asdict(settings)})
            log(f"update {update}: {decisions} decisions, {episodes} episodes, 12-step reward "
                f"{row['reward_12_mean']:+.5f}, sigma {np.mean(row['sigma']):.3f}, kl {stats['approx_kl']:.4f}, "
                f"{row['decisions_per_second']:.2f} decisions/s")
    finally:
        envs.close()
        for s, handler in previous.items():
            signal.signal(s, handler)
    return row


def _ppo_update(agent, optimizer, settings, rng, b_obs, b_actions, b_logprobs, b_advantages, b_returns, b_values):
    """CleanRL L248-305 (no target_kl)."""
    import torch
    batch = b_obs.shape[0]
    minibatch = batch // settings.num_minibatches
    clipfracs, approx_kl, pg_loss, v_loss, entropy_loss = [], torch.tensor(0.0), None, None, None
    for _ in range(settings.update_epochs):
        order = rng.permutation(batch)
        for start in range(0, batch, minibatch):
            mb = torch.as_tensor(order[start:start + minibatch], device=b_obs.device)
            _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb], b_actions[mb])
            logratio = newlogprob - b_logprobs[mb]
            ratio = logratio.exp()
            with torch.no_grad():
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs.append(((ratio - 1.0).abs() > settings.clip_coef).float().mean().item())
            mb_adv = b_advantages[mb]
            if settings.norm_adv:
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
            pg_loss = torch.max(-mb_adv * ratio,
                                -mb_adv * torch.clamp(ratio, 1 - settings.clip_coef, 1 + settings.clip_coef)).mean()
            newvalue = newvalue.view(-1)
            if settings.clip_vloss:
                unclipped = (newvalue - b_returns[mb]) ** 2
                clipped = b_values[mb] + torch.clamp(newvalue - b_values[mb], -settings.clip_coef, settings.clip_coef)
                v_loss = 0.5 * torch.max(unclipped, (clipped - b_returns[mb]) ** 2).mean()
            else:
                v_loss = 0.5 * ((newvalue - b_returns[mb]) ** 2).mean()
            entropy_loss = entropy.mean()
            loss = pg_loss - settings.ent_coef * entropy_loss + v_loss * settings.vf_coef
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), settings.max_grad_norm)
            optimizer.step()
    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    var_y = np.var(y_true)
    return {"approx_kl": float(approx_kl), "clipfrac": float(np.mean(clipfracs)), "pg_loss": pg_loss.item(),
            "v_loss": v_loss.item(), "entropy": entropy_loss.item(),
            "explained_variance": float("nan") if var_y == 0 else float(1 - np.var(y_true - y_pred) / var_y)}


def export(checkpoint, out_json=None):
    """The actor of a checkpoint as a policies/b2rl.py model (dict; written to out_json when given)."""
    from smart_building.policies.b2rl import DECISION_STEPS, LAYOUTS, MODEL_FORMAT, layout_features
    from smart_building.policies.g36 import C3_COMBINATIONS, SLOW_PARAMETER_RANGES
    checkpoint = Path(checkpoint)
    state = load_checkpoint(checkpoint)
    settings = Settings(**{f.name: state["settings"][f.name] for f in fields(Settings) if f.name in state["settings"]})
    FEATURES, PARAMETERS = layout_features(settings.layout), LAYOUTS[settings.layout]["parameters"]
    linear = sorted({key.rsplit(".", 1)[0] for key in state["agent"] if key.startswith("actor_mean.")},
                    key=lambda k: int(k.split(".")[1]))
    layers = [{"weight": state["agent"][f"{k}.weight"].double().tolist(),
               "bias": state["agent"][f"{k}.bias"].double().tolist()} for k in linear]
    import hashlib
    model = {"format": MODEL_FORMAT, **({"layout": settings.layout} if settings.layout != "v1" else {}),
             **({"anchor": dict(settings.anchor)} if settings.anchor else {}),
             "features": list(FEATURES), "parameters": list(PARAMETERS),
             "ranges": {k: list(SLOW_PARAMETER_RANGES[k]) for k in PARAMETERS}, "interval": DECISION_STEPS,
             "obs_mean": state["obs_norm"]["mean"], "obs_var": state["obs_norm"]["var"],
             "obs_epsilon": settings.obs_epsilon, "obs_clip": settings.obs_clip, "layers": layers,
             "log_std": state["agent"]["actor_logstd"].double().flatten().tolist(),
             "rule": {"class": "G36PolicyB2R2" if not settings.rule else settings.rule,
                      "b2r2_combination": C3_COMBINATIONS["laa_occrow_dampfloor_b2r2"]},
             "source": {"checkpoint": checkpoint.as_posix(), "update": state["update"],
                        "decisions": state["decisions"], "seed": settings.seed,
                        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}}
    if out_json is not None:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(model) + "\n", encoding="utf-8")
    return model
