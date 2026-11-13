"""B1-RL stage-3 submission policy: the fold ensemble of the j9 PPO agents (docs/plan/b1-rl-stage3.md §8;
handoff/RUNBOOK-b1-chain.md §4.4).

Each agent file is read with learned/ppo.py load_agent, as ppo.policy_class reads one agent (the interface the INNER
and S4 evaluations use, remote/tasks.py run_rl_task), and each agent's continuous output is its mean action computed as
ppo.PPOPolicy computes it (anchor "student": the actor's tanh mean, no observation clip, scattered to the action vector).
The rest follows policies/b1l_ensemble_v5.py, the j7b-v5s student ensemble:
  - one rule controller instance per episode, the agents' own rule controller (v4 + readback,
    G36PolicyC3LaaOccrowDampfloorReadback; every agent file must name the class given here): on/off (D-L07, D-RL04 (a))
    and the D-T02 fallback; for agents whose switches are learned and anchored on it (switches "bernoulli",
    switch_prior "rule"; reviewer 2026-10-25) its decision d of the step anchors each agent's switch probability
    sigmoid(head + gain x (2 d - 1)), and a switch is on when the agents' mean probability is at least 1/2 (one agent:
    ppo.PPOPolicy's logit >= 0); one ensemble takes agents of one switch mode;
  - its B1-R internal state (learned/privileged.py policy_features) is read once per step, before its get_action, and
    the same features are appended to every agent's tokens, in ppo.PPOPolicy's order (state, rule controller's call,
    history push, networks);
  - the continuous action is the mean of the agents' outputs (float64 mean of the float32 outputs); D-T03: a non-finite
    mean is a failure, the action is clipped to the action spec and returned as float32;
  - D-T02: if the state cannot be read, any agent raises or the mean is not finite, the step uses the rule controller's
    action.
Agents with the same history length share one observation window (the same buffer contents and token tables, so each
gets exactly the inputs its own ppo.PPOPolicy would build). With one agent the ensemble acts as ppo.policy_class bit for
bit; with N agents the continuous action is the mean of the N single-agent actions (tests/test_b1rl_ensemble.py).
"""
import numpy as np

from smart_building.learned import ppo
from smart_building.learned.privileged import FEATURES, PRIVILEGED_FORMAT, append_features, policy_features


def check_agent(path, settings, switches_class):
    """Refuse an agent the ensemble does not combine: stage 3's agents only (anchor "student", on/off from the rule
    controller), whose rule controller is the class used here, and a B1-R state entry of learned/privileged.py's format."""
    switches = settings.get("switches")
    anchored = switches == "bernoulli" and settings.get("switch_prior", "none") == "rule"
    if settings.get("anchor") != "student" or not (switches == "backbone" or anchored):
        raise ValueError(f"{path}: anchor {settings.get('anchor')!r}, switches {switches!r}, switch_prior "
                         f"{settings.get('switch_prior', 'none')!r} (the stage-3 ensemble takes student-anchored agents "
                         "with the rule controller's on/off, or switches learned and anchored on it)")
    rule = settings.get("rule_controller")
    name = rule.get("class") if isinstance(rule, dict) else rule
    if name != switches_class.__name__:
        raise ValueError(f"{path}: the agent's rule controller is {name!r}, not {switches_class.__name__!r}")
    privileged = settings.get("privileged")
    if privileged and (privileged.get("format") != PRIVILEGED_FORMAT
                       or {k: tuple(v) for k, v in privileged.get("features", {}).items()} != FEATURES):
        raise ValueError(f"{path}: B1-R state entry {privileged!r} is not {PRIVILEGED_FORMAT}")


def load_agents(paths, switches_class):
    """[(agent, settings, history)] of saved j9 agent files (ppo.load_agent), each checked by check_agent."""
    agents = []
    for path in paths:
        agent, settings, history = ppo.load_agent(path)
        check_agent(path, settings, switches_class)
        agents.append((agent, settings, history))
    modes = {settings.get("switches") for _, settings, _ in agents}
    if len(modes) > 1:
        raise ValueError(f"the agents mix switch modes {sorted(modes)}; one ensemble takes one kind")
    return agents


class EnsembleRLPolicy:
    """The competition interface over N loaded j9 agents and one rule controller class."""

    name = "b1_rl_ensemble"

    def __init__(self, building_meta_data, action_spec, agents, switches_class):
        import torch
        from smart_building.learned.tokens import HistoryBuffer, TokenLayout
        if not agents:
            raise ValueError("the ensemble needs at least one agent")
        torch.set_num_threads(1)                                                   # as ppo.PPOPolicy (REQ-014 (7))
        self.windows = {}
        for _, _, history in agents:
            if history not in self.windows:
                layout = TokenLayout(building_meta_data, action_spec, history)
                self.windows[history] = (layout, layout.spec(), HistoryBuffer(history, layout.n_fields))
        self.members = [(agent.eval(), bool(settings.get("privileged")), history,
                         np.float32(settings.get("switch_prior_gain", 4.0))) for agent, settings, history in agents]
        self.learned_switches = agents[0][1].get("switches") == "bernoulli"    # reviewer 2026-10-25
        self.switches = switches_class(building_meta_data, action_spec)          # D-L07, D-RL04 (a)
        continuous = action_spec["continuous_action"]
        self._low = np.asarray(continuous.low, dtype=np.float32)
        self._high = np.asarray(continuous.high, dtype=np.float32)
        self.fallback_steps = 0

    def _output(self, agent, privileged, history, tokens, features, prior=None):
        """(continuous output, switch probabilities or None) of one agent: ppo.PPOPolicy.get_action's computation for
        anchor "student"; with `prior` (gain x (2 d - 1), float32, the AHUs then the hot water) the probabilities
        sigmoid(head + prior), the logits summed in float32 as ppo.PPOPolicy sums them."""
        import torch
        layout, tables, _ = self.windows[history]
        if privileged:
            if features is None:
                raise RuntimeError("B1-R state unavailable")
            tokens = append_features(tokens, features)
        inputs = {k: torch.from_numpy(v).unsqueeze(0) for k, v in tokens.items()}
        probabilities = None
        with torch.no_grad():
            mean = agent.mean_action(inputs)
            if prior is not None:
                f = ppo.encode(agent.actor, inputs, agent.observation_clip)
                logits = np.concatenate([agent.switch_heads[k](f[k]).squeeze(-1)[0].numpy() for k in ("ahu", "hws")])
                probabilities = 1.0 / (1.0 + np.exp(-np.asarray(logits + prior, dtype=np.float64)))
        return (ppo.scatter(tables, {k: v[0].numpy() for k, v in mean.items()}, layout.index.n_continuous),
                probabilities)

    def get_action(self, obs):
        from smart_building.learned.tokens import TOKEN_TYPES
        try:
            features = policy_features(self.switches)                            # before this step's B1-R call
        except Exception:  # noqa: BLE001 - D-T02 below
            features = None
        rule = self.switches.get_action(obs)
        try:
            windows = {history: buffer.push(obs) for history, (_, _, buffer) in self.windows.items()}
            tokens = {history: ppo.window_tokens(self.windows[history][1], window, TOKEN_TYPES)
                      for history, window in windows.items()}
            decision = np.asarray(rule["discrete_action"], dtype=np.float32)
            results = [self._output(agent, privileged, history, tokens[history], features,
                                    gain * (np.float32(2.0) * decision - np.float32(1.0)) if self.learned_switches
                                    else None)
                       for agent, privileged, history, gain in self.members]
            outputs = [np.asarray(output, dtype=np.float64) for output, _ in results]
            continuous = np.mean(outputs, axis=0)
            if not np.all(np.isfinite(continuous)):
                raise FloatingPointError("non-finite ensemble action")          # D-T03
            discrete = np.asarray(rule["discrete_action"], dtype=np.int32)
            if self.learned_switches:                                          # reviewer 2026-10-25
                probability = np.mean([p for _, p in results], axis=0)
                if not np.all(np.isfinite(probability)):
                    raise FloatingPointError("non-finite switch probability")
                discrete = (probability >= 0.5).astype(np.int32)
        except Exception:  # noqa: BLE001 - D-T02: the rule controller's action for this step
            self.fallback_steps += 1
            return rule
        return {"continuous_action": np.clip(continuous, self._low, self._high).astype(np.float32),
                "discrete_action": discrete}


def ensemble_policy_class(paths, switches_class):
    """A competition policy class (building_meta_data, action_spec) over saved j9 agent files; the files are read once."""
    agents = load_agents(paths, switches_class)

    class BoundEnsembleRL(EnsembleRLPolicy):
        def __init__(self, building_meta_data, action_spec):
            super().__init__(building_meta_data, action_spec, agents, switches_class)

    return BoundEnsembleRL
