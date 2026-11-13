"""Entity transformer of the student B1-L and its competition wrapper (WP4: D-L01-D-L07).

D-L01  Tokens of one type share a linear tokenizer.
D-L02  No zone index or positional embedding; full attention (no mask).
D-L03  AHU and hot-water tokens always exist (tokens.py).
D-L04  Decoders shared within a type; the decoder input is the encoder output
       concatenated with the token's own (normalised) features (AMORPHEUS
       conditioning, notes/06 §B `TransformerActor.py:52-55`); tanh to [-1, 1],
       trained against the teacher's normalised action without mjrl's output
       shift / scale (REQ-018 (2), approved); the last layer starts small.
D-L05  3 layers, 2-4 heads, d 64-128, FFN 256, LayerNorm.
D-L06  History length H in {12, 64, 288} (tokens.py); H = 64 (REQ-018 (1), 2026-10-17).
D-L07  The on/off (discrete) actions come from the B1-R teacher.

REQ-014 as settled by docs/plan/baseline-v1-stage45.md §2 (2026-10-17):
  (1, 2) every input feature is normalised with the mean and standard deviation
         of the first-iteration demonstrations, (x - mean) / (std + 1e-8), no
         clipping (DAPG / mjrl `behavior_cloning.py:52-61`, `fc_network.py:46`,
         notes/10 §K); one set of statistics per token type, shared by every
         token of the type (D-L02 permutation equivariance); static features
         the same way. The statistics are buffers of the model: frozen, saved
         and loaded with it (set_normalization). Input features whose standard
         deviation is 0 in those demonstrations are left out of the inputs
         (REQ-018 (6)): the kept positions are the input_index_<type> buffers.
  (4)    last layer orthogonal, gain 0.01, bias 0 (Building2Building
         `amorpheus.py:177-188`, notes/10 §H).
  (5, 6) AMORPHEUS: d 128, 2 heads, FFN 256, 3 layers, tokenizer output x
         sqrt(d), post-norm layers plus a final LayerNorm, dropout 0 (notes/10 §G).
  (7)    one torch thread at inference.
The per-type log_std buffers hold the log of the demonstrations' action standard
deviation (mjrl `set_variance_with_data`, D-L09); the student acts with its mean.

StudentPolicy follows the competition interface. D-T02: when the network
raises or returns non-finite values the step uses the teacher's action; D-T03:
the action is checked for finite values and clipped to the action spec.
"""
from dataclasses import asdict, dataclass
import io
import math

import numpy as np
import torch
from torch import nn

from smart_building.learned.tokens import OUTPUT_DIMS, TOKEN_TYPES, HistoryBuffer, TokenLayout

NORMALIZATION_EPS = 1e-8          # mjrl fc_network.py:46
LOG_STD_EPS = 1e-12               # mjrl behavior_cloning.py:68-72 (set_variance_with_data)
MODEL_FORMAT = "b1-l-student/1"


@dataclass(frozen=True)
class StudentConfig:
    history: int = 64                  # D-L06: 12, 64 or 288; 64 (REQ-018 (1))
    d_model: int = 128                 # REQ-014 (5): AMORPHEUS
    heads: int = 2                     # REQ-014 (5)
    layers: int = 3                    # D-L05
    ffn: int = 256                     # D-L05
    final_init_gain: float = 0.01      # REQ-014 (4): orthogonal, gain 0.01
    dropout: float = 0.0               # REQ-014 (6), set explicitly (PyTorch's default is 0.1)
    torch_threads: int | None = 1      # REQ-014 (7): one thread at inference; None leaves torch's setting

    def __post_init__(self):
        if not 64 <= self.d_model <= 128 or not 2 <= self.heads <= 4 or self.layers != 3 or self.ffn != 256:
            raise ValueError("outside the D-L05 ranges")
        if self.d_model % self.heads:
            raise ValueError("d_model must be divisible by heads")


def keep_from_state(state_dict):
    """The kept input positions {type: list} recorded in a state dict (None if it has none)."""
    if state_dict is None or not any(name.startswith("input_index_") for name in state_dict):
        return None
    return {k: state_dict[f"input_index_{k}"].tolist() for k in TOKEN_TYPES}


class EntityTransformer(nn.Module):
    """Typed tokens -> shared transformer -> typed decoders (D-L01, D-L02, D-L04, D-L05).

    input_dims are the raw token input sizes (H x dynamic + static features); keep gives, per
    type, the input positions the network reads (default all; REQ-018 (6) drops those whose
    standard deviation is 0 in the first-iteration demonstrations)."""

    def __init__(self, input_dims, config, keep=None):
        super().__init__()
        self.config = config
        self.kinds = TOKEN_TYPES
        self.input_dims = dict(input_dims)
        keep = keep if keep is not None else {k: range(input_dims[k]) for k in self.kinds}
        self.kept_dims = {k: len(keep[k]) for k in self.kinds}
        self.scale = math.sqrt(config.d_model)
        for k in self.kinds:                                   # REQ-014 (1, 2), REQ-018 (6): frozen inputs
            index = torch.as_tensor(list(keep[k]), dtype=torch.long)
            if index.numel() and (int(index.min()) < 0 or int(index.max()) >= input_dims[k]):
                raise ValueError(f"{k}: kept positions outside 0..{input_dims[k] - 1}")
            self.register_buffer(f"input_index_{k}", index)
            self.register_buffer(f"input_mean_{k}", torch.zeros(self.kept_dims[k]))
            self.register_buffer(f"input_std_{k}", torch.ones(self.kept_dims[k]))
        self.tokenizers = nn.ModuleDict({k: nn.Linear(self.kept_dims[k], config.d_model) for k in self.kinds})
        layer = nn.TransformerEncoderLayer(config.d_model, config.heads, dim_feedforward=config.ffn,
                                           dropout=config.dropout, batch_first=True, norm_first=False)
        self.encoder = nn.TransformerEncoder(layer, config.layers, norm=nn.LayerNorm(config.d_model),
                                             enable_nested_tensor=False)
        self.decoders = nn.ModuleDict({k: nn.Linear(config.d_model + self.kept_dims[k], OUTPUT_DIMS[k])
                                       for k in self.kinds if OUTPUT_DIMS[k]})
        for k, decoder in self.decoders.items():
            nn.init.orthogonal_(decoder.weight, gain=config.final_init_gain)          # REQ-014 (4)
            nn.init.zeros_(decoder.bias)
            self.register_buffer(f"log_std_{k}", torch.zeros(OUTPUT_DIMS[k]))       # D-L09

    def set_normalization(self, mean, std):
        """Freeze the input statistics {type: (kept dims,)} of the kept positions (first-iteration
        demonstrations)."""
        for k in self.kinds:
            getattr(self, f"input_mean_{k}").copy_(torch.as_tensor(np.asarray(mean[k]), dtype=torch.float32))
            getattr(self, f"input_std_{k}").copy_(torch.as_tensor(np.asarray(std[k]), dtype=torch.float32))

    def set_log_std(self, action_std):
        """log_std = log(std + 1e-12) of the demonstrations' actions per type and output (mjrl, D-L09)."""
        for k in self.decoders:
            value = np.log(np.asarray(action_std[k], dtype=np.float64) + LOG_STD_EPS)
            getattr(self, f"log_std_{k}").copy_(torch.as_tensor(value, dtype=torch.float32))

    def normalized(self, tokens):
        """Kept positions of the raw inputs, then (x - mean) / (std + 1e-8), no clipping."""
        return {k: (tokens[k][..., getattr(self, f"input_index_{k}")] - getattr(self, f"input_mean_{k}"))
                / (getattr(self, f"input_std_{k}") + NORMALIZATION_EPS) for k in self.kinds}

    def forward(self, tokens):
        """tokens: {type: (batch, count, input_dim)} raw -> {type: (batch, count, out_dim)} in [-1, 1]."""
        x = self.normalized(tokens)
        counts = [x[k].shape[1] for k in self.kinds]
        embedded = torch.cat([self.tokenizers[k](x[k]) * self.scale for k in self.kinds], dim=1)
        encoded = self.encoder(embedded)                       # D-L02: no positions, no mask
        parts = torch.split(encoded, counts, dim=1)
        out = {}
        for k, part in zip(self.kinds, parts):
            if k in self.decoders:
                out[k] = torch.tanh(self.decoders[k](torch.cat([part, x[k]], dim=-1)))
        return out


def save_student(path, model, *, teacher=None, extra=None):
    """One file: format, StudentConfig, state dict (with the frozen statistics), teacher spec."""
    payload = {"format": MODEL_FORMAT, "config": asdict(model.config), "input_dims": model.input_dims,
               "state_dict": model.state_dict(), "teacher": teacher, **(extra or {})}
    torch.save(payload, path)


def state_bytes(model):
    """The state dict as bytes (to hand a model to a rollout process)."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getvalue()


def load_state_bytes(data):
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)


def load_student(path):
    """(StudentConfig, state dict, teacher spec) from save_student's file."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != MODEL_FORMAT:
        raise ValueError(f"{path}: not a {MODEL_FORMAT} file")
    return StudentConfig(**payload["config"]), payload["state_dict"], payload.get("teacher")


class StudentNetwork:
    """The network side of the student for one building: history, tokens, mean action (no switches)."""

    def __init__(self, building_meta_data, action_spec, config, state_dict=None):
        self.config = config
        if config.torch_threads:
            torch.set_num_threads(config.torch_threads)                       # REQ-014 (7)
        self.layout = TokenLayout(building_meta_data, action_spec, config.history)
        self.model = EntityTransformer(self.layout.input_dims, config, keep_from_state(state_dict))
        if state_dict is not None:
            self.model.load_state_dict(state_dict)
        self.model.eval()
        self.history = HistoryBuffer(config.history, self.layout.n_fields)

    def mean_action(self, obs):
        """Continuous action (normalised, unclipped) for this observation; pushes it into the history."""
        window = self.history.push(obs)
        inputs = self.layout.inputs(window)
        with torch.no_grad():
            outputs = self.model({k: torch.from_numpy(v).unsqueeze(0) for k, v in inputs.items()})
        return self.layout.scatter({k: v[0].numpy() for k, v in outputs.items()})


class StudentPolicy:
    """B1-L with the competition interface (untrained unless a state dict is given).

    switches_class is the B1-R teacher whose on/off actions the student uses (D-L07);
    the default is the B1-R champion C3 (G36PolicyV3Ti900, stage45 §1).
    """

    name = "b1_l"

    def __init__(self, building_meta_data, action_spec, config=None, state_dict=None, switches_class=None):
        self.config = config if config is not None else StudentConfig()
        self.network = StudentNetwork(building_meta_data, action_spec, self.config, state_dict)
        self.layout, self.model, self.history = self.network.layout, self.network.model, self.network.history
        if switches_class is None:
            from smart_building.policies.g36 import G36PolicyV3Ti900 as switches_class
        self.switches = switches_class(building_meta_data, action_spec)       # D-L07
        continuous = action_spec["continuous_action"]
        self._low = np.asarray(continuous.low, dtype=np.float32)
        self._high = np.asarray(continuous.high, dtype=np.float32)
        self.fallback_steps = 0

    def continuous_action(self, obs):
        return self.network.mean_action(obs)

    def get_action(self, obs):
        rule = self.switches.get_action(obs)                                   # D-L07 (and D-T02 fallback)
        try:
            continuous = self.continuous_action(obs)
            if not np.all(np.isfinite(continuous)):
                raise FloatingPointError("non-finite student action")      # D-T03
        except Exception:  # noqa: BLE001 - D-T02: the teacher's action for this step
            self.fallback_steps += 1
            return rule
        return {"continuous_action": np.clip(continuous, self._low, self._high).astype(np.float32),
                "discrete_action": rule["discrete_action"]}


def student_policy_class(path, switches_class=None):
    """A competition policy class (building_meta_data, action_spec) bound to a saved student file."""
    config, state_dict, _ = load_student(path)

    class BoundStudent(StudentPolicy):
        def __init__(self, building_meta_data, action_spec):
            super().__init__(building_meta_data, action_spec, config, state_dict, switches_class)

    return BoundStudent
