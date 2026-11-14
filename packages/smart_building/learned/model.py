"""Entity transformer of the student B1-L and its competition wrapper.

       concatenated with the token's own (normalised) features (AMORPHEUS
       trained against the teacher's normalised action without mjrl's output

  (1, 2) every input feature is normalised with the mean and standard deviation
         of the first-iteration demonstrations, (x - mean) / (std + 1e-8), no
         clipping (DAPG / mjrl `behavior_cloning.py:52-61`, `fc_network.py:46`,
         token of the type; static features
         the same way. The statistics are buffers of the model: frozen, saved
         and loaded with it (set_normalization). Input features whose standard
         deviation is 0 in those demonstrations are left out of the inputs
  (4)    last layer orthogonal, gain 0.01, bias 0 (Building2Building
  (5, 6) AMORPHEUS: d 128, 2 heads, FFN 256, 3 layers, tokenizer output x
         sqrt(d), post-norm layers plus a final LayerNorm, dropout 0.
  (7)    one torch thread at inference.
The per-type log_std buffers hold the log of the demonstrations' action standard
deviation; the student acts with its mean.

the action is checked for finite values and clipped to the action spec."""
from dataclasses import asdict, dataclass
import io
import math

import numpy as np
import torch
from torch import nn

from smart_building.learned.tokens import OUTPUT_DIMS, TOKEN_TYPES, HistoryBuffer, TokenLayout

NORMALIZATION_EPS = 1e-8
LOG_STD_EPS = 1e-12
MODEL_FORMAT = "b1-l-student/1"


@dataclass(frozen=True)
class StudentConfig:
    history: int = 64
    d_model: int = 128
    heads: int = 2
    layers: int = 3
    ffn: int = 256
    final_init_gain: float = 0.01
    dropout: float = 0.0
    torch_threads: int | None = 1

    def __post_init__(self):
        if not 64 <= self.d_model <= 128 or not 2 <= self.heads <= 4 or self.layers != 3 or self.ffn != 256:
            raise ValueError("outside the ranges")
        if self.d_model % self.heads:
            raise ValueError("d_model must be divisible by heads")


def keep_from_state(state_dict):
    """The kept input positions {type: list} recorded in a state dict (None if it has none)."""
    if state_dict is None or not any(name.startswith("input_index_") for name in state_dict):
        return None
    return {k: state_dict[f"input_index_{k}"].tolist() for k in TOKEN_TYPES}


class EntityTransformer(nn.Module):
    """Typed tokens -> shared transformer -> typed decoders.

    input_dims are the raw token input sizes (H x dynamic + static features); keep gives, per
    standard deviation is 0 in the first-iteration demonstrations)."""

    def __init__(self, input_dims, config, keep=None):
        super().__init__()
        self.config = config
        self.kinds = TOKEN_TYPES
        self.input_dims = dict(input_dims)
        keep = keep if keep is not None else {k: range(input_dims[k]) for k in self.kinds}
        self.kept_dims = {k: len(keep[k]) for k in self.kinds}
        self.scale = math.sqrt(config.d_model)
        for k in self.kinds:
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
            nn.init.orthogonal_(decoder.weight, gain=config.final_init_gain)
            nn.init.zeros_(decoder.bias)
            self.register_buffer(f"log_std_{k}", torch.zeros(OUTPUT_DIMS[k]))

    def set_normalization(self, mean, std):
        """Freeze the input statistics {type: (kept dims,)} of the kept positions (first-iteration
        demonstrations)."""
        for k in self.kinds:
            getattr(self, f"input_mean_{k}").copy_(torch.as_tensor(np.asarray(mean[k]), dtype=torch.float32))
            getattr(self, f"input_std_{k}").copy_(torch.as_tensor(np.asarray(std[k]), dtype=torch.float32))

    def set_log_std(self, action_std):
        """log_std = log(std + 1e-12) of the demonstrations' actions per type and output."""
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
        encoded = self.encoder(embedded)
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
            torch.set_num_threads(config.torch_threads)
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

    switches_class is the B1-R teacher whose on/off actions the student uses;
    the default is the B1-R champion C3."""

    name = "b1_l"

    def __init__(self, building_meta_data, action_spec, config=None, state_dict=None, switches_class=None):
        self.config = config if config is not None else StudentConfig()
        self.network = StudentNetwork(building_meta_data, action_spec, self.config, state_dict)
        self.layout, self.model, self.history = self.network.layout, self.network.model, self.network.history
        if switches_class is None:
            from smart_building.policies.g36 import G36PolicyV3Ti900 as switches_class
        self.switches = switches_class(building_meta_data, action_spec)
        continuous = action_spec["continuous_action"]
        self._low = np.asarray(continuous.low, dtype=np.float32)
        self._high = np.asarray(continuous.high, dtype=np.float32)
        self.fallback_steps = 0

    def continuous_action(self, obs):
        return self.network.mean_action(obs)

    def get_action(self, obs):
        rule = self.switches.get_action(obs)
        try:
            continuous = self.continuous_action(obs)
            if not np.all(np.isfinite(continuous)):
                raise FloatingPointError("non-finite student action")
        except Exception:
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
