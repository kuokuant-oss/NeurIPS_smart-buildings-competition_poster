"""Pipeline 5 option specification (2026-11-04, docs/plan/geo-pipeline5.md §2): data/runs/b2rl/geo5/options.json ->
validated option levels -> models on the base gt_u1b, single or combined.

Option level fields:
  id         unique level name (model name; records are <id>__<spec>.json)
  option     the option it is a level of (levels of one option are mutually exclusive in a combination)
  layer      building | ahu | room           (split-plot: building / ahu = whole plot, room = sub-plot; the tree stage)
  category   physics | competition           (paper track = physics only; competition track = both)
  build      {"rule_config": {...overrides of the base rule_config...},
              "wrappers": [{"name": <registered wrapper or "module:function">, "args": {...}}]}
             (shorthand: "wrapper" + "args")
  conditions optional {"cities": [...], "seasons": [...], "oat_c": [lo, hi], "note": ...}: where the option is
             expected to help; the screen also reports it on those scenarios. A deployed condition must be inside
             the build (an observable switch such as sat_cost_model_min_oat_k): the policy never sees the city.
  source, expected, ms_per_step   free text / numbers for the report (catalogue reference, expected size, cost)
Wrappers change the model dict; register new option code in WRAPPERS or name a function "module:function" (module
on .tmp/setup). Standard library only."""
import copy
import hashlib
import importlib
import json
from pathlib import Path

from geo5_lib import ROOT

BASE_MODEL = ROOT / "data/runs/b2rl/search/models/model-gt_u1b.json"
LAYERS = ("building", "ahu", "room")
CATEGORIES = ("physics", "competition")
WHOLE = ("building", "ahu")
SEASONS = ("winter", "spring", "summer", "autumn")
CITIES = ("atlanta", "denver", "fairbanks", "phoenix", "singapore")
REUSE_MODEL_DIRS = [ROOT / "data/runs/b2rl/options/models", ROOT / "data/runs/b2rl/search/models"]
REUSE_RECORD_DIRS = {"options": ROOT / "data/runs/b2rl/options"}


# ---------------------------------------------------------------- wrappers
def room_option_add(model, args):
    """Add to a per-room setting in every option of the room policy (e.g. the occupied margins, opt_margin_03)."""
    for option in model["room_policy"]["options"]:
        for key, add in args.items():
            option[key] = round(float(option.get(key, 0.0)) + float(add), 6)
    return model


def room_option_set(model, args):
    """Set a per-room setting in every option of the room policy."""
    for option in model["room_policy"]["options"]:
        option.update(copy.deepcopy(args))
    return model


WRAPPERS = {"room_option_add": room_option_add, "room_option_set": room_option_set}


def wrapper_fn(name):
    if name in WRAPPERS:
        return WRAPPERS[name]
    if ":" in name:
        module, fn = name.split(":", 1)
        return getattr(importlib.import_module(module), fn)
    raise ValueError(f"unknown wrapper {name!r} (register it in geo5_options.WRAPPERS or use module:function)")


# ---------------------------------------------------------------- spec
def load(path):
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    levels = spec["options"]
    seen = set()
    for lv in levels:
        for key in ("id", "layer", "category", "build"):
            if key not in lv:
                raise ValueError(f"option level {lv.get('id')}: missing {key}")
        if lv["id"] in seen:
            raise ValueError(f"duplicate option level {lv['id']}")
        seen.add(lv["id"])
        if lv["layer"] not in LAYERS or lv["category"] not in CATEGORIES:
            raise ValueError(f"{lv['id']}: layer {lv['layer']} / category {lv['category']}")
        lv.setdefault("option", lv["id"])
        b = lv["build"]
        if "wrapper" in b:
            b.setdefault("wrappers", []).append({"name": b.pop("wrapper"), "args": b.pop("args", {})})
        b.setdefault("rule_config", {})
        b.setdefault("wrappers", [])
        for w in b["wrappers"]:
            wrapper_fn(w["name"])
        cond = lv.get("conditions") or {}
        for c in cond.get("cities", []):
            if c not in CITIES:
                raise ValueError(f"{lv['id']}: city {c}")
        for s in cond.get("seasons", []):
            if s not in SEASONS:
                raise ValueError(f"{lv['id']}: season {s}")
    spec.setdefault("base", {"name": "gt_u1b", "model": BASE_MODEL.relative_to(ROOT).as_posix()})
    return spec


def by_id(spec):
    return {lv["id"]: lv for lv in spec["options"]}


def base_model(spec):
    return json.loads((ROOT / spec["base"]["model"]).read_text(encoding="utf-8"))


def conflicts(levels):
    """Pairs of levels that cannot be on together: levels of one option, or two levels setting the same rule_config
    key (or the same per-room key with room_option_set) to different values."""
    out = []
    for i, a in enumerate(levels):
        for b in levels[i + 1:]:
            if a["option"] == b["option"]:
                out.append((a["id"], b["id"], "same option"))
                continue
            ra, rb = a["build"]["rule_config"], b["build"]["rule_config"]
            clash = [k for k in set(ra) & set(rb) if ra[k] != rb[k]]
            sa = {k for w in a["build"]["wrappers"] if w["name"] == "room_option_set" for k in w["args"]}
            sb = {k for w in b["build"]["wrappers"] if w["name"] == "room_option_set" for k in w["args"]}
            if clash or sa & sb:
                out.append((a["id"], b["id"], f"same switch {sorted(set(clash) | (sa & sb))}"))
    return out


def build(base, levels, name):
    """The base model with every level's rule_config overrides and wrappers applied in order."""
    model = copy.deepcopy(base)
    bad = conflicts(levels)
    if bad:
        raise ValueError(f"{name}: conflicting levels {bad}")
    rc = dict(model.get("rule_config") or {})
    for lv in levels:
        clash = set(lv["build"]["rule_config"]) & set(base.get("rule_config") or {})
        if clash:
            raise ValueError(f"{lv['id']}: {sorted(clash)} already in the base rule_config")
        rc.update(copy.deepcopy(lv["build"]["rule_config"]))
    model["rule_config"] = rc
    for lv in levels:
        for w in lv["build"]["wrappers"]:
            model = wrapper_fn(w["name"])(model, w.get("args", {}))
    model.setdefault("source", {})["geo5"] = {"base": "gt_u1b", "levels": [lv["id"] for lv in levels],
                                              "spec": "data/runs/b2rl/geo5/options.json"}
    return model


def behaviour_sha(model):
    """sha256 of everything but the free-text source: two models with the same sha behave the same."""
    body = {k: v for k, v in model.items() if k != "source"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()


def write_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(model) + "\n"
    if not path.is_file() or path.read_text(encoding="utf-8") != text:      # never rewrite a model in use
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    return path


_INDEX = None


def reuse_index():
    """behaviour sha -> earlier model names whose full-month records may stand in (options/ and search/ models)."""
    global _INDEX
    if _INDEX is None:
        _INDEX = {}
        for d in REUSE_MODEL_DIRS:
            for p in sorted(d.glob("model-*.json")):
                try:
                    sha = behaviour_sha(json.loads(p.read_text(encoding="utf-8")))
                except (OSError, ValueError):
                    continue
                _INDEX.setdefault(sha, []).append(p.stem.replace("model-", ""))
    return _INDEX
