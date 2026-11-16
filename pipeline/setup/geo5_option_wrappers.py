"""data/runs/b2rl/geo5/options.json as "geo5_option_wrappers:<function>"). Each takes the model dict and its args and
returns the changed model; geo5_options.build applies them after the rule_config overrides.

rule_set             rule_config keys that the base gt_u1b already sets (geo5_options.build refuses a level's
                     rule_config key that is in the base, so these go through a wrapper): {key: value, ...}
room_option_replace  every room-policy option whose `key` equals `from` gets `to` ({"key", "from", "to"}); e.g. the
                     room tree's 0.4 K cooling margin set back to 0 (O527 level 1)
room_vent_off        the per-room 62.1 minimum removed from every room-policy option (O551, as ub_make_models vent0)
slow_band            one band of the slow layer's outside-air schedule set to a value: {"parameter", "band":
                     cold | mild | hot, "value"} (the schedule's bands: 24 h mean below 0 C, 5-20 C, above 25 C; linear
                     in between). The hourly slow layer overwrites rule_config values of its own parameters
                     (occupied_margin_k, t_opt_max_s, ...), so a level that changes one must change the schedule."""
import copy

BANDS = ("cold", "mild", "hot")
EDGES = (0.0, 0.25, 1.0, 1.25)
VENT_KEYS = ("zone_min_ra_lps_m2", "zone_min_rp_lps_m2", "zone_min_oa_divide")


def rule_set(model, args):
    model["rule_config"] = {**(model.get("rule_config") or {}), **copy.deepcopy(args)}
    return model


def room_option_replace(model, args):
    key, old, new = args["key"], args["from"], args["to"]
    for option in model["room_policy"]["options"]:
        if key in option and option[key] == old:
            option[key] = new
    return model


def room_option_relax(model, args):
    """Option audit: as room_option_replace, and the
    options changed also get args["mark"] = from - to (e.g. room_cool_relax_k: how far the target moved toward the
    edge, which the A12 comfort guard then keeps for its near test)."""
    key, old, new, mark = args["key"], args["from"], args["to"], args["mark"]
    for option in model["room_policy"]["options"]:
        if key in option and option[key] == old:
            option[key] = new
            option[mark] = round(float(old) - float(new), 9)
    return model


def room_vent_off(model, args):
    for option in model["room_policy"]["options"]:
        for key in VENT_KEYS:
            option.pop(key, None)
    return model


def _default(model, name):
    """The slow-layer parameter's default d of the two-sided map: the rule's own value (G36Config + the B2R2
    combination + the model's rule_config; the schedule models have no anchor)."""
    from smart_building.policies.g36 import C3_COMBINATIONS, G36PolicyB2R2, c3_configured_config
    cfg = c3_configured_config({**C3_COMBINATIONS[G36PolicyB2R2.combination], **(model.get("rule_config") or {})})
    if model.get("anchor") and name in model["anchor"]:
        return float(model["anchor"][name])
    return float(getattr(cfg, name))


def _row_value(weight, bias, x):
    h = [max(x - e, 0.0) for e in EDGES]
    return bias + sum(w * v for w, v in zip(weight, h))


def slow_band(model, args):
    """Set one band of one parameter. The schedule model is the two-layer net of the outside-air schedules (first
    layer: the four hinge units on outside_air_24h); the parameter's output row is rewritten so that its other bands
    keep their values."""
    from smart_building.policies.g36 import SLOW_PARAMETER_RANGES
    name, band, value = args["parameter"], args["band"], float(args["value"])
    if band not in BANDS:
        raise ValueError(f"unknown band {band!r}")
    if model.get("layout") not in ("v2", "v3"):
        raise ValueError("slow_band: the two-sided parameter map of layouts v2 / v3 only")
    params = list(model["parameters"])
    row = params.index(name)
    first, second = model["layers"]
    feature = list(model["features"]).index("outside_air_24h")
    for unit, edge in zip(first["weight"], EDGES):
        if [i for i, w in enumerate(unit) if w != 0.0] != [feature] or abs(unit[feature] - 1.0) > 1e-12:
            raise ValueError("slow_band: first layer is not the outside-air hinge schedule")
    if [round(-b, 9) for b in first["bias"]] != list(EDGES):
        raise ValueError("slow_band: first layer biases are not the schedule edges")
    if model.get("obs_mean") and any(model["obs_mean"]) or any(v != 1.0 for v in model.get("obs_var", [])) \
            or float(model.get("obs_epsilon", 0.0)) != 0.0:
        raise ValueError("slow_band: the features are normalised")
    weight, bias = list(second["weight"][row]), float(second["bias"][row])
    outs = {"cold": _row_value(weight, bias, -1.0), "mild": _row_value(weight, bias, 0.5),
            "hot": _row_value(weight, bias, 2.0)}
    low, high = (float(v) for v in SLOW_PARAMETER_RANGES[name])
    d = _default(model, name)
    if not low <= value <= high:
        raise ValueError(f"{name} = {value} outside [{low}, {high}]")
    a = (value - d) / (high - d) if value >= d else (value - d) / (d - low)
    outs[band] = a
    c, m, h = outs["cold"], outs["mild"], outs["hot"]
    second["weight"][row] = [(m - c) / 0.25, -(m - c) / 0.25, (h - m) / 0.25, -(h - m) / 0.25]
    second["bias"][row] = c
    return model
