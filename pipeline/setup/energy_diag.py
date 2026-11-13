"""Where does the energy and carbon part of the cost go? (2026-10-30; the official S4 first 30 gate showed hot, humid
scenarios — singapore, the atlanta building 31750 — gaining 0-8 % against the original rule while the rest gain 20-70 %.)

One whole episode per policy and scenario on the fast simulator (rom; SB_FASTSIM=rom, the official equipment and reward
code) with SetpointEnergyCarbonRegretFunction.compute_reward wrapped: per step, every device's rate from the RewardInfo
is priced and weighted exactly as the reward does (energy weight x cost / the step's max cost, carbon weight x
emission / the step's max emission), so the components add up to the episode's energy and carbon parts (checked).
Components: blower (fans), ac_cool / ac_heat (air conditioning rate below / above 0), pump (boiler and heat-pump
pumps), gas (boiler), hp_heat; each split by occupied (total occupancy > 0) and unoccupied steps, plus comfort the
same way. Writes data/runs/b2rl/energy-diag/<policy>__<spec>.json; prints one line per episode.

python3 -B pipeline/launch/launch.py pipeline/setup/energy_diag.py <workers> <model.json|rule>,... <spec> [<spec> ...]"""
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages"))
OUT = ROOT / "data/runs/b2rl/energy-diag"
PARTS = ("blower", "ac_cool", "ac_heat", "pump", "gas", "hp_heat")


def episode(job):
    policy, spec = job
    os.environ["SB_FASTSIM"] = "rom"
    from smart_building.learned.b2rl_env import setup_simulator, spec_dir
    setup_simulator(ROOT)
    from smart_buildings.smart_control.reward import setpoint_energy_carbon_regret as regret
    from smart_buildings.smart_control.utils import conversion_utils
    from smart_building.evaluation.gateway_episode import run_gateway_episode
    from smart_building.policies.b2rl import b2rl_policy_class
    from smart_building.policies.g36 import G36PolicyB2R2

    acc = {f"{k}_{when}": 0.0 for k in ("comfort", *(f"{p}_{x}" for p in PARTS for x in ("energy", "carbon")))
           for when in ("occ", "unocc")}
    acc["steps_occ"] = acc["steps_unocc"] = 0
    seen = set()
    original = regret.SetpointEnergyCarbonRegretFunction.compute_reward

    def wrapped(self, info):
        response = original(self, info)
        stamp = (info.start_timestamp.seconds, info.start_timestamp.nanos)
        if stamp in seen:
            return response
        seen.add(stamp)
        start = conversion_utils.proto_to_pandas_timestamp(info.start_timestamp)
        end = conversion_utils.proto_to_pandas_timestamp(info.end_timestamp)
        rates = dict.fromkeys(PARTS, 0.0)
        for a in info.air_handler_reward_infos.values():
            rates["blower"] += a.blower_electrical_energy_rate
            ac = a.air_conditioning_electrical_energy_rate
            rates["ac_cool" if ac < 0 else "ac_heat"] += abs(ac)
        for b in info.boiler_reward_infos.values():
            rates["pump"] += b.pump_electrical_energy_rate
            rates["gas"] += b.natural_gas_heating_energy_rate
        for h in info.heat_pump_reward_infos.values():
            rates["pump"] += abs(h.pump_electrical_energy_rate)
            rates["hp_heat"] += abs(h.electricity_heating_energy_rate)
        elec, gas = self._electricity_energy_cost, self._natural_gas_energy_cost
        max_cost = elec.cost(start, end, self._max_electricity_rate) + gas.cost(start, end, self._max_natural_gas_rate)
        max_carbon = (elec.carbon(start, end, self._max_electricity_rate)
                      + gas.carbon(start, end, self._max_natural_gas_rate))
        when = "occ" if response.total_occupancy > 0.0 else "unocc"
        acc[f"steps_{when}"] += 1
        acc[f"comfort_{when}"] += self._productivity_weight * response.normalized_productivity_regret
        for p, rate in rates.items():
            model = gas if p == "gas" else elec
            if max_cost > 0.0:
                acc[f"{p}_energy_{when}"] -= self._energy_cost_weight * model.cost(start, end, rate) / max_cost
            if max_carbon > 0.0:
                acc[f"{p}_carbon_{when}"] -= self._carbon_emission_weight * model.carbon(start, end, rate) / max_carbon
        return response

    regret.SetpointEnergyCarbonRegretFunction.compute_reward = wrapped
    base = G36PolicyB2R2 if policy == "rule" else b2rl_policy_class(ROOT / policy)
    steps = int(os.environ.get("ENERGY_MAX_STEPS", "0")) or None                    # smoke runs only
    row = run_gateway_episode(spec_dir(spec), base, policy_key="energy-diag", label=spec, max_steps=steps)
    check = {"comfort": row["comfort_part"] - acc["comfort_occ"] - acc["comfort_unocc"],
             **{k: row[f"{k}_part"] - sum(acc[f"{p}_{k}_{w}"] for p in PARTS for w in ("occ", "unocc"))
                for k in ("energy", "carbon")}}
    return policy, spec, {"cost": -row["mean_reward"], "steps": row["steps"], "comfort_part": row["comfort_part"],
                          "energy_part": row["energy_part"], "carbon_part": row["carbon_part"], "check": check, **acc}


def main():
    workers, policies, specs = int(sys.argv[1]), sys.argv[2].split(","), sys.argv[3:]
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(p, s) for p in policies for s in specs]
    with mp.get_context("spawn").Pool(min(workers, len(jobs)), maxtasksperchild=1) as pool:
        for policy, spec, r in pool.imap_unordered(episode, jobs):
            name = "rule" if policy == "rule" else Path(policy).stem.replace("model-", "")
            (OUT / f"{name}__{spec.replace('/', '__')}.json").write_text(json.dumps(r, indent=1) + "\n",
                                                                          encoding="utf-8")
            parts = {p: r[f"{p}_energy_occ"] + r[f"{p}_energy_unocc"] + r[f"{p}_carbon_occ"] + r[f"{p}_carbon_unocc"]
                     for p in PARTS}
            print(f"{name:14s} {spec:34s} cost {r['cost']:.5f} comfort {r['comfort_part']:7.1f} "
                  + " ".join(f"{p} {v:7.1f}" for p, v in parts.items() if v)
                  + f"  check {max(abs(v) for v in r['check'].values()):.1e}", flush=True)


if __name__ == "__main__":
    main()
