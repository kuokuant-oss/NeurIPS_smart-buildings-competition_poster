"""Model-based planning layers on top of the rule (pipeline 5, 2026-11-04; docs/plan/option-set-20261104.md O505,
O544, O545). Imported lazily by policies/g36.py only when one of those options is on; numpy only, except the hourly
planner, which imports scipy.optimize.linprog (HiGHS) when it is built.

GroupModel   per AHU, a first-order model of the group's mean zone temperature learned online by recursive least
             squares: dT per 5-min step = a (T_out - T) + b q / 10 (+ c, held at 0), q = the heat (kW) the group's
             boxes delivered to their zones in the step that just ended (observed dampers, valves, supply air, static
             pressure and tank temperature, the D-R09 equations of zone.py). a = envelope / ventilation loss rate,
             b = response to heat.
DailyStart   O505: once a night per AHU, 3 h before the occupied start, the candidates {G36 lead - 1 h, G36 lead,
             + 1 h, + 2 h} x warm-up airflow {the rule's, 1.0 x the cooling maximum} are simulated with the model to the
             occupied start (warm-up at full airflow until the target is reached, then holding); the cheapest
             candidate (gas + 2.55 x electricity) that reaches the target within tol_k is chosen, else the warmest.
DailyLayer   O544: DailyStart, plus each evening per AHU: tonight's setback depth (0, 1 or 2 K below the eco heating
             edge, the deepest from which the model still recovers by the start with the largest candidate), and for
             tomorrow a static pressure cap (1.15 x the 95th percentile of today's occupied pressure commands, within
             [200 Pa, the cap in force]) and a heating airflow budget (1.2 x the 95th percentile of today's occupied
             heating airflow share above the minimum, within [0.5, 1]).
HourlyLP     O545: every `every_steps` steps a linear program over the next `horizon_h` hours, all AHUs at once (one
             thermal node per AHU), hourly heat and cooling (kW, within each AHU's capacity) and temperatures; cost =
             gas (heat / 0.9) + 2.55 x electricity (cooling / 3.5) + comfort_weight x the kelvin-hours outside the
             occupied targets at every hour boundary that starts or ends a scheduled occupied hour (+ 10 x below the
             freeze-off limit at the other boundaries) + a terminal value (the kelvins the last temperature is below
             what the AHU can still recover by the next occupied start). The hourly model is the exact 12-step
             discretisation of the group model (hourly_matrices). The node is the group's coldest zone on the heating
             side and its warmest zone on the cooling side, as the mean shifted by the current spread (heat_extra /
             cool_extra, policies/g36.py). Only the first hour's heating / cooling decision before arrival is used
             (warm-up / cool-down per AHU, policies/g36.py; never a zone setpoint in occupied hours). HiGHS with a
             time limit; any failure, timeout or untrained model leaves the rule alone for that hour.

2026-11-05 fixes (O545 screen +40%, diagnosis data/runs/b2rl/geo5/diag/o545/): the occupied comfort constraint was on
the temperature at the END of each occupied hour (06:00 itself unconstrained: the plan cancelled the rule's warm-up /
cool-down and heated in the first occupied hour); hourly forward Euler (12 a, 12 b: unstable for a > 1/12); the plan
targeted the group mean, not the worst zone; the plan's next-hour mean was forced onto every zone's setpoints in
occupied hours (policies/g36.py).
"""
import time

import numpy as np

from smart_building.g36 import zone
from smart_building.g36.modes import is_occupied, seconds_to_next_occupied

AIR_KJ = zone.AIR_DENSITY * zone.AIR_HEAT_CAPACITY / 1000.0     # kJ / (m3 K)
STEP_S = 300.0
GAS_EFFICIENCY, COP, ELECTRICITY_WEIGHT = 0.9, 3.5, 2.55        # as the cost-based SAT (G36Controller A12)
Q_SCALE = 10.0                                                  # model regressor q / 10 kW (conditioning)
SPREAD_MAX_K = 3.0                                              # O545: largest zone spread a plan allows for


def box_heat_kw(index, o, t_prev):
    """Heat (kW) each box delivered to its zone in the step that just ended, from the read-back actuators."""
    groups = index.zone_group
    return_temp = np.array([t_prev[s].mean() for s in index.group_slices])
    supply = zone.estimate_supply_air(return_temp, o.oat, o.recirculation, o.sat_setpoint, o.ahu_on)
    pressure = o.static_pressure[groups]
    demand = zone.box_flow_demand(o.damper, pressure)
    ratio = zone.supply_flow_ratio(demand, groups, index.n_ahu)
    discharge = zone.estimate_discharge(o.damper, o.valve, pressure, ratio[groups], supply[groups], o.water_temp,
                                        o.water_dp)
    on = np.asarray(o.ahu_on, dtype=bool)[groups]
    return np.where(on, AIR_KJ * demand * ratio[groups] * (discharge - t_prev), 0.0)


class GroupModel:
    """O505 / O544 / O545: per-AHU recursive least squares of dT = a (T_out - T) + b q / 10 + c per 5-min step,
    fitted on `window`-step increments (2026-11-05: 12 = one hour; 1 = the first version, per step, whose loss rate
    took up the fast air relaxation after heating pulses)."""

    def __init__(self, index, forget=0.998, min_updates=288, window=12):
        self.index = index
        g = index.n_ahu
        self.theta = np.zeros((g, 3))
        self.p = np.tile(np.eye(3) * 10.0, (g, 1, 1))
        self.forget, self.min_updates = float(forget), int(min_updates)
        self.window = max(1, int(window))
        self.updates = np.zeros(g, dtype=int)
        self.heat_seen = np.zeros(g, dtype=int)          # steps with q > 1 kW (b identifiable)
        self.last = None
        self.acc_x, self.acc_q, self.acc_n, self.acc_t0 = np.zeros(g), np.zeros(g), 0, None

    def mean_temp(self, temp):
        return np.array([float(np.mean(temp[s])) for s in self.index.group_slices])

    def observe(self, o, temp):
        t = np.asarray(temp, dtype=float)
        if self.last is not None:
            t_prev = self.last
            q = np.bincount(self.index.zone_group, weights=box_heat_kw(self.index, o, t_prev),
                            minlength=self.index.n_ahu)
            tg_prev, tg = self.mean_temp(t_prev), self.mean_temp(t)
            oat = np.broadcast_to(np.asarray(o.oat, dtype=float), (self.index.n_ahu,))
            # c is held at 0 (its regressor is 0): with a nearly constant outdoor-indoor difference the constant and
            # the loss rate are confounded (first fits on fairbanks_2 gave c ~ 0.24 K per step against a large a);
            # internal gains then show up in a lower a while people are in.
            if self.acc_n == 0:
                self.acc_t0 = tg_prev
            self.acc_x = self.acc_x + (oat - tg_prev)
            self.acc_q = self.acc_q + q / Q_SCALE
            self.acc_n += 1
            if self.acc_n >= self.window:                           # one update per window of steps
                x = np.stack([self.acc_x, self.acc_q, np.zeros_like(q)], axis=1)
                y = tg - self.acc_t0
                self.acc_x, self.acc_q, self.acc_n = np.zeros_like(self.acc_x), np.zeros_like(self.acc_q), 0
                lam = self.forget ** self.window
                px = np.einsum("gij,gj->gi", self.p, x)
                gain = px / (lam + np.einsum("gi,gi->g", x, px))[:, None]
                self.theta = self.theta + gain * (y - np.einsum("gi,gi->g", x, self.theta))[:, None]
                self.p = (self.p - np.einsum("gi,gj->gij", gain, px)) / lam
                self.p[:, 2, :] = 0.0                               # c held at 0: no covariance of its own
                self.p[:, :, 2] = 0.0
                trace = np.einsum("gii->g", self.p)
                self.p = self.p * np.minimum(1.0, 100.0 / np.maximum(trace, 1e-12))[:, None, None]
                bad = ~(np.all(np.isfinite(self.theta), axis=1) & np.all(np.isfinite(self.p), axis=(1, 2)))
                if bad.any():
                    self.theta[bad] = 0.0
                    self.p[bad] = np.eye(3) * 10.0
                    self.updates[bad] = 0
            self.updates += 1
            self.heat_seen += np.abs(q) > 1.0
        self.last = t.copy()

    def valid(self):
        """Groups whose model is usable: enough steps, heat seen, a physical sign (loss and heat response > 0)."""
        a, b = self.theta[:, 0], self.theta[:, 1]
        return (self.updates >= self.min_updates) & (self.heat_seen >= 24) & (a > 1e-5) & (a < 0.2) & (b > 1e-5)

    def step(self, g, t, oat, q):
        a, b, c = self.theta[g]
        return t + a * (oat - t) + b * q / Q_SCALE + c

    def hold_heat(self, g, t, oat):
        """Heat (kW) that keeps group g at t."""
        a, b, c = self.theta[g]
        return max(0.0, -(a * (oat - t) + c) * Q_SCALE / b) if b > 0 else 0.0

    def hold_cool(self, g, t, oat):
        """Cooling (kW) that keeps group g at t."""
        a, b, c = self.theta[g]
        return max(0.0, (a * (oat - t) + c) * Q_SCALE / b) if b > 0 else 0.0

    def delivered_kwh(self, g, t0, oat, seconds, target, q_max, cooling=False):
        """O545 check: model estimate of the heat (cooling) the rule's warm-up (cool-down) delivers from now for
        `seconds`: full capacity q_max until the group node reaches target, then what holds it there."""
        t, kwh = float(t0), 0.0
        for _ in range(int(round(seconds / STEP_S))):
            if cooling:
                q = q_max if t > target else min(q_max, self.hold_cool(g, target, oat))
                t = self.step(g, t, oat, -q)
            else:
                q = q_max if t < target else min(q_max, self.hold_heat(g, target, oat))
                t = self.step(g, t, oat, q)
            kwh += q * STEP_S / 3600.0
        return kwh


def hourly_matrices(theta, steps=12):
    """O545: exact discretisation over `steps` 5-min steps (one hour) of dT = a (T_out - T) + b q / Q_SCALE + c with
    T_out and q held: T(k+1) = A T(k) + (1 - A) T_out + B q + C, A = (1 - a)^steps, B = b / Q_SCALE x S, C = c x S,
    S = sum_j (1 - a)^j = (1 - A) / a (steps when a = 0). Forward Euler (12 a, 12 b) overstates the response and is
    unstable for a > 1 / 12."""
    a, b, c = (np.asarray(theta[:, i], dtype=float) for i in range(3))
    a_c = np.clip(a, 0.0, 0.999)
    big_a = (1.0 - a_c) ** steps
    s = np.where(a_c > 1e-9, (1.0 - big_a) / np.maximum(a_c, 1e-9), float(steps))
    return big_a, b / Q_SCALE * s, c * s


def heat_capacity_kw(index, flow_fraction, t_zone, discharge_k=zone.MAX_DISCHARGE_TEMPERATURE):
    """Per AHU, the box heat (kW) at a heating airflow fraction of every zone, discharging at the limit."""
    return AIR_KJ * flow_fraction * index.group_sizes * zone.VAV_MAX_FLOW * np.maximum(discharge_k - t_zone, 0.0)


class DailyStart:
    """O505: once a night per AHU, choose (warm-up lead s, warm-up airflow fraction or None = the rule's)."""

    OFFSETS_H = (-1.0, 0.0, 1.0, 2.0)
    COMFORT_KWH = 3.0     # gas kWh per zone-kelvin-hour below the band at the start (calibrated 2026-11-05)

    def __init__(self, index, flows=(None, 1.0), tol_k=0.2, max_lead_s=4 * 3600.0, decision_extra_s=7200.0):
        self.index = index
        self.flows = tuple(flows)
        self.tol_k, self.max_lead_s, self.decision_extra_s = float(tol_k), float(max_lead_s), float(decision_extra_s)
        self.choice = None            # (lead s per group, flow per group (nan = rule), valid per group)
        self.decided_for = None       # the occupied start (absolute minutes) the choice is for
        self.decisions = 0
        self.last_candidates = None   # diagnostics

    def candidates(self, rule_lead_s):
        return [float(np.clip(rule_lead_s + 3600.0 * h, 0.0, self.max_lead_s)) for h in self.OFFSETS_H]

    def simulate(self, model, g, t0, oat, to_start_s, lead_s, flow, rule_flow, target, floor_k=None,
                 floor_within_s=float("inf")):
        """(temperature at the occupied start, cost) of one candidate: float, then full heat until the target.
        floor_k: while floating, the rule's setback heating holds the node there once the start is at most
        floor_within_s away (2026-11-05; None = a free float, the first version)."""
        steps = int(round(to_start_s / STEP_S))
        on_from = steps - int(round(lead_s / STEP_S))
        frac = float(rule_flow if flow is None else flow)
        size = float(self.index.group_sizes[g]) * zone.VAV_MAX_FLOW          # m3/s at full design flow
        fan_kw = 300.0 * frac * size / 0.9 / 1000.0                           # about 300 Pa, 90% efficient
        b = float(model.theta[g, 1])
        t, cost = float(t0), 0.0
        for k in range(steps):
            q = 0.0
            if k >= on_from:
                full = AIR_KJ * frac * size * max(zone.MAX_DISCHARGE_TEMPERATURE - t, 0.0)
                q = full if t < target else min(full, model.hold_heat(g, target, oat))
                cost += (q / GAS_EFFICIENCY + ELECTRICITY_WEIGHT * fan_kw) * STEP_S / 3600.0
            elif floor_k is not None and to_start_s - k * STEP_S <= floor_within_s and b > 0.0:
                free = model.step(g, t, oat, 0.0)
                if free < floor_k:                                   # the rule's setback heating
                    full = AIR_KJ * frac * size * max(zone.MAX_DISCHARGE_TEMPERATURE - t, 0.0)
                    q = min(full, (floor_k - free) * Q_SCALE / b)
                    cost += (q / GAS_EFFICIENCY + ELECTRICITY_WEIGHT * fan_kw) * STEP_S / 3600.0
            t = model.step(g, t, oat, q)
        return t, cost

    def decide(self, model, temp, oat, to_start_s, target, rule_lead_s, rule_flow, spread_k=None, floor_k=None,
               floor_within_s=float("inf")):
        """Choose for every AHU; groups without a valid model keep the rule (valid False). spread_k: the group's
        current spread below its mean, so the candidates are judged on its coldest zone; floor_k / floor_within_s:
        the rule's setback heating during the float (simulate)."""
        g_n = self.index.n_ahu
        valid = model.valid()
        t_now = np.asarray(temp, dtype=float)
        tg = model.mean_temp(t_now)
        oat_g = np.broadcast_to(np.asarray(oat, dtype=float), (g_n,))
        lead = np.zeros(g_n)
        flow = np.full(g_n, np.nan)
        table = []
        if spread_k is not None:                       # 2026-11-05: energy + priced comfort, coldest-zone guard
            spread = np.asarray(spread_k, dtype=float)
            for g in range(g_n):
                if not valid[g]:
                    continue
                offs = t_now[self.index.group_slices[g]] - tg[g]
                s_g = max(float(spread[g]), 1e-6)
                share = np.clip((offs + s_g) / s_g, 0.0, 1.0)          # 0 = the coldest zone, 1 = the mean or above
                size = float(self.index.group_sizes[g]) * zone.VAV_MAX_FLOW
                a, b, _ = model.theta[g]
                rule_s = float(rule_lead_s[g])
                best = None
                for cand_lead in self.candidates(rule_s):
                    for f in self.flows:
                        t_mean, cost = self.simulate(model, g, tg[g], oat_g[g], to_start_s, cand_lead, f, rule_flow,
                                                     target, floor_k, floor_within_s)
                        t_cold, _ = self.simulate(model, g, tg[g] - spread[g], oat_g[g], to_start_s, cand_lead, f,
                                                  rule_flow, target, floor_k, floor_within_s)
                        frac = float(rule_flow if f is None else f)
                        full = AIR_KJ * frac * size * max(zone.MAX_DISCHARGE_TEMPERATURE - target, 0.0)
                        rise = max(12.0 * (b * full / Q_SCALE - a * max(target - oat_g[g], 0.0)), 0.5)   # K / h
                        short = np.maximum(target - (t_cold + share * (t_mean - t_cold)), 0.0)
                        total = cost + self.COMFORT_KWH * float(np.sum(short ** 2)) / (2.0 * rise)
                        later = cand_lead < rule_s - 1.0
                        ok = (not later) or t_cold >= target - self.tol_k
                        table.append((g, cand_lead, f, round(t_mean, 3), round(t_cold, 3), round(cost, 3),
                                      round(total, 3), ok))
                        if ok and (best is None or total < best[0]):
                            best = (total, cand_lead, f)
                if best is None:                       # nothing admissible: the rule's own lead and airflow
                    best = (0.0, rule_s, None)
                lead[g] = best[1]
                flow[g] = np.nan if best[2] is None else best[2]
            self.choice = (lead, flow, valid)
            self.decisions += 1
            self.last_candidates = table
            return self.choice
        for g in range(g_n):
            if not valid[g]:
                continue
            best = None
            for cand_lead in self.candidates(float(rule_lead_s[g])):
                for f in self.flows:
                    t_end, cost = self.simulate(model, g, tg[g], oat_g[g], to_start_s, cand_lead, f, rule_flow,
                                                target, floor_k, floor_within_s)
                    ok = t_end >= target - self.tol_k
                    key = (0, cost) if ok else (1, -t_end)
                    table.append((g, cand_lead, f, round(t_end, 3), round(cost, 3), ok))
                    if best is None or key < best[0]:
                        best = (key, cand_lead, f)
            lead[g] = best[1]
            flow[g] = np.nan if best[2] is None else best[2]
        self.choice = (lead, flow, valid)
        self.decisions += 1
        self.last_candidates = table
        return self.choice


class DailyLayer:
    """O544: tonight's setback depth, tomorrow's static pressure cap and heating airflow budget per AHU, from the
    model and today's occupied operation (the start choice is DailyStart's)."""

    DEPTHS_K = (2.0, 1.0, 0.0)
    PARTS = ("all", "setback", "sp_cap", "budget")

    def __init__(self, index, parts="all"):
        if parts not in self.PARTS:
            raise ValueError(f"unknown O544 part {parts!r}")
        self.index, self.parts = index, parts
        g = index.n_ahu
        self.depth = np.zeros(g)
        self.sp_cap = np.full(g, np.inf)
        self.budget = np.ones(g)
        self.sp_today = [[] for _ in range(g)]
        self.share_today = [[] for _ in range(g)]
        self.evenings = 0

    def record(self, occupied_running, pressure, heat_share):
        for g in np.flatnonzero(occupied_running):
            self.sp_today[g].append(float(pressure[g]))
            if np.isfinite(heat_share[g]):
                self.share_today[g].append(float(heat_share[g]))

    def evening(self, model, starter, temp, oat, eco_heat, target, max_lead_s, rule_flow, sp_top, flow=1.0):
        """At the end of the occupied day: tonight's depth, tomorrow's cap and budget (the parts in force; flow:
        the warm-up airflow of the depth check, None = the rule's)."""
        valid = model.valid()
        oat_g = np.broadcast_to(np.asarray(oat, dtype=float), (self.index.n_ahu,))
        for g in range(self.index.n_ahu):
            if self.sp_today[g] and self.parts in ("all", "sp_cap"):
                self.sp_cap[g] = float(np.clip(1.15 * np.percentile(self.sp_today[g], 95), 200.0, sp_top))
            if self.share_today[g] and self.parts in ("all", "budget"):
                self.budget[g] = float(np.clip(1.2 * np.percentile(self.share_today[g], 95), 0.5, 1.0))
            self.depth[g] = 0.0
            if valid[g] and self.parts in ("all", "setback"):
                for d in self.DEPTHS_K:        # deepest first: recoverable from eco - d with the largest lead
                    t_end, _ = starter.simulate(model, g, eco_heat - d, oat_g[g], max_lead_s, max_lead_s, flow,
                                                rule_flow, target)
                    if t_end >= target - starter.tol_k:
                        self.depth[g] = d
                        break
            self.sp_today[g], self.share_today[g] = [], []
        self.evenings += 1


class HourlyLP:
    """O545: rolling linear program over the next horizon_h hours (one node per AHU); see the module docstring."""

    def __init__(self, index, horizon_h=12, every_steps=12, time_limit_s=0.5, comfort_weight=100.0):
        from scipy.optimize import linprog   # noqa: F401 - fail at construction (the option then stays off)
        from scipy import sparse            # noqa: F401
        self.index = index
        self.horizon, self.every = int(horizon_h), int(every_steps)
        self.time_limit, self.comfort_weight = float(time_limit_s), float(comfort_weight)
        self.plan = None                  # dict of per-AHU arrays for the hour in force
        self.solve_ms, self.failures, self.solves = [], 0, 0
        self.steps = 0

    def due(self):
        due = self.steps % self.every == 0
        self.steps += 1
        return due

    def solve(self, model, t_group, oat_forecast, occupied_h, heat_target, cool_target, freeze_k, q_heat_max,
              q_cool_max, hours_after, heat_extra=None, cool_extra=None):
        """Plan; returns the first hour (heat kW, cool kW, mean temperature after one hour) per AHU and the heat /
        cooling (kWh) planned before the next occupied hour, or None. heat_extra / cool_extra (K per AHU): the group's
        current spread below / above its mean, so that the comfort and freeze limits hold for its coldest / warmest
        zone. Boundary j (j hours from now) carries the occupied limits when the hour before or after it is a
        scheduled occupied hour (so the temperature at the occupied start is constrained). The plan keeps every
        hour (heat_h, cool_h, t_h: AHU x hour) and `before`, the hours before the next occupied hour."""
        from scipy import sparse
        from scipy.optimize import linprog
        t0 = time.perf_counter()
        g_n, h_n = self.index.n_ahu, self.horizon
        valid = model.valid()
        if not valid.any():
            self.plan = None
            return None
        big_a, big_b, big_c = hourly_matrices(model.theta)
        lo_x = np.zeros(g_n) if heat_extra is None else np.asarray(heat_extra, dtype=float)
        hi_x = np.zeros(g_n) if cool_extra is None else np.asarray(cool_extra, dtype=float)
        occ = np.asarray(occupied_h, dtype=bool)
        boundary = np.array([bool(occ[k] or (occ[k + 1] if k + 1 < h_n else hours_after <= 0.0))
                             for k in range(h_n)])
        before = int(np.argmax(occ)) if occ.any() else h_n                # hours before the next occupied hour
        nv = 5                                                     # Qh, Qc, T(k+1), s_lo, s_hi per (g, k)
        n = g_n * h_n * nv + g_n                                   # + terminal slack per g

        def idx(g, k, j):
            return (g * h_n + k) * nv + j

        cost = np.zeros(n)
        lo, hi = np.zeros(n), np.full(n, np.inf)
        rows, cols, vals, rhs_eq = [], [], [], []
        urows, ucols, uvals, rhs_ub = [], [], [], []
        eq = ub = 0
        for g in range(g_n):
            a_g, b_g, c_g = float(big_a[g]), float(big_b[g]), float(big_c[g])
            for k in range(h_n):
                qh, qc, tk1, sl, sh = (idx(g, k, j) for j in range(nv))
                cost[qh] = 1.0 / GAS_EFFICIENCY
                cost[qc] = ELECTRICITY_WEIGHT / COP
                hi[qh], hi[qc] = q_heat_max[g] if valid[g] else 0.0, q_cool_max[g] if valid[g] else 0.0
                lo[tk1], hi[tk1] = -np.inf, np.inf
                w = self.comfort_weight if boundary[k] else 10.0 * self.comfort_weight
                cost[sl] = w
                cost[sh] = self.comfort_weight if boundary[k] else 0.0
                # T(k+1) - A T(k) - B Qh + B Qc = (1 - A) Tout(k) + C
                rows += [eq, eq, eq]
                cols += [tk1, qh, qc]
                vals += [1.0, -b_g, b_g]
                if k == 0:
                    rhs_eq.append(a_g * t_group[g] + (1.0 - a_g) * oat_forecast[k] + c_g)
                else:
                    rows.append(eq)
                    cols.append(idx(g, k - 1, 2))
                    vals.append(-a_g)
                    rhs_eq.append((1.0 - a_g) * oat_forecast[k] + c_g)
                eq += 1
                low_k = (heat_target if boundary[k] else freeze_k) + lo_x[g]
                urows += [ub, ub]                                  # -T(k+1) - s_lo <= -low
                ucols += [tk1, sl]
                uvals += [-1.0, -1.0]
                rhs_ub.append(-low_k)
                ub += 1
                if boundary[k]:                                    # T(k+1) - s_hi <= cool target
                    urows += [ub, ub]
                    ucols += [tk1, sh]
                    uvals += [1.0, -1.0]
                    rhs_ub.append(cool_target - hi_x[g])
                    ub += 1
            term = g_n * h_n * nv + g
            cost[term] = self.comfort_weight
            goal = heat_target + lo_x[g]
            recover = max(0.0, b_g * q_heat_max[g] - (1.0 - a_g) * max(goal - oat_forecast[-1], 0.0) + c_g)
            need = goal - recover * max(hours_after, 0.0)
            urows += [ub, ub]                                      # -T(H) - s_term <= -need
            ucols += [idx(g, h_n - 1, 2), term]
            uvals += [-1.0, -1.0]
            rhs_ub.append(-need)
            ub += 1
        a_eq = sparse.csr_matrix((vals, (rows, cols)), shape=(eq, n))
        a_ub = sparse.csr_matrix((uvals, (urows, ucols)), shape=(ub, n))
        try:
            result = linprog(cost, A_ub=a_ub, b_ub=np.array(rhs_ub), A_eq=a_eq, b_eq=np.array(rhs_eq),
                             bounds=np.stack([lo, hi], axis=1), method="highs",
                             options={"time_limit": self.time_limit, "presolve": True})
        except Exception:  # noqa: BLE001 - the rule stays in charge this hour
            result = None
        self.solve_ms.append(1000.0 * (time.perf_counter() - t0))
        self.solves += 1
        if result is None or result.status != 0 or result.x is None:
            self.failures += 1
            self.plan = None
            return None
        x = result.x
        xs = x[:g_n * h_n * nv].reshape(g_n, h_n, nv)
        plan = {"heat_kw": xs[:, 0, 0].copy(), "cool_kw": xs[:, 0, 1].copy(), "t_next": xs[:, 0, 2].copy(),
                "heat_h": xs[:, :, 0].copy(), "cool_h": xs[:, :, 1].copy(), "t_h": xs[:, :, 2].copy(),
                "before": before, "heat_extra": lo_x.copy(), "cool_extra": hi_x.copy(),
                "valid": valid.copy(), "heat_max": np.asarray(q_heat_max, dtype=float),
                "cool_max": np.asarray(q_cool_max, dtype=float)}
        self.plan = plan
        return plan


def hourly_schedule(minute_of_day, weekday, morning_hour, evening_hour, horizon_h):
    """Scheduled occupancy of each of the next horizon_h hours (the hour's middle) and the hours from the horizon's end
    to the next occupied start (0 when the horizon ends inside the schedule)."""
    occupied = []
    for k in range(horizon_h):
        m = int(minute_of_day) + 60 * k + 30
        day, minute = divmod(m, 1440)
        occupied.append(is_occupied(minute, (int(weekday) + day) % 7, morning_hour, evening_hour))
    day, minute = divmod(int(minute_of_day) + 60 * horizon_h, 1440)
    after = seconds_to_next_occupied(minute, (int(weekday) + day) % 7, morning_hour, evening_hour) / 3600.0
    return np.array(occupied, dtype=bool), after
