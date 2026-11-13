"""Thermal cores of the fast simulator (docs/plan/fastsim.md).

The official step (smart_control/simulator/finite_difference_solver.py) is backward Euler on the control-volume
field T plus one interior-mass node per occupied CV, with the VAV air streams coupled implicitly:

    (E/dt + K) x' = (E/dt) x + B u + P q,        q = G (T_supply - S T'),

x = (T, T_mass), B u the zone powers in input_q and the ambient forcing, P the diffuser shares, S the VAV sensing
weights. The official code eliminates the mass nodes into the diagonal, factorizes the matrix once per episode
and spends two triangular solves per step (the second one folds the air side back in).

ExactCore keeps that discretisation and the full field; it pays one triangular solve per step and gets the
air-side correction from the precomputed response W = A^-1 P (a dense n x k matrix product instead of the second
solve). RomCore replaces the field with a Galerkin reduced model on a rational Krylov basis of the same
discretisation (rom.py): the dynamics become diagonal in the reduced modes, so a step costs O(r k).

Both read and write zone-level quantities only through this module: zone powers (what apply_thermal_power_zone
assigns), zone means (get_zone_average_temps), sensed means (the VAV thermostats) and the field mean.
"""
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

# right-hand sides per triangular solve when building W = A^-1 P (as the official _RESPONSE_BLOCK)
_BLOCK = 16


class ThermalData:
    """Geometry, materials and couplings of one simulator, read once from the official solver and building."""

    def __init__(self, simulator):
        from smart_buildings.smart_control.simulator import constants
        solver, building = simulator._solver, simulator.building
        self.simulator = simulator
        self.solver = solver
        self.shape = tuple(building.temp.shape)
        self.n_cells = int(building.temp.size)
        self.solved = np.asarray(solver._solved)
        self.n = len(self.solved)
        self.dt = float(simulator.time_step_sec)
        self.capacitance_rate = np.asarray(solver._capacitance_rate, dtype=np.float64)
        self.mass_coupling = np.asarray(solver._mass_coupling, dtype=np.float64)
        self.mass_retained = np.asarray(solver._mass_retained, dtype=np.float64)
        self.mass_gained = np.asarray(solver._mass_gained, dtype=np.float64)
        self.dirichlet = np.asarray(solver._dirichlet_conductance, dtype=np.float64)
        self.conduction = solver._conduction
        self.applied = scipy.sparse.csc_matrix(solver._applied)
        self.sensed = scipy.sparse.csr_matrix(solver._sensed)
        self.n_vav = self.applied.shape[1]
        self.zones = [z for z in building.room_dict if not constants.is_non_physical_space(z)]
        self.zone_index = {z: i for i, z in enumerate(self.zones)}
        position = np.full(self.n_cells, -1, dtype=np.int64)
        position[self.solved] = np.arange(self.n)
        zone_of = np.full(self.n, len(self.zones), dtype=np.int64)   # len(zones) = no zone (a zero slot)
        sizes = np.zeros(len(self.zones), dtype=np.int64)
        rows, cols, vals = [], [], []
        for i, zone in enumerate(self.zones):
            r, c = np.asarray(building.room_dict[zone], dtype=np.int64).reshape(-1, 2).T
            p = position[r * self.shape[1] + c]
            if (p < 0).any():
                raise ValueError(f"zone {zone} has cells pinned to ambient")
            zone_of[p] = i
            sizes[i] = len(p)
            rows.append(np.full(len(p), i))
            cols.append(p)
            vals.append(np.full(len(p), 1.0 / len(p)))
        self.zone_of = zone_of
        self.zone_sizes = sizes
        # zone means over solved positions, as the mean get_zone_temp_stats takes over the zone's CVs
        self.zone_mean = scipy.sparse.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(self.zones), self.n))
        self.mass = self.mass_coupling > 0.0
        self.n_exterior = self.n_cells - self.n
        # AHUs without zones read the whole-field mean for recirculation (simulator_flexible_floor_plan.py)
        self.initial_temp = np.asarray(building.temp, dtype=np.float64).copy()

    def exposure(self, h):
        """W/K from each solved CV to ambient at convection coefficient h (the official cached computation)."""
        return np.asarray(self.solver.ambient_conductance(h), dtype=np.float64)

    def matrix(self, h):
        """The official timestep matrix (E/dt + K with the mass nodes eliminated) at convection coefficient h."""
        return (self.conduction + scipy.sparse.diags(self.exposure(h))).tocsc()

    def factorize(self, h):
        """SuperLU of the timestep matrix: symmetric mode without pivoting (the matrix is SPD and diagonally
        dominant), the official MMD ordering; measured about a third faster per solve than the official pivoted
        factorization with the same fill."""
        return scipy.sparse.linalg.splu(self.matrix(h), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                                        options={"SymmetricMode": True})

    def expand_zone_power(self, zone_power):
        """input_q over solved positions from zone powers: power / n_zone on each CV of the zone, 0 elsewhere."""
        rate = np.zeros(len(self.zones) + 1)
        rate[:-1] = zone_power / self.zone_sizes
        return rate[self.zone_of]


class ExactCore:
    """The official discretisation on the full field with one triangular solve per step.

    State: the field over solved CVs and the mass-node temperatures (as the official solver keeps them). The
    simulator's building.temp grid is updated every step so the official convection shuffle can act on it.
    """

    kind = "exact"

    def __init__(self, data, response_dtype=np.float64):
        self.data = data
        self.response_dtype = response_dtype
        self.h = None
        self.factor = self.response = self.sensed_response = self.exposure = None
        self.zone_power = np.zeros(len(data.zones))
        self.temp = None
        self.mass_temp = np.zeros(data.n)
        self._means = None
        self.steps = 0

    def __getstate__(self):
        """Snapshots (sim/snapshot.py) cannot pickle a SuperLU factorization; drop it with the response it
        produced, so a restored core refactorizes the same matrix on its next step (bit for bit)."""
        state = dict(self.__dict__)
        state.update(h=None, factor=None, response=None, sensed_response=None, reward_cache=None)
        return state

    def prepare(self, h):
        if self.h == h and self.factor is not None:
            return
        d = self.data
        self.factor = d.factorize(h)
        self.exposure = d.exposure(h)
        response = np.empty((d.n, d.n_vav), dtype=np.float64)
        for start in range(0, d.n_vav, _BLOCK):
            response[:, start:start + _BLOCK] = self.factor.solve(d.applied[:, start:start + _BLOCK].toarray())
        self.sensed_response = d.sensed @ response
        self.response = np.asfortranarray(response.astype(self.response_dtype))
        self.h = h

    def reset(self, grid):
        """The building was reset: grid is its new temperature field; zone powers back to zero (input_q)."""
        d = self.data
        self.temp = np.asarray(grid, dtype=np.float64).reshape(-1)[d.solved].copy()
        self.mass_temp = np.where(d.mass, self.temp, 0.0)
        self.zone_power[:] = 0.0
        self._means = None
        self.steps = 0

    def set_zone_power(self, index, power):
        self.zone_power[index] = power

    def load_field(self, grid):
        """The grid was changed outside the solve (the convection shuffle): take its solved values back."""
        self.temp = np.asarray(grid, dtype=np.float64).reshape(-1)[self.data.solved].copy()
        self._means = None

    def step(self, ambient, h, conductance, supply):
        """One timestep; returns the new full grid (exterior CVs at ambient), as the official solve does."""
        d = self.data
        self.prepare(h)
        forcing = (d.capacitance_rate * self.temp + d.expand_zone_power(self.zone_power)
                   + ambient * (d.dirichlet + self.exposure))
        forcing += d.mass_coupling * self.mass_temp
        uncoupled = self.factor.solve(forcing)
        conductance = np.asarray(conductance, dtype=np.float64).reshape(-1)
        supply = np.asarray(supply, dtype=np.float64).reshape(-1)
        power = np.linalg.solve(np.eye(len(conductance)) + conductance[:, np.newaxis] * self.sensed_response,
                                conductance * (supply - d.sensed @ uncoupled))
        solved = uncoupled + self.response @ power.astype(self.response_dtype)
        self.mass_temp *= d.mass_retained
        self.mass_temp += d.mass_gained * solved
        self.temp = solved
        self._means = None
        self.steps += 1
        grid = np.full(d.n_cells, ambient)
        grid[d.solved] = solved
        return grid.reshape(d.shape)

    def zone_means(self):
        if self._means is None:
            self._means = self.data.zone_mean @ self.temp
        return self._means

    def sensed_means(self):
        return self.data.sensed @ self.temp

    def field_mean(self, ambient):
        d = self.data
        return (self.temp.sum() + ambient * d.n_exterior) / d.n_cells

    def bounds(self, ambient):
        """min and max over the whole grid (the official divergence guard)."""
        lo, hi = float(self.temp.min()), float(self.temp.max())
        if self.data.n_exterior:
            lo, hi = min(lo, ambient), max(hi, ambient)
        return lo, hi


class RomCore:
    """Reduced thermal model (rom.RomModel) in diagonal modal form; zone-level inputs and outputs only."""

    kind = "rom"

    def __init__(self, data, model):
        self.data = data
        self.model = model
        self.h = model.h
        self.zone_power = np.zeros(len(data.zones))
        self.w = model.w0.copy()
        self._means = None
        self.steps = 0
        m = model
        # (I + G H) q = G (T_supply - S y): H = S diag(beta) P in modal coordinates
        self._sensed_response = (m.sensed * m.beta[np.newaxis, :]) @ m.applied

    def __getstate__(self):
        state = dict(self.__dict__)
        state["reward_cache"] = None
        return state

    def reset(self, grid=None):
        self.w = self.model.w0.copy()
        self.zone_power[:] = 0.0
        self._means = None
        self.steps = 0

    def set_zone_power(self, index, power):
        self.zone_power[index] = power

    def step(self, ambient, h, conductance, supply):
        m = self.model
        if h != self.h:
            raise ValueError(f"reduced model built for convection coefficient {self.h}, the step asks for {h}")
        drive = m.zone_input @ self.zone_power + m.ambient_input * ambient
        y = m.decay * self.w + m.beta * drive
        conductance = np.asarray(conductance, dtype=np.float64).reshape(-1)
        supply = np.asarray(supply, dtype=np.float64).reshape(-1)
        power = np.linalg.solve(np.eye(len(conductance)) + conductance[:, np.newaxis] * self._sensed_response,
                                conductance * (supply - m.sensed @ y))
        self.w = y + m.beta * (m.applied @ power)
        self._means = None
        self.steps += 1
        return None

    def zone_means(self):
        if self._means is None:
            self._means = self.model.zone @ self.w
        return self._means

    def sensed_means(self):
        return self.model.sensed @ self.w

    def field_mean(self, ambient):
        return float(self.model.field @ self.w) + ambient * self.data.n_exterior / self.data.n_cells

    def bounds(self, ambient):
        means = self.zone_means()
        return min(float(means.min()), ambient), max(float(means.max()), ambient)
