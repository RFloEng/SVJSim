"""The simulation core: state, equations of motion, and the integrator.

The vehicle is a multibody system with `6 + N` degrees of freedom — six for the
sprung mass and one wheel-travel coordinate per station — plus a spin state per
wheel and the driveline. Rather than hand-deriving load-transfer equations, the
equations of motion are assembled from each body's velocity Jacobian::

    M(q) u̇ = Q_external(q, u) - Q_bias(q, u)

with generalized velocities `u = [v_O, ω, j̇₁ … j̇_N]`, the first two in body-frame
components at the SVJ origin. Two things follow from doing it this way:

* jacking, anti-dive and anti-squat are not parameters. A tire force at the
  contact patch is projected onto the travel coordinate through `∂p/∂j` from
  the K&C tables, so whatever the real linkage does, the model does too.
* nothing is specific to four wheels. Trucks, trailers and three-wheelers use
  the same assembly with a different `N`.

Ground contact is solved in the inertial frame: the tire is a rigid disc of its
unloaded radius, tilted by camber and steer, and the vertical force comes from
how far that disc penetrates the road surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Sequence

import numpy as np

from .model import SimVehicle, Station
from .util import G, cross3, euler_rates, rot_from_euler, skew

# Below this speed slip ratio and slip angle are evaluated against a floor
# rather than the true forward speed, which would otherwise divide by zero.
V_LOW = 1.0

# Speed below which extra slip-velocity damping is blended in. Dividing slip by
# a speed floor leaves the wheel-spin equation almost undamped near standstill —
# a lightly damped mode near 80 Hz that no practical time step can integrate.
# Pacejka's remedy (Tyre and Vehicle Dynamics §8.5) is a damper across the
# contact patch, faded out once the tire is rolling properly.
V_DAMP_BLEND = 5.0

# Target time constant of the damped wheel-spin mode at standstill.
SPIN_DAMPING_TAU = 0.05

# Target time constant for `body_damping` (see VehicleDynamics.__init__ and
# its use in _wheel_forces) to absorb the vehicle's own translational
# momentum at near-zero speed — a separate, much larger-scale coefficient
# from SPIN_DAMPING_TAU, which only has to hold back a wheel's rotational
# inertia, not the whole car.
BODY_DAMPING_TAU = 0.05

# `body_damping` needs its own, much narrower fade-out than V_DAMP_BLEND.
# V_DAMP_BLEND's 5 m/s (18 km/h) window is sized for the wheel-spin term,
# which only has to fight a wheel's own small inertia and is gone long before
# that speed matters. body_damping is sized to hold back the *whole vehicle*
# — reusing the same wide window meant every wheel, including the two that
# are not even driven, dragged with several thousand newtons of numerical
# friction all the way up to 18 km/h, comfortably swamping the driven wheels'
# actual tractive force and making the car nearly impossible to launch. It
# only needs to act in the last few centimetres per second before standstill.
V_BODY_DAMP_BLEND = 0.15

# Slip speed at which the clutch reaches its full torque capacity (rad/s). Small
# enough that a closed clutch barely slips, large enough that the resulting
# engine equation stays integrable at a 1 ms step.
CLUTCH_SLIP_REFERENCE = 8.0


# ── Inputs and environment ────────────────────────────────────────────

@dataclass
class Controls:
    """What the driver is doing, independent of who or what the driver is."""

    steering: float = 0.0      # steering wheel angle (rad), positive turns right
    throttle: float = 0.0      # 0 to 1
    brake: float = 0.0         # 0 to 1
    gear: int = 1              # 0 is neutral
    clutch: float = 1.0        # 1 fully engaged

    def clipped(self) -> "Controls":
        return replace(self,
                       throttle=float(np.clip(self.throttle, 0.0, 1.0)),
                       brake=float(np.clip(self.brake, 0.0, 1.0)),
                       clutch=float(np.clip(self.clutch, 0.0, 1.0)))


@dataclass
class Road:
    """A road surface. Flat and uniform unless a height or friction map is given."""

    friction: float = 1.0
    height_fn: Callable[[float, float], float] | None = None
    friction_fn: Callable[[float, float], float] | None = None

    def height(self, x: float, y: float) -> float:
        """Surface height as an inertial Z (positive down), so uphill is negative."""
        return self.height_fn(x, y) if self.height_fn else 0.0

    def mu(self, x: float, y: float) -> float:
        return self.friction_fn(x, y) if self.friction_fn else self.friction


@dataclass
class Environment:
    road: Road = field(default_factory=Road)
    air_density: float = 1.225
    wind: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gravity: float = G


# ── State ─────────────────────────────────────────────────────────────

class State:
    """Packed simulation state for a vehicle with `n` wheel stations.

    Layout, all in SI units::

        0:3      position of the SVJ origin in the inertial frame
        3:6      roll, pitch, yaw (SAE 3-2-1)
        6:9      linear velocity of that origin, body-frame components
        9:12     angular velocity, body frame
        12     : 12+n      jounce (suspension compression, + is wheel up)
        12+n   : 12+2n     jounce rate
        12+2n  : 12+3n     wheel spin speed
        12+3n  : 12+4n     lagged slip angle
        12+4n  : 12+5n     lagged slip ratio
        12+5n              engine speed (rad/s)
    """

    def __init__(self, n: int, data: np.ndarray | None = None):
        self.n = n
        self.data = np.zeros(12 + 5 * n + 1) if data is None else np.asarray(data, dtype=float)

    def copy(self) -> "State":
        return State(self.n, self.data.copy())

    @property
    def size(self) -> int:
        return len(self.data)

    # Slices are properties so the solver reads like the equations do.
    position = property(lambda s: s.data[0:3])
    angles = property(lambda s: s.data[3:6])
    velocity = property(lambda s: s.data[6:9])
    omega = property(lambda s: s.data[9:12])

    @property
    def jounce(self) -> np.ndarray:
        return self.data[12:12 + self.n]

    @property
    def jounce_rate(self) -> np.ndarray:
        return self.data[12 + self.n:12 + 2 * self.n]

    @property
    def spin(self) -> np.ndarray:
        return self.data[12 + 2 * self.n:12 + 3 * self.n]

    @property
    def slip_angle(self) -> np.ndarray:
        return self.data[12 + 3 * self.n:12 + 4 * self.n]

    @property
    def slip_ratio(self) -> np.ndarray:
        return self.data[12 + 4 * self.n:12 + 5 * self.n]

    @property
    def engine_speed(self) -> float:
        return float(self.data[-1])

    @engine_speed.setter
    def engine_speed(self, value: float) -> None:
        self.data[-1] = value

    @property
    def speed(self) -> float:
        """Forward speed along the body X axis (m/s)."""
        return float(self.velocity[0])

    @property
    def ground_speed(self) -> float:
        return float(np.hypot(self.velocity[0], self.velocity[1]))

    @property
    def yaw_rate(self) -> float:
        return float(self.omega[2])

    @property
    def sideslip(self) -> float:
        """Body sideslip angle at the reference point (rad)."""
        return float(np.arctan2(self.velocity[1], max(V_LOW, abs(self.velocity[0]))))


# ── Per-station derived quantities ────────────────────────────────────

@dataclass
class WheelOutput:
    """Everything computed at one station during a derivative evaluation."""

    name: str
    Fz: float = 0.0
    Fx: float = 0.0                # body-frame longitudinal force
    Fy: float = 0.0                # body-frame lateral force
    Fx_tire: float = 0.0           # force along the wheel's own heading
    Mz: float = 0.0
    slip_angle: float = 0.0
    slip_ratio: float = 0.0
    camber: float = 0.0
    steer: float = 0.0
    deflection: float = 0.0
    load_ratio: float = 0.0        # fraction of available grip in use
    contact_body: np.ndarray = field(default_factory=lambda: np.zeros(3))
    brake_torque: float = 0.0
    drive_torque: float = 0.0
    grounded: bool = True


# ── The model ─────────────────────────────────────────────────────────

class VehicleDynamics:
    """Assembles and evaluates the equations of motion for a `SimVehicle`."""

    def __init__(self, vehicle: SimVehicle, environment: Environment | None = None):
        self.vehicle = vehicle
        self.env = environment or Environment()
        self.stations: list[Station] = vehicle.stations
        self.n = len(self.stations)
        self.dof = 6 + self.n
        self.partner_index = [
            next((k for k, o in enumerate(self.stations) if o.name == s.arb_partner), None)
            for s in self.stations
        ]
        # Wheel spin inertia including the share of driveline inertia reflected
        # through the final drive is applied in `_driveline_torque`.
        self.spin_inertia = np.array([max(0.05, s.spin_inertia) for s in self.stations])
        # Contact-patch damping sized to critically damp each wheel's spin mode
        # at standstill, where the slip-based force gives almost no damping.
        self.slip_damping = np.array([
            self.spin_inertia[i] / (max(0.05, s.tire.unloaded_radius) ** 2 * SPIN_DAMPING_TAU)
            for i, s in enumerate(self.stations)
        ])
        # A second damping coefficient at the vehicle's own mass scale (see its
        # use in `_wheel_forces`): `slip_damping` above is sized from a wheel's
        # rotational inertia and is far too weak to hold the car itself against
        # a small sustained force at rest.
        self.body_damping = np.array([
            (vehicle.total_mass / max(1, self.n)) / BODY_DAMPING_TAU
            for _ in self.stations
        ])

    # -- kinematic helpers ---------------------------------------------

    def _station_geometry(self, station: Station, jounce: float, rack: float,
                          h: float = 5e-4) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Wheel-centre position, its first and second jounce derivatives, and
        the full K&C sample at the operating point."""
        s0 = station.kc.sample_all(jounce, rack)
        s_up = station.kc.sample_all(jounce + h, rack)
        s_dn = station.kc.sample_all(jounce - h, rack)
        p0, p_up, p_dn = s0[:3], s_up[:3], s_dn[:3]
        dp = (p_up - p_dn) / (2 * h)
        d2p = (p_up - 2 * p0 + p_dn) / (h * h)
        return p0.copy(), dp, d2p, s0

    # -- main evaluation -----------------------------------------------

    def derivatives(self, state: State, controls: Controls,
                    collect: bool = False) -> tuple[np.ndarray, list[WheelOutput] | None]:
        """Time derivative of the whole state vector.

        Returns the derivative array and, when `collect` is set, the per-station
        detail the telemetry recorder needs.
        """
        controls = controls.clipped()
        v = self.vehicle
        n = self.n
        R = rot_from_euler(*state.angles)
        omega = state.omega
        v_O = state.velocity
        rack = v.steering.rack_from_wheel(controls.steering)
        gravity_body = R.T @ np.array([0.0, 0.0, self.env.gravity])

        M = np.zeros((self.dof, self.dof))
        Q = np.zeros(self.dof)
        bias = np.zeros(self.dof)

        # --- sprung mass ----------------------------------------------
        c = v.sprung_cg
        J_s = np.zeros((3, self.dof))
        J_s[:, 0:3] = np.eye(3)
        J_s[:, 3:6] = -skew(c)
        M += v.sprung_mass * (J_s.T @ J_s)
        M[3:6, 3:6] += v.sprung_inertia

        bias_s = cross3(omega, v_O) + cross3(omega, cross3(omega, c))
        bias += v.sprung_mass * (J_s.T @ bias_s)
        bias[3:6] += cross3(omega, v.sprung_inertia @ omega)
        Q += v.sprung_mass * (J_s.T @ gravity_body)

        # --- wheel stations -------------------------------------------
        outputs: list[WheelOutput] = []
        spin_torque = np.zeros(n)
        drive = self._driveline_torque(state, controls)

        for i, station in enumerate(self.stations):
            j, jdot = float(state.jounce[i]), float(state.jounce_rate[i])
            p, dp, d2p, sample = self._station_geometry(station, j, rack)

            J_u = np.zeros((3, self.dof))
            J_u[:, 0:3] = np.eye(3)
            J_u[:, 3:6] = -skew(p)
            J_u[:, 6 + i] = dp
            mass_u = station.unsprung_mass
            M += mass_u * (J_u.T @ J_u)

            p_dot = dp * jdot
            bias_u = (cross3(omega, v_O)
                      + cross3(omega, cross3(omega, p))
                      + 2.0 * cross3(omega, p_dot)
                      + d2p * jdot * jdot)
            bias += mass_u * (J_u.T @ bias_u)
            Q += mass_u * (J_u.T @ gravity_body)

            beam_j = (float(state.jounce[station.beam_partner])
                      if station.beam_partner is not None else 0.0)
            out = self._wheel_forces(state, station, i, j, jdot, p, dp, R, rack,
                                     controls, float(drive[i]), sample, beam_j)
            outputs.append(out)

            # Tire force acts at the contact patch, not the hub — that offset is
            # what turns a lateral force into jacking and a braking force into
            # anti-dive, through the same Jacobian.
            cp = out.contact_body
            J_c = np.zeros((3, self.dof))
            J_c[:, 0:3] = np.eye(3)
            J_c[:, 3:6] = -skew(cp)
            J_c[:, 6 + i] = dp
            force = np.array([out.Fx, out.Fy, -out.Fz])
            Q += J_c.T @ force
            Q[3:6] += np.array([0.0, 0.0, out.Mz])

            partner = self.partner_index[i]
            partner_j = float(state.jounce[partner]) if partner is not None else 0.0
            suspension = station.suspension_force(j, jdot, partner_j)
            Q[6 + i] -= suspension

            spin_torque[i] = (out.drive_torque - out.brake_torque
                              - out.Fx_tire * self._rolling_radius(station, out.Fz))

        # --- aerodynamics ---------------------------------------------
        speed = float(np.linalg.norm(v_O[:2]))
        aero_force, _, _ = v.aero.forces(speed, state.sideslip)
        cop = v.aero.centre_of_pressure
        J_a = np.zeros((3, self.dof))
        J_a[:, 0:3] = np.eye(3)
        J_a[:, 3:6] = -skew(cop)
        Q += J_a.T @ aero_force

        # --- solve -----------------------------------------------------
        try:
            udot = np.linalg.solve(M, Q - bias)
        except np.linalg.LinAlgError:
            udot = np.linalg.lstsq(M, Q - bias, rcond=None)[0]

        # --- pack derivatives ------------------------------------------
        d = np.zeros(state.size)
        d[0:3] = R @ v_O
        d[3:6] = euler_rates(state.angles[0], state.angles[1], omega)
        d[6:12] = udot[0:6]
        d[12:12 + n] = state.jounce_rate
        d[12 + n:12 + 2 * n] = udot[6:6 + n]
        d[12 + 2 * n:12 + 3 * n] = spin_torque / self.spin_inertia

        for i, out in enumerate(outputs):
            station = self.stations[i]
            vx = max(V_LOW, abs(self._forward_speed(state, station, i)))
            d[12 + 3 * n + i] = (vx / max(1e-3, station.tire.relaxation_lateral)) * \
                (out.slip_angle - float(state.slip_angle[i]))
            d[12 + 4 * n + i] = (vx / max(1e-3, station.tire.relaxation_longitudinal)) * \
                (out.slip_ratio - float(state.slip_ratio[i]))

        d[-1] = self._engine_derivative(state, controls)
        return d, (outputs if collect else None)

    # -- contact and tire ----------------------------------------------

    def _forward_speed(self, state: State, station: Station, index: int) -> float:
        return float(state.velocity[0])

    def _rolling_radius(self, station: Station, Fz: float) -> float:
        return station.tire.effective_radius(max(0.0, Fz))

    def _wheel_forces(self, state: State, station: Station, index: int,
                      jounce: float, jounce_rate: float, p: np.ndarray, dp: np.ndarray,
                      R: np.ndarray, rack: float, controls: Controls,
                      drive_torque: float, sample: np.ndarray,
                      partner_jounce: float = 0.0) -> WheelOutput:
        out = WheelOutput(name=station.name)
        camber, steer = station.resolve(sample, jounce, partner_jounce)
        out.camber, out.steer = camber, steer

        # Wheel spin axis, nominally +Y, tilted by camber then steered.
        axis_body = np.array([
            -np.sin(steer) * np.cos(camber),
            np.cos(steer) * np.cos(camber),
            np.sin(camber),
        ])
        axis_world = R @ axis_body

        wc_world = state.position + R @ p
        down = np.array([0.0, 0.0, 1.0])
        plane_down = down - float(down @ axis_world) * axis_world
        norm = float(np.linalg.norm(plane_down))
        if norm < 1e-9:
            return out
        plane_down /= norm

        road_z = self.env.road.height(float(wc_world[0]), float(wc_world[1]))
        if plane_down[2] <= 1e-6:
            return out
        contact_distance = (road_z - float(wc_world[2])) / plane_down[2]
        R0 = station.tire.unloaded_radius
        deflection = R0 - contact_distance
        if deflection <= 0.0:
            out.grounded = False
            out.deflection = 0.0
            # Keep the contact point under the wheel so the Jacobian stays finite.
            out.contact_body = p + R0 * (R.T @ plane_down)
            return out

        loaded_radius = max(0.05 * R0, contact_distance)
        contact_world = wc_world + loaded_radius * plane_down
        contact_body = R.T @ (contact_world - state.position)
        out.contact_body = contact_body
        out.deflection = deflection

        # Velocity of the wheel-carrier material point at the contact patch.
        lever = contact_body - p
        v_carrier = (state.velocity + cross3(state.omega, contact_body)
                     + dp * jounce_rate)
        deflection_rate = -float((R @ v_carrier)[2])
        Fz = station.tire.normal_force(deflection, deflection_rate) * station.wheel_count
        if Fz <= 0.0:
            out.grounded = False
            return out
        out.Fz = Fz

        # Tire frame: heading is the wheel plane projected onto the road.
        heading_body = np.array([np.cos(steer), np.sin(steer), 0.0])
        lateral_body = np.array([-np.sin(steer), np.cos(steer), 0.0])
        Vx = float(v_carrier @ heading_body)
        Vy = float(v_carrier @ lateral_body)
        speed_floor = max(V_LOW, abs(Vx))

        spin = float(state.spin[index])
        r_eff = station.tire.effective_radius(Fz / max(1, station.wheel_count))
        out.slip_ratio = float(np.clip((spin * r_eff - Vx) / speed_floor, -10.0, 10.0))
        out.slip_angle = float(np.arctan2(-Vy, speed_floor))

        # The lagged states are what the tire actually responds to; relaxation
        # keeps forces continuous through transients and at a standstill.
        alpha = float(state.slip_angle[index])
        kappa = float(state.slip_ratio[index])

        mu = self.env.road.mu(float(contact_world[0]), float(contact_world[1]))
        per_wheel = Fz / max(1, station.wheel_count)
        flip = -1.0 if station.mirror_tire else 1.0
        forces = station.tire.forces(per_wheel, kappa, flip * alpha, flip * camber,
                                     Vx, mu_scale=mu)
        scale = station.wheel_count
        fx_tire = forces.Fx * scale
        fy_tire = forces.Fy * scale * flip
        out.Mz = forces.Mz * scale * flip

        # Low-speed contact-patch damping (see V_DAMP_BLEND). Friction opposes
        # how the patch slides over the road, so a wheel spinning faster than
        # the road speed is pushed forwards — the same sign as the slip force it
        # stands in for. It fades out as the tire starts rolling and is capped by
        # the available friction so it can never invent grip. Sized to the
        # wheel's own (small) spin inertia — it feeds spin_torque below, so it
        # has to stay small enough not to over-stiffen that equation at the
        # real-time step size (see SPIN_DAMPING_TAU).
        blend = max(0.0, 1.0 - abs(Vx) / V_DAMP_BLEND)
        if blend > 0.0:
            c = self.slip_damping[index] * blend
            limit = mu * station.tire.mu_reference * Fz
            fx_tire += float(np.clip(c * (spin * r_eff - Vx), -limit, limit))
            fy_tire += float(np.clip(-c * Vy, -limit, limit))

        # Rotate tire-frame forces into the body frame through the steer angle.
        out.Fx = fx_tire * np.cos(steer) - fy_tire * np.sin(steer)
        out.Fy = fx_tire * np.sin(steer) + fy_tire * np.cos(steer)
        out.Fx_tire = fx_tire       # along the wheel, which is what spins it

        # A second, much larger, whole-vehicle-mass-scaled damping term acts on
        # the body-frame force only — not fed back into spin_torque, so it
        # cannot destabilise the wheel-spin equation the way scaling `c` above
        # up to this size would (spin_inertia is on the order of 1 kg·m²; the
        # car is a thousand-plus kg). Without a term at this scale, a genuinely
        # tiny sustained out-of-balance force at rest — even just gravity's
        # component along the body's own, very slightly pitched, equilibrium
        # axis — creeps the car for many seconds, because the wheel-inertia-
        # scaled term above is far too weak to resist the car's own momentum.
        # The tire model has no true static friction/stiction below the slip
        # threshold; this stands in for it. Its own fade-out (V_BODY_DAMP_BLEND)
        # is much narrower than the wheel-spin term's — see that constant.
        body_blend = max(0.0, 1.0 - abs(Vx) / V_BODY_DAMP_BLEND)
        if body_blend > 0.0:
            c_body = self.body_damping[index] * body_blend
            limit_body = mu * station.tire.mu_reference * Fz
            out.Fx += float(np.clip(-c_body * v_carrier[0], -limit_body, limit_body))
            out.Fy += float(np.clip(-c_body * v_carrier[1], -limit_body, limit_body))

        grip = max(1e-6, mu * station.tire.mu_reference * Fz)
        out.load_ratio = float(np.hypot(fx_tire, fy_tire) / grip)

        out.brake_torque = self._brake_torque(station, controls, spin)
        out.drive_torque = drive_torque
        return out

    def _brake_torque(self, station: Station, controls: Controls, spin: float) -> float:
        """Signed brake torque: always opposing rotation, never driving the wheel.

        The `tanh` blend keeps the torque continuous through zero speed, where a
        hard `sign` would chatter between time steps once the wheel locks.
        """
        if station.brake is None or controls.brake <= 0.0:
            return 0.0
        is_front = station.axle == min(s.axle for s in self.stations)
        pressure = self.vehicle.brakes.line_pressure(controls.brake, is_front)
        return float(station.brake.torque(pressure) * np.tanh(spin / 0.5))

    # -- driveline ------------------------------------------------------

    def _clutch_torque(self, state: State, controls: Controls) -> float:
        """Torque the clutch carries between engine and gearbox input.

        Modelled as a friction plate rather than a rigid connection, because a
        rigid one cannot launch: at a standing start it drags the engine to zero
        rpm, where the torque curve extrapolates to a negative value and the car
        will not pull away. Torque saturates at the clutch's rated capacity and
        varies smoothly with slip, so lock-up needs no special case.
        """
        v = self.vehicle
        if v.driveline is None or controls.clutch <= 0.0:
            return 0.0
        ratio = v.driveline.ratio(controls.gear)
        if ratio <= 0.0:
            return 0.0
        driven = [i for i, s in enumerate(self.stations)
                  if s.name in v.driveline.driven_stations]
        if not driven:
            return 0.0
        input_speed = float(np.mean([state.spin[i] for i in driven])) * ratio
        slip = state.engine_speed - input_speed
        capacity = v.driveline.clutch_capacity * controls.clutch
        return float(capacity * np.tanh(slip / CLUTCH_SLIP_REFERENCE))

    def _driveline_torque(self, state: State, controls: Controls) -> np.ndarray:
        """Torque delivered to each station's wheel, in newton-metres."""
        torque = np.zeros(self.n)
        v = self.vehicle
        if v.engine is None or v.driveline is None or not v.driveline.driven_stations:
            return torque
        ratio = v.driveline.ratio(controls.gear)
        if ratio <= 0.0:
            return torque

        shaft = self._clutch_torque(state, controls) * ratio * v.driveline.efficiency

        driven = [i for i, s in enumerate(self.stations) if s.name in v.driveline.driven_stations]
        if not driven:
            return torque
        share = shaft / len(driven)
        for i in driven:
            torque[i] = share

        # A limited-slip differential moves torque towards the slower wheel.
        lock = v.driveline.diff_lock_power if shaft >= 0 else v.driveline.diff_lock_coast
        if lock > 0.0 and len(driven) == 2:
            a, b = driven
            delta = float(state.spin[a] - state.spin[b])
            transfer = lock * abs(share) * np.tanh(delta * 2.0) + \
                v.driveline.diff_preload * np.tanh(delta * 10.0)
            torque[a] -= transfer
            torque[b] += transfer
        return torque

    def _engine_derivative(self, state: State, controls: Controls) -> float:
        """Engine speed from its own torque less what the clutch takes away."""
        v = self.vehicle
        if v.engine is None:
            return 0.0
        rpm = state.engine_speed * 60.0 / (2 * np.pi)
        torque = v.engine.torque(rpm, controls.throttle)

        # Idle governor: below idle the ECU adds fuel, which is also what keeps
        # the engine off the negative end of an extrapolated torque curve.
        idle = v.engine.idle_rpm * 2 * np.pi / 60.0
        if state.engine_speed < idle:
            torque += (idle - state.engine_speed) * v.engine.inertia * 8.0

        return float((torque - self._clutch_torque(state, controls))
                     / max(0.02, v.engine.inertia))

    # -- setting up -----------------------------------------------------

    def initial_state(self, speed: float = 0.0, gear: int = 1) -> State:
        """A state with the vehicle at its design attitude, rolling at `speed`."""
        state = State(self.n)
        state.velocity[0] = speed
        # Lift the body so each tire carries its static load.
        offsets = []
        for station in self.stations:
            r_loaded = station.tire.unloaded_radius - station.static_load / max(
                1.0, station.tire.vertical_stiffness * station.wheel_count)
            offsets.append(-r_loaded - station.position0[2])
        state.position[2] = float(np.mean(offsets))
        for i, station in enumerate(self.stations):
            r_loaded = station.tire.unloaded_radius - station.static_load / max(
                1.0, station.tire.vertical_stiffness * station.wheel_count)
            state.spin[i] = speed / max(0.05, r_loaded)
        if self.vehicle.engine and self.vehicle.driveline:
            ratio = self.vehicle.driveline.ratio(gear)
            idle = self.vehicle.engine.idle_rpm * 2 * np.pi / 60.0
            state.engine_speed = max(idle, float(np.mean(state.spin)) * ratio)
        return state

    def trim(self, speed: float = 0.0, gear: int = 1, duration: float = 3.0,
             dt: float = 0.001) -> State:
        """Settle the vehicle onto its suspension before a maneuver starts.

        Integrates from the design attitude with the body's horizontal motion
        held fixed, so the suspension and tires reach equilibrium without the
        vehicle wandering. Extra heave damping is applied for the first half to
        kill the initial transient quickly.
        """
        state = self.initial_state(speed, gear)
        # Trim in neutral with the clutch out: engine braking through the gears
        # would otherwise drag the driven wheels while the vehicle is meant to
        # be standing still.
        controls = Controls(gear=0, throttle=0.0, clutch=0.0)
        steps = int(duration / dt)
        frozen = [0, 1, 5, 6, 7, 11]      # x, y, yaw, and their accelerations
        for step in range(steps):
            state = _rk4_constrained(self, state, controls, dt, frozen)
            if step < steps // 2:
                state.velocity[2] *= 0.90
                state.omega[0:2] *= 0.90
                state.jounce_rate[:] *= 0.90
            state.velocity[0] = speed
        return state


# ── Integration ───────────────────────────────────────────────────────

def _rk4_constrained(dynamics: "VehicleDynamics", state: State, controls: Controls,
                     dt: float, frozen: Sequence[int]) -> State:
    """RK4 with selected derivative components held at zero, used during trim."""
    def f(y: np.ndarray) -> np.ndarray:
        d, _ = dynamics.derivatives(State(state.n, y), controls)
        d[list(frozen)] = 0.0
        return d

    y0 = state.data
    k1 = f(y0)
    k2 = f(y0 + 0.5 * dt * k1)
    k3 = f(y0 + 0.5 * dt * k2)
    k4 = f(y0 + dt * k3)
    return State(state.n, y0 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))


def step_rk4(dynamics: VehicleDynamics, state: State, controls: Controls,
             dt: float) -> State:
    """One classical Runge-Kutta 4 step."""
    y0 = state.data
    k1, _ = dynamics.derivatives(state, controls)
    k2, _ = dynamics.derivatives(State(state.n, y0 + 0.5 * dt * k1), controls)
    k3, _ = dynamics.derivatives(State(state.n, y0 + 0.5 * dt * k2), controls)
    k4, _ = dynamics.derivatives(State(state.n, y0 + dt * k3), controls)
    return State(state.n, y0 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))


def step_euler(dynamics: VehicleDynamics, state: State, controls: Controls,
               dt: float) -> State:
    """Semi-implicit (symplectic) Euler: cheap, and stable for an oscillator.

    Plain forward Euler — `state += dt * derivatives(state)` in one shot — is
    unconditionally unstable for any oscillatory mode: its numerical
    amplification factor for a purely imaginary eigenvalue has magnitude
    greater than 1 at *any* step size, so energy leaks in every step. Left
    running on a stationary vehicle this shows up as the heave/pitch mode
    slowly growing on its own — a car that visibly rocks and creeps forward
    while "parked", which is exactly what an earlier version of this function
    did.

    The fix is the standard symplectic-Euler split: advance every rate-like
    state component (`v_O`, `omega`, jounce rate, wheel spin, the slip lag
    states, engine speed) from the derivatives evaluated at the start of the
    step, then advance every position-like component (position, attitude,
    jounce) using those just-updated rates instead of the step's starting
    ones. For a harmonic oscillator this keeps the numerical energy bounded
    rather than growing every step, at the same one-evaluation-per-step cost
    as the naive version.
    """
    n = state.n
    d, _ = dynamics.derivatives(state, controls)
    data = state.data.copy()

    # Rate-like components: each is governed by its own self-contained ODE
    # (nothing here needs a "look-ahead" rate), so advancing them from the
    # derivative just computed is the same as plain Euler would do.
    data[6:12] += dt * d[6:12]                    # v_O, omega
    data[12 + n:] += dt * d[12 + n:]               # jounce_rate, spin,
                                                    # slip states, engine speed

    # Position-like components use the rates just updated above rather than
    # the ones the step started with — this half of the split is what makes
    # the integrator symplectic.
    updated = State(n, data)
    R = rot_from_euler(*state.angles)
    data[0:3] += dt * (R @ updated.velocity)
    data[3:6] += dt * euler_rates(state.angles[0], state.angles[1], updated.omega)
    data[12:12 + n] += dt * updated.jounce_rate

    return State(n, data)
