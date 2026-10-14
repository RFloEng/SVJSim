"""Building a solver-ready vehicle from an SVJ document.

This is the preprocessing step: read the file, solve the suspension geometry
into K&C tables, build tire models, decompose the mass into sprung and unsprung
parts, and work out the static load at every wheel station. What comes out is a
`SimVehicle` — numbers and callables only, with no dictionary lookups left for
the solver to do at run time.

Conventions that the rest of SVJSim depends on:

* the body frame is the SVJ frame (SAE J670, Z down, origin at axle 1 on the
  ground) and hardpoints are used unconverted
* `jounce` is suspension compression, positive when the wheel moves up towards
  the body, measured from the design position the hardpoints describe
* spring force is expressed as *static load + rate x jounce*, so the vehicle
  stands at its design attitude at rest. This is how a suspension is defined in
  practice, and it avoids depending on free length and preload values that are
  usually missing or estimated

Everything the build had to assume, estimate or work around is recorded in
`SimVehicle.report`, so a result can always be traced back to the data it came
from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .driveline import (CentreSpec, Differential, axle_number, infer_driven,
                        parse_centre, parse_differential)
from .kinematics import (AxleKinematics, KCTable, StationKinematics, from_svj_camber,
                         from_svj_toe, side_sign)
from .tires import TireModel, build_tire
from .util import (G, Curve, clean, dig, inertia_matrix, num, parallel_axis,
                   vec3)


# ── Build report ──────────────────────────────────────────────────────

@dataclass
class BuildReport:
    """Everything assumed, estimated or corrected while building the model."""

    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    estimates: list[str] = field(default_factory=list)

    def note(self, msg: str) -> None:
        self.notes.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def estimate(self, msg: str) -> None:
        self.estimates.append(msg)

    def extend(self, messages: Iterable[str], kind: str = "warnings") -> None:
        getattr(self, kind).extend(messages)

    def dedupe(self) -> None:
        """Collapse messages repeated across stations into one entry each."""
        for kind in ("notes", "warnings", "estimates"):
            seen: dict[str, None] = {}
            for msg in getattr(self, kind):
                seen.setdefault(msg, None)
            setattr(self, kind, list(seen))

    @property
    def confidence(self) -> str:
        """How much of this model came from the file rather than from guesses."""
        if not self.estimates and not self.warnings:
            return "high"
        if len(self.estimates) + len(self.warnings) <= 4:
            return "medium"
        return "low"


# ── Force elements ────────────────────────────────────────────────────

@dataclass
class Spring:
    """Wheel-rate spring, referenced to the static load at the design position."""

    rate: float                       # wheel rate (N/m)
    static_force: float = 0.0         # load carried at the design position (N)
    curve: Curve | None = None        # optional non-linear wheel-rate curve

    def force(self, jounce: float) -> float:
        if self.curve is not None:
            return self.static_force + self.curve(jounce) - self.curve(0.0)
        return self.static_force + self.rate * jounce


@dataclass
class Damper:
    """Damper resolved to the wheel, from the SVJ bump and rebound curves."""

    bump: Curve
    rebound: Curve
    motion_ratio: float = 1.0

    def force(self, jounce_rate: float) -> float:
        """Force at the wheel resisting travel; positive opposes compression."""
        v = jounce_rate * self.motion_ratio
        f = self.bump(abs(v)) if v >= 0.0 else -self.rebound(abs(v))
        return f * self.motion_ratio

    def equivalent_rate(self, velocity: float = 0.1) -> float:
        """Linear-equivalent damping at the wheel, in N·s/m.

        Taken as the secant through a representative velocity rather than the
        slope at the origin. Dampers are strongly digressive — the MX-5 template
        is four times stiffer just off zero than at 0.5 m/s — so an origin slope
        overstates the damping ratio several-fold.
        """
        v = max(1e-4, velocity)
        return (self.force(v) - self.force(-v)) / (2 * v)


@dataclass
class TravelStop:
    """Bump or rebound stop: free gap, then a rising rate."""

    gap: float
    rate: float = 0.0
    curve: Curve | None = None

    def force(self, penetration: float) -> float:
        if penetration <= 0.0:
            return 0.0
        if self.curve is not None:
            return self.curve(penetration)
        return self.rate * penetration


@dataclass
class Brake:
    """Per-station friction brake, driven by hydraulic line pressure."""

    torque_per_pressure: float        # Nm per Pa of line pressure
    mu_curve: Curve | None = None
    thermal_mass: float = 1.0e4
    disc_inertia: float = 0.0

    def torque(self, pressure: float, temperature: float = 100.0) -> float:
        scale = 1.0
        if self.mu_curve is not None and len(self.mu_curve) > 1:
            nominal = self.mu_curve(100.0)
            if nominal > 1e-6:
                scale = self.mu_curve(temperature) / nominal
        return max(0.0, pressure) * self.torque_per_pressure * scale


# ── Station ───────────────────────────────────────────────────────────

@dataclass
class Station:
    """One wheel station: geometry, tire, springing and brake, ready to solve."""

    name: str
    axle: int
    side: str
    position0: np.ndarray             # design wheel centre, body frame
    unsprung_mass: float
    spin_inertia: float
    kc: KCTable
    tire: TireModel
    spring: Spring
    damper: Damper
    brake: Brake | None
    bump_stop: TravelStop | None
    rebound_stop: TravelStop | None
    static_camber: float              # absolute SAE rotation about +X (rad)
    static_toe: float                 # absolute SAE rotation about +Z (rad)
    steered: bool
    driven: bool
    wheel_count: int = 1
    mirror_tire: bool = False         # see SimVehicle notes on ply steer
    camber_ref: float = 0.0           # geometric camber at the design position
    toe_ref: float = 0.0              # geometric toe at the design position
    beam_partner: int | None = None   # index of the station sharing a rigid axle
    beam_track: float = 0.0           # distance between the two wheel centres (m)
    beam_side: float = 1.0            # +1 for the left station, -1 for the right
    beam_roll_centre_z: float | None = None  # axle's solved roll centre, body-frame Z (m)
    arb_partner: str | None = None
    arb_rate: float = 0.0             # wheel-rate equivalent (N/m)
    static_load: float = 0.0          # vertical load at rest (N)

    @property
    def sign(self) -> float:
        return side_sign(float(self.position0[1]))

    def wheel_center(self, jounce: float, rack: float = 0.0) -> np.ndarray:
        return self.kc.wheel_center(jounce, rack)

    def camber(self, jounce: float, rack: float = 0.0) -> float:
        """Absolute camber: static alignment plus whatever the geometry adds."""
        return self.static_camber + self.kc.camber_at(jounce, rack) - self.camber_ref

    def steer(self, jounce: float, rack: float = 0.0) -> float:
        """Absolute steer angle: static toe, bump steer and rack input combined."""
        return self.static_toe + self.kc.toe_at(jounce, rack) - self.toe_ref

    def resolve(self, sample: np.ndarray) -> tuple[float, float]:
        """Camber and steer from a `KCTable.sample_all` row, avoiding a re-lookup.

        Beam-axle stations need no special case here any more: their `kc` is
        an `AxleKinematics`-built table already indexed by (own jounce,
        partner jounce) instead of (own jounce, rack), so `sample` already
        reflects the axle's actual solved roll — the caller (`solver.py`) is
        what passes partner jounce in place of rack when it builds `sample`.
        """
        return (self.static_camber + float(sample[3]) - self.camber_ref,
                self.static_toe + float(sample[4]) - self.toe_ref)

    @property
    def is_beam_axle(self) -> bool:
        return self.beam_partner is not None

    @property
    def camber_gain(self) -> float:
        """Camber change per metre of wheel travel in pure bump, rad/m.

        Zero on a rigid beam axle: both wheels rise together and the axle does
        not rotate. Its camber comes from roll instead, which a single-station
        sweep cannot see.
        """
        return 0.0 if self.is_beam_axle else self.kc.camber_gain

    def roll_centre_height(self) -> float | None:
        """Roll centre above ground in metres, or None when it is not derivable.

        A beam axle's roll centre is a property of the whole axle (its
        locating linkage — Panhard rod, Watt's link, trailing-link axes — not
        of one wheel's travel), solved once for the axle by `AxleKinematics`
        at build time (`beam_roll_centre_z`) since that needs both sides'
        links together. Converting to "height above ground" is deferred to
        here because tyres aren't built yet when that solve runs.
        """
        if self.is_beam_axle:
            if self.beam_roll_centre_z is None or self.tire is None:
                return None
            ground_z = float(self.position0[2]) + self.tire.unloaded_radius
            return ground_z - self.beam_roll_centre_z
        return self.kc.roll_centre_height(self.tire.unloaded_radius)

    def suspension_force(self, jounce: float, jounce_rate: float,
                         partner_jounce: float = 0.0) -> float:
        """Total wheel-referenced force resisting compression, in newtons."""
        force = self.spring.force(jounce) + self.damper.force(jounce_rate)
        if self.arb_rate and self.arb_partner:
            force += self.arb_rate * (jounce - partner_jounce) / 2.0
        if self.bump_stop is not None:
            force += self.bump_stop.force(jounce - self.bump_stop.gap)
        if self.rebound_stop is not None:
            force -= self.rebound_stop.force(-jounce - self.rebound_stop.gap)
        return force


# ── Subsystems ────────────────────────────────────────────────────────

@dataclass
class Steering:
    """Rack-and-pinion steering: driver angle in, rack displacement out."""

    rack_per_wheel_angle: float       # m of rack per rad of road wheel steer
    overall_ratio: float              # steering wheel angle per road wheel angle
    max_rack: float
    lock_to_lock: float

    def rack_from_wheel(self, steering_wheel_angle: float) -> float:
        rack = steering_wheel_angle / max(1e-6, self.overall_ratio) * self.rack_per_wheel_angle
        return float(np.clip(rack, -self.max_rack, self.max_rack))

    @property
    def max_steer_angle(self) -> float:
        return self.max_rack / max(1e-9, self.rack_per_wheel_angle)


@dataclass
class Aerodynamics:
    """Drag, lift and their front/rear split, with optional yaw sensitivity."""

    frontal_area: float
    air_density: float
    Cd: float
    Cl_front: float
    Cl_rear: float
    centre_of_pressure: np.ndarray
    Cd_vs_yaw: Curve | None = None
    Cs_vs_yaw: Curve | None = None
    Cs: float = 0.0

    def forces(self, speed: float, yaw_angle: float = 0.0) -> tuple[np.ndarray, float, float]:
        """Body-frame force plus the front and rear lift contributions.

        Returns `(force, lift_front, lift_rear)`. Lift is positive upwards, so
        downforce is negative, matching SVJ's SAE J1594 sign convention.
        """
        q = 0.5 * self.air_density * speed ** 2 * self.frontal_area
        cd = self.Cd_vs_yaw(np.degrees(yaw_angle)) if self.Cd_vs_yaw else self.Cd
        cs = self.Cs_vs_yaw(np.degrees(yaw_angle)) if self.Cs_vs_yaw else self.Cs
        drag = -q * cd                      # acts rearward, i.e. -X
        side = q * cs
        lift_front = q * self.Cl_front
        lift_rear = q * self.Cl_rear
        # Z is down, so upward lift is negative Z.
        return np.array([drag, side, -(lift_front + lift_rear)]), lift_front, lift_rear


@dataclass
class BrakeSystem:
    """Pedal-to-line-pressure chain and the front/rear split (SVJ §12.6)."""

    max_line_pressure: float
    bias_front: float
    abs_enabled: bool = False
    abs_slip_target: tuple[float, float] = (0.08, 0.14)

    def line_pressure(self, pedal: float, axle_is_front: bool) -> float:
        share = self.bias_front if axle_is_front else (1.0 - self.bias_front)
        return float(np.clip(pedal, 0.0, 1.0)) * self.max_line_pressure * share * 2.0


@dataclass
class Engine:
    torque_curve: Curve
    idle_rpm: float
    max_rpm: float
    inertia: float

    def torque(self, rpm: float, throttle: float) -> float:
        wot = self.torque_curve(float(np.clip(rpm, 0.0, self.max_rpm)))
        drag = -0.12 * abs(wot) if wot else -20.0
        if rpm > self.max_rpm:
            return drag
        return drag + (wot - drag) * float(np.clip(throttle, 0.0, 1.0))


@dataclass
class Driveline:
    """Gearbox, final drive and the layout downstream of it, reduced to what the
    solver needs. `axle_diffs` holds each axle's own differential by axle number;
    an axle without one uses `default_diff`. `centre` is what sits between the
    front and rear axles, if anything. `filter_count` is how many low-pass
    filters the lock controllers need integrated alongside the vehicle state."""

    gear_ratios: list[float]
    final_drive: float
    efficiency: float
    driven_stations: list[str]
    axle_diffs: dict[int, Differential] = field(default_factory=dict)
    default_diff: Differential = field(default_factory=Differential)
    centre: CentreSpec | None = None
    filter_count: int = 0
    shift_time: float = 0.2
    inertia: float = 0.0
    clutch_capacity: float = 400.0     # maximum torque the clutch can carry (Nm)

    def ratio(self, gear: int) -> float:
        if gear <= 0 or gear > len(self.gear_ratios):
            return 0.0
        return self.gear_ratios[gear - 1] * self.final_drive


# ── The assembled vehicle ─────────────────────────────────────────────

@dataclass
class SimVehicle:
    name: str
    stations: list[Station]
    sprung_mass: float
    sprung_cg: np.ndarray
    sprung_inertia: np.ndarray
    total_mass: float
    total_cg: np.ndarray
    wheelbase: float
    steering: Steering
    aero: Aerodynamics
    brakes: BrakeSystem
    engine: Engine | None
    driveline: Driveline | None
    report: BuildReport
    benchmarks: list[dict[str, Any]] = field(default_factory=list)
    validation: dict[str, Any] = field(default_factory=dict)
    source: dict[str, Any] = field(default_factory=dict)

    def station(self, name: str) -> Station:
        for s in self.stations:
            if s.name == name:
                return s
        raise KeyError(name)

    @property
    def station_names(self) -> list[str]:
        return [s.name for s in self.stations]

    @property
    def front_stations(self) -> list[Station]:
        first = min(s.axle for s in self.stations)
        return [s for s in self.stations if s.axle == first]

    @property
    def rear_stations(self) -> list[Station]:
        last = max(s.axle for s in self.stations)
        return [s for s in self.stations if s.axle == last]

    @property
    def unsprung_mass(self) -> float:
        return sum(s.unsprung_mass for s in self.stations)

    def benchmark(self, kpi: str) -> dict[str, Any] | None:
        for entry in self.benchmarks:
            if str(entry.get("id", "")).lower() == kpi.lower():
                return entry
        return None

    def sprung_corner_mass(self, station: Station) -> float:
        """Sprung mass carried at one station. NaN if the station carries none."""
        mass = station.static_load / G - station.unsprung_mass
        return mass if mass > 1.0 else float("nan")

    def _ride_rate(self, station: Station) -> float:
        k_tire = station.tire.vertical_stiffness
        return station.spring.rate * k_tire / max(1.0, station.spring.rate + k_tire)

    def ride_frequency(self, station: Station) -> float:
        """Undamped ride frequency in Hz, spring and tire acting in series."""
        return float(np.sqrt(self._ride_rate(station) / self.sprung_corner_mass(station))
                     / (2 * np.pi))

    def damping_ratio(self, station: Station) -> float:
        critical = 2.0 * np.sqrt(self._ride_rate(station) * self.sprung_corner_mass(station))
        return float(station.damper.equivalent_rate() / critical)


# ── Building ──────────────────────────────────────────────────────────

def _station_axle(name: str) -> int:
    if name.upper().startswith("A"):
        digits = "".join(c for c in name[1:] if c.isdigit())
        return int(digits) if digits else 1
    return 1 if name.upper().startswith("F") else 2


def _station_side(name: str) -> str:
    return name.strip().upper()[-1]


def _static_loads(stations: Sequence[tuple[str, np.ndarray, float]],
                  total_mass: float, cg: np.ndarray, report: BuildReport,
                  balance_pitch: bool = True) -> dict[str, float]:
    """Vertical load at each station for a rigid body on linear springs.

    Three statics equations — total weight, pitch moment and roll moment — are
    solved for the heave, pitch and roll of the sprung mass. With more than
    three stations the distribution is genuinely indeterminate without the
    stiffnesses, which is exactly how a multi-axle truck behaves: the stiffer
    the spring, the more load it takes.

    `balance_pitch` is dropped for trailers, whose weight is partly carried at a
    king pin or drawbar that SVJ does not yet describe (§23.8); balancing pitch
    on the axles alone would pile the whole load onto the front axle.
    """
    weight = total_mass * G
    names = [n for n, _, _ in stations]
    x = np.array([p[0] for _, p, _ in stations])
    y = np.array([p[1] for _, p, _ in stations])
    k = np.array([max(1.0, rate) for _, _, rate in stations])

    # F_i = k_i * (z + a*x_i + b*y_i); solve for (z, a, b).
    if balance_pitch:
        A = np.array([
            [k.sum(), (k * x).sum(), (k * y).sum()],
            [(k * x).sum(), (k * x * x).sum(), (k * x * y).sum()],
            [(k * y).sum(), (k * x * y).sum(), (k * y * y).sum()],
        ])
        rhs = np.array([weight, weight * cg[0], weight * cg[1]])
    else:
        A = np.array([
            [k.sum(), 0.0, (k * y).sum()],
            [0.0, 1.0, 0.0],
            [(k * y).sum(), 0.0, (k * y * y).sum()],
        ])
        rhs = np.array([weight, 0.0, weight * cg[1]])
    try:
        z, a, b = np.linalg.solve(A, rhs)
        loads = k * (z + a * x + b * y)
    except np.linalg.LinAlgError:
        loads = np.full(len(names), weight / len(names))

    if np.any(loads < -1.0):
        lifted = [n for n, v in zip(names, loads) if v < -1.0]
        report.warn(
            f"static load solve wanted to pull {', '.join(lifted)} downwards, which means the "
            f"centre of gravity sits outside the wheelbase for the given spring rates. "
            f"Loads were clipped at zero and rescaled — check `chassis.center_of_gravity`")
    if np.any(loads < 0):
        loads = np.clip(loads, 0.0, None)
        loads *= weight / max(1e-6, loads.sum())
    return dict(zip(names, (float(v) for v in loads)))


def _build_spring(corner: dict[str, Any], report: BuildReport,
                  station: str, motion_ratio: float) -> Spring:
    spec = clean(dig(corner, "spring", default={}) or {})
    rate = num(spec, "rate", default=None)
    curve = Curve.from_svj(spec.get("rate_curve"))
    if rate is None and curve is None:
        rate = 25000.0
        report.estimate(f"{station}: no spring rate — assumed {rate / 1000:.0f} kN/m at the wheel")
    declared_mr = num(spec, "motion_ratio", default=None)
    if declared_mr and motion_ratio and np.isfinite(motion_ratio) and motion_ratio > 0:
        if abs(declared_mr - motion_ratio) / declared_mr > 0.05:
            report.warn(
                f"{station}: declared spring motion ratio {declared_mr:.3f} differs from the "
                f"{motion_ratio:.3f} the hardpoints give ({abs(declared_mr / motion_ratio - 1) * 100:.0f}% "
                f"apart); SVJ §9.3 says the geometry wins, so the declared value is only reported")
    return Spring(rate=rate or 25000.0, curve=curve)


def _build_damper(corner: dict[str, Any], report: BuildReport,
                  station: str, motion_ratio: float) -> Damper:
    spec = clean(dig(corner, "damper", default={}) or {})
    bump = Curve.from_svj(spec.get("bump_curve"))
    rebound = Curve.from_svj(spec.get("rebound_curve"))
    if bump is None or rebound is None:
        # A car with no damper data would oscillate forever; 0.35 of critical is
        # a conventional road-car value and is flagged as invented.
        bump = bump or Curve([[0.0, 0.0], [1.0, 3000.0]])
        rebound = rebound or Curve([[0.0, 0.0], [1.0, 4500.0]])
        report.estimate(f"{station}: no damper curves — substituted a typical road-car damper")
    declared = num(spec, "motion_ratio", default=None)
    ratio = motion_ratio if (motion_ratio and np.isfinite(motion_ratio) and motion_ratio > 0) \
        else (declared or 1.0)
    return Damper(bump=bump, rebound=rebound, motion_ratio=ratio)


def _build_stops(corner: dict[str, Any], key: str) -> TravelStop | None:
    spec = clean(dig(corner, key, default=None) or {})
    if not spec:
        return None
    gap = num(spec, "gap", default=None)
    if gap is None:
        return None
    return TravelStop(gap=gap, rate=num(spec, "rate", default=0.0) or 0.0,
                      curve=Curve.from_svj(spec.get("rate_curve")))


def _build_brake(corner: dict[str, Any], master_area: float) -> Brake | None:
    spec = clean(dig(corner, "brake", default=None) or {})
    if not spec:
        return None
    disc = clean(spec.get("disc") or {})
    caliper = clean(spec.get("caliper") or {})
    pad = clean(spec.get("pad") or {})
    piston_area = num(caliper, "piston_area_total", default=None) or 0.0025
    radius = num(disc, "effective_radius", default=None)
    if radius is None:
        outer = num(disc, "outer_diameter", default=0.28) / 2.0
        inner = num(disc, "inner_diameter", default=None)
        radius = (outer + inner / 2.0) / 2.0 if inner else outer * 0.85
    mu = num(pad, "mu_cold", default=None) or 0.40
    # SVJ §12.6: torque = pressure * piston area * 2 faces * mu * effective radius.
    gain = piston_area * 2.0 * mu * radius
    thermal = (num(disc, "thermal_mass", default=None)
               or (num(disc, "mass", default=5.0) * num(disc, "specific_heat", default=460.0)))
    return Brake(torque_per_pressure=gain,
                 mu_curve=Curve.from_svj(pad.get("mu_curve")),
                 thermal_mass=thermal or 1.0e4,
                 disc_inertia=num(disc, "rotational_inertia", default=0.0) or 0.0)


def _build_aero(doc: dict[str, Any], report: BuildReport,
                wheelbase: float, cg: np.ndarray) -> Aerodynamics:
    aero = clean(dig(doc, "aerodynamics", default={}) or {})
    ref = clean(aero.get("reference") or {})
    coeff = clean(aero.get("coefficients") or {})
    area = num(ref, "frontal_area", default=None)
    if area is None:
        area = 2.0
        report.estimate(f"no aerodynamic reference area — assumed {area:.1f} m²")
    Cd = num(coeff, "Cd", default=None)
    if Cd is None:
        Cd = 0.33
        report.estimate(f"no drag coefficient — assumed Cd {Cd:.2f}")
    Cl = num(coeff, "Cl", default=0.0) or 0.0
    Cl_front = num(coeff, "Cl_front", default=None)
    Cl_rear = num(coeff, "Cl_rear", default=None)
    if Cl_front is None or Cl_rear is None:
        Cl_front = Cl_rear = Cl / 2.0
        if Cl:
            report.note("lift split front/rear was not given — assumed 50/50 (SVJ §13.3)")
    cop = vec3(aero, "center_of_pressure", default=[cg[0], 0.0, cg[2]])
    maps = aero.get("maps") or {}
    return Aerodynamics(
        frontal_area=area, air_density=num(ref, "air_density", default=1.225) or 1.225,
        Cd=Cd, Cl_front=Cl_front, Cl_rear=Cl_rear, centre_of_pressure=cop,
        Cd_vs_yaw=Curve.from_svj(maps.get("Cd_vs_yaw")),
        Cs_vs_yaw=Curve.from_svj(maps.get("Cs_vs_yaw")),
        Cs=num(coeff, "Cs", default=0.0) or 0.0)


def _build_brake_system(doc: dict[str, Any], report: BuildReport) -> tuple[BrakeSystem, float]:
    brakes = clean(dig(doc, "brakes", default={}) or {})
    mc = clean(brakes.get("master_cylinder") or {})
    bore = (num(mc, "bore_primary", default=None) or num(mc, "bore", default=None) or 0.022)
    area = np.pi * (bore / 2.0) ** 2
    max_pressure = num(brakes, "line_pressure_max", default=None)
    if max_pressure is None:
        max_pressure = 1.0e7
        report.estimate("no maximum brake line pressure — assumed 100 bar")
    bias = num(brakes, "bias", default=None)
    if bias is None:
        bias = 0.62
        report.estimate(f"no brake bias — assumed {bias * 100:.0f}% front")
    abs_spec = clean(brakes.get("abs") or {})
    target = abs_spec.get("slip_target") or [0.08, 0.14]
    return BrakeSystem(
        max_line_pressure=max_pressure, bias_front=bias,
        abs_enabled=bool(abs_spec.get("enabled")),
        abs_slip_target=(float(target[0]), float(target[1]))), area


def _build_powertrain(doc: dict[str, Any], stations: Sequence[Station],
                      report: BuildReport) -> tuple[Engine | None, Driveline | None]:
    pt = clean(dig(doc, "powertrain", default={}) or {})
    if not pt:
        report.warn("no powertrain in the file — the vehicle can only coast and brake")
        return None, None

    engine_spec = clean(pt.get("engine") or {})
    torque_curve = Curve.from_svj(engine_spec.get("torque_curve"))
    engine = None
    if torque_curve is not None:
        engine = Engine(
            torque_curve=torque_curve,
            idle_rpm=num(engine_spec, "idle_rpm", default=800.0) or 800.0,
            max_rpm=num(engine_spec, "max_rpm", default=None) or float(torque_curve.x[-1]),
            inertia=num(engine_spec, "inertia", default=0.15) or 0.15)
    else:
        report.warn("no engine torque curve — no tractive effort will be produced")

    gearbox = clean(pt.get("gearbox") or {})
    ratios = [float(r) for r in (gearbox.get("ratios") or []) if isinstance(r, (int, float))]
    diffs = [clean(d) for d in (pt.get("differentials") or []) if isinstance(d, dict)]
    driven = infer_driven(pt, doc, stations, diffs, report)
    for station in stations:                # the flag follows what the driveline drives
        station.driven = station.name in driven

    # Axle differentials: those not located at the centre. Each goes to the axle
    # it names (`axle_ref`, or `location` front/rear); the first one also stands
    # in for any driven axle that has none of its own, as the single differential
    # of a car always did.
    axle_numbers = sorted({s.axle for s in stations})
    counter = [0]
    axle_entries = [d for d in diffs
                    if str(d.get("location", "")).lower() not in ("center", "inter_axle")]
    parsed: list[tuple[int | None, Differential]] = []
    for entry in axle_entries:
        location = str(entry.get("location", "")).lower()
        axle = (axle_number(entry.get("axle_ref")) if entry.get("axle_ref")
                else axle_numbers[0] if location == "front"
                else axle_numbers[-1] if location == "rear" else None)
        parsed.append((axle if axle in axle_numbers else None,
                       parse_differential(entry, counter, report,
                                          f"differential {entry.get('id') or location or 'axle'}")))
    axle_diffs: dict[int, Differential] = {}
    for axle, diff in parsed:
        if axle is not None and axle not in axle_diffs:
            axle_diffs[axle] = diff
    default_diff = parsed[0][1] if parsed else Differential()
    final_drive = (num(axle_entries[0], "final_drive", default=None) or 1.0) if axle_entries else 1.0
    if axle_entries:
        drives = {round(num(d, "final_drive", default=final_drive) or final_drive, 3)
                  for d in axle_entries}
        if len(drives) > 1:
            report.warn("the axle differentials have different final drives "
                        f"({', '.join(f'{v:g}' for v in sorted(drives))}); SVJSim uses one "
                        f"overall ratio ({final_drive:g}) for the whole driveline")
    centre = parse_centre(pt, diffs, counter, report)
    if not ratios:
        report.warn("no gearbox ratios - treated as a single direct gear")
        ratios = [1.0]

    driven_axles = {s.axle for s in stations if s.name in set(driven)}
    if centre is not None and len(driven_axles) > 2 and centre.kind != "fixed":
        report.warn("a centre differential or coupling is declared on a vehicle with "
                    f"{len(driven_axles)} driven axles; SVJSim models the centre only "
                    "between two, so the torque is shared by wheel count instead")
    if centre is not None and len(driven_axles) < 2 and centre.kind != "fixed":
        report.note("a centre differential or coupling is declared but only one axle is "
                    "driven, so it has no effect")
    if centre is not None and centre.kind == "coupling":
        report.note(
            "centre coupling: " + (f"{centre.mechanism}, " if centre.mechanism else "")
            + f"{centre.ramp_torque:g} to {centre.max_torque:g} Nm. SVJ does not name the "
              "primary axle, so it is taken as the one with the larger torque_split share "
              "(the rear when none is given)")

    driveline = Driveline(
        gear_ratios=ratios, final_drive=final_drive,
        efficiency=num(gearbox, "efficiency", default=0.95) or 0.95,
        driven_stations=driven,
        axle_diffs=axle_diffs, default_diff=default_diff, centre=centre,
        filter_count=counter[0],
        shift_time=num(gearbox, "shift_time", default=0.2) or 0.2,
        inertia=(num(gearbox, "inertia_output", default=0.0) or 0.0)
        + ((num(axle_entries[0], "inertia", default=0.0) or 0.0) if axle_entries else 0.0),
        clutch_capacity=num(pt, "clutch", "max_torque", default=None)
        or _default_clutch_capacity(engine))
    if not driven:
        report.warn("no driven stations could be identified from half_shafts or the layout")
    return engine, driveline


# Topologies where one rigid body spans the axle, so the two wheels cannot
# camber independently. `torsion_beam` is deliberately absent: its beam twists,
# which is closer to two trailing arms with a roll spring between them.
BEAM_TOPOLOGIES = frozenset({"solid_axle", "de_dion"})


def _beam_axle_pairs(names: list[str], suspension: dict[str, Any]) -> dict[int, tuple[str, str]]:
    """Which station names share a rigid axle, keyed by axle number.

    Single source of truth for "is this a beam axle" pairing, used both to
    build a coupled `AxleKinematics` solve (in `build_vehicle`, before
    `Station` objects exist to check track against) and by `_link_beam_axles`
    below (once they do) to set the `beam_*` bookkeeping fields.
    """
    by_axle: dict[int, list[str]] = {}
    for name in names:
        topology = str(dig(suspension.get(name, {}), "topology", "system_type", default=""))
        if topology in BEAM_TOPOLOGIES:
            by_axle.setdefault(_station_axle(name), []).append(name)
    return {axle: (members[0], members[1]) for axle, members in by_axle.items()
            if len(members) == 2}


def _link_beam_axles(stations: list[Station], suspension: dict[str, Any],
                     report: BuildReport) -> None:
    """Pair up the two sides of each rigid axle so their camber stays coupled."""
    index_of = {s.name: i for i, s in enumerate(stations)}
    for axle, (name_a, name_b) in _beam_axle_pairs([s.name for s in stations], suspension).items():
        a, b = stations[index_of[name_a]], stations[index_of[name_b]]
        track = abs(float(a.position0[1] - b.position0[1]))
        if track < 1e-6:
            continue
        a.beam_partner, b.beam_partner = index_of[name_b], index_of[name_a]
        a.beam_track = b.beam_track = track
        a.beam_side = 1.0 if a.side == "L" else -1.0
        b.beam_side = 1.0 if b.side == "L" else -1.0
        report.note(
            f"axle A{axle} is a rigid beam: its two wheels' camber, toe and roll centre "
            f"come from the axle's own solved kinematics, not from each side's own linkage")


def _default_clutch_capacity(engine: Engine | None) -> float:
    """Clutches are sized with margin over peak engine torque; 1.6x is typical."""
    if engine is None:
        return 400.0
    return 1.6 * max(400.0 / 1.6, float(np.max(engine.torque_curve.y)))


def _build_steering(doc: dict[str, Any], stations: Sequence[Station],
                    report: BuildReport) -> Steering:
    spec = clean(dig(doc, "steering", default={}) or {})
    overall = num(spec, "overall_ratio", default=None) or 15.0
    steered = [s for s in stations if s.steered]

    geometric = [s.kc.steer_ratio for s in steered if abs(s.kc.steer_ratio) > 1e-6]
    if geometric:
        rack_per_wheel = 1.0 / float(np.mean(np.abs(geometric)))
    else:
        rack_per_wheel = 0.13 / 0.55
        if steered:
            report.warn("steered stations produced no rack-to-steer gain — using a nominal rack ratio")

    max_rack = min([s.kc.rack[-1] for s in steered if len(s.kc.rack) > 1], default=0.0)
    if max_rack <= 0:
        max_rack = (num(spec, "rack_travel", default=0.13) or 0.13) / 2.0

    declared_max = num(spec, "max_steer_angle", default=None)
    geometric_max = max_rack / max(1e-9, rack_per_wheel)
    if declared_max and abs(geometric_max - declared_max) / declared_max > 0.10:
        report.warn(
            f"steering geometry gives {np.degrees(geometric_max):.1f}° of road wheel lock but the "
            f"file declares {np.degrees(declared_max):.1f}°; the tie-rod hardpoints and "
            f"`rack_travel`/`overall_ratio` disagree")
    return Steering(rack_per_wheel_angle=rack_per_wheel, overall_ratio=overall,
                    max_rack=max_rack,
                    lock_to_lock=num(spec, "lock_to_lock", default=2 * np.pi) or 2 * np.pi)


def static_alignment(doc: dict[str, Any], station: str, y: float) -> tuple[float, float]:
    """A station's static camber and toe, as absolute SAE rotations in radians.

    Read straight from the document's `alignment` block, in whichever sign
    convention the file declares (§9.6). A function of its own so the viewer
    can ask what a *different* document says — the file as loaded, against the
    version with the user's setup edits applied — without building a vehicle.
    """
    corner = clean((doc.get("suspension") or {}).get(station) or {})
    alignment = clean(corner.get("alignment") or {})
    convention = str(dig(doc, "_metadata", "alignment_convention",
                         default="relative_to_centerline"))
    camber_raw = num(alignment, "camber", default=0.0) or 0.0
    toe_raw = num(alignment, "toe", default=0.0) or 0.0
    if convention == "absolute":
        return camber_raw, toe_raw
    return from_svj_camber(camber_raw, y), from_svj_toe(toe_raw, y)


def build_vehicle(source: Any, *, prefer_tire: str | None = None,
                  kc_samples: int = 21) -> SimVehicle:
    """Assemble a `SimVehicle` from an `svj.Vehicle` or a raw SVJ dictionary."""
    doc = getattr(source, "data", source)
    vehicle = source if hasattr(source, "stations") else None
    report = BuildReport()

    suspension = clean(doc.get("suspension") or {})
    names = sorted(suspension.keys(), key=lambda n: (_station_axle(n), _station_side(n)))
    if not names:
        raise ValueError("SVJ document has no suspension stations to simulate")

    chassis = clean(doc.get("chassis") or {})
    total_mass = num(chassis, "mass_total", default=None)
    if total_mass is None:
        raise ValueError("chassis.mass_total is required to simulate a vehicle")
    total_cg = vec3(chassis, "center_of_gravity")
    if total_cg is None:
        raise ValueError("chassis.center_of_gravity is required to simulate a vehicle")
    wheelbase = num(chassis, "wheelbase", default=2.5) or 2.5
    unsprung_spec = clean(chassis.get("mass_unsprung_per_corner") or {})

    axles = {str(a.get("id", "")): clean(a) for a in (doc.get("axles") or [])
             if isinstance(a, dict)}
    steering_spec = clean(doc.get("steering") or {})
    primary_axle = str(steering_spec.get("axle_ref") or "A1")
    extra_steered = {str(a.get("axle_ref", "")) for a in
                     (steering_spec.get("additional_axles") or []) if isinstance(a, dict)}
    rack_travel = (num(steering_spec, "rack_travel", default=0.13) or 0.13) / 2.0

    tire_sets = clean(dig(doc, "tires", "sets", default={}) or {})

    # -- first pass: kinematics, so motion ratios are known before springs ----
    solvers: dict[str, StationKinematics] = {}
    tables: dict[str, KCTable] = {}
    axle_roll_centre_z: dict[str, float] = {}

    def _make_solver(name: str) -> tuple[StationKinematics, bool]:
        corner = clean(suspension[name])
        axle_id = f"A{_station_axle(name)}"
        declared_steered = bool(axles.get(axle_id, {}).get("steered"))
        is_front = _station_axle(name) == min(_station_axle(n) for n in names)
        steered = declared_steered or axle_id in extra_steered or \
            (axle_id == primary_axle and is_front and not axles)
        return StationKinematics(name, corner, samples=kc_samples,
                                 rack_limit=rack_travel if steered else 0.0), steered

    beam_pairs = _beam_axle_pairs(names, suspension)
    paired_with: dict[str, str] = {}
    for name_a, name_b in beam_pairs.values():
        paired_with[name_a] = name_b
        paired_with[name_b] = name_a

    handled: set[str] = set()
    for name in names:
        if name in handled:
            continue
        partner = paired_with.get(name)
        if partner is not None:
            # SVJ doesn't say which of a pair's two data structures "is" the
            # left one; sort by actual side so AxleKinematics's L/R roles
            # match the real stations regardless of file/dict order.
            left_name, right_name = ((name, partner) if _station_side(name) == "L"
                                     else (partner, name))
            left_solver, left_steered = _make_solver(left_name)
            right_solver, right_steered = _make_solver(right_name)
            if left_steered or right_steered:
                report.warn(
                    f"{left_name}/{right_name}: steering on a rigid beam axle is not "
                    f"simulated yet — treated as unsteered")
            axle = AxleKinematics(left_solver, right_solver)
            left_table, right_table, roll_centre_z = axle.build_tables()
            tables[left_name], tables[right_name] = left_table, right_table
            solvers[left_name], solvers[right_name] = left_solver, right_solver
            axle_roll_centre_z[left_name] = axle_roll_centre_z[right_name] = roll_centre_z
            report.extend(left_solver.warnings + right_solver.warnings + axle.warnings)
            handled.add(left_name)
            handled.add(right_name)
            continue
        solver, _ = _make_solver(name)
        tables[name] = solver.build_table()
        solvers[name] = solver
        report.extend(solver.warnings)

    # -- masses ---------------------------------------------------------------
    stations: list[Station] = []
    for name in names:
        corner = clean(suspension[name])
        table = tables[name]
        wheel = clean(corner.get("wheel") or {})
        multiplicity = int(num(wheel, "multiplicity", default=1) or 1)

        unsprung = num(unsprung_spec, name, default=None)
        if unsprung is None:
            unsprung = (num(wheel, "mass", default=None) or 20.0) + 15.0
            # SVJ §9.2.4: axle_body.mass "MUST be included (split 50/50) in
            # mass_unsprung_per_corner" -- a file that already declares its
            # own mass_unsprung_per_corner is trusted to have done that
            # itself; this only applies when SVJSim is estimating instead,
            # so the axle housing isn't silently missing from an estimate.
            axle_mass = num(corner, "topology", "axle_body", "mass", default=None)
            if axle_mass:
                unsprung += axle_mass / 2.0
            report.estimate(
                f"{name}: no unsprung mass given — assumed {unsprung:.0f} kg" +
                (f" (including half the {axle_mass:.0f} kg axle body)" if axle_mass else ""))

        spin = num(wheel, "rotational_inertia", default=None)
        if spin is None:
            radius = 0.33
            spin = 0.5 * (num(wheel, "mass", default=20.0) or 20.0) * radius ** 2
            report.estimate(f"{name}: no wheel spin inertia — estimated {spin:.2f} kg·m²")
        brake_disc_inertia = num(corner, "brake", "disc", "rotational_inertia", default=0.0) or 0.0

        y = float(table.wheel_center0[1])
        static_camber, static_toe = static_alignment(doc, name, y)

        axle_id = f"A{_station_axle(name)}"
        driven = bool(axles.get(axle_id, {}).get("driven")) or \
            name in clean(dig(doc, "powertrain", "half_shafts", default={}) or {})

        stations.append(Station(
            name=name, axle=_station_axle(name), side=_station_side(name),
            position0=table.wheel_center0, unsprung_mass=unsprung,
            spin_inertia=spin + brake_disc_inertia, kc=table,
            tire=None, spring=None, damper=None, brake=None,
            bump_stop=_build_stops(corner, "bump_stop"),
            rebound_stop=_build_stops(corner, "rebound_stop"),
            static_camber=static_camber, static_toe=static_toe,
            steered=solvers[name].steered, driven=driven,
            wheel_count=max(1, multiplicity),
            # Tire data is fitted from one wheel on a test rig, so its odd terms
            # — ply steer and conicity — carry that wheel's handedness. Fitting
            # them unmirrored to both sides of a car makes it pull to one side
            # while standing still. Left-hand stations evaluate the mirrored
            # tire, which is how the same casing behaves fitted the other way up.
            mirror_tire=_station_side(name) == "L",
            camber_ref=table.camber_at(0.0, 0.0), toe_ref=table.toe_at(0.0, 0.0),
            beam_roll_centre_z=axle_roll_centre_z.get(name)))

    sprung_mass = total_mass - sum(s.unsprung_mass for s in stations)
    if sprung_mass <= 0:
        raise ValueError(f"unsprung masses ({total_mass - sprung_mass:.0f} kg) exceed "
                         f"mass_total ({total_mass:.0f} kg)")

    # Sprung CG follows from the total CG once the unsprung masses are removed.
    moment = total_mass * total_cg - sum(s.unsprung_mass * s.position0 for s in stations)
    sprung_cg = moment / sprung_mass
    sprung_inertia = _sprung_inertia(doc, stations, total_mass, total_cg,
                                     sprung_mass, sprung_cg, report)

    # -- static loads, then everything that depends on them -------------------
    stiffness_guess = []
    for s in stations:
        corner = clean(suspension[s.name])
        rate = num(corner, "spring", "rate", default=None) or 25000.0
        stiffness_guess.append((s.name, s.position0, rate))

    vehicle_class = str(dig(doc, "vehicle_info", "vehicle_class", default="")).lower()
    is_trailer = "trailer" in vehicle_class
    if is_trailer:
        report.note(
            "this is a trailer: part of its weight rests on a king pin or drawbar that SVJ does "
            "not describe yet (§23.8), so load is distributed across its own axles only and "
            "longitudinal balance is not enforced")
    elif total_cg[0] > 0.0:
        report.warn(
            f"centre of gravity is {total_cg[0]:.2f} m *ahead* of axle 1. SVJ puts the origin at "
            f"axle 1 with X forward, so a vehicle's CG.x must be negative — this looks like a "
            f"sign error in `chassis.center_of_gravity`")
    if total_cg[2] > 0.0:
        report.warn(
            f"centre of gravity is at Z = {total_cg[2]:+.2f} m, which is *below* ground in SAE "
            f"J670 (Z points down). Expect a sign error in `chassis.center_of_gravity`")

    loads = _static_loads(stiffness_guess, total_mass, total_cg, report,
                          balance_pitch=not is_trailer)

    brake_system, master_area = _build_brake_system(doc, report)
    for s in stations:
        corner = clean(suspension[s.name])
        table = s.kc
        s.static_load = loads[s.name]

        tire_spec = clean(corner.get("tire") or {})
        set_ref = str(tire_spec.get("set_ref", ""))
        tire_set = tire_sets.get(set_ref, {})
        if set_ref and set_ref not in tire_sets:
            report.warn(f"{s.name}: tire set '{set_ref}' is not in tires.sets")
        if not tire_set:
            report.estimate(f"{s.name}: no tire set referenced — using tire size from the wheel")
        s.tire = build_tire(set_ref or f"{s.name}_tire", tire_set, wheel=corner.get("wheel"),
                            pressure_override=num(tire_spec, "pressure", default=None),
                            reference_load=s.static_load / max(1, s.wheel_count),
                            prefer=prefer_tire)
        if s.tire.notes:
            report.extend(s.tire.notes, "estimates")

        spring_mr = table.spring_motion_ratio()
        damper_mr = table.damper_motion_ratio()
        s.spring = _build_spring(corner, report, s.name, spring_mr)
        # The spring carries the sprung share only: the tire holds up the
        # unsprung mass directly, below the spring.
        s.spring.static_force = s.static_load - s.unsprung_mass * G
        s.damper = _build_damper(corner, report, s.name, damper_mr)
        s.brake = _build_brake(corner, master_area)

        arb = clean(corner.get("arb") or {})
        if arb:
            rate = num(arb, "wheel_rate_equiv", default=None)
            if rate is None:
                report.warn(
                    f"{s.name}: anti-roll bar gives `bar_rate` but no `wheel_rate_equiv`, and the "
                    f"lever length cannot be derived from the hardpoints — the bar is ignored")
                rate = 0.0
            s.arb_rate = rate
            partner = [o for o in stations
                       if o.axle == s.axle and o.name != s.name]
            s.arb_partner = partner[0].name if partner else None

    _link_beam_axles(stations, suspension, report)
    steering = _build_steering(doc, stations, report)
    aero = _build_aero(doc, report, wheelbase, total_cg)
    engine, driveline = _build_powertrain(doc, stations, report)

    name = " ".join(str(x) for x in (
        dig(doc, "vehicle_info", "make", default=""),
        dig(doc, "vehicle_info", "model", default=""),
        dig(doc, "vehicle_info", "year", default="")) if x).strip() or "Unnamed vehicle"

    sim = SimVehicle(
        name=name, stations=stations, sprung_mass=sprung_mass, sprung_cg=sprung_cg,
        sprung_inertia=sprung_inertia, total_mass=total_mass, total_cg=total_cg,
        wheelbase=wheelbase, steering=steering, aero=aero, brakes=brake_system,
        engine=engine, driveline=driveline, report=report,
        benchmarks=[clean(b) for b in (doc.get("benchmarks") or []) if isinstance(b, dict)],
        validation=clean(doc.get("validation") or {}), source=doc)

    _sanity_check(sim, report)
    report.dedupe()
    return sim


def _sprung_inertia(doc: dict[str, Any], stations: Sequence[Station], total_mass: float,
                    total_cg: np.ndarray, sprung_mass: float, sprung_cg: np.ndarray,
                    report: BuildReport) -> np.ndarray:
    """Inertia of the sprung mass about its own CG.

    Preferred route is summing `mass_bodies`, which SVJ defines as the sprung
    decomposition. Otherwise the vehicle inertia has the unsprung masses removed
    from it, which is only as good as the tensor the file supplies.
    """
    bodies = [clean(b) for b in (dig(doc, "chassis", "mass_bodies", default=[]) or [])
              if isinstance(b, dict)]
    if bodies:
        total = np.zeros((3, 3))
        mass_sum = 0.0
        for body in bodies:
            mass = num(body, "mass", default=None)
            position = vec3(body, "position")
            if mass is None or position is None:
                continue
            own = inertia_matrix(body.get("inertia") or {})
            total += parallel_axis(own, mass, position - sprung_cg)
            mass_sum += mass
        if mass_sum > 0 and np.trace(total) > 0:
            if abs(mass_sum - sprung_mass) / sprung_mass > 0.05:
                report.warn(
                    f"mass_bodies sum to {mass_sum:.0f} kg but mass_total minus unsprung is "
                    f"{sprung_mass:.0f} kg ({abs(mass_sum / sprung_mass - 1) * 100:.0f}% apart); "
                    f"inertia is scaled to the sprung mass")
            return total * (sprung_mass / mass_sum)

    vehicle_inertia = inertia_matrix(dig(doc, "chassis", "inertia", default={}) or {})
    if np.trace(vehicle_inertia) <= 0:
        # Empirical fallback: a car's yaw inertia is close to m*a*b (Ixx and Iyy
        # scaled from typical passenger-car ratios).
        wheelbase = num(doc, "chassis", "wheelbase", default=2.5) or 2.5
        track = num(doc, "chassis", "track_front", default=1.5) or 1.5
        izz = sprung_mass * (0.42 * wheelbase) ** 2
        vehicle_inertia = np.diag([sprung_mass * (0.35 * track) ** 2, izz * 0.95, izz])
        report.estimate("no inertia tensor — estimated from wheelbase and track "
                        "(radius of gyration 0.42 x wheelbase in yaw)")
        return vehicle_inertia

    # Remove each unsprung mass's contribution about the total CG, then shift.
    sprung_about_total = vehicle_inertia.copy()
    for s in stations:
        sprung_about_total -= parallel_axis(np.zeros((3, 3)), s.unsprung_mass,
                                            s.position0 - total_cg)
    shift = sprung_cg - total_cg
    result = sprung_about_total - sprung_mass * (float(shift @ shift) * np.eye(3)
                                                 - np.outer(shift, shift))
    if np.any(np.linalg.eigvalsh(result) <= 0):
        report.warn("removing the unsprung masses left a non-physical sprung inertia tensor; "
                    "the vehicle inertia in the file is too small for its unsprung mass. "
                    "Using the vehicle tensor unmodified")
        return vehicle_inertia
    return result


def _sanity_check(sim: SimVehicle, report: BuildReport) -> None:
    """Flag physically implausible results before anyone runs a maneuver on them."""
    for s in sim.stations:
        if np.isnan(sim.sprung_corner_mass(s)):
            report.warn(f"{s.name}: carries no sprung load at rest, so ride frequency and "
                        f"damping cannot be evaluated there")
            continue
        f = sim.ride_frequency(s)
        if not 0.4 < f < 4.0:
            report.warn(f"{s.name}: ride frequency {f:.2f} Hz is outside the plausible "
                        f"0.4–4 Hz range — check the spring rate and unsprung mass")
        zeta = sim.damping_ratio(s)
        if not 0.10 < zeta < 1.20:
            report.warn(f"{s.name}: damping ratio {zeta:.2f} at 0.1 m/s is outside the "
                        f"0.10–1.20 a road or race car runs — check the damper curves "
                        f"and whether they are damper-shaft or wheel referenced")
    total = sum(s.static_load for s in sim.stations)
    expected = sim.total_mass * G
    if abs(total - expected) / expected > 0.02:
        report.warn(f"static loads sum to {total:.0f} N against a weight of {expected:.0f} N")
