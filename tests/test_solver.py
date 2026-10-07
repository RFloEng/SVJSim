"""Physics checks on the assembled vehicle: does it obey the laws it should?

These are the tests that would catch a sign error or a missing term — the class
of bug that still produces plausible-looking numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

from svjsim.solver import Controls, State, VehicleDynamics, step_rk4

G = 9.80665


# ── Building the model ────────────────────────────────────────────────

def test_mass_decomposition_adds_up(vehicle):
    assert vehicle.sprung_mass + vehicle.unsprung_mass == pytest.approx(vehicle.total_mass)


def test_sprung_cg_is_consistent_with_the_total(vehicle):
    """Sprung and unsprung masses about their own centres must recover the total."""
    moment = vehicle.sprung_mass * vehicle.sprung_cg + sum(
        s.unsprung_mass * s.position0 for s in vehicle.stations)
    assert moment / vehicle.total_mass == pytest.approx(vehicle.total_cg, abs=1e-9)


def test_static_loads_carry_the_weight_and_balance_the_cg(vehicle):
    loads = np.array([s.static_load for s in vehicle.stations])
    assert loads.sum() == pytest.approx(vehicle.total_mass * G, rel=1e-6)
    x = np.array([s.position0[0] for s in vehicle.stations])
    assert float(loads @ x / loads.sum()) == pytest.approx(vehicle.total_cg[0], abs=1e-6)


def test_a_symmetric_car_has_symmetric_corner_loads(vehicle):
    for axle in {s.axle for s in vehicle.stations}:
        pair = [s for s in vehicle.stations if s.axle == axle]
        assert pair[0].static_load == pytest.approx(pair[1].static_load, rel=1e-9)


def test_front_carries_more_than_rear_for_a_forward_cg(vehicle):
    front = sum(s.static_load for s in vehicle.front_stations)
    rear = sum(s.static_load for s in vehicle.rear_stations)
    assert front > rear


def test_damping_ratio_matches_the_analytic_quarter_car(vehicle):
    station = vehicle.stations[0]
    mass = vehicle.sprung_corner_mass(station)
    k_tire = station.tire.vertical_stiffness
    k = station.spring.rate * k_tire / (station.spring.rate + k_tire)
    expected = station.damper.equivalent_rate() / (2.0 * np.sqrt(k * mass))
    assert vehicle.damping_ratio(station) == pytest.approx(expected, rel=1e-9)
    assert vehicle.ride_frequency(station) == pytest.approx(
        np.sqrt(k / mass) / (2 * np.pi), rel=1e-9)


def test_benchmarks_are_carried_through_from_the_file(vehicle):
    assert vehicle.benchmark("accel_0_100kph")["value"] == 6.0
    assert vehicle.benchmark("nonexistent") is None


# ── Static equilibrium ────────────────────────────────────────────────

@pytest.fixture(scope="module")
def trimmed(request):
    from svjsim.model import build_vehicle
    from svjsim.solver import VehicleDynamics
    from conftest import svj_document
    vehicle = build_vehicle(svj_document())
    dynamics = VehicleDynamics(vehicle)
    return dynamics, dynamics.trim(speed=0.0, duration=2.0)


def test_trim_reaches_equilibrium(trimmed):
    dynamics, state = trimmed
    derivative, _ = dynamics.derivatives(state, Controls(gear=0, clutch=0.0))
    # Residual accelerations should be a rounding error, not a drift.
    assert np.max(np.abs(derivative[6:12])) < 0.02


def test_trim_puts_the_right_load_on_every_tire(trimmed):
    dynamics, state = trimmed
    _, wheels = dynamics.derivatives(state, Controls(gear=0, clutch=0.0), collect=True)
    for wheel, station in zip(wheels, dynamics.stations):
        assert wheel.Fz == pytest.approx(station.static_load, rel=0.01)
    total = sum(w.Fz for w in wheels)
    assert total == pytest.approx(dynamics.vehicle.total_mass * G, rel=1e-3)


def test_a_symmetric_car_does_not_roll_or_pull_at_rest(trimmed):
    dynamics, state = trimmed
    assert abs(float(state.angles[0])) < 1e-6, "no roll"
    _, wheels = dynamics.derivatives(state, Controls(gear=0, clutch=0.0), collect=True)
    assert abs(sum(w.Fy for w in wheels)) < 1.0, "no net side force"
    assert abs(sum(w.Fx for w in wheels)) < 1.0, "no net drive or drag force"


def test_camber_thrust_cancels_across_an_axle(trimmed):
    """Static negative camber makes side force on both wheels; it must cancel."""
    dynamics, state = trimmed
    _, wheels = dynamics.derivatives(state, Controls(gear=0, clutch=0.0), collect=True)
    by_axle: dict[int, list[float]] = {}
    for wheel, station in zip(wheels, dynamics.stations):
        by_axle.setdefault(station.axle, []).append(wheel.Fy)
    for forces in by_axle.values():
        assert sum(forces) == pytest.approx(0.0, abs=1.0)
        assert abs(forces[0]) > 1.0, "there should be camber thrust to cancel"


# ── Dynamic behaviour ─────────────────────────────────────────────────

def _run(dynamics: VehicleDynamics, state: State, controls: Controls,
         duration: float, dt: float = 0.002) -> State:
    for _ in range(int(duration / dt)):
        state = step_rk4(dynamics, state, controls, dt)
    return state


def test_steering_left_and_right_mirror_each_other(dynamics):
    base = dynamics.trim(speed=20.0, gear=3, duration=1.2)
    right = _run(dynamics, base.copy(), Controls(steering=0.35, gear=3), 2.0)
    left = _run(dynamics, base.copy(), Controls(steering=-0.35, gear=3), 2.0)
    assert float(right.omega[2]) == pytest.approx(-float(left.omega[2]), rel=1e-3)
    assert float(right.angles[0]) == pytest.approx(-float(left.angles[0]), rel=1e-3)
    assert float(right.velocity[1]) == pytest.approx(-float(left.velocity[1]), rel=1e-2)


def test_steering_right_yaws_right_and_transfers_load_left(dynamics):
    state = dynamics.trim(speed=20.0, gear=3, duration=1.2)
    state = _run(dynamics, state, Controls(steering=0.5, gear=3), 2.0)
    _, wheels = dynamics.derivatives(state, Controls(steering=0.5, gear=3), collect=True)
    loads = {w.name: w.Fz for w in wheels}
    assert float(state.omega[2]) > 0, "positive steer must give positive yaw rate"
    assert loads["FL"] > loads["FR"], "load transfers to the outside of the turn"
    assert loads["RL"] > loads["RR"]
    # Rolling outward about +X means the outside (left) drops: negative roll.
    assert float(state.angles[0]) < 0


def test_braking_transfers_load_to_the_front(dynamics):
    state = dynamics.trim(speed=25.0, gear=4, duration=1.2)
    controls = Controls(brake=0.6, gear=0, clutch=0.0)
    before = dynamics.derivatives(state, controls, collect=True)[1]
    state = _run(dynamics, state, controls, 1.0)
    after = dynamics.derivatives(state, controls, collect=True)[1]

    front_before = sum(w.Fz for w, s in zip(before, dynamics.stations) if s.axle == 1)
    front_after = sum(w.Fz for w, s in zip(after, dynamics.stations) if s.axle == 1)
    assert front_after > front_before
    assert float(state.velocity[0]) < 25.0, "the vehicle must slow down"
    assert float(state.angles[1]) > 0 or front_after > front_before, "the nose dives"


def test_throttle_accelerates_and_squats_the_rear(dynamics):
    state = dynamics.trim(speed=10.0, gear=2, duration=1.2)
    controls = Controls(throttle=0.8, gear=2)
    state = _run(dynamics, state, controls, 1.5)
    assert float(state.velocity[0]) > 10.0
    _, wheels = dynamics.derivatives(state, controls, collect=True)
    rear = sum(w.Fz for w, s in zip(wheels, dynamics.stations) if s.axle == 2)
    front = sum(w.Fz for w, s in zip(wheels, dynamics.stations) if s.axle == 1)
    static_rear = sum(s.static_load for s in dynamics.vehicle.rear_stations)
    assert rear > static_rear, "accelerating must load the driven axle"
    assert front + rear == pytest.approx(dynamics.vehicle.total_mass * G, rel=0.06)


def test_launch_from_a_standstill_actually_builds_speed(dynamics):
    """A standing-start launch must not be swamped by low-speed damping.

    Regression test: the low-speed numerical damping added at the contact
    patch to stop a parked car creeping (see `V_BODY_DAMP_BLEND` in
    solver.py) once shared its fade-out distance with the wheel-spin damping
    term's much wider window. Sized for the whole vehicle's mass rather than
    a wheel's own inertia, at that width it dragged on *every* wheel —
    including the two that were not even driven — all the way up to 18 km/h,
    comfortably exceeding the driven wheels' actual tractive force and
    making the car nearly unable to move under full throttle. Every existing
    throttle test happened to start already above that speed, so none of
    them caught it.
    """
    state = dynamics.trim(speed=0.0, gear=0, duration=1.0)
    controls = Controls(throttle=1.0, gear=1)
    state = _run(dynamics, state, controls, 2.0)
    speed_kph = float(state.velocity[0]) * 3.6
    assert speed_kph > 15.0, (
        f"full throttle from a stop reached only {speed_kph:.1f} km/h in 2 s — "
        f"low-speed damping may be swamping the driven wheels' tractive force")


def test_coasting_is_only_slowed_by_drag_and_rolling_resistance(dynamics):
    state = dynamics.trim(speed=30.0, gear=0, duration=1.2)
    controls = Controls(gear=0, clutch=0.0)
    state = _run(dynamics, state, controls, 3.0)
    lost = 30.0 - float(state.velocity[0])
    assert 0.0 < lost < 4.0, f"lost {lost:.2f} m/s coasting for 3 s, which is implausible"
    assert abs(float(state.position[1])) < 0.2, "a symmetric car must coast straight"


def test_the_vehicle_stays_on_the_ground_and_finite(dynamics):
    state = dynamics.trim(speed=20.0, gear=3, duration=1.2)
    controls = Controls(steering=0.6, throttle=0.5, gear=3)
    for _ in range(1500):
        state = step_rk4(dynamics, state, controls, 0.002)
        assert np.all(np.isfinite(state.data))
    _, wheels = dynamics.derivatives(state, controls, collect=True)
    assert all(w.Fz >= 0 for w in wheels), "a tire can never pull the car down"


def _solid_axle_corner(y: float, *, panhard: bool, x: float = -2.60) -> dict:
    """A real solid-axle corner: trailing links + a Panhard rod on one side.

    The trailing links' inboard and outboard points share the same y, so
    each rod's own axis has zero y-component and (to first order) adds
    negligible sensitivity to a y-perturbation of the axle -- they locate it
    longitudinally without measurably fighting the Panhard rod for lateral
    location. That is what lets the classic "roll centre sits on the
    Panhard rod's own line" hand construction be checked against the solver
    to good precision, the same way test_kinematics.py's
    test_axle_roll_centre_matches_panhard_rod_line does at the lower level.
    """
    hardpoints = {
        "wheel_center": [x, y, -0.35],
        "upper_link_end": [x, y, -0.45],
        "lower_link_end": [x, y, -0.20],
    }
    links = [
        {"name": "upper_link", "type": "rod",
         "inboard_points": [[x + 1.0, y, -0.45]], "outboard_ref": "hardpoints.upper_link_end"},
        {"name": "lower_link", "type": "rod",
         "inboard_points": [[x + 1.0, y, -0.20]], "outboard_ref": "hardpoints.lower_link_end"},
    ]
    if panhard:
        hardpoints["panhard_axle_end"] = [x, -0.775, -0.40]
        links.append({"name": "panhard_rod", "type": "rod",
                     "inboard_points": [[x, 0.05, -0.30]],
                     "outboard_ref": "hardpoints.panhard_axle_end"})
    return {
        "topology": {"system_type": "solid_axle", "upright": {"hardpoints": hardpoints},
                     "links": links, "axle_body": {"id": "rear_axle", "mass": 70.0,
                                                    "position": [x, 0.0, -0.32]}},
        "wheel": {"rim_diameter": 0.4318, "rim_width": 0.1905, "mass": 21.0,
                  "rotational_inertia": 0.85, "set_ref": "road_225_45r17"},
        "spring": {"type": "coil", "rate": 30000.0, "motion_ratio": 0.75,
                   "inboard_mount": [x, y * 0.4, -0.60], "outboard_mount": [x, y, -0.20]},
        "damper": {"type": "monotube", "motion_ratio": 0.75,
                   "bump_curve": [[0.0, 0.0], [0.1, 220.0], [0.5, 700.0]],
                   "rebound_curve": [[0.0, 0.0], [0.1, 320.0], [0.5, 1000.0]],
                   "inboard_mount": [x, y * 0.4, -0.60], "outboard_mount": [x, y, -0.20]},
        "alignment": {"camber": 0.0, "toe": 0.0},
        "bump_stop": {"gap": 0.06, "rate": 250000.0},
        "rebound_stop": {"gap": 0.07, "rate": 60000.0},
        "brake": {
            "disc": {"type": "vented", "outer_diameter": 0.30, "inner_diameter": 0.20,
                     "effective_radius": 0.128, "mass": 7.0, "specific_heat": 460.0,
                     "rotational_inertia": 0.10},
            "caliper": {"type": "fixed", "pistons": 4, "piston_area_total": 0.0032},
            "pad": {"material": "semi_metallic", "mu_cold": 0.40},
        },
        "tire": {"set_ref": "road_225_45r17", "pressure": 230000.0},
    }


def test_beam_axle_camber_follows_axle_roll_not_wheel_travel(document):
    """A rigid axle cannot camber one wheel without rolling the whole axle."""
    from svjsim.model import build_vehicle

    document["suspension"]["RL"] = _solid_axle_corner(-0.775, panhard=True)
    document["suspension"]["RR"] = _solid_axle_corner(0.775, panhard=False)
    vehicle = build_vehicle(document)
    left = vehicle.station("RL")
    right = vehicle.station("RR")
    assert left.is_beam_axle and right.is_beam_axle
    assert left.camber_gain == 0.0, "pure bump must not camber a beam axle"

    # Now genuinely solved, from both sides' links and the Panhard rod
    # together -- not the None the old single-station-only version returned.
    rc = left.roll_centre_height()
    assert rc is not None
    # Hand construction: the classic Panhard-rod roll centre sits on the
    # rod's own line, at the height where it crosses the vehicle centreline.
    y_in, z_in, y_out, z_out = 0.05, -0.30, -0.775, -0.40
    slope = (z_out - z_in) / (y_out - y_in)
    expected_z = z_in - y_in * slope                      # body-frame Z
    ground_z = float(left.position0[2]) + left.tire.unloaded_radius
    assert rc == pytest.approx(ground_z - expected_z, abs=2e-3)
    assert right.roll_centre_height() == pytest.approx(rc, abs=1e-6), \
        "one roll centre for the whole axle, not one per side"

    # Both wheels rising together (pure bump): the axle does not roll.
    sample_bump = left.kc.sample_all(0.03, 0.03)
    camber_bump, _ = left.resolve(sample_bump)
    assert camber_bump == pytest.approx(left.static_camber, abs=1e-6)

    # One wheel up and the other down by the same amount (roll): both wheels
    # take the same absolute camber, since they are rigid extensions of the
    # one solved axle body -- this is now read straight from the table
    # (second axis = partner jounce, per Station.resolve's docstring), not
    # approximated from a `(jL - jR) / track` formula.
    camber_roll, _ = left.resolve(left.kc.sample_all(0.03, -0.03))
    camber_other, _ = right.resolve(right.kc.sample_all(-0.03, 0.03))
    assert camber_roll != pytest.approx(left.static_camber, abs=1e-4), \
        "a real roll angle should show up as real camber"
    assert camber_other == pytest.approx(camber_roll, abs=1e-6)

    # The vehicle still simulates: a beam axle's coupled solve must not make
    # the dynamics blow up.
    dynamics = VehicleDynamics(vehicle)
    state = dynamics.trim(speed=20.0, gear=2, duration=1.0)
    controls = Controls(throttle=0.3, gear=2, clutch=1.0)
    for _ in range(200):
        state = step_rk4(dynamics, state, controls, 0.002)
        assert np.all(np.isfinite(state.data))


def test_lateral_grip_is_bounded_by_the_tire(dynamics):
    """The car cannot pull more lateral g than its tires can make."""
    state = dynamics.trim(speed=25.0, gear=3, duration=1.2)
    peak = 0.0
    controls = Controls(steering=1.2, throttle=0.25, gear=3)
    for _ in range(1200):
        state = step_rk4(dynamics, state, controls, 0.002)
        derivative, _ = dynamics.derivatives(state, controls)
        ay = derivative[7] + state.velocity[0] * state.omega[2]
        peak = max(peak, abs(ay) / G)
    assert 0.5 < peak < 1.6, f"peak lateral acceleration {peak:.2f} g is implausible"
