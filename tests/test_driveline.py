"""Driven wheels, differentials, the centre device and lock control (SVJ v0.99.3)."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from svjsim.driveline import (DrivePlan, Differential, LockChain, LockStage, pair_torques,
                              parse_lock_control)
from svjsim.model import BuildReport, build_vehicle
from svjsim.solver import Controls, State, VehicleDynamics, step_rk4
from svjsim.util import Curve


def _plan(vehicle) -> DrivePlan:
    return VehicleDynamics(vehicle).drive


def _awd(document, **transfer_case):
    """The reference car with both axles driven and an optional centre device."""
    doc = copy.deepcopy(document)
    pt = doc["powertrain"]
    pt["layout"] = "AWD"
    pt.pop("half_shafts")
    pt["differentials"] = [
        {"id": "diff_front", "location": "front", "type": "open", "final_drive": 3.45},
        {"id": "diff_rear", "location": "rear", "type": "open", "final_drive": 3.45},
    ]
    if transfer_case:
        pt["transfer_case"] = transfer_case
    return doc


def _state(vehicle, spins, speed=20.0):
    state = VehicleDynamics(vehicle).initial_state(speed=speed, gear=3)
    state.spin[:] = spins
    return state


# ── pair_torques ──────────────────────────────────────────────────────

def test_pair_torques_matches_the_original_limited_slip_formula():
    share, delta, lock, preload = 600.0, 0.7, 0.35, 50.0
    a, b = pair_torques(share, share, delta, lock, preload)
    transfer = lock * share * np.tanh(delta * 2.0) + preload * np.tanh(delta * 10.0)
    assert (a, b) == pytest.approx((share - transfer, share + transfer))


def test_the_faster_output_loses_torque_and_the_total_is_kept():
    a, b = pair_torques(500.0, 500.0, 2.0, 0.5, 0.0)       # a is faster
    assert a < 500.0 < b
    assert a + b == pytest.approx(1000.0)


def test_a_lock_cannot_push_the_donor_negative_on_an_unequal_split():
    a, b = pair_torques(200.0, 800.0, 5.0, 1.0, 0.0)       # a faster, small share
    assert a >= -1e-9 and a + b == pytest.approx(1000.0)


# ── which wheels are driven ───────────────────────────────────────────

@pytest.mark.parametrize("layout, expected", [
    ("FF", {"FL", "FR"}), ("FR", {"RL", "RR"}), ("MR", {"RL", "RR"}),
    ("AWD", {"FL", "FR", "RL", "RR"}), ("4WD", {"FL", "FR", "RL", "RR"})])
def test_driven_wheels_come_from_the_layout_when_nothing_names_them(document, layout, expected):
    document["powertrain"].pop("half_shafts")
    document["powertrain"]["layout"] = layout
    vehicle = build_vehicle(document)
    assert set(vehicle.driveline.driven_stations) == expected
    assert any("taken from the layout" in n for n in vehicle.report.notes)


def test_half_shafts_still_win_over_the_layout(document):
    document["powertrain"]["layout"] = "AWD"
    assert set(build_vehicle(document).driveline.driven_stations) == {"RL", "RR"}


def test_driven_wheels_fall_back_to_where_the_differentials_sit(document):
    document["powertrain"].pop("half_shafts")
    document["powertrain"].pop("layout")
    document["vehicle_info"].pop("drive_type")
    assert set(build_vehicle(document).driveline.driven_stations) == {"RL", "RR"}


def test_the_axle_differential_is_not_mistaken_for_an_inter_axle_one(document):
    """An inter-axle differential has no final drive; taking it for the axle's
    left every such truck with a final drive of 1."""
    document["powertrain"]["differentials"].insert(
        0, {"id": "inter", "location": "inter_axle", "type": "lsd_viscous"})
    assert build_vehicle(document).driveline.final_drive == pytest.approx(3.45)


# ── lock control ──────────────────────────────────────────────────────

def _stage(input, curve, add=False, tau=0.0, lo=None, hi=None, index=-1):
    return LockStage(input=input, add=add, curve=Curve(curve), tau=tau, lo=lo, hi=hi,
                     filter_index=index)


INPUTS = {"gear": 1.0, "brake": 0.0, "throttle": 1.0, "speed": 10.0, "rear_speed_ratio": 1.0,
          "oversteer_factor": 0.0, "slip_angle_front": 0.0, "slip_angle_rear": 0.0}


def test_a_lock_chain_runs_from_each_static_lock_separately():
    """The spec's own example (0.4 / 0.2, gear then brake multipliers) only makes
    sense if the chain starts from lock_power and from lock_coast in turn."""
    diff = Differential(kind="lsd_clutch", lock_power=0.4, lock_coast=0.2, chains=[LockChain(
        "lock", [_stage("gear", [[1, 1.0], [2, 0.8], [3, 0.6]]),
                 _stage("brake", [[0, 1], [1, 0]])])])
    power, coast, _ = diff.resolve({**INPUTS, "gear": 2.0}, None, None)
    assert (power, coast) == pytest.approx((0.4 * 0.8, 0.2 * 0.8))
    power, coast, _ = diff.resolve({**INPUTS, "gear": 2.0, "brake": 1.0}, None, None)
    assert (power, coast) == pytest.approx((0.0, 0.0))


def test_the_running_value_may_leave_the_range_but_the_result_may_not():
    diff = Differential(kind="lsd_clutch", lock_power=0.4, chains=[LockChain(
        "lock_power", [_stage("gear", [[1, 12.8], [2, 9.9]])])])
    assert diff.resolve(INPUTS, None, None)[0] == 1.0          # 0.4 * 12.8, clamped on exit


def test_stage_clamps_act_after_their_own_stage():
    diff = Differential(kind="lsd_clutch", lock_power=0.5, chains=[LockChain(
        "lock_power", [_stage("speed", [[0, 0.0], [20, 1.0]], add=True, hi=0.7),
                       _stage("throttle", [[0, 0.0], [1, 1.0]])])])
    # 0.5 + 0.5 = 1.0 -> clamped to 0.7 by the first stage -> times throttle 1.0
    assert diff.resolve(INPUTS, None, None)[0] == pytest.approx(0.7)


def test_separate_power_and_coast_chains_are_independent():
    diff = Differential(kind="lsd_clutch", lock_power=0.35, lock_coast=0.15, chains=[
        LockChain("lock_power", [_stage("throttle", [[0, 0.5], [1, 1.0]])]),
        LockChain("lock_coast", [_stage("speed", [[0, 1.0], [40, 0.6]])])])
    power, coast, _ = diff.resolve({**INPUTS, "throttle": 0.0, "speed": 40.0}, None, None)
    assert (power, coast) == pytest.approx((0.35 * 0.5, 0.15 * 0.6))


def test_preload_is_a_torque_and_is_not_clamped_to_one():
    diff = Differential(kind="lsd_clutch", preload=50.0, chains=[LockChain(
        "preload", [_stage("throttle", [[0, 1.0], [1, 3.0]])])])
    assert diff.resolve(INPUTS, None, None)[2] == pytest.approx(150.0)


def test_open_locked_and_spool_differentials_ignore_control():
    chain = LockChain("lock", [_stage("throttle", [[0, 0.0], [1, 0.5]])])
    assert Differential(kind="open", lock_power=0.4, chains=[chain]).resolve(INPUTS, None, None) == (0.0, 0.0, 0.0)
    assert Differential(kind="locked", chains=[chain]).resolve(INPUTS, None, None)[:2] == (1.0, 1.0)


def test_inputs_the_simulator_cannot_supply_are_skipped_and_reported():
    report = BuildReport()
    chains = parse_lock_control(
        {"target": "lock", "stages": [
            {"input": "x_my_signal", "combine": "multiply", "curve": [[0, 1], [1, 0]]},
            {"input": "throttle", "combine": "multiply", "curve": [[0, 1], [1, 0]]}]},
        [0], report, "test")
    assert [s.input for s in chains[0].stages] == ["throttle"]
    assert any("x_my_signal" in w for w in report.warnings)


def test_filter_slots_are_numbered_across_every_differential(document):
    stage = {"input": "brake", "combine": "multiply", "curve": [[0, 1], [1, 0]],
             "filter_time_constant": 0.1}
    doc = _awd(document, type="full_time",
               center_diff={"type": "lsd_clutch", "location": "center", "lock_power": 0.4,
                            "lock_control": {"target": "lock", "stages": [dict(stage)]}})
    doc["powertrain"]["differentials"][1]["type"] = "lsd_clutch"
    doc["powertrain"]["differentials"][1]["lock_power"] = 0.3
    doc["powertrain"]["differentials"][1]["lock_control"] = {"target": "lock", "stages": [dict(stage)]}
    vehicle = build_vehicle(doc)
    assert vehicle.driveline.filter_count == 2
    state = VehicleDynamics(vehicle).initial_state(speed=10.0, gear=2)
    assert list(state.filters) == [1.0, 1.0]                 # brake 0 -> stage output 1


def test_a_filter_follows_its_first_order_equation(document):
    """Brake held at 1 drives the stage output to 0 from 1: h(t) = exp(-t/tau)."""
    doc = _awd(document, type="full_time",
               center_diff={"type": "lsd_clutch", "location": "center", "lock_power": 0.4,
                            "lock_control": {"target": "lock", "stages": [
                                {"input": "brake", "combine": "multiply",
                                 "curve": [[0, 1], [1, 0]], "filter_time_constant": 0.1}]}})
    vehicle = build_vehicle(doc)
    dynamics = VehicleDynamics(vehicle)
    state = dynamics.initial_state(speed=25.0, gear=3)
    controls = Controls(throttle=0.0, brake=1.0, gear=3, clutch=0.0)
    for _ in range(100):
        state = step_rk4(dynamics, state, controls, 0.002)
    assert state.filters[0] == pytest.approx(np.exp(-0.2 / 0.1), abs=1e-4)


# ── torque distribution ───────────────────────────────────────────────

def _torques(vehicle, spins, shaft=1000.0, controls=None):
    plan = _plan(vehicle)
    state = _state(vehicle, spins)
    out, _ = plan.torques(state, controls or Controls(throttle=1.0, gear=3), shaft,
                          state.filters if state.filters.size else None)
    return dict(zip([s.name for s in vehicle.stations], out))


def test_without_a_centre_device_the_shaft_is_shared_by_wheel_count(document):
    t = _torques(build_vehicle(_awd(document)), [30, 30, 30, 30])
    assert t == pytest.approx({"FL": 250.0, "FR": 250.0, "RL": 250.0, "RR": 250.0})


def test_a_declared_torque_split_is_honoured(document):
    t = _torques(build_vehicle(_awd(document, type="part_time", torque_split=[0.4, 0.6])),
                 [30, 30, 30, 30])
    assert t["FL"] + t["FR"] == pytest.approx(400.0) and t["RL"] + t["RR"] == pytest.approx(600.0)


def test_a_rear_biased_car_with_one_driven_axle_gives_it_everything(document):
    t = _torques(build_vehicle(document), [30, 30, 30, 30])
    assert t["FL"] == t["FR"] == 0.0 and t["RL"] + t["RR"] == pytest.approx(1000.0)


def test_a_centre_differential_moves_torque_to_the_slower_axle(document):
    vehicle = build_vehicle(_awd(document, type="full_time", torque_split=[0.5, 0.5],
                                 center_diff={"type": "lsd_clutch", "location": "center",
                                              "lock_power": 0.5}))
    even = _torques(vehicle, [30, 30, 30, 30])
    front_fast = _torques(vehicle, [34, 34, 30, 30])
    assert even["FL"] + even["FR"] == pytest.approx(500.0)
    assert front_fast["FL"] + front_fast["FR"] < 500.0 < front_fast["RL"] + front_fast["RR"]
    assert sum(front_fast.values()) == pytest.approx(1000.0)


def test_a_coupling_carries_nothing_with_matched_speeds_and_is_capped_when_they_differ(document):
    vehicle = build_vehicle(_awd(document, type="on_demand",
                                 coupling={"mechanism": "clutch_pack", "ramp_torque": 40.0,
                                           "max_torque": 100.0}))
    final = vehicle.driveline.final_drive
    matched = _torques(vehicle, [30, 30, 30, 30], shaft=5000.0)
    assert matched["FL"] + matched["FR"] == pytest.approx(0.0, abs=1e-6)       # rear is primary
    slipping = _torques(vehicle, [30, 30, 60, 60], shaft=5000.0)               # rear spinning up
    front = slipping["FL"] + slipping["FR"]
    assert 0.0 < front <= 100.0 * final + 1e-6                                   # rated at the case
    assert sum(slipping.values()) == pytest.approx(5000.0)


def test_a_coupling_cannot_hand_over_more_than_the_engine_gives(document):
    vehicle = build_vehicle(_awd(document, type="on_demand",
                                 coupling={"ramp_torque": 40.0, "max_torque": 5000.0}))
    t = _torques(vehicle, [30, 30, 90, 90], shaft=300.0)
    assert t["FL"] + t["FR"] == pytest.approx(300.0, rel=1e-3)


def test_the_coupling_rating_is_referred_to_the_wheels_through_the_final_drive(document):
    vehicle = build_vehicle(_awd(document, type="on_demand",
                                 coupling={"ramp_torque": 100.0, "max_torque": 100.0}))
    t = _torques(vehicle, [30, 30, 90, 90], shaft=9000.0)
    assert t["FL"] + t["FR"] == pytest.approx(100.0 * 3.45, rel=1e-3)


def test_a_centre_lock_scales_the_coupling(document):
    base = dict(type="on_demand", coupling={"ramp_torque": 100.0, "max_torque": 100.0})
    full = _torques(build_vehicle(_awd(document, **base)), [30, 30, 90, 90], shaft=9000.0)
    scaled = _torques(build_vehicle(_awd(document, **base, center_diff={
        "type": "lsd_clutch", "location": "center", "lock_power": 0.25})), [30, 30, 90, 90], shaft=9000.0)
    assert scaled["FL"] + scaled["FR"] == pytest.approx(0.25 * (full["FL"] + full["FR"]), rel=1e-3)


def test_the_axle_differential_still_shares_torque_between_its_wheels(document):
    doc = _awd(document)
    doc["powertrain"]["differentials"][1].update(type="lsd_clutch", lock_power=0.4)
    t = _torques(build_vehicle(doc), [30, 30, 31, 29])         # RL faster than RR
    assert t["RL"] < t["RR"] and t["RL"] + t["RR"] == pytest.approx(500.0)


def test_a_vehicle_with_no_filters_has_the_classic_state_layout(vehicle):
    state = VehicleDynamics(vehicle).initial_state(speed=5.0)
    assert state.size == 12 + 5 * len(vehicle.stations) + 1 and state.filters.size == 0


def test_the_new_awd_example_drives_off_and_stays_finite(document):
    doc = _awd(document, type="on_demand",
               coupling={"mechanism": "clutch_pack", "ramp_torque": 40.0, "max_torque": 800.0},
               center_diff={"type": "lsd_clutch", "location": "center", "lock_power": 0.4,
                            "lock_control": {"target": "lock", "stages": [
                                {"input": "rear_speed_ratio", "combine": "add",
                                 "curve": [[1.0, 0.0], [1.1, 0.2]], "min": 0.0, "max": 1.0}]}})
    vehicle = build_vehicle(doc)
    dynamics = VehicleDynamics(vehicle)
    state = dynamics.trim(speed=15.0, gear=3, duration=0.5, dt=0.002)
    controls = Controls(throttle=0.6, gear=3, clutch=1.0)
    for _ in range(500):
        state = step_rk4(dynamics, state, controls, 0.002)
    assert np.all(np.isfinite(state.data)) and state.speed > 14.0
