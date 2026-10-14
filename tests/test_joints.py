"""Link joint types (SVJ 9.2.3.2): inference, planning, and the constraints they give."""

from __future__ import annotations

import copy
import itertools

import numpy as np
import pytest

from svjsim.joints import (JointPair, infer_joint_types, plan_link, resolve_joints)
from svjsim.kinematics import Link, StationKinematics, UprightPose


# ── the inference table ───────────────────────────────────────────────

SYSTEMS = ["double_wishbone", "macpherson", "chapman_strut", "multi_link", "custom",
           "trailing_arm", "semi_trailing_arm", "torsion_beam", "parallelogram",
           "pendulum_axle", "swing_axle", "solid_axle", "de_dion", None]
LINK_TYPES = ["arm", "rod", "strut", "pivot", "torque_rod", None, "leaf"]


@pytest.mark.parametrize("system, link", list(itertools.product(SYSTEMS, LINK_TYPES)))
def test_inference_matches_the_reference_implementation(system, link):
    """The spec says every converter must infer identically; svj-py is the reference."""
    multibody = pytest.importorskip("svj.multibody")
    assert infer_joint_types(system, link) == multibody.infer_joint_types(system, link)


def test_explicit_joints_win_over_inferred_ones():
    spec = {"type": "arm", "joint_type": "fixed", "inboard_joint_type": "spherical"}
    joints = resolve_joints(spec, "double_wishbone")
    assert (joints.outboard, joints.inboard) == ("fixed", "spherical")
    assert (joints.outboard_source, joints.inboard_source) == ("explicit", "explicit")


def test_a_half_stated_pair_is_completed_from_the_table():
    joints = resolve_joints({"type": "arm", "joint_type": "fixed"}, "trailing_arm")
    assert (joints.outboard, joints.inboard) == ("fixed", "revolute")
    assert (joints.outboard_source, joints.inboard_source) == ("explicit", "inferred")


def test_an_unknown_joint_name_is_ignored_and_reported():
    issues: list[str] = []
    joints = resolve_joints({"type": "rod", "joint_type": "ball"}, "double_wishbone", issues)
    assert joints.outboard == "spherical" and "ball" in issues[0]


# ── planning ──────────────────────────────────────────────────────────

def _plan(kind, inboard, outboard, points=1, wheel_center=False, system="double_wishbone"):
    return plan_link(kind=kind, joints=JointPair(outboard, inboard, "explicit", "explicit"),
                     inboard_count=points, at_wheel_center=wheel_center, system_type=system)


@pytest.mark.parametrize("kind, inboard, outboard, points, wheel_center, system, model", [
    ("rod", "spherical", "spherical", 1, False, "double_wishbone", "distance"),
    ("arm", "revolute", "spherical", 2, False, "double_wishbone", "distance"),
    ("strut", "fixed", "translational", 1, False, "macpherson", "strut"),
    ("arm", "revolute", "revolute", 2, True, "trailing_arm", "carrier"),
    ("arm", "revolute", "fixed", 1, False, "trailing_arm", "carrier"),       # arm_axle_seat
    ("arm", None, "fixed", 1, False, "trailing_arm", "carrier"),
    ("pivot", None, "revolute", 2, False, "pendulum_axle", "carrier"),
    ("arm", "revolute", "revolute", 1, False, "parallelogram", "revolute_pair"),
    ("arm", "cylindrical", "spherical", 2, False, "double_wishbone", "cylinder"),
    ("rod", "fixed", "spherical", 1, False, "double_wishbone", "pinned"),
    ("arm", "spherical", "fixed", 1, False, "double_wishbone", "pinned"),
    ("arm", "revolute", "universal", 2, False, "double_wishbone", "distance"),
    ("arm", "spherical", "revolute", 1, False, "double_wishbone", "distance"),   # no axis to use
])
def test_planned_constraint_model(kind, inboard, outboard, points, wheel_center, system, model):
    assert _plan(kind, inboard, outboard, points, wheel_center, system).model == model


def test_joints_it_cannot_use_are_reduced_to_a_distance_and_say_so():
    plan = _plan("arm", "revolute", "universal", points=2)
    assert plan.model == "distance" and any("universal" in n for n in plan.notes)
    plan = _plan("rod", "spherical", "planar")
    assert plan.model == "distance" and any("planar" in n for n in plan.notes)


def test_a_beam_axle_link_stays_a_plain_distance_whatever_its_joints():
    assert _plan("arm", "revolute", "fixed", 2, system="solid_axle").model == "distance"


def test_nothing_stated_and_nothing_to_infer_keeps_the_topology_rule():
    joints = JointPair(None, None, None, None)
    kwargs = dict(kind="arm", joints=joints, inboard_count=2, system_type="swing_axle")
    assert plan_link(at_wheel_center=True, **kwargs).model == "carrier"
    assert plan_link(at_wheel_center=False, **kwargs).model == "distance"


def test_a_ball_joint_on_a_carrier_arm_is_flagged_so_the_station_can_overrule_it():
    assert _plan("arm", "revolute", "spherical", 2, True, "semi_trailing_arm").demoted_carrier


# ── the constraints, against hand calculations ────────────────────────

def _corner(system, links, hardpoints=None):
    points = {"wheel_center": [0.0, -0.8, -0.3]}
    points.update(hardpoints or {})
    return {"topology": {"system_type": system, "upright": {"id": "u", "hardpoints": points},
                         "links": links}}


def _wheel_centre(solver, jounce, rack=0.0):
    pose, residual = solver.solve(jounce, rack)
    return pose.transform(solver.wheel_center0), pose, residual


def test_a_trailing_arm_rigid_with_the_upright_swings_the_wheel_on_a_circle():
    """`fixed` outboard at a hardpoint that is not the wheel centre: the upright is
    still carried by the arm, so the wheel centre moves on a circle about the pivot."""
    pivot = np.array([0.8, -0.35, -0.45])
    corner = _corner("trailing_arm", [
        {"name": "trailing_arm", "type": "arm", "inboard_joint_type": "revolute",
         "joint_type": "fixed", "inboard_points": [pivot.tolist()],
         "outboard_ref": "hardpoints.arm_axle_seat"}],
        {"arm_axle_seat": [0.0, -0.7, -0.3]})
    solver = StationKinematics("RL", corner, samples=5)
    assert solver.mode == "revolute"
    radius = np.linalg.norm(solver.wheel_center0 - pivot)
    for jounce in (-0.06, -0.02, 0.03, 0.07):
        wc, _, residual = _wheel_centre(solver, jounce)
        assert wc[2] == pytest.approx(solver.wheel_center0[2] - jounce, abs=1e-9)
        assert np.linalg.norm(wc - pivot) == pytest.approx(radius, abs=1e-9)
        assert wc[1] == pytest.approx(solver.wheel_center0[1], abs=1e-9)    # lateral pivot axis


def test_a_pendulum_pin_turns_the_upright_about_its_two_inboard_points():
    p1, p2 = np.array([0.0, -0.1, -0.4]), np.array([-0.5, -0.1, -0.4])      # axis along -X
    corner = _corner("pendulum_axle", [
        {"name": "pendulum_pin", "type": "pivot", "joint_type": "revolute",
         "inboard_points": [p1.tolist(), p2.tolist()], "outboard_ref": "hardpoints.pendulum_pivot"}],
        {"pendulum_pivot": [-0.25, -0.1, -0.4]})
    solver = StationKinematics("A1L", corner, samples=5)
    assert solver.mode == "revolute"
    axis = (p2 - p1) / np.linalg.norm(p2 - p1)
    rel0 = solver.wheel_center0 - p1
    radial0 = np.linalg.norm(rel0 - (rel0 @ axis) * axis)
    for jounce in (-0.05, 0.05):
        wc, pose, _ = _wheel_centre(solver, jounce)
        rel = wc - p1
        assert np.linalg.norm(rel - (rel @ axis) * axis) == pytest.approx(radial0, abs=1e-9)
        assert rel @ axis == pytest.approx(rel0 @ axis, abs=1e-9)
        # The rotation vector is along the pin: rolling, no pitch or yaw.
        assert np.linalg.norm(np.cross(pose.theta, axis)) == pytest.approx(0.0, abs=1e-9)


def _parallelogram(beam=0.65, gap=0.22):
    pivots = {"upper": [-2.2, -0.42, -0.71], "lower": [-2.2, -0.42, -0.49]}
    ends = {"upper_beam_end": [-2.2 - beam, -0.42, -0.71], "lower_beam_end": [-2.2 - beam, -0.42, -0.49]}
    corner = _corner("parallelogram", [
        {"name": "upper_beam", "type": "arm", "inboard_points": [pivots["upper"]],
         "outboard_ref": "hardpoints.upper_beam_end"},
        {"name": "lower_beam", "type": "arm", "inboard_points": [pivots["lower"]],
         "outboard_ref": "hardpoints.lower_beam_end"}],
        {**ends, "wheel_center": [-2.9, -1.025, -0.51]})
    return StationKinematics("A2L", corner, samples=5), beam


def test_a_parallelogram_keeps_the_housing_level_while_the_wheel_moves_on_an_arc():
    solver, beam = _parallelogram()
    assert [l.model for l in solver.links] == ["revolute_pair", "revolute_pair"]
    for jounce in (-0.08, -0.03, 0.04, 0.09):
        wc, pose, residual = _wheel_centre(solver, jounce)
        assert residual < 1e-9
        assert np.linalg.norm(pose.theta) == pytest.approx(0.0, abs=1e-9)   # no pitch, roll or yaw
        # The beam ends are level with their pivots, so a vertical move h swings each
        # through asin(h / beam) and draws the housing towards the pivots by
        # beam - sqrt(beam^2 - h^2), the same whichever way it moves.
        assert wc[0] - solver.wheel_center0[0] == pytest.approx(
            beam - np.sqrt(beam ** 2 - jounce ** 2), abs=1e-9)
        assert wc[2] == pytest.approx(solver.wheel_center0[2] - jounce, abs=1e-9)
        assert wc[1] == pytest.approx(solver.wheel_center0[1], abs=1e-9)    # stays in its plane


def test_a_parallelograms_beam_arc_has_the_right_radius():
    solver, beam = _parallelogram()
    pivot = np.array([-2.2, -0.42, -0.71])
    wc0 = np.array([-2.85, -0.42, -0.71])                                    # the upper beam's end
    for jounce in (-0.08, 0.09):
        pose, _ = solver.solve(jounce, 0.0)
        end = pose.transform(wc0)
        assert np.linalg.norm(end - pivot) == pytest.approx(beam, abs=1e-9)


def test_parallel_revolute_links_share_constraints_without_a_warning():
    solver, _ = _parallelogram()
    assert not any("over-defined" in w for w in solver.warnings)
    assert any("duplicate the other links' rotation and plane constraints" in n for n in solver.notes)


def test_a_cylindrical_inboard_joint_leaves_the_arm_free_to_slide_along_its_axis():
    inboard = [[0.1, -0.3, -0.5], [-0.1, -0.3, -0.5]]
    link = Link(name="arm", kind="arm", inboard=[np.array(p) for p in inboard],
                outboard=np.array([0.0, -0.7, -0.5]), model="cylinder")
    pose = UprightPose.identity(np.array([0.0, -0.8, -0.3]))
    assert link.residuals(pose, 0.0).shape == (1,)
    assert link.residuals(pose, 0.0)[0] == pytest.approx(0.0, abs=1e-12)
    slid = UprightPose(np.array([0.05, 0.0, 0.0]), np.zeros(3), pose.origin)   # along the axis
    assert link.residuals(slid, 0.0)[0] == pytest.approx(0.0, abs=1e-12)
    lifted = UprightPose(np.array([0.0, 0.0, -0.02]), np.zeros(3), pose.origin)
    # The joint sits 0.4 m from the axis; lifting it 20 mm moves it to sqrt(0.4^2 + 0.02^2).
    assert link.residuals(lifted, 0.0)[0] == pytest.approx(np.hypot(0.4, 0.02) - 0.4, abs=1e-12)


def test_a_link_rigid_with_the_chassis_pins_its_ball_joint():
    link = Link(name="stub", kind="rod", inboard=[np.array([0.0, -0.3, -0.5])],
                outboard=np.array([0.0, -0.7, -0.5]), model="pinned")
    pose = UprightPose.identity(np.array([0.0, -0.8, -0.3]))
    assert np.allclose(link.residuals(pose, 0.0), 0.0)
    moved = UprightPose(np.array([0.0, 0.0, -0.02]), np.zeros(3), pose.origin)
    assert np.allclose(link.residuals(moved, 0.0), [0.0, 0.0, -0.02])


def test_a_rigid_upright_hung_from_one_ball_joint_pins_the_pivot():
    link = Link(name="hanger", kind="arm", inboard=[np.array([0.0, -0.3, -0.5])],
                outboard=np.array([0.0, -0.7, -0.5]), model="pinned", carrier_hangs=True)
    assert np.allclose(link.pin, [0.0, -0.3, -0.5])


# ── the carrier fallback ──────────────────────────────────────────────

def _semi_trailing(extra_rods=0, outboard="spherical"):
    hardpoints = {"wheel_center": [0.0, -0.8, -0.3], "a": [0.12, -0.7, -0.4],
                  "b": [-0.12, -0.7, -0.2], "c": [0.0, -0.6, -0.25]}
    links = [{"name": "semi_trailing_arm", "type": "arm", "inboard_joint_type": "revolute",
              "joint_type": outboard, "outboard_ref": "hardpoints.wheel_center",
              "inboard_points": [[0.6, -0.3, -0.35], [0.45, -0.1, -0.35]]}]
    anchors = {"a": [0.5, -0.3, -0.5], "b": [-0.4, -0.35, -0.1], "c": [0.1, -0.25, -0.6]}
    for name in ("a", "b", "c")[:extra_rods]:
        links.append({"name": f"rod_{name}", "type": "rod", "inboard_points": [anchors[name]],
                      "outboard_ref": f"hardpoints.{name}"})
    return _corner("semi_trailing_arm", links, hardpoints)


def test_a_ball_joint_that_would_leave_the_upright_loose_is_overruled_and_reported():
    solver = StationKinematics("RL", _semi_trailing(extra_rods=0), samples=5)
    assert solver.mode == "revolute"
    assert any("treated as carrying the upright" in w for w in solver.warnings)
    assert not any("under-located" in w for w in solver.warnings)    # no stale complaint


def test_a_ball_joint_is_respected_when_other_links_locate_the_upright():
    solver = StationKinematics("RL", _semi_trailing(extra_rods=3), samples=5)
    assert solver.mode == "link"
    assert not any("treated as carrying the upright" in w for w in solver.warnings)


def test_a_stated_rigid_outboard_joint_is_a_carrier_without_any_fallback():
    solver = StationKinematics("RL", _semi_trailing(extra_rods=0, outboard="fixed"), samples=5)
    assert solver.mode == "revolute" and not solver.warnings


# ── through the whole build ───────────────────────────────────────────

def test_stated_joints_reach_the_build_report(document):
    from svjsim.model import build_vehicle

    doc = copy.deepcopy(document)
    corner = doc["suspension"]["RL"]["topology"]
    corner["system_type"] = "trailing_arm"
    corner["links"] = [{"name": "trailing_arm", "type": "arm", "inboard_joint_type": "revolute",
                        "joint_type": "fixed", "inboard_points": [[0.6, -0.35, -0.4]],
                        "outboard_ref": "hardpoints.wheel_center"}]
    vehicle = build_vehicle(doc)
    assert any("trailing_arm" in n and "carries the upright rigidly" in n
               for n in vehicle.report.notes)
    assert any("lateral" in n and "axis is assumed" in n for n in vehicle.report.notes)


def test_default_wishbone_joints_add_no_noise_to_the_report(vehicle):
    assert not [n for n in vehicle.report.notes if n.startswith("link ")]
