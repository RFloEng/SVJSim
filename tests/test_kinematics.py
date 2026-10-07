"""Kinematics solver checks against geometry that can be worked out by hand.

The reference case is a front-view double wishbone whose roll centre can be
constructed graphically: project both arm axes to their intersection (the
instantaneous centre), draw a line from the contact patch through it, and read
the height where that line crosses the vehicle centreline.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from svjsim.kinematics import AxleKinematics, StationKinematics, to_svj_camber

TIRE_RADIUS = 0.32


def wishbone_corner(*, upper_inboard_z: float = -0.48, upper_outboard_z: float = -0.52,
                    lower_inboard_z: float = -0.15, lower_outboard_z: float = -0.16) -> dict:
    """A left-front double wishbone with independently placeable arm heights."""
    return {
        "topology": {
            "system_type": "double_wishbone",
            "upright": {
                "id": "upright_FL",
                "hardpoints": {
                    "upper_ball_joint": [0.0, -0.60, upper_outboard_z],
                    "lower_ball_joint": [0.0, -0.62, lower_outboard_z],
                    "wheel_center": [0.0, -0.75, -0.32],
                    "steering_tie_rod_end": [0.10, -0.60, -0.25],
                },
            },
            "links": [
                {"name": "upper_wishbone", "type": "arm",
                 "inboard_points": [[0.12, -0.38, upper_inboard_z], [-0.12, -0.38, upper_inboard_z]],
                 "outboard_ref": "hardpoints.upper_ball_joint"},
                {"name": "lower_wishbone", "type": "arm",
                 "inboard_points": [[0.14, -0.30, lower_inboard_z], [-0.14, -0.30, lower_inboard_z]],
                 "outboard_ref": "hardpoints.lower_ball_joint"},
                {"name": "steering_tie_rod", "type": "rod",
                 "inboard_points": [[0.10, -0.20, -0.25]],
                 "outboard_ref": "hardpoints.steering_tie_rod_end"},
            ],
        },
    }


def instant_centre(corner: dict) -> tuple[float, float]:
    """Front-view intersection of the two arm axes, by hand, in (y, z)."""
    links = {l["name"]: l for l in corner["topology"]["links"]}
    hp = corner["topology"]["upright"]["hardpoints"]
    lines = []
    for name, ref in (("upper_wishbone", "upper_ball_joint"),
                      ("lower_wishbone", "lower_ball_joint")):
        y_in, z_in = links[name]["inboard_points"][0][1], links[name]["inboard_points"][0][2]
        y_out, z_out = hp[ref][1], hp[ref][2]
        slope = (z_out - z_in) / (y_out - y_in)
        lines.append((slope, z_in - slope * y_in))
    (m1, c1), (m2, c2) = lines
    y = (c2 - c1) / (m1 - m2)
    return y, m1 * y + c1


def test_solver_satisfies_every_constraint():
    k = StationKinematics("FL", wishbone_corner(), travel_limits=(-0.05, 0.05))
    for jounce in np.linspace(-0.05, 0.05, 11):
        pose, residual = k.solve(float(jounce))
        assert residual < 1e-9, f"constraint violation {residual:.2e} m at {jounce:+.3f} m"
        # The driving constraint must actually place the wheel where asked.
        assert pose.wheel_center[2] == pytest.approx(-0.32 - jounce, abs=1e-9)


def test_roll_centre_matches_graphical_construction():
    corner = wishbone_corner()
    y_ic, z_ic = instant_centre(corner)
    y_cp = -0.75                       # zero camber at design, so contact is below the hub
    expected = -(-z_ic * y_cp / (y_ic - y_cp))

    k = StationKinematics("FL", corner, travel_limits=(-0.05, 0.05))
    k.build_table(probe=False)
    assert k.table.roll_centre_height(TIRE_RADIUS) == pytest.approx(expected, abs=1e-3)


def test_parallel_equal_arms_give_no_camber_gain():
    """Parallel arms of equal length translate the upright, so camber holds.

    Not identically zero: the tie rod is a fixed length, so it rotates the
    upright a fraction of a degree through travel and drags camber with it.
    """
    corner = wishbone_corner(upper_inboard_z=-0.48, upper_outboard_z=-0.48,
                             lower_inboard_z=-0.16, lower_outboard_z=-0.16)
    corner["topology"]["upright"]["hardpoints"]["upper_ball_joint"] = [0.0, -0.62, -0.48]
    corner["topology"]["links"][1]["inboard_points"] = [[0.14, -0.38, -0.16],
                                                        [-0.14, -0.38, -0.16]]
    k = StationKinematics("FL", corner, travel_limits=(-0.04, 0.04))
    k.build_table(probe=False)
    assert abs(k.table.camber_gain) * 0.1 < 1e-3, "under 0.001 deg per 100 mm"


def test_short_upper_arm_gains_negative_camber_in_bump():
    """The classic SLA layout: a shorter top arm pulls the wheel top inboard."""
    corner = wishbone_corner()
    k = StationKinematics("FL", corner, travel_limits=(-0.05, 0.05))
    k.build_table(probe=False)
    gain_svj = to_svj_camber(k.table.camber_gain, -0.75)
    assert gain_svj < 0, "bump travel should add negative (inboard) camber"
    # Half a degree to three degrees per 100 mm covers every road car layout.
    assert 0.005 < abs(gain_svj) * 0.1 < 0.06


def test_tie_rod_length_sets_bump_steer_direction():
    """A tie rod mounted low relative to the arms steers the wheel as it rises."""
    high = wishbone_corner()
    high["topology"]["upright"]["hardpoints"]["steering_tie_rod_end"] = [0.10, -0.60, -0.40]
    high["topology"]["links"][2]["inboard_points"] = [[0.10, -0.20, -0.40]]
    low = wishbone_corner()

    tables = []
    for corner in (high, low):
        k = StationKinematics("FL", corner, travel_limits=(-0.04, 0.04))
        k.build_table(probe=False)
        tables.append(k.table.bump_steer)
    assert tables[0] != pytest.approx(tables[1], abs=1e-4), \
        "tie rod height must change the bump-steer gradient"


def test_over_constrained_linkage_still_moves():
    """Extra links (SVJ's own 5-link guidance) must not freeze the suspension."""
    corner = wishbone_corner()
    corner["topology"]["links"].append(
        {"name": "extra_lateral", "type": "rod",
         "inboard_points": [[0.05, -0.25, -0.30]],
         "outboard_ref": "hardpoints.wheel_center"})
    corner["topology"]["links"].append(
        {"name": "extra_toe", "type": "rod",
         "inboard_points": [[-0.05, -0.25, -0.30]],
         "outboard_ref": "hardpoints.wheel_center"})
    k = StationKinematics("FL", corner, travel_limits=(-0.04, 0.04))
    k.build_table(probe=False)
    travel = k.table.wheel_center(0.03)[2] - k.table.wheel_center(-0.03)[2]
    assert travel == pytest.approx(-0.06, abs=1e-6)
    assert any("over-defined" in w for w in k.warnings)


def test_trailing_arm_uses_revolute_mode_and_holds_camber():
    corner = {
        "topology": {
            "system_type": "trailing_arm",
            "upright": {"id": "upright_RL",
                        "hardpoints": {"wheel_center": [-2.30, -0.70, -0.30]}},
            "links": [{"name": "trailing_arm", "type": "arm",
                       "inboard_points": [[-1.80, -0.55, -0.20], [-1.80, -0.85, -0.20]],
                       "outboard_ref": "hardpoints.wheel_center"}],
        },
    }
    k = StationKinematics("RL", corner, travel_limits=(-0.05, 0.05))
    assert k.mode == "revolute"
    k.build_table(probe=False)
    # A pure trailing arm pivots on a lateral axis: no camber, no toe change.
    assert abs(k.table.camber_gain) < 1e-9
    assert abs(k.table.bump_steer) < 1e-9
    # ...but the wheel does swing fore and aft through its arc.
    assert abs(k.table.recession) > 1e-3


# ── Beam axles ───────────────────────────────────────────────────────
#
# The reference case here is a solid axle located by a Panhard rod: for
# infinitesimal roll, the classic hand construction is that the roll centre
# sits on the rod's own line, at the height where that line crosses the
# vehicle centreline (y=0) -- the rod is the only thing giving the axle any
# lateral stiffness, so for a small enough roll its geometry alone decides
# where the axle can move sideways.
#
# The trailing links below give inboard and outboard points the *same* y, so
# each rod's own axis lies in the X-Z plane with zero y-component -- to first
# order they add negligible sensitivity to a y-perturbation of the axle, so
# they locate the axle longitudinally without measurably fighting the
# Panhard rod for lateral/roll location. That is what lets the classic
# formula be checked to good precision without solving the whole multi-link
# mechanism by hand.

TRACK = 1.5
PANHARD_INBOARD = np.array([0.0, 0.05, -0.30])
PANHARD_OUTBOARD_Y = -TRACK / 2
PANHARD_OUTBOARD_Z = -0.40


def solid_axle_pair() -> tuple[dict, dict]:
    def corner(y: float, *, panhard: bool) -> dict:
        hardpoints = {
            "wheel_center": [0.0, y, -0.35],
            "upper_link_end": [0.0, y, -0.45],
            "lower_link_end": [0.0, y, -0.20],
        }
        links = [
            {"name": "upper_link", "type": "rod",
             "inboard_points": [[1.0, y, -0.45]], "outboard_ref": "hardpoints.upper_link_end"},
            {"name": "lower_link", "type": "rod",
             "inboard_points": [[1.0, y, -0.20]], "outboard_ref": "hardpoints.lower_link_end"},
        ]
        if panhard:
            hardpoints["panhard_axle_end"] = [0.0, PANHARD_OUTBOARD_Y, PANHARD_OUTBOARD_Z]
            links.append({"name": "panhard_rod", "type": "rod",
                         "inboard_points": [PANHARD_INBOARD.tolist()],
                         "outboard_ref": "hardpoints.panhard_axle_end"})
        return {"topology": {"system_type": "solid_axle",
                             "upright": {"hardpoints": hardpoints}, "links": links,
                             "axle_body": {"id": "test_axle"}}}

    return corner(-TRACK / 2, panhard=True), corner(TRACK / 2, panhard=False)


def test_axle_roll_centre_matches_panhard_rod_line():
    left_corner, right_corner = solid_axle_pair()
    left = StationKinematics("RL", left_corner, samples=5)
    right = StationKinematics("RR", right_corner, samples=5)
    axle = AxleKinematics(left, right)
    # 5 nominal link residuals (upper+lower per side, plus the Panhard rod)
    # for the 4 the axle actually needs beyond each side's own driven
    # travel -- one gets dropped as redundant, same as an over-defined
    # single corner (test_over_constrained_linkage_still_moves).
    assert len(axle.active) == 4, "4 link residuals expected beyond each side's own travel"
    assert any("redundant" in w for w in axle.warnings)
    assert not any("under-located" in w for w in axle.warnings)

    y_in, z_in = PANHARD_INBOARD[1], PANHARD_INBOARD[2]
    y_out, z_out = PANHARD_OUTBOARD_Y, PANHARD_OUTBOARD_Z
    slope = (z_out - z_in) / (y_out - y_in)
    expected = z_in - y_in * slope

    assert axle.roll_centre_z() == pytest.approx(expected, abs=1e-4)


def test_axle_camber_shared_and_zero_in_pure_bump():
    left_corner, right_corner = solid_axle_pair()
    left = StationKinematics("RL", left_corner, samples=5, travel_limits=(-0.05, 0.05))
    right = StationKinematics("RR", right_corner, samples=5, travel_limits=(-0.05, 0.05))
    left_table, right_table, _ = AxleKinematics(left, right).build_tables()

    # Pure bump (both sides travel together): the axle does not roll.
    assert left_table.camber_at(0.03, 0.03) == pytest.approx(left_table.camber_at(0.0, 0.0), abs=1e-9)

    # Roll (sides travel oppositely): both wheels take the same absolute
    # camber, since they are rigid extensions of the one solved body.
    left_camber = left_table.camber_at(0.03, -0.03)
    right_camber = right_table.camber_at(-0.03, 0.03)
    assert left_camber == pytest.approx(right_camber, abs=1e-9)
    assert abs(left_camber) > 1e-3, "a real roll angle should show up as real camber"


def test_axle_under_located_without_lateral_device_warns():
    """Trailing links alone locate the axle longitudinally but not laterally."""
    left_corner, right_corner = solid_axle_pair()
    del left_corner["topology"]["links"][-1]  # drop the only lateral device (Panhard rod)
    del left_corner["topology"]["upright"]["hardpoints"]["panhard_axle_end"]
    left = StationKinematics("RL", left_corner, samples=5)
    right = StationKinematics("RR", right_corner, samples=5)
    axle = AxleKinematics(left, right)
    assert len(axle.active) < 4
    assert any("under-located" in w for w in axle.warnings)
