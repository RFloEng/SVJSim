"""A minimal, complete, symmetric SVJ vehicle for the tests to run against.

Built in code rather than read from disk so the tests do not depend on the SVJ
repository being checked out next door, and so a test can change one number and
see what it does.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

WHEELBASE = 2.60
TRACK = 1.55
MASS = 1400.0
UNSPRUNG = 40.0


def _corner(x: float, y: float, *, steered: bool) -> dict[str, Any]:
    """A double wishbone at longitudinal station `x`, on the side given by `y`."""
    side = 1.0 if y > 0 else -1.0
    links = [
        {"name": "upper_wishbone", "type": "arm",
         "inboard_points": [[x + 0.12, side * 0.36, -0.52], [x - 0.12, side * 0.36, -0.52]],
         "outboard_ref": "hardpoints.upper_ball_joint"},
        {"name": "lower_wishbone", "type": "arm",
         "inboard_points": [[x + 0.14, side * 0.28, -0.17], [x - 0.14, side * 0.28, -0.17]],
         "outboard_ref": "hardpoints.lower_ball_joint"},
    ]
    hardpoints = {
        "upper_ball_joint": [x, side * 0.60, -0.56],
        "lower_ball_joint": [x, side * 0.64, -0.18],
        "wheel_center": [x, y, -0.32],
        "damper_outboard": [x, side * 0.50, -0.20],
    }
    if steered:
        hardpoints["steering_tie_rod_end"] = [x + 0.11, side * 0.60, -0.27]
        links.append({"name": "steering_tie_rod", "type": "rod",
                      "inboard_points": [[x + 0.11, side * 0.20, -0.27]],
                      "outboard_ref": "hardpoints.steering_tie_rod_end"})

    return {
        "topology": {
            "system_type": "double_wishbone",
            "upright": {"id": f"upright_{x}_{y}", "mass": 9.0, "hardpoints": hardpoints},
            "links": links,
        },
        "wheel": {"rim_diameter": 0.4318, "rim_width": 0.1905, "mass": 21.0,
                  "rotational_inertia": 0.85, "set_ref": "road_225_45r17"},
        "spring": {"type": "coil", "rate": 30000.0, "motion_ratio": 0.75,
                   "inboard_mount": [x, side * 0.30, -0.60],
                   "outboard_mount": [x, side * 0.50, -0.20]},
        "damper": {"type": "monotube", "motion_ratio": 0.75,
                   "bump_curve": [[0.0, 0.0], [0.1, 220.0], [0.5, 700.0]],
                   "rebound_curve": [[0.0, 0.0], [0.1, 320.0], [0.5, 1000.0]],
                   "inboard_mount": [x, side * 0.30, -0.60],
                   "outboard_mount": [x, side * 0.50, -0.20]},
        "alignment": {"camber": -0.0175, "toe": 0.0005, "caster": 0.10},
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


def svj_document() -> dict[str, Any]:
    front_x, rear_x = 0.0, -WHEELBASE
    half = TRACK / 2.0
    return {
        "_metadata": {"specification": "SVJ", "version": "0.99.1",
                      "coordinate_system": "SAE_J670", "units": "SI",
                      "alignment_convention": "relative_to_centerline"},
        "vehicle_info": {"make": "Test", "model": "Reference", "year": 2026,
                         "drive_type": "RWD"},
        "chassis": {
            "mass_total": MASS,
            "mass_unsprung_per_corner": {"FL": UNSPRUNG, "FR": UNSPRUNG,
                                         "RL": UNSPRUNG, "RR": UNSPRUNG},
            "wheelbase": WHEELBASE, "track_front": TRACK, "track_rear": TRACK,
            "center_of_gravity": [-1.20, 0.0, -0.50],
            "inertia": {"Ixx": 480.0, "Iyy": 2100.0, "Izz": 2250.0, "Ixz": 0.0},
        },
        "steering": {
            "type": "rack_and_pinion", "rack_position": [0.11, 0.0, -0.27],
            "rack_travel": 0.140, "overall_ratio": 15.0, "lock_to_lock": 16.0,
            "tie_rod_inboard": {"left": [0.11, -0.20, -0.27],
                                "right": [0.11, 0.20, -0.27]},
        },
        "suspension": {
            "FL": _corner(front_x, -half, steered=True),
            "FR": _corner(front_x, half, steered=True),
            "RL": _corner(rear_x, -half, steered=False),
            "RR": _corner(rear_x, half, steered=False),
        },
        "tires": {"sets": {"road_225_45r17": {
            "description": "Reference summer tire",
            "size_code": "225/45R17",
            "dimensions": {"section_width": 0.225, "aspect_ratio": 0.45,
                           "rim_diameter_code": 17, "overall_diameter": 0.634},
            "construction": {"vertical_stiffness": 230000.0, "vertical_damping": 420.0,
                             "rolling_radius": 0.305, "mass": 9.5},
            "reference": {"load": 4000.0, "pressure": 230000.0},
            "relaxation": {"lateral": 0.30, "longitudinal": 0.10},
            "tmeasy": {"Fz0": 4000.0, "dFy0": 62000.0, "Fy_max": 4400.0,
                       "alpha_max": 0.13, "Fy_slide": 3800.0, "alpha_slide": 0.30,
                       "dFx0": 75000.0, "Fx_max": 4600.0, "kappa_max": 0.11,
                       "Fx_slide": 4000.0, "kappa_slide": 0.32,
                       "camber_stiffness": 2000.0,
                       "pneumatic_trail_max": 0.030, "pneumatic_trail_slide": 0.004},
        }}},
        "brakes": {
            "pedal": {"ratio": 4.0}, "booster": {"type": "vacuum", "boost_ratio": 3.2},
            "master_cylinder": {"bore_primary": 0.0223, "bore_secondary": 0.0223},
            "bias": 0.62, "line_pressure_max": 1.1e7,
            "abs": {"enabled": False},
        },
        "powertrain": {
            "layout": "FR",
            "engine": {"idle_rpm": 800, "max_rpm": 6800, "inertia": 0.16,
                       "torque_curve": [[800, 130], [2000, 230], [4000, 280],
                                        [5500, 265], [6800, 220]]},
            "clutch": {"type": "single_plate", "max_torque": 420.0},
            "gearbox": {"type": "manual", "ratios": [3.6, 2.2, 1.55, 1.2, 1.0, 0.82],
                        "efficiency": 0.95, "shift_time": 0.2},
            "differentials": [{"id": "diff_rear", "location": "rear", "type": "lsd_clutch",
                               "final_drive": 3.45, "preload": 50.0,
                               "lock_power": 0.35, "lock_coast": 0.20}],
            "half_shafts": {"RL": {"diff_ref": "diff_rear"},
                            "RR": {"diff_ref": "diff_rear"}},
        },
        "aerodynamics": {
            "reference": {"frontal_area": 2.05, "air_density": 1.225},
            "coefficients": {"Cd": 0.32, "Cl": 0.05, "Cl_front": 0.03, "Cl_rear": 0.02},
            "center_of_pressure": [-1.20, 0.0, -0.45],
        },
        "benchmarks": [
            {"id": "accel_0_100kph", "value": 6.0, "unit": "s", "type": "target"},
            {"id": "skidpad_lateral_g", "value": 0.95, "unit": "g", "type": "target"},
        ],
    }


@pytest.fixture
def document() -> dict[str, Any]:
    return copy.deepcopy(svj_document())


@pytest.fixture
def vehicle(document):
    from svjsim.model import build_vehicle
    return build_vehicle(document)


@pytest.fixture
def dynamics(vehicle):
    from svjsim.solver import VehicleDynamics
    return VehicleDynamics(vehicle)
