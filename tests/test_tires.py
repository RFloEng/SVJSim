"""Tire model checks: shape of the force curves, and behaviour no model may break."""

from __future__ import annotations

import math

import numpy as np
import pytest

from svjsim.tires import (BrushTire, PacejkaTire, TMeasyTire, TireDimensions,
                          build_tire, parse_size_code)

DIMS = TireDimensions(section_width=0.225, aspect_ratio=0.45,
                      rim_diameter=0.4318, overall_diameter=0.634, tread_width=0.19)
COMMON = dict(vertical_stiffness=220000.0, vertical_damping=400.0,
              rolling_radius=0.305, pressure=220000.0)


def every_model() -> list:
    return [
        TMeasyTire("tmeasy", DIMS, {"Fz0": 4000.0, "Fy_max": 4400.0, "Fx_max": 4600.0},
                   **COMMON),
        BrushTire("brush", DIMS, {"mu_static": 1.1, "tread_stiffness_lat": 25e6},
                  **COMMON),
        PacejkaTire("pacejka", DIMS,
                    {"model": "MF52",
                     "lateral": {"pCy1": 1.5, "pDy1": 1.15, "pDy2": -0.1, "pKy1": 20.0,
                                 "pKy2": 1.8, "pKy4": 2.0, "pEy1": -0.5},
                     "longitudinal": {"pCx1": 1.6, "pDx1": 1.2, "pDx2": -0.08,
                                      "pKx1": 22.0, "pEx1": 0.3}},
                    {"load": 4000.0, "pressure": 220000.0}, **COMMON),
    ]


@pytest.mark.parametrize("size,expected", [
    ("245/35R19", (0.245, 0.35, 19 * 0.0254)),
    ("315/80R22.5", (0.315, 0.80, 22.5 * 0.0254)),
    ("ST235/80R16", (0.235, 0.80, 16 * 0.0254)),
    ("14.00R20", (14 * 0.0254, 1.0, 20 * 0.0254)),
])
def test_size_code_parsing(size, expected):
    assert parse_size_code(size) == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("tire", every_model(), ids=lambda t: t.kind)
def test_zero_slip_gives_no_force(tire):
    f = tire.forces(4000.0, 0.0, 0.0, 0.0, 20.0)
    assert abs(f.Fx) < 1.0
    assert abs(f.Fy) < 1.0


@pytest.mark.parametrize("tire", every_model(), ids=lambda t: t.kind)
def test_unloaded_tire_makes_no_force(tire):
    f = tire.forces(0.0, 0.2, 0.1, 0.0, 20.0)
    assert f == (0.0, 0.0, 0.0, 0.0, 0.0)


@pytest.mark.parametrize("tire", every_model(), ids=lambda t: t.kind)
def test_lateral_force_is_odd_in_slip_angle(tire):
    """Steering left and right must give mirrored force: no built-in bias."""
    pos = tire.forces(4000.0, 0.0, 0.06, 0.0, 20.0)
    neg = tire.forces(4000.0, 0.0, -0.06, 0.0, 20.0)
    assert pos.Fy == pytest.approx(-neg.Fy, rel=1e-6)


@pytest.mark.parametrize("tire", every_model(), ids=lambda t: t.kind)
def test_longitudinal_force_is_near_odd_in_slip_ratio(tire):
    """Drive and brake forces mirror each other, up to the slip definition.

    Not exactly odd for the brush model: it works in slip relative to contact
    patch speed, `kappa / (1 + kappa)`, under which braking at -0.05 is a larger
    slip than driving at +0.05. That asymmetry is the physics, not a defect.
    """
    pos = tire.forces(4000.0, 0.05, 0.0, 0.0, 20.0)
    neg = tire.forces(4000.0, -0.05, 0.0, 0.0, 20.0)
    assert pos.Fx == pytest.approx(-neg.Fx, rel=0.12)


@pytest.mark.parametrize("tire", every_model(), ids=lambda t: t.kind)
def test_lateral_force_peaks_then_falls_or_saturates(tire):
    alpha = np.linspace(0.0, 0.5, 200)
    fy = np.array([tire.forces(4000.0, 0.0, float(a), 0.0, 20.0).Fy for a in alpha])
    peak = int(np.argmax(fy))
    assert 0.02 < alpha[peak] < 0.35, "peak slip angle outside any plausible range"
    assert fy[-1] <= fy[peak] + 1.0, "force must not keep climbing past the peak"
    assert np.all(np.diff(fy[:peak]) > -1.0), "curve must rise monotonically to the peak"


@pytest.mark.parametrize("tire", every_model(), ids=lambda t: t.kind)
def test_grip_is_bounded_by_a_sane_friction_coefficient(tire):
    for Fz in (2000.0, 4000.0, 8000.0):
        alpha = np.linspace(0.0, 0.4, 80)
        peak = max(tire.forces(Fz, 0.0, float(a), 0.0, 20.0).Fy for a in alpha)
        assert 0.4 < peak / Fz < 2.0, f"{tire.kind} mu={peak / Fz:.2f} at {Fz} N"


@pytest.mark.parametrize("tire", every_model(), ids=lambda t: t.kind)
def test_combined_slip_never_exceeds_pure_slip_envelope(tire):
    """Braking hard must cost cornering force — the friction circle."""
    pure = tire.forces(4000.0, 0.0, 0.10, 0.0, 20.0)
    combined = tire.forces(4000.0, -0.25, 0.10, 0.0, 20.0)
    assert abs(combined.Fy) < abs(pure.Fy)
    envelope = max(abs(tire.forces(4000.0, 0.0, float(a), 0.0, 20.0).Fy)
                   for a in np.linspace(0, 0.4, 60))
    assert math.hypot(combined.Fx, combined.Fy) <= envelope * 1.35


@pytest.mark.parametrize("tire", every_model(), ids=lambda t: t.kind)
def test_cornering_stiffness_grows_with_load(tire):
    def stiffness(Fz: float) -> float:
        return (tire.forces(Fz, 0.0, 0.005, 0.0, 20.0).Fy
                - tire.forces(Fz, 0.0, -0.005, 0.0, 20.0).Fy) / 0.01

    assert stiffness(6000.0) > stiffness(3000.0) > stiffness(1500.0)


@pytest.mark.parametrize("tire", every_model(), ids=lambda t: t.kind)
def test_friction_falls_as_load_rises(tire):
    """Load sensitivity is what makes lateral load transfer cost grip."""
    def mu(Fz: float) -> float:
        return max(tire.forces(Fz, 0.0, float(a), 0.0, 20.0).Fy
                   for a in np.linspace(0, 0.4, 60)) / Fz

    assert mu(8000.0) <= mu(4000.0) + 1e-9


def test_aligning_torque_opposes_slip_and_collapses_at_the_limit():
    tire = every_model()[0]
    small = tire.forces(4000.0, 0.0, 0.03, 0.0, 20.0)
    assert small.Mz < 0, "self-aligning torque must steer the wheel back"
    large = tire.forces(4000.0, 0.0, 0.35, 0.0, 20.0)
    assert abs(large.Mz) < abs(small.Mz), "trail collapses once the tire slides"


def test_vertical_force_never_pulls():
    tire = every_model()[0]
    assert tire.normal_force(-0.01) == 0.0
    assert tire.normal_force(0.02) == pytest.approx(0.02 * 220000.0)
    assert tire.loaded_radius(4400.0) < tire.unloaded_radius


def test_synthesised_tire_is_marked_estimated_and_usable():
    tire = build_tire("nameless", {"size_code": "225/45R17"}, reference_load=4200.0)
    assert tire.source == "estimated"
    assert tire.notes and "synthesised" in tire.notes[0]
    peak = max(tire.forces(4200.0, 0.0, float(a), 0.0, 20.0).Fy
               for a in np.linspace(0, 0.4, 60))
    assert 0.8 < peak / 4200.0 < 1.5


def test_file_model_is_preferred_over_synthesis():
    tire_set = {"size_code": "225/45R17",
                "tmeasy": {"Fz0": 4000.0, "Fy_max": 4200.0}}
    tire = build_tire("has_data", tire_set)
    assert tire.source == "file"
    assert tire.kind == "tmeasy"
