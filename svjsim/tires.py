"""Tire force models: Pacejka Magic Formula, TMeasy, brush, and a synthesised fallback.

All models share one interface so the solver never branches on tire type::

    forces(Fz, kappa, alpha, gamma, Vx) -> TireForces(Fx, Fy, Mz, Mx, My)

Sign conventions (ISO/"adapted SAE", which is what fitted `.tir` coefficient sets
assume, and what SVJ stores):

* ``Fz``    vertical load, positive in compression
* ``kappa`` longitudinal slip ratio, positive under drive, so ``Fx`` is positive
* ``alpha`` slip angle in radians, positive giving positive ``Fy``
* ``gamma`` inclination angle in radians, positive when the wheel top leans +Y

Most SVJ files in the wild carry tire *dimensions* but no fitted force model —
of the vehicles shipped with the standard, only the MX-5 template has Pacejka
coefficients. `synthesise` builds a TMeasy model from size, pressure and load
rating so those vehicles can still be driven, and marks itself `estimated` so
reports never present guessed grip as measured data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

from .util import Curve, clean, dig, num

EPS = 1e-6


class TireForces(NamedTuple):
    Fx: float      # longitudinal force at the contact patch (N)
    Fy: float      # lateral force (N)
    Mz: float      # self-aligning torque (Nm)
    Mx: float      # overturning moment (Nm)
    My: float      # rolling resistance moment (Nm)


@dataclass
class TireDimensions:
    section_width: float
    aspect_ratio: float
    rim_diameter: float
    overall_diameter: float
    tread_width: float

    @property
    def unloaded_radius(self) -> float:
        return self.overall_diameter / 2.0


class TireModel:
    """Base class: geometry, vertical behaviour and the force interface."""

    kind = "base"

    def __init__(self, name: str, dims: TireDimensions, *,
                 vertical_stiffness: float, vertical_damping: float,
                 rolling_radius: float, pressure: float, source: str = "measured",
                 relaxation_lateral: float = 0.3, relaxation_longitudinal: float = 0.1,
                 mu_reference: float = 1.0, notes: tuple[str, ...] = ()):
        self.name = name
        self.dims = dims
        self.vertical_stiffness = vertical_stiffness
        self.vertical_damping = vertical_damping
        self.rolling_radius = rolling_radius
        self.pressure = pressure
        self.source = source
        self.relaxation_lateral = relaxation_lateral
        self.relaxation_longitudinal = relaxation_longitudinal
        self.mu_reference = mu_reference
        self.notes = list(notes)

    @property
    def unloaded_radius(self) -> float:
        return self.dims.unloaded_radius

    def normal_force(self, deflection: float, deflection_rate: float = 0.0) -> float:
        """Vertical force from radial deflection. A tire can push but never pull."""
        if deflection <= 0.0:
            return 0.0
        return max(0.0, self.vertical_stiffness * deflection +
                   self.vertical_damping * deflection_rate)

    def loaded_radius(self, Fz: float) -> float:
        return self.unloaded_radius - max(0.0, Fz) / max(self.vertical_stiffness, 1.0)

    def effective_radius(self, Fz: float) -> float:
        """Rolling radius: about a third of the way from loaded back to unloaded."""
        r0, rl = self.unloaded_radius, self.loaded_radius(Fz)
        return rl + (r0 - rl) * 2.0 / 3.0

    def forces(self, Fz: float, kappa: float, alpha: float, gamma: float,
               Vx: float = 10.0, mu_scale: float = 1.0) -> TireForces:
        raise NotImplementedError


# ── Pacejka Magic Formula (MF 5.2 / 6.2) ──────────────────────────────

class PacejkaTire(TireModel):
    """Pacejka Magic Formula, MF 5.2 and MF 6.2 coefficient sets.

    Follows Pacejka, *Tyre and Vehicle Dynamics*, 3rd ed., §4.3.2. Scaling
    factors default to 1 and structural coefficients to 0, per SVJ §11.6.1,
    except the shape factors, where a zero would silently kill all force —
    those fall back to their conventional values.

    Combined slip uses the MF weighting functions when the file supplies the
    `combined` group, and a friction ellipse when it does not. The MX-5 template
    is an example of the latter.
    """

    kind = "pacejka"

    # SVJ §11.6.1 says absent coefficients default to zero, which is right for
    # most of them — a zero simply switches that effect off. These are the
    # exceptions, where a zero would silently produce no force at all, so the
    # conventional value is used instead.
    _DEFAULTS = {
        "pCy1": 1.3, "pCx1": 1.65, "qCz1": 1.1,
        "pKy1": 20.0, "pKy2": 2.0, "pKy4": 2.0, "pKx1": 20.0,
        "qBz1": 10.0, "rBx1": 13.0, "rBx2": 10.0, "rCx1": 1.0,
        "rBy1": 10.0, "rBy2": 7.0, "rCy1": 1.0, "qsy1": 0.01,
    }

    def __init__(self, name: str, dims: TireDimensions, coefficients: dict[str, Any],
                 reference: dict[str, Any], **kwargs: Any):
        super().__init__(name, dims, **kwargs)
        self._cache: dict[tuple[str, str], float] = {}
        self.model = str(coefficients.get("model", "MF52")).upper()
        self.groups = {g: clean(coefficients.get(g) or {}) for g in
                       ("lateral", "longitudinal", "aligning", "overturning",
                        "rolling_resistance", "combined")}
        self.Fz0 = (num(reference, "load", default=None)
                    or num(coefficients, "reference_load_N", default=None) or 4000.0)
        self.p0 = num(reference, "pressure", default=None) or self.pressure or 200000.0
        self.has_combined = bool(self.groups["combined"])
        if not self.has_combined:
            self.notes.append(
                "no `combined` Pacejka group — combined slip falls back to a friction ellipse")

    def c(self, group: str, key: str, default: float = 0.0) -> float:
        """Resolved coefficient value, cached on first use.

        The force routines read around seventy coefficients per evaluation, so
        resolving the type check and default every time showed up clearly in a
        profile. Values are memoised into a plain dict the second time round.
        """
        cache = self._cache
        hit = cache.get((group, key))
        if hit is not None:
            return hit
        val = self.groups.get(group, {}).get(key)
        if val is None or isinstance(val, bool) or not isinstance(val, (int, float)):
            val = self._DEFAULTS.get(key, default)
        val = float(val)
        cache[(group, key)] = val
        return val

    # -- pure slip ------------------------------------------------------

    def _pure_lateral(self, Fz: float, alpha: float, gamma: float,
                      dfz: float, dpi: float, mu_scale: float) -> tuple[float, float, float]:
        """Returns (Fy0, Kya, mu_y) at pure side slip."""
        Cy = self.c("lateral", "pCy1")
        mu_y = ((self.c("lateral", "pDy1") + self.c("lateral", "pDy2") * dfz)
                * (1.0 + self.c("lateral", "pPy3") * dpi
                   + self.c("lateral", "pPy4") * dpi ** 2)
                * (1.0 - self.c("lateral", "pDy3") * gamma ** 2)) * mu_scale
        Dy = mu_y * Fz

        pKy2 = self.c("lateral", "pKy2", 2.0) or 2.0
        Kya = (self.c("lateral", "pKy1", 20.0) * self.Fz0
               * (1.0 + self.c("lateral", "pPy1") * dpi)
               * (1.0 - self.c("lateral", "pKy3") * abs(gamma))
               * np.sin(self.c("lateral", "pKy4", 2.0)
                        * np.arctan(Fz / max(EPS, (pKy2 + self.c("lateral", "pKy5") * gamma ** 2)
                                             * self.Fz0 * (1.0 + self.c("lateral", "pPy2") * dpi)))))
        By = Kya / max(EPS, Cy * Dy)

        Svy_gamma = Fz * (self.c("lateral", "pVy3") + self.c("lateral", "pVy4") * dfz) * gamma
        Svy = Fz * (self.c("lateral", "pVy1") + self.c("lateral", "pVy2") * dfz) + Svy_gamma
        Ky_gamma = Fz * (self.c("lateral", "pKy6") + self.c("lateral", "pKy7") * dfz) \
            * (1.0 + self.c("lateral", "pPy5") * dpi)
        Shy = (self.c("lateral", "pHy1") + self.c("lateral", "pHy2") * dfz) \
            + (Ky_gamma * gamma - Svy_gamma) / max(EPS, Kya)

        ay = alpha + Shy
        Ey = ((self.c("lateral", "pEy1") + self.c("lateral", "pEy2") * dfz)
              * (1.0 + self.c("lateral", "pEy5") * gamma ** 2
                 - (self.c("lateral", "pEy3") + self.c("lateral", "pEy4") * gamma) * np.sign(ay)))
        Ey = min(Ey, 1.0)
        Bay = By * ay
        Fy0 = Dy * np.sin(Cy * np.arctan(Bay - Ey * (Bay - np.arctan(Bay)))) + Svy
        return float(Fy0), float(Kya), float(mu_y)

    def _pure_longitudinal(self, Fz: float, kappa: float, gamma: float,
                           dfz: float, dpi: float, mu_scale: float) -> tuple[float, float]:
        Cx = self.c("longitudinal", "pCx1")
        mu_x = ((self.c("longitudinal", "pDx1") + self.c("longitudinal", "pDx2") * dfz)
                * (1.0 + self.c("longitudinal", "pPx3") * dpi
                   + self.c("longitudinal", "pPx4") * dpi ** 2)
                * (1.0 - self.c("longitudinal", "pDx3") * gamma ** 2)) * mu_scale
        Dx = mu_x * Fz
        Kxk = (Fz * (self.c("longitudinal", "pKx1", 20.0)
                     + self.c("longitudinal", "pKx2") * dfz)
               * np.exp(np.clip(self.c("longitudinal", "pKx3") * dfz, -20.0, 20.0))
               * (1.0 + self.c("longitudinal", "pPx1") * dpi
                  + self.c("longitudinal", "pPx2") * dpi ** 2))
        Bx = Kxk / max(EPS, Cx * Dx)

        Shx = self.c("longitudinal", "pHx1") + self.c("longitudinal", "pHx2") * dfz
        Svx = Fz * (self.c("longitudinal", "pVx1") + self.c("longitudinal", "pVx2") * dfz)
        kx = kappa + Shx
        Ex = ((self.c("longitudinal", "pEx1") + self.c("longitudinal", "pEx2") * dfz
               + self.c("longitudinal", "pEx3") * dfz ** 2)
              * (1.0 - self.c("longitudinal", "pEx4") * np.sign(kx)))
        Ex = min(Ex, 1.0)
        Bkx = Bx * kx
        Fx0 = Dx * np.sin(Cx * np.arctan(Bkx - Ex * (Bkx - np.arctan(Bkx)))) + Svx
        return float(Fx0), float(mu_x)

    # -- aligning torque ------------------------------------------------

    def _aligning(self, Fz: float, alpha: float, gamma: float, dfz: float,
                  Fy0: float, Kya: float, mu_y: float) -> float:
        if not self.groups["aligning"]:
            return 0.0
        R0 = self.unloaded_radius
        Sht = (self.c("aligning", "qHz1") + self.c("aligning", "qHz2") * dfz
               + (self.c("aligning", "qHz3") + self.c("aligning", "qHz4") * dfz) * gamma)
        at = alpha + Sht
        Bt = ((self.c("aligning", "qBz1", 10.0) + self.c("aligning", "qBz2") * dfz
               + self.c("aligning", "qBz3") * dfz ** 2)
              * (1.0 + self.c("aligning", "qBz4") * gamma
                 + self.c("aligning", "qBz5") * abs(gamma)))
        Ct = self.c("aligning", "qCz1")
        Dt = (Fz * (R0 / max(EPS, self.Fz0))
              * (self.c("aligning", "qDz1") + self.c("aligning", "qDz2") * dfz)
              * (1.0 + self.c("aligning", "qDz3") * gamma
                 + self.c("aligning", "qDz4") * gamma ** 2))
        Et = ((self.c("aligning", "qEz1") + self.c("aligning", "qEz2") * dfz
               + self.c("aligning", "qEz3") * dfz ** 2)
              * (1.0 + (self.c("aligning", "qEz4") + self.c("aligning", "qEz5") * gamma)
                 * (2.0 / np.pi) * np.arctan(Bt * Ct * at)))
        Et = min(Et, 1.0)
        Bat = Bt * at
        trail = Dt * np.cos(Ct * np.arctan(Bat - Et * (Bat - np.arctan(Bat)))) * np.cos(alpha)

        Br = self.c("aligning", "qBz9") + self.c("aligning", "qBz10") * Bt * Ct
        Dr = (Fz * R0 * ((self.c("aligning", "qDz6") + self.c("aligning", "qDz7") * dfz)
                         + (self.c("aligning", "qDz8") + self.c("aligning", "qDz9") * dfz) * gamma
                         + (self.c("aligning", "qDz10")
                            + self.c("aligning", "qDz11") * dfz) * gamma * abs(gamma))
              * np.cos(alpha))
        residual = Dr * np.cos(np.arctan(Br * at))
        return float(-trail * Fy0 + residual)

    # -- public ---------------------------------------------------------

    def forces(self, Fz: float, kappa: float, alpha: float, gamma: float,
               Vx: float = 10.0, mu_scale: float = 1.0) -> TireForces:
        if Fz <= EPS:
            return TireForces(0.0, 0.0, 0.0, 0.0, 0.0)
        dfz = (Fz - self.Fz0) / max(EPS, self.Fz0)
        dpi = (self.pressure - self.p0) / max(EPS, self.p0) if self.p0 else 0.0

        Fy0, Kya, mu_y = self._pure_lateral(Fz, alpha, gamma, dfz, dpi, mu_scale)
        Fx0, mu_x = self._pure_longitudinal(Fz, kappa, gamma, dfz, dpi, mu_scale)

        if self.has_combined:
            Fx, Fy = self._combined_mf(Fz, kappa, alpha, gamma, dfz, Fx0, Fy0, mu_y)
        else:
            Fx, Fy = _friction_ellipse(Fx0, Fy0, mu_x * Fz, mu_y * Fz)

        Mz = self._aligning(Fz, alpha, gamma, dfz, Fy0, Kya, mu_y)
        Mx = self._overturning(Fz, gamma, Fy)
        My = self._rolling_resistance(Fz, Vx)
        return TireForces(Fx, Fy, Mz, Mx, My)

    def _combined_mf(self, Fz: float, kappa: float, alpha: float, gamma: float,
                     dfz: float, Fx0: float, Fy0: float,
                     mu_y: float) -> tuple[float, float]:
        Bxa = ((self.c("combined", "rBx1", 13.0) + self.c("combined", "rBx3") * gamma ** 2)
               * np.cos(np.arctan(self.c("combined", "rBx2", 10.0) * kappa)))
        Cxa = self.c("combined", "rCx1", 1.0)
        Shxa = self.c("combined", "rHx1")
        Gxa = _weighting(Bxa, Cxa, alpha + Shxa, Shxa)

        Byk = ((self.c("combined", "rBy1", 10.0) + self.c("combined", "rBy4") * gamma ** 2)
               * np.cos(np.arctan(self.c("combined", "rBy2", 7.0)
                                  * (alpha - self.c("combined", "rBy3")))))
        Cyk = self.c("combined", "rCy1", 1.0)
        Shyk = self.c("combined", "rHy1") + self.c("combined", "rHy2") * dfz
        Gyk = _weighting(Byk, Cyk, kappa + Shyk, Shyk)

        Dvyk = (mu_y * Fz * (self.c("combined", "rVy1") + self.c("combined", "rVy2") * dfz
                             + self.c("combined", "rVy3") * gamma)
                * np.cos(np.arctan(self.c("combined", "rVy4") * alpha)))
        Svyk = Dvyk * np.sin(self.c("combined", "rVy5") * np.arctan(
            self.c("combined", "rVy6") * kappa))
        return float(Gxa * Fx0), float(Gyk * Fy0 + Svyk)

    def _overturning(self, Fz: float, gamma: float, Fy: float) -> float:
        if not self.groups["overturning"]:
            return 0.0
        R0 = self.unloaded_radius
        return float(Fz * R0 * (self.c("overturning", "qsx1")
                                - self.c("overturning", "qsx2") * gamma
                                + self.c("overturning", "qsx3") * Fy / max(EPS, self.Fz0)))

    def _rolling_resistance(self, Fz: float, Vx: float) -> float:
        g = self.groups["rolling_resistance"]
        if not g:
            return 0.0
        R0 = self.unloaded_radius
        v = abs(Vx) / 16.7
        return float(-np.sign(Vx) * Fz * R0 *
                     (self.c("rolling_resistance", "qsy1", 0.01)
                      + self.c("rolling_resistance", "qsy3") * v
                      + self.c("rolling_resistance", "qsy4") * v ** 4))


def _weighting(B: float, C: float, arg: float, shift: float) -> float:
    """Pacejka cosine weighting function, normalised to 1 at zero slip."""
    top = np.cos(C * np.arctan(B * arg))
    bottom = np.cos(C * np.arctan(B * shift))
    if abs(bottom) < EPS:
        return 1.0
    return float(np.clip(top / bottom, 0.0, 1.0))


def _friction_ellipse(Fx0: float, Fy0: float, Fx_max: float,
                      Fy_max: float) -> tuple[float, float]:
    """Scale pure-slip forces onto the friction ellipse.

    Used when a file has no `combined` coefficient group. It reproduces the
    essential coupling — grip spent longitudinally is unavailable laterally —
    without pretending to the fidelity of fitted weighting functions.
    """
    nx = Fx0 / max(EPS, abs(Fx_max))
    ny = Fy0 / max(EPS, abs(Fy_max))
    demand = np.hypot(nx, ny)
    if demand <= 1.0:
        return float(Fx0), float(Fy0)
    return float(Fx0 / demand), float(Fy0 / demand)


# ── TMeasy ────────────────────────────────────────────────────────────

class TMeasyTire(TireModel):
    """TMeasy (Rill): a force-slip curve defined by slope, peak and slide plateau.

    Parameters are physically meaningful, so this is also the model `synthesise`
    generates when a file has tire dimensions but no fitted coefficients.
    """

    kind = "tmeasy"

    def __init__(self, name: str, dims: TireDimensions, params: dict[str, Any], **kwargs: Any):
        super().__init__(name, dims, **kwargs)
        p = clean(params)
        self.Fz0 = num(p, "Fz0", default=None) or 4000.0
        self.dFy0 = num(p, "dFy0", default=None) or 15.0 * self.Fz0
        self.Fy_max = num(p, "Fy_max", default=None) or self.mu_reference * self.Fz0
        self.alpha_max = num(p, "alpha_max", default=None) or 0.14
        self.Fy_slide = num(p, "Fy_slide", default=None) or 0.85 * self.Fy_max
        self.alpha_slide = num(p, "alpha_slide", default=None) or 0.35
        self.dFx0 = num(p, "dFx0", default=None) or 18.0 * self.Fz0
        self.Fx_max = num(p, "Fx_max", default=None) or self.mu_reference * self.Fz0
        self.kappa_max = num(p, "kappa_max", default=None) or 0.11
        self.Fx_slide = num(p, "Fx_slide", default=None) or 0.85 * self.Fx_max
        self.kappa_slide = num(p, "kappa_slide", default=None) or 0.4
        self.camber_stiffness = num(p, "camber_stiffness", default=None) or 0.6 * self.Fz0
        self.trail_max = num(p, "pneumatic_trail_max", default=None) or 0.03
        self.trail_slide = num(p, "pneumatic_trail_slide", default=None) or 0.0
        self.rolling_resistance = 0.012
        # SVJ §11.10.3 load sensitivities, in force per newton of extra load.
        self.dFy0_dFz = num(p, "dFy0_dFz", default=None)
        self.Fy_max_dFz = num(p, "Fy_max_dFz", default=None)

    @staticmethod
    def _curve(slip: float, slope: float, peak: float, slip_peak: float,
               slide: float, slip_slide: float) -> float:
        """Rising / peak / falling / plateau force-slip characteristic."""
        s = abs(slip)
        sign = np.sign(slip)
        if s <= EPS:
            return 0.0
        if s < slip_peak:
            # Cubic through the origin with the given slope, flat at the peak.
            x = s / slip_peak
            k = slope * slip_peak / max(EPS, peak)
            shape = k * x - (2.0 * k - 3.0) * x ** 2 + (k - 2.0) * x ** 3
            return float(sign * peak * shape)
        if s < slip_slide:
            x = (s - slip_peak) / max(EPS, slip_slide - slip_peak)
            blend = 0.5 * (1.0 + np.cos(np.pi * x))       # smooth peak → plateau
            return float(sign * (slide + (peak - slide) * blend))
        return float(sign * slide)

    def _load_scale(self, Fz: float) -> float:
        """Degressive load sensitivity: grip per newton falls as load rises."""
        ratio = Fz / max(EPS, self.Fz0)
        return ratio * (1.0 - 0.08 * (ratio - 1.0))

    def _peak_and_stiffness(self, Fz: float) -> tuple[float, float, float]:
        """Peak lateral force, sliding force and cornering stiffness at this load.

        Uses the file's explicit load derivatives when it has them, and a
        degressive ratio otherwise.
        """
        excess = Fz - self.Fz0
        if self.Fy_max_dFz is not None:
            Fy_max = max(EPS, self.Fy_max + self.Fy_max_dFz * excess)
        else:
            Fy_max = self.Fy_max * self._load_scale(Fz)
        slide = Fy_max * (self.Fy_slide / max(EPS, self.Fy_max))
        if self.dFy0_dFz is not None:
            stiffness = max(EPS, self.dFy0 + self.dFy0_dFz * excess)
        else:
            stiffness = self.dFy0 * Fz / max(EPS, self.Fz0)
        return Fy_max, slide, stiffness

    def forces(self, Fz: float, kappa: float, alpha: float, gamma: float,
               Vx: float = 10.0, mu_scale: float = 1.0) -> TireForces:
        if Fz <= EPS:
            return TireForces(0.0, 0.0, 0.0, 0.0, 0.0)
        scale = self._load_scale(Fz) * mu_scale
        stiffness_scale = Fz / max(EPS, self.Fz0)

        Fy_max, Fy_slide, dFy0 = self._peak_and_stiffness(Fz)
        Fy_max, Fy_slide = Fy_max * mu_scale, Fy_slide * mu_scale
        Fx_max, Fx_slide = self.Fx_max * scale, self.Fx_slide * scale

        Fy0 = self._curve(alpha, dFy0, Fy_max, self.alpha_max, Fy_slide, self.alpha_slide)
        Fy0 += self.camber_stiffness * stiffness_scale * gamma
        Fx0 = self._curve(kappa, self.dFx0 * stiffness_scale, Fx_max,
                          self.kappa_max, Fx_slide, self.kappa_slide)

        Fx, Fy = _friction_ellipse(Fx0, Fy0, Fx_max, Fy_max)
        blend = min(1.0, abs(alpha) / max(EPS, self.alpha_slide))
        trail = self.trail_max + (self.trail_slide - self.trail_max) * blend
        Mz = -trail * Fy
        My = -np.sign(Vx) * self.rolling_resistance * Fz * self.effective_radius(Fz)
        return TireForces(Fx, Fy, float(Mz), 0.0, float(My))


# ── Brush ─────────────────────────────────────────────────────────────

class BrushTire(TireModel):
    """Brush model: contact patch of elastic bristles, forces from first principles."""

    kind = "brush"

    def __init__(self, name: str, dims: TireDimensions, params: dict[str, Any], **kwargs: Any):
        super().__init__(name, dims, **kwargs)
        p = clean(params)
        self.contact_length_ref = num(p, "contact_length", default=None) or 0.15
        self.width = num(p, "contact_width", default=None) or dims.tread_width
        self.cy = num(p, "tread_stiffness_lat", default=None) or 6.0e6
        self.cx = num(p, "tread_stiffness_lon", default=None) or 8.0e6
        self.mu_static = num(p, "mu_static", default=None) or self.mu_reference
        self.mu_kinetic = num(p, "mu_kinetic", default=None) or 0.9 * self.mu_static

    def half_contact_length(self, Fz: float) -> float:
        """Half the contact patch length, from how far the carcass is squashed.

        A tire of radius R flattened by `d` touches the road over a chord of
        half-length `sqrt(R*d)`. Deriving it from load rather than using the
        declared constant is what makes brush cornering stiffness rise with
        load, as a real tire's does.
        """
        deflection = max(EPS, Fz / max(EPS, self.vertical_stiffness))
        return float(np.sqrt(self.unloaded_radius * deflection))

    def forces(self, Fz: float, kappa: float, alpha: float, gamma: float,
               Vx: float = 10.0, mu_scale: float = 1.0) -> TireForces:
        if Fz <= EPS:
            return TireForces(0.0, 0.0, 0.0, 0.0, 0.0)
        mu = self.mu_static * mu_scale
        a = self.half_contact_length(Fz)
        # Parabolic-pressure brush: C = 2 * (stiffness per unit area) * width * a^2.
        Cy = 2.0 * self.cy * self.width * a ** 2
        Cx = 2.0 * self.cx * self.width * a ** 2
        sy = np.tan(alpha)
        sx = kappa / max(EPS, 1.0 + kappa)
        sigma = np.hypot(Cx * sx, Cy * sy)
        peak = mu * Fz
        if sigma <= EPS:
            return TireForces(0.0, 0.0, 0.0, 0.0, 0.0)
        theta = sigma / max(EPS, 3.0 * peak)
        if theta < 1.0:
            magnitude = sigma * (1.0 - theta + theta ** 2 / 3.0)
        else:
            magnitude = peak * self.mu_kinetic / max(EPS, self.mu_static)
        Fx = magnitude * (Cx * sx) / sigma
        Fy = magnitude * (Cy * sy) / sigma
        trail = a / 3.0 * max(0.0, 1.0 - theta)
        return TireForces(float(Fx), float(Fy), float(-trail * Fy), 0.0,
                          float(-np.sign(Vx) * 0.012 * Fz * self.effective_radius(Fz)))


# ── Building a model from an SVJ tire set ─────────────────────────────

# Rough friction levels by service class, used only when a file supplies no
# force model at all. Marked `estimated` wherever they surface.
_MU_BY_CLASS = {"passenger": 1.05, "performance": 1.25, "racing": 1.55,
                "truck": 0.75, "offroad": 0.80, "trailer": 0.75}


_SIZE_CODE = re.compile(
    r"(?:[A-Z]{1,3})?\s*(\d+(?:\.\d+)?)\s*(?:/\s*(\d+(?:\.\d+)?))?\s*[RDB-]\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE)


def parse_size_code(code: str) -> tuple[float, float, float] | None:
    """Parse a tire size code into (section width m, aspect ratio, rim diameter m).

    Handles the metric passenger and truck forms (`245/35R19`, `315/80R22.5`,
    `ST235/80R16`) and the imperial off-road form (`14.00R20`), where the
    leading number is a width in inches and there is no aspect ratio.
    """
    match = _SIZE_CODE.search(str(code or ""))
    if not match:
        return None
    first, aspect, rim = match.groups()
    rim_m = float(rim) * 0.0254
    if aspect is None:
        # Imperial: "14.00R20" is a 14-inch-wide tire, conventionally near-square.
        return float(first) * 0.0254, 1.0, rim_m
    return float(first) / 1000.0, float(aspect) / 100.0, rim_m


def _dimensions(tire_set: dict[str, Any], wheel: dict[str, Any] | None = None) -> TireDimensions:
    d = clean(dig(tire_set, "dimensions", default={}) or {})
    wheel = clean(wheel or {})
    parsed = parse_size_code(tire_set.get("size_code", ""))

    width = (num(d, "section_width", default=None)
             or num(wheel, "tire_section_width", default=None)
             or (parsed[0] if parsed else 0.205))
    aspect = (num(d, "aspect_ratio", default=None)
              or num(wheel, "tire_aspect_ratio", default=None)
              or (parsed[1] if parsed else 0.5))
    rim = (num(tire_set, "rim", "diameter", default=None)
           or num(wheel, "rim_diameter", default=None)
           or (num(d, "rim_diameter_code", default=None) or 0.0) * 0.0254
           or (parsed[2] if parsed else None)
           or (num(d, "overall_diameter", default=0.0) - 2.0 * width * aspect)
           or 17.0 * 0.0254)
    overall = (num(d, "overall_diameter", default=None)
               or num(wheel, "tire_outer_diameter", default=None)
               or rim + 2.0 * width * aspect)
    tread = num(d, "tread_width", default=None) or 0.85 * width
    return TireDimensions(width, aspect, rim, overall, tread)


def _estimate_vertical_stiffness(dims: TireDimensions, pressure: float) -> float:
    """Radial stiffness from inflation pressure and contact geometry.

    Calibrated against the MX-5 template (205/45R17 at 200 kPa declares
    210 kN/m; this returns about 201 kN/m) and gives the right order for
    truck tires at 850 kPa.
    """
    return 4.0 * pressure * float(np.sqrt(dims.unloaded_radius * dims.section_width))


def _service_class(tire_set: dict[str, Any], dims: TireDimensions) -> str:
    """Guess what a tire is for, to pick a friction level when nothing is fitted."""
    text = " ".join(str(tire_set.get(k, "")) for k in ("description", "size_code")).lower()
    # Ordered: a "drive/trailer axle tyre" is a truck tire, not a trailer tire.
    for key, name in (("race", "racing"), ("slick", "racing"),
                      ("offroad", "offroad"), ("off-road", "offroad"), ("xzl", "offroad"),
                      ("drive", "truck"), ("steer", "truck"), ("truck", "truck"),
                      ("trailer", "trailer"),
                      ("sport", "performance"), ("summer", "performance")):
        if key in text:
            return name
    if dims.rim_diameter > 0.50 or dims.section_width > 0.30:
        return "truck"
    return "passenger"


def _load_rating(tire_set: dict[str, Any]) -> float | None:
    """Rated load in newtons from the tire's load index, if it carries one."""
    index = num(tire_set, "construction", "load_index", default=None)
    if index is None:
        return None
    # ETRTO load index table is logarithmic; this fit is within a few percent
    # across the passenger and truck range (index 60 = 250 kg, 150 = 3350 kg).
    kilograms = 45.0 * (1.0293 ** float(index))
    return kilograms * 9.80665


def synthesise(name: str, tire_set: dict[str, Any], dims: TireDimensions,
               pressure: float, rated_load: float | None) -> tuple[dict[str, Any], list[str]]:
    """Invent TMeasy parameters from tire size, pressure and service class.

    Every SVJ vehicle shipped with the standard except the MX-5 describes its
    tires by size alone. Refusing to simulate them would make the tool useless
    on most real files, so a plausible tire is generated from published rules of
    thumb: cornering stiffness scales with rated load, peak slip angle with
    aspect ratio, and friction with what the tire is for.
    """
    service = _service_class(tire_set, dims)
    mu = _MU_BY_CLASS[service]
    # Best reference load available: the corner's real static load, else the
    # load index, else the load the contact patch can carry at this pressure.
    Fz0 = rated_load or (dims.section_width * dims.unloaded_radius * pressure * 0.30)

    # Stiff, low-profile tires reach their peak at smaller slip angles.
    alpha_max = float(np.clip(0.06 + 0.22 * dims.aspect_ratio, 0.08, 0.20))
    cornering_stiffness_ratio = float(np.clip(24.0 - 22.0 * dims.aspect_ratio, 9.0, 18.0))

    params = {
        "Fz0": Fz0,
        "dFy0": cornering_stiffness_ratio * Fz0,
        "Fy_max": mu * Fz0,
        "alpha_max": alpha_max,
        "Fy_slide": 0.82 * mu * Fz0,
        "alpha_slide": alpha_max * 2.8,
        "dFx0": 1.25 * cornering_stiffness_ratio * Fz0,
        "Fx_max": 1.05 * mu * Fz0,
        "kappa_max": 0.09 + 0.06 * dims.aspect_ratio,
        "Fx_slide": 0.85 * mu * Fz0,
        "kappa_slide": 0.35,
        "camber_stiffness": 0.5 * Fz0,
    }
    size = tire_set.get("size_code") or (
        f"{dims.section_width * 1000:.0f}/{dims.aspect_ratio * 100:.0f}"
        f"R{dims.rim_diameter / 0.0254:g}")
    note = (f"tire '{name}' has no fitted force model — TMeasy parameters synthesised from "
            f"size {size} as a '{service}' tire (mu={mu:.2f}, reference load "
            f"{Fz0 / 1000:.1f} kN). Estimated, not measured.")
    return params, [note]


def build_tire(name: str, tire_set: dict[str, Any], *,
               wheel: dict[str, Any] | None = None,
               pressure_override: float | None = None,
               reference_load: float | None = None,
               prefer: str | None = None) -> TireModel:
    """Build the best available force model for one SVJ tire set.

    Preference order is Pacejka, then TMeasy, then brush, then a synthesised
    TMeasy — matching SVJ's own fidelity guide (§11.12.1). `prefer` forces a
    particular model where a set carries several, which is how the MX-5's four
    models can be compared against each other.

    `reference_load` is the station's static vertical load. It is the most
    reliable anchor for a synthesised tire — better than the load index, which
    is a legal rating rather than an operating point, and far better than
    guessing from contact geometry.
    """
    tire_set = tire_set or {}
    dims = _dimensions(tire_set, wheel)
    pressure = (pressure_override or num(tire_set, "pressure", default=None) or 220000.0)
    construction = clean(dig(tire_set, "construction", default={}) or {})

    vertical = (num(construction, "vertical_stiffness", default=None)
                or _estimate_vertical_stiffness(dims, pressure))
    damping = num(construction, "vertical_damping", default=None) or 0.002 * vertical
    rolling_radius = (num(construction, "rolling_radius", default=None)
                      or 0.98 * dims.unloaded_radius)
    rated = reference_load or _load_rating(tire_set)

    common = dict(
        vertical_stiffness=vertical,
        vertical_damping=damping,
        rolling_radius=rolling_radius,
        pressure=pressure,
        relaxation_lateral=num(tire_set, "relaxation", "lateral", default=None) or 0.3,
        relaxation_longitudinal=num(tire_set, "relaxation", "longitudinal", default=None) or 0.1,
        mu_reference=_MU_BY_CLASS[_service_class(tire_set, dims)],
    )

    available = [k for k in ("pacejka", "tmeasy", "brush") if tire_set.get(k)]
    choice = prefer if (prefer in available) else (available[0] if available else None)

    if choice == "pacejka":
        return PacejkaTire(name, dims, tire_set["pacejka"],
                           clean(dig(tire_set, "reference", default={}) or {}),
                           source="file", **common)
    if choice == "tmeasy":
        return TMeasyTire(name, dims, tire_set["tmeasy"], source="file", **common)
    if choice == "brush":
        return BrushTire(name, dims, tire_set["brush"], source="file", **common)

    params, notes = synthesise(name, tire_set, dims, pressure, rated)
    return TMeasyTire(name, dims, params, source="estimated", notes=tuple(notes), **common)
