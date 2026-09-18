"""Shared constants, SVJ data helpers and small vector utilities.

Reference frame throughout SVJSim is SAE J670, identical to SVJ itself:
X forward, Y right, Z **down**, origin at the centre of axle 1 at ground level
in the design attitude. Gravity therefore acts along +Z. Hardpoints read from
an SVJ file are used without any transformation.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

G = 9.80665
RHO_AIR_ISA = 1.225
KMH = 1.0 / 3.6


# ── SVJ document access ───────────────────────────────────────────────

def clean(obj: Any) -> Any:
    """Drop SVJ metadata keys (`_est`, `_source`, `_note`, ...) from a dict.

    Provenance keys sit alongside real data everywhere in SVJ, including inside
    otherwise numeric maps like `mass_unsprung_per_corner`.
    """
    if isinstance(obj, dict):
        return {k: v for k, v in obj.items() if not k.startswith("_")}
    return obj


def merge_patch(target: Any, patch: Any) -> Any:
    """Apply an RFC 7396 JSON Merge Patch, as SVJ override files use (§3.4).

    A null value in the patch removes the key; anything else replaces it, with
    objects merged recursively. The target is not modified.
    """
    if not isinstance(patch, dict):
        return patch
    result = dict(target) if isinstance(target, dict) else {}
    for key, value in patch.items():
        if value is None:
            result.pop(key, None)
        else:
            result[key] = merge_patch(result.get(key), value)
    return result


def dig(doc: Any, *path: str | int, default: Any = None) -> Any:
    """Walk a nested dict/list by key path, returning `default` if absent."""
    cur = doc
    for key in path:
        if isinstance(key, int):
            if not isinstance(cur, (list, tuple)) or not -len(cur) <= key < len(cur):
                return default
            cur = cur[key]
        else:
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
    return default if cur is None else cur


def num(doc: Any, *path: str | int, default: float | None = None) -> float | None:
    """`dig` restricted to a finite real number."""
    val = dig(doc, *path, default=None)
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return default
    val = float(val)
    return val if np.isfinite(val) else default


def vec3(doc: Any, *path: str | int, default: Sequence[float] | None = None) -> np.ndarray | None:
    """Read an `[x, y, z]` array. SVJ v0.96 also allows `{position, orientation}`."""
    val = dig(doc, *path, default=None)
    if isinstance(val, dict):
        val = val.get("position")
    if not isinstance(val, (list, tuple)) or len(val) != 3:
        return None if default is None else np.asarray(default, dtype=float)
    try:
        out = np.asarray([float(c) for c in val], dtype=float)
    except (TypeError, ValueError):
        return None if default is None else np.asarray(default, dtype=float)
    return out if np.all(np.isfinite(out)) else (None if default is None else np.asarray(default, float))


# ── Curves ────────────────────────────────────────────────────────────

class Curve:
    """A piecewise-linear SVJ curve: `[[x, y], ...]`, sorted on load.

    Interpolation is linear between points. Outside the range the curve
    extrapolates linearly from the two nearest points, per SVJ §9.4.
    """

    __slots__ = ("x", "y")

    def __init__(self, points: Iterable[Sequence[float]]):
        pts = [(float(p[0]), float(p[1])) for p in points
               if isinstance(p, (list, tuple)) and len(p) >= 2]
        if not pts:
            raise ValueError("curve needs at least one point")
        pts.sort(key=lambda p: p[0])
        self.x = np.array([p[0] for p in pts], dtype=float)
        self.y = np.array([p[1] for p in pts], dtype=float)

    def __len__(self) -> int:
        return len(self.x)

    def __call__(self, q: float) -> float:
        x, y = self.x, self.y
        if len(x) == 1:
            return float(y[0])
        if q <= x[0]:
            slope = (y[1] - y[0]) / (x[1] - x[0]) if x[1] != x[0] else 0.0
            return float(y[0] + slope * (q - x[0]))
        if q >= x[-1]:
            slope = (y[-1] - y[-2]) / (x[-1] - x[-2]) if x[-1] != x[-2] else 0.0
            return float(y[-1] + slope * (q - x[-1]))
        return float(np.interp(q, x, y))

    def at(self, q: np.ndarray) -> np.ndarray:
        """Vectorised evaluation, clamped rather than extrapolated at the ends."""
        return np.interp(q, self.x, self.y)

    @property
    def peak(self) -> tuple[float, float]:
        i = int(np.argmax(self.y))
        return float(self.x[i]), float(self.y[i])

    @classmethod
    def from_svj(cls, points: Any) -> "Curve | None":
        """Build from an SVJ curve, accepting both the pair-array and map forms.

        SVJ aero maps (§13.4) use `{"values": [{"x": .., "value": ..}, ...]}`
        while damper/torque curves use plain `[[x, y], ...]`.
        """
        if isinstance(points, dict):
            values = points.get("values")
            if isinstance(values, list):
                pairs = [(v.get("x"), v.get("value")) for v in values
                         if isinstance(v, dict) and v.get("x") is not None
                         and v.get("value") is not None]
                return cls(pairs) if pairs else None
            return None
        if isinstance(points, (list, tuple)) and points:
            try:
                return cls(points)
            except (ValueError, TypeError, IndexError):
                return None
        return None


# ── Rigid-body maths ──────────────────────────────────────────────────

def skew(v: np.ndarray) -> np.ndarray:
    """Cross-product matrix: `skew(a) @ b == cross(a, b)`."""
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cross product of two 3-vectors.

    `np.cross` spends most of its time normalising axis arguments, which for
    length-3 vectors costs several times the arithmetic. The solver evaluates
    this a dozen times per station per step, so it is worth writing out.
    """
    return np.array([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ])


def rot_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Body→inertial rotation for the SAE 3-2-1 (yaw, pitch, roll) sequence."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])


def euler_rates(roll: float, pitch: float, omega: np.ndarray) -> np.ndarray:
    """Body angular velocity (p, q, r) → Euler angle rates (roll, pitch, yaw)."""
    p, q, r = omega
    cr, sr = np.cos(roll), np.sin(roll)
    cp, tp = np.cos(pitch), np.tan(pitch)
    cp = cp if abs(cp) > 1e-9 else 1e-9
    return np.array([
        p + (q * sr + r * cr) * tp,
        q * cr - r * sr,
        (q * sr + r * cr) / cp,
    ])


def inertia_matrix(spec: Any) -> np.ndarray:
    """Build the symmetric inertia tensor from an SVJ `inertia` object (§7.1).

    SVJ stores products of inertia with the sign already folded into the
    off-diagonal terms of the tensor, i.e. the matrix is
    `[[Ixx, -Ixy, -Ixz], [-Ixy, Iyy, -Iyz], [-Ixz, -Iyz, Izz]]`.
    """
    ixx = num(spec, "Ixx", default=0.0) or 0.0
    iyy = num(spec, "Iyy", default=0.0) or 0.0
    izz = num(spec, "Izz", default=0.0) or 0.0
    ixy = num(spec, "Ixy", default=0.0) or 0.0
    ixz = num(spec, "Ixz", default=0.0) or 0.0
    iyz = num(spec, "Iyz", default=0.0) or 0.0
    return np.array([
        [ixx, -ixy, -ixz],
        [-ixy, iyy, -iyz],
        [-ixz, -iyz, izz],
    ])


def parallel_axis(inertia: np.ndarray, mass: float, offset: np.ndarray) -> np.ndarray:
    """Translate an inertia tensor from a body's own CG by `offset` (Steiner)."""
    d = np.asarray(offset, dtype=float)
    return inertia + mass * (float(d @ d) * np.eye(3) - np.outer(d, d))
