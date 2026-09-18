"""Suspension kinematics — solving SVJ hardpoint geometry into K&C tables.

SVJ is one of the few interchange formats that carries complete suspension
hardpoints, so SVJSim solves the linkage rather than asking the user for
roll-centre heights and bump-steer gradients. For each wheel station this
module sweeps wheel travel (and steering rack travel, where the station is
steered), solves the upright pose that satisfies every link constraint, and
tabulates the result.

What comes out of the sweep:

* wheel-centre path in the body frame — its slope `dy/dj` *is* the jacking
  term, and `dx/dj` *is* the anti-dive/anti-squat term, so the solver never
  needs an explicit roll centre or anti-percentage
* camber and toe versus travel and rack position (bump steer)
* spring, damper and ARB motion ratios measured from the moving mounts

Two solver modes cover the SVJ topologies:

`link`      Upright pose (3 translation + 3 rotation) solved by damped
            least-squares Newton against the link constraints. Redundant
            constraints (beam axles) are least-squares fitted; slack ones keep
            the pose closest to the previous step.
`revolute`  Topologies where one arm rigidly carries the upright — trailing
            and semi-trailing arms, swing and pendulum axles. The assembly has
            a single degree of freedom about the arm's pivot axis, so one
            angle is solved instead of six numbers.

Angle sign conventions are absolute SAE J670 rotations, not the vehicle-relative
signs SVJ uses for static alignment (§9.6); `to_svj_camber` / `to_svj_toe`
convert for reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .util import dig, num, vec3

# Topologies where an `arm` link rigidly carries the upright, making the
# station a single revolute joint rather than a linkage to be solved.
CARRIER_TOPOLOGIES = frozenset({
    "trailing_arm", "semi_trailing_arm", "torsion_beam",
    "swing_axle", "pendulum_axle",
})

# Axles where one rigid body spans both wheels. Each side's links do not locate
# its own upright — the beam does — so an under-constrained solve is expected
# here, and camber is resolved from the axle's roll instead (see `model.py`).
BEAM_TOPOLOGIES = frozenset({"solid_axle", "de_dion"})

# Upright hardpoints an arm/rod link may drive, in the order we prefer them
# when a link does not name its `outboard_ref` explicitly.
_MOUNT_TOLERANCE = 5e-3  # m — how close a spring mount must sit to a hardpoint


def side_sign(y: float) -> float:
    """+1 for a right-hand station, -1 for a left-hand one (SVJ Y is positive right)."""
    return -1.0 if y < 0 else 1.0


def to_svj_camber(camber_abs: float, y: float) -> float:
    """Absolute camber (rotation about +X) → SVJ relative-to-centreline sign."""
    return camber_abs * side_sign(y)


def to_svj_toe(toe_abs: float, y: float) -> float:
    """Absolute steer angle (rotation about +Z) → SVJ toe (positive = toe-in)."""
    return -toe_abs * side_sign(y)


def from_svj_camber(camber_svj: float, y: float) -> float:
    return camber_svj * side_sign(y)


def from_svj_toe(toe_svj: float, y: float) -> float:
    return -toe_svj * side_sign(y)


def rotation_vector_matrix(theta: np.ndarray) -> np.ndarray:
    """Rodrigues rotation matrix for a rotation vector (axis * angle)."""
    angle = float(np.linalg.norm(theta))
    if angle < 1e-12:
        return np.eye(3)
    axis = theta / angle
    K = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def rotation_about_axis(point: np.ndarray, origin: np.ndarray,
                        axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotate `point` about the line through `origin` with direction `axis`."""
    return origin + rotation_vector_matrix(axis * angle) @ (point - origin)


# ── Link constraints ──────────────────────────────────────────────────

@dataclass
class Link:
    """One SVJ suspension link, reduced to the constraints it imposes."""

    name: str
    kind: str                      # "arm" | "rod" | "strut"
    inboard: list[np.ndarray]      # chassis-side pickup points, design position
    outboard: np.ndarray           # upright-side hardpoint, design position
    outboard_ref: str = ""
    steered: bool = False          # inboard point translates with the rack
    carrier: bool = False          # rigidly carries the upright (revolute mode)
    strut_axis: np.ndarray | None = None
    lengths: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.lengths = [float(np.linalg.norm(self.outboard - p)) for p in self.inboard]
        if self.kind == "strut" and self.strut_axis is None and self.inboard:
            axis = self.outboard - self.inboard[0]
            n = np.linalg.norm(axis)
            self.strut_axis = axis / n if n > 1e-9 else np.array([0.0, 0.0, -1.0])

    def inboard_points(self, rack: float) -> list[np.ndarray]:
        """Inboard pickups at the current rack displacement (+Y = rack moves right)."""
        if not self.steered or rack == 0.0:
            return self.inboard
        shift = np.array([0.0, rack, 0.0])
        return [p + shift for p in self.inboard]

    def residuals(self, pose_point: "UprightPose", rack: float) -> np.ndarray:
        """Constraint violation for the current upright pose (zero when satisfied)."""
        out = pose_point.transform(self.outboard)
        inb = self.inboard_points(rack)

        if self.kind == "strut":
            # The strut telescopes, so its length is free; what is constrained is
            # that the upright-fixed strut axis still passes through the top mount.
            axis_pt = pose_point.transform(self.outboard + self.strut_axis * 0.25)
            u = axis_pt - out
            w = inb[0] - out
            u_hat = u / max(np.linalg.norm(u), 1e-12)
            perp = w - float(w @ u_hat) * u_hat
            e1, e2 = _basis_perpendicular(self.strut_axis)
            return np.array([float(perp @ e1), float(perp @ e2)])

        # Arms and rods hold their outboard joint at a fixed distance from each
        # inboard pickup; two spheres intersect in the arc an A-arm sweeps.
        return np.array([
            float(np.linalg.norm(out - p) - L)
            for p, L in zip(inb, self.lengths)
        ])


def _basis_perpendicular(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors spanning the plane perpendicular to `axis`."""
    a = axis / max(np.linalg.norm(axis), 1e-12)
    seed = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(a, seed)
    e1 /= max(np.linalg.norm(e1), 1e-12)
    return e1, np.cross(a, e1)


@dataclass
class UprightPose:
    """Rigid displacement of the upright from its design position.

    A point `p0` fixed to the upright moves to `wheel_center0 + d + R(theta) @ (p0 - wheel_center0)`,
    so `d` is exactly the wheel-centre displacement.
    """

    d: np.ndarray
    theta: np.ndarray
    origin: np.ndarray

    def transform(self, p0: np.ndarray) -> np.ndarray:
        return self.origin + self.d + rotation_vector_matrix(self.theta) @ (p0 - self.origin)

    @property
    def wheel_center(self) -> np.ndarray:
        return self.origin + self.d

    def spin_axis(self) -> np.ndarray:
        """Wheel spin axis, nominally +Y for both sides, rotated by the pose."""
        return rotation_vector_matrix(self.theta) @ np.array([0.0, 1.0, 0.0])

    @staticmethod
    def identity(origin: np.ndarray) -> "UprightPose":
        return UprightPose(np.zeros(3), np.zeros(3), origin)


# ── Solved sample ─────────────────────────────────────────────────────

@dataclass
class KCSample:
    """The kinematic state of one station at one (jounce, rack) operating point."""

    jounce: float
    rack: float
    wheel_center: np.ndarray
    camber: float           # absolute rotation about +X (rad)
    toe: float              # absolute rotation about +Z (rad), positive = nose right
    spring_length: float
    damper_length: float
    arb_travel: float
    residual: float         # worst violation among the solved constraints (m)
    geometry_error: float   # worst violation including dropped redundant ones (m)
    pose_d: np.ndarray      # upright translation from design (m)
    pose_theta: np.ndarray  # upright rotation vector from design (rad)


class StationKinematics:
    """Solves and tabulates the kinematics of one wheel station."""

    def __init__(self, station: str, corner: dict[str, Any], *,
                 travel_limits: tuple[float, float] | None = None,
                 rack_limit: float = 0.0, samples: int = 25, rack_samples: int = 9):
        self.station = station
        self.corner = corner
        self.topology = str(dig(corner, "topology", "system_type", default="custom"))
        self.warnings: list[str] = []

        hardpoints = dig(corner, "topology", "upright", "hardpoints", default={}) or {}
        self.hardpoints = {
            k: v for k, v in ((k, vec3(hardpoints, k)) for k in hardpoints)
            if v is not None
        }
        wc = self.hardpoints.get("wheel_center")
        if wc is None:
            raise ValueError(f"{station}: no wheel_center hardpoint — cannot solve kinematics")
        self.wheel_center0 = wc
        self.side = side_sign(float(wc[1]))

        self.links = self._build_links()
        self.mode = "revolute" if any(l.carrier for l in self.links) else "link"
        self.steered = any(l.steered for l in self.links)
        self.active: list[tuple[int, int]] = []
        self._axis_cache: tuple[np.ndarray, np.ndarray] | None = None
        if self.mode == "link":
            self._select_constraints()

        self.spring_mounts = self._mount_pair("spring")
        self.damper_mounts = self._mount_pair("damper")
        self.arb_mount = self._arb_mount()

        self.rack_limit = rack_limit if self.steered else 0.0
        self.samples = samples
        self.travel_limits = travel_limits or self._declared_travel()
        self.rack_grid = (np.linspace(-self.rack_limit, self.rack_limit, rack_samples)
                          if self.rack_limit > 0 else np.zeros(1))
        self.jounce_grid = np.linspace(*self.travel_limits, samples)
        self.table: KCTable | None = None

    def _declared_travel(self) -> tuple[float, float]:
        """Travel envelope from the bump and rebound stops, with headroom.

        The stops are where the suspension is *meant* to run out of travel, so
        the tables need to cover a little beyond them but not so far that the
        linkage is asked to reach somewhere it physically cannot.
        """
        bump = num(self.corner, "bump_stop", "gap", default=None)
        rebound = num(self.corner, "rebound_stop", "gap", default=None)
        up = (bump if bump and bump > 0 else 0.07) * 1.35
        down = (rebound if rebound and rebound > 0 else 0.07) * 1.35
        return (-min(down, 0.30), min(up, 0.30))

    def probe_travel_limits(self, tolerance: float = 1e-6) -> tuple[float, float]:
        """Shrink the travel range to what the linkage can actually reach.

        Estimated hardpoints frequently describe a linkage that binds before the
        bump stop does. Rather than tabulating poses that do not satisfy the
        constraints, the range is bisected down to the last converging travel.
        """
        lo, hi = self.travel_limits

        def reaches(j: float) -> bool:
            try:
                return self.solve(j)[1] < tolerance
            except (np.linalg.LinAlgError, ValueError):
                return False

        new_hi, new_lo = _bisect_limit(reaches, hi), -_bisect_limit(reaches, lo)
        if new_hi < hi - 1e-6 or new_lo > lo + 1e-6:
            self.warnings.append(
                f"{self.station}: linkage binds before the stops — travel tables clipped to "
                f"{new_lo * 1000:+.0f}/{new_hi * 1000:+.0f} mm (stops asked for "
                f"{lo * 1000:+.0f}/{hi * 1000:+.0f} mm); check the hardpoints")
        self.travel_limits = (new_lo, new_hi)
        self.jounce_grid = np.linspace(new_lo, new_hi, self.samples)
        return self.travel_limits

    def probe_rack_limit(self, tolerance: float = 1e-6) -> float:
        """Shrink the rack sweep to the travel the steering linkage can follow.

        The tie rod can run out of reach before the declared rack travel is used
        up — an inconsistency between `steering.rack_travel` and the tie-rod
        hardpoints, which is common in estimated data. Clipping here keeps the
        rack problem from truncating the wheel-travel tables as well.
        """
        if self.rack_limit <= 0:
            return 0.0
        original = self.rack_limit

        def reaches(r: float) -> bool:
            try:
                return self.solve(0.0, r)[1] < tolerance
            except (np.linalg.LinAlgError, ValueError):
                return False

        limit = min(_bisect_limit(reaches, original), _bisect_limit(reaches, -original))
        if limit < original - 1e-6:
            self.warnings.append(
                f"{self.station}: tie rod cannot follow the full rack travel — rack sweep "
                f"clipped to ±{limit * 1000:.0f} mm of the declared ±{original * 1000:.0f} mm; "
                f"check `steering.rack_travel` against the tie-rod hardpoints")
        self.rack_limit = limit
        self.rack_grid = (np.linspace(-limit, limit, len(self.rack_grid))
                          if limit > 0 else np.zeros(1))
        return limit

    # -- construction --------------------------------------------------

    def _build_links(self) -> list[Link]:
        links: list[Link] = []
        carrier_topology = self.topology in CARRIER_TOPOLOGIES
        for spec in dig(self.corner, "topology", "links", default=[]) or []:
            name = str(spec.get("name", "link"))
            kind = str(spec.get("type", "arm"))
            inboard = [vec3(spec, "inboard_points", i)
                       for i in range(len(spec.get("inboard_points", []) or []))]
            inboard = [p for p in inboard if p is not None]
            ref = str(spec.get("outboard_ref", "") or "")
            key = ref.split(".")[-1] if ref else ""
            outboard = self.hardpoints.get(key)
            if outboard is None or not inboard:
                self.warnings.append(
                    f"{self.station}: link '{name}' has no resolvable outboard hardpoint "
                    f"('{ref}') or no inboard points — ignored")
                continue
            steered = "tie_rod" in name.lower() or key == "steering_tie_rod_end"
            carrier = carrier_topology and kind == "arm" and key == "wheel_center"
            links.append(Link(name=name, kind=kind, inboard=inboard, outboard=outboard,
                              outboard_ref=key, steered=steered, carrier=carrier))
        if not links:
            self.warnings.append(f"{self.station}: no usable links — treated as rigid")
        return links

    def _all_residuals(self, pose: UprightPose, rack: float) -> np.ndarray:
        if not self.links:
            return np.zeros(0)
        return np.concatenate([l.residuals(pose, rack) for l in self.links])

    def _select_constraints(self) -> None:
        """Choose a maximal independent subset of the link constraints.

        A wheel station has six rigid-body freedoms, one of which is the travel
        being commanded, so five independent link constraints define it. Files
        routinely declare more: SVJ's own multi-link guidance (§9.2.1) suggests
        two A-arms *plus* three rods, which is seven. Taken literally that
        linkage is rigid and will not move at all.

        Rather than reject such a file, the constraints are ranked by how much
        new direction each one adds to the constraint Jacobian at the design
        position, and the redundant ones are dropped. Under-defined stations
        keep whatever they have; the remaining freedoms resolve to the pose
        closest to design.
        """
        labels: list[tuple[int, int]] = [
            (li, ci) for li, link in enumerate(self.links)
            for ci in range(len(link.residuals(UprightPose.identity(self.wheel_center0), 0.0)))
        ]
        if not labels:
            return
        pose0 = UprightPose.identity(self.wheel_center0)
        base = self._all_residuals(pose0, 0.0)
        J = _numeric_jacobian(
            lambda v: self._all_residuals(UprightPose(v[:3], v[3:], self.wheel_center0), 0.0),
            np.zeros(6), base)

        # Greedy Gram-Schmidt: repeatedly take the constraint with the largest
        # component orthogonal to everything already selected.
        chosen: list[int] = []
        basis: list[np.ndarray] = []
        remaining = list(range(len(labels)))
        while remaining and len(chosen) < 5:
            best_i, best_res, best_vec = None, 0.0, None
            for i in remaining:
                v = J[i].astype(float)
                for b in basis:
                    v = v - float(v @ b) * b
                norm = float(np.linalg.norm(v))
                if norm > best_res:
                    best_i, best_res, best_vec = i, norm, v
            if best_i is None or best_res < 1e-6 * max(1.0, float(np.linalg.norm(J))):
                break
            chosen.append(best_i)
            basis.append(best_vec / best_res)
            remaining.remove(best_i)

        self.active = [labels[i] for i in chosen]
        if remaining:
            dropped = sorted({self.links[labels[i][0]].name for i in remaining})
            self.warnings.append(
                f"{self.station}: {len(labels)} link constraints for a 5-constraint linkage — "
                f"redundant ones on {', '.join(dropped)} dropped. The topology is "
                f"over-defined; a true multi-link wants `rod` links, not `arm` links")
        elif len(chosen) < 5 and self.topology not in BEAM_TOPOLOGIES:
            self.warnings.append(
                f"{self.station}: only {len(chosen)} independent link constraints "
                f"(5 expected) — the upright is under-located and free directions "
                f"are held at their design values")

    def _mount_pair(self, section: str) -> tuple[np.ndarray, np.ndarray, str] | None:
        """Return (inboard, outboard, attachment) for a spring or damper."""
        inboard = vec3(self.corner, section, "inboard_mount")
        outboard = vec3(self.corner, section, "outboard_mount")
        if inboard is None or outboard is None:
            return None
        return inboard, outboard, self._attachment_of(outboard)

    def _arb_mount(self) -> tuple[np.ndarray, str] | None:
        point = vec3(self.corner, "arb", "drop_link_mount")
        if point is None:
            return None
        return point, self._attachment_of(point)

    def _attachment_of(self, point: np.ndarray) -> str:
        """Which moving body a spring/damper/ARB mount rides on.

        SVJ gives mount coordinates but not the body they belong to. A mount that
        coincides with an upright hardpoint rides on the upright; otherwise the
        nearest arm link that could carry it is used, since dampers most often
        pick up on the lower wishbone.
        """
        for key, hp in self.hardpoints.items():
            if float(np.linalg.norm(point - hp)) < _MOUNT_TOLERANCE:
                return "upright"
        best, best_dist = "upright", float("inf")
        for link in self.links:
            if link.kind != "arm" or len(link.inboard) < 2:
                continue
            axis = link.inboard[1] - link.inboard[0]
            n = np.linalg.norm(axis)
            if n < 1e-9:
                continue
            axis = axis / n
            rel = point - link.inboard[0]
            dist = float(np.linalg.norm(rel - float(rel @ axis) * axis))
            span = float(np.linalg.norm(link.outboard - link.inboard[0]))
            if dist < best_dist and dist <= span * 1.25:
                best, best_dist = f"link:{link.name}", dist
        return best

    # -- solving -------------------------------------------------------

    def solve(self, jounce: float, rack: float = 0.0,
              guess: UprightPose | None = None) -> tuple[UprightPose, float]:
        """Solve the upright pose at a given wheel travel and rack displacement.

        Jounce is compression, positive when the wheel moves up relative to the
        body; since Z points down, the wheel-centre Z target is `z0 - jounce`.
        """
        if self.mode == "revolute":
            return self._solve_revolute(jounce, rack)
        return self._solve_links(jounce, rack, guess)

    def _solve_links(self, jounce: float, rack: float,
                     guess: UprightPose | None) -> tuple[UprightPose, float]:
        pose = UprightPose(
            d=np.array(guess.d, copy=True) if guess else np.zeros(3),
            theta=np.array(guess.theta, copy=True) if guess else np.zeros(3),
            origin=self.wheel_center0,
        )
        x = np.concatenate([pose.d, pose.theta])

        def residual(v: np.ndarray) -> np.ndarray:
            p = UprightPose(v[:3], v[3:], self.wheel_center0)
            per_link = [link.residuals(p, rack) for link in self.links]
            active = [per_link[li][ci] for li, ci in self.active]
            # Driving constraint: the wheel centre sits at the commanded height.
            active.append(p.d[2] + jounce)
            return np.asarray(active)

        for _ in range(60):
            r = residual(x)
            err = float(np.max(np.abs(r)))
            if err < 1e-10:
                break
            J = _numeric_jacobian(residual, x, r)
            # Damped least squares: tolerates both redundant constraints (beam
            # axles) and slack ones (under-defined custom topologies).
            lam = 1e-9
            JtJ = J.T @ J + lam * np.eye(J.shape[1])
            step = np.linalg.solve(JtJ, J.T @ r)
            scale = 1.0
            for _ in range(20):
                trial = x - scale * step
                if float(np.max(np.abs(residual(trial)))) < err:
                    break
                scale *= 0.5
            else:
                break
            x = x - scale * step

        pose = UprightPose(x[:3], x[3:], self.wheel_center0)
        return pose, float(np.max(np.abs(residual(x))))

    def _revolute_axis(self) -> tuple[np.ndarray, np.ndarray]:
        """Pivot origin and unit axis for a carrier arm, with a sanity check.

        The axis is the line through the arm's inboard bushings. If that line is
        (near) vertical the wheel cannot rise by rotating about it at all — the
        bushings have been stacked in Z instead of spread in Y, which happens in
        hand-written files. A lateral axis is substituted so the station still
        moves, which is what a trailing arm is meant to do.
        """
        if self._axis_cache is not None:
            return self._axis_cache
        carrier = next(l for l in self.links if l.carrier)
        origin = carrier.inboard[0]
        axis = (carrier.inboard[1] - carrier.inboard[0]) if len(carrier.inboard) >= 2 \
            else np.array([0.0, 1.0, 0.0])
        n = float(np.linalg.norm(axis))
        axis = axis / n if n > 1e-9 else np.array([0.0, 1.0, 0.0])

        # Sensitivity of wheel height to rotation about this axis.
        lever = self.wheel_center0 - origin
        if abs(float(np.cross(axis, lever)[2])) < 1e-3:
            self.warnings.append(
                f"{self.station}: the '{carrier.name}' pivot axis is {axis.round(3).tolist()}, "
                f"about which the wheel cannot move vertically — its inboard bushings differ "
                f"only in Z. Substituting a lateral pivot axis; fix the inboard_points")
            axis = np.array([0.0, 1.0, 0.0])
            origin = np.mean(carrier.inboard, axis=0)
        self._axis_cache = (origin, axis)
        return self._axis_cache

    def _solve_revolute(self, jounce: float, rack: float) -> tuple[UprightPose, float]:
        origin, axis = self._revolute_axis()
        target_z = self.wheel_center0[2] - jounce

        def height_error(angle: float) -> float:
            return float(rotation_about_axis(self.wheel_center0, origin, axis, angle)[2] - target_z)

        angle, err = 0.0, height_error(0.0)
        for _ in range(60):
            if abs(err) < 1e-12:
                break
            step = 1e-6
            deriv = (height_error(angle + step) - err) / step
            if abs(deriv) < 1e-12:
                break
            angle -= err / deriv
            err = height_error(angle)

        R = rotation_vector_matrix(axis * angle)
        new_wc = rotation_about_axis(self.wheel_center0, origin, axis, angle)
        return UprightPose(d=new_wc - self.wheel_center0, theta=axis * angle,
                           origin=self.wheel_center0), abs(err)

    # -- sampling ------------------------------------------------------

    def sample(self, jounce: float, rack: float = 0.0,
               guess: UprightPose | None = None) -> tuple[KCSample, UprightPose]:
        pose, residual = self.solve(jounce, rack, guess)
        axis = pose.spin_axis()
        camber = float(np.arcsin(np.clip(axis[2], -1.0, 1.0)))
        toe = float(np.arctan2(-axis[0], axis[1]))
        full = self._all_residuals(pose, rack)
        geometry_error = float(np.max(np.abs(full))) if full.size else 0.0
        return KCSample(
            jounce=jounce,
            rack=rack,
            wheel_center=pose.wheel_center,
            camber=camber,
            toe=toe,
            spring_length=self._element_length(self.spring_mounts, pose, rack),
            damper_length=self._element_length(self.damper_mounts, pose, rack),
            arb_travel=self._arb_travel(pose, rack),
            residual=residual,
            geometry_error=geometry_error,
            pose_d=np.array(pose.d, copy=True),
            pose_theta=np.array(pose.theta, copy=True),
        ), pose

    def _moving_point(self, point: np.ndarray, attachment: str,
                      pose: UprightPose, rack: float) -> np.ndarray:
        if attachment == "upright" or not attachment.startswith("link:"):
            return pose.transform(point)
        name = attachment.split(":", 1)[1]
        link = next((l for l in self.links if l.name == name), None)
        if link is None or len(link.inboard) < 2:
            return pose.transform(point)
        # The arm is a rigid body pivoting on its inboard axis; its rotation
        # follows from where the solved outboard joint ended up.
        origin = link.inboard[0]
        axis = link.inboard[1] - link.inboard[0]
        n = np.linalg.norm(axis)
        if n < 1e-9:
            return pose.transform(point)
        axis = axis / n
        target = pose.transform(link.outboard)
        v0 = link.outboard - origin
        v1 = target - origin
        v0p = v0 - float(v0 @ axis) * axis
        v1p = v1 - float(v1 @ axis) * axis
        n0, n1 = np.linalg.norm(v0p), np.linalg.norm(v1p)
        if n0 < 1e-9 or n1 < 1e-9:
            return point
        cos_a = float(np.clip((v0p @ v1p) / (n0 * n1), -1.0, 1.0))
        sin_a = float(np.cross(v0p, v1p) @ axis / (n0 * n1))
        return rotation_about_axis(point, origin, axis, float(np.arctan2(sin_a, cos_a)))

    def _element_length(self, mounts: tuple[np.ndarray, np.ndarray, str] | None,
                        pose: UprightPose, rack: float) -> float:
        if mounts is None:
            return float("nan")
        inboard, outboard, attachment = mounts
        return float(np.linalg.norm(self._moving_point(outboard, attachment, pose, rack) - inboard))

    def _arb_travel(self, pose: UprightPose, rack: float) -> float:
        if self.arb_mount is None:
            return float("nan")
        point, attachment = self.arb_mount
        moved = self._moving_point(point, attachment, pose, rack)
        return float(point[2] - moved[2])       # upward travel of the drop link

    def build_table(self, probe: bool = True) -> "KCTable":
        if probe:
            self.probe_travel_limits()
            self.probe_rack_limit()
        rows: list[list[KCSample]] = []
        for rack in self.rack_grid:
            guess: UprightPose | None = None
            row: list[KCSample] = []
            # Sweep outwards from the design position so each solve starts from
            # its neighbour — Newton then converges in two or three iterations.
            zero = int(np.argmin(np.abs(self.jounce_grid)))
            ordered = list(range(zero, len(self.jounce_grid))) + list(range(zero - 1, -1, -1))
            samples: dict[int, KCSample] = {}
            for count, idx in enumerate(ordered):
                if count == len(self.jounce_grid) - zero:
                    guess = None       # restarting at the design point, sweeping down
                sample, guess = self.sample(float(self.jounce_grid[idx]), float(rack), guess)
                samples[idx] = sample
            row = [samples[i] for i in range(len(self.jounce_grid))]
            rows.append(row)
        table = KCTable(self, rows)
        self.table = table
        return table


def _bisect_limit(reaches, limit: float) -> float:
    """Largest fraction of `limit` (same sign) for which `reaches` holds."""
    if limit == 0.0 or reaches(limit):
        return abs(limit)
    good, bad = 0.0, limit
    for _ in range(24):
        mid = 0.5 * (good + bad)
        if reaches(mid):
            good = mid
        else:
            bad = mid
    return abs(good) * 0.98      # stay clear of the binding point itself


def _numeric_jacobian(fn, x: np.ndarray, f0: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    J = np.empty((len(f0), len(x)))
    for i in range(len(x)):
        dx = np.array(x, copy=True)
        dx[i] += eps
        J[:, i] = (fn(dx) - f0) / eps
    return J


# ── Tabulated result ──────────────────────────────────────────────────

class KCTable:
    """Interpolatable kinematics for one station over jounce and rack travel."""

    def __init__(self, solver: StationKinematics, rows: Sequence[Sequence[KCSample]]):
        self.station = solver.station
        self.topology = solver.topology
        self.steered = solver.steered
        self.wheel_center0 = solver.wheel_center0
        self.jounce = np.asarray(solver.jounce_grid, dtype=float)
        self.rack = np.asarray(solver.rack_grid, dtype=float)

        shape = (len(self.rack), len(self.jounce))
        self.wc_x = np.empty(shape)
        self.wc_y = np.empty(shape)
        self.wc_z = np.empty(shape)
        self.camber = np.empty(shape)
        self.toe = np.empty(shape)
        self.spring_length = np.empty(shape)
        self.damper_length = np.empty(shape)
        self.arb_travel = np.empty(shape)
        self.residual = np.empty(shape)
        self.geometry_error = np.empty(shape)
        # Full upright pose per operating point, so a viewer can move every
        # hardpoint and link rather than just the wheel centre.
        self.pose_d = np.empty(shape + (3,))
        self.pose_theta = np.empty(shape + (3,))
        for i, row in enumerate(rows):
            for k, s in enumerate(row):
                self.wc_x[i, k] = s.wheel_center[0]
                self.wc_y[i, k] = s.wheel_center[1]
                self.wc_z[i, k] = s.wheel_center[2]
                self.camber[i, k] = s.camber
                self.toe[i, k] = s.toe
                self.spring_length[i, k] = s.spring_length
                self.damper_length[i, k] = s.damper_length
                self.arb_travel[i, k] = s.arb_travel
                self.residual[i, k] = s.residual
                self.geometry_error[i, k] = s.geometry_error
                self.pose_d[i, k] = s.pose_d
                self.pose_theta[i, k] = s.pose_theta

        self._mid = int(np.argmin(np.abs(self.rack)))
        self._zero = int(np.argmin(np.abs(self.jounce)))

        # Every channel stacked into one array so a single bilinear lookup
        # returns all of them. The solver reads these on every derivative
        # evaluation for every station, so this is the hot path.
        self._channels = np.stack([
            self.wc_x, self.wc_y, self.wc_z, self.camber, self.toe,
            self.spring_length, self.damper_length, self.arb_travel,
        ])

    # -- interpolation --------------------------------------------------

    @staticmethod
    def _bracket(axis: np.ndarray, value: float) -> tuple[int, int, float]:
        """Indices either side of `value` and the blend between them, clamped."""
        if len(axis) == 1:
            return 0, 0, 0.0
        hi = int(np.searchsorted(axis, value))
        if hi <= 0:
            return 0, 0, 0.0
        if hi >= len(axis):
            return len(axis) - 1, len(axis) - 1, 0.0
        span = axis[hi] - axis[hi - 1]
        return hi - 1, hi, float((value - axis[hi - 1]) / span) if span else 0.0

    def sample_all(self, jounce: float, rack: float = 0.0) -> np.ndarray:
        """Bilinear lookup of every channel at once.

        Order: wheel centre x, y, z, camber, toe, spring length, damper length,
        ARB travel. Values outside the tabulated range clamp to the edge, which
        is where the travel stops should already have taken over.
        """
        rl, rh, rb = self._bracket(self.rack, rack)
        jl, jh, jb = self._bracket(self.jounce, jounce)
        c = self._channels
        low = c[:, rl, jl] * (1.0 - jb) + c[:, rl, jh] * jb
        if rl == rh:
            return low
        high = c[:, rh, jl] * (1.0 - jb) + c[:, rh, jh] * jb
        return low * (1.0 - rb) + high * rb

    def _interp(self, grid: np.ndarray, jounce: float, rack: float) -> float:
        rl, rh, rb = self._bracket(self.rack, rack)
        jl, jh, jb = self._bracket(self.jounce, jounce)
        low = grid[rl, jl] * (1.0 - jb) + grid[rl, jh] * jb
        if rl == rh:
            return float(low)
        high = grid[rh, jl] * (1.0 - jb) + grid[rh, jh] * jb
        return float(low * (1.0 - rb) + high * rb)

    def wheel_center(self, jounce: float, rack: float = 0.0) -> np.ndarray:
        return self.sample_all(jounce, rack)[:3].copy()

    def camber_at(self, jounce: float, rack: float = 0.0) -> float:
        return self._interp(self.camber, jounce, rack)

    def toe_at(self, jounce: float, rack: float = 0.0) -> float:
        return self._interp(self.toe, jounce, rack)

    def jacobian(self, jounce: float, rack: float = 0.0, h: float = 1e-4) -> np.ndarray:
        """d(wheel centre)/d(jounce) — the jacking and anti-effect term."""
        return (self.wheel_center(jounce + h, rack) - self.wheel_center(jounce - h, rack)) / (2 * h)

    def motion_ratio(self, grid: np.ndarray, jounce: float = 0.0,
                     rack: float = 0.0, h: float = 2e-3) -> float:
        """d(element length)/d(jounce): spring or damper travel per wheel travel."""
        hi = self._interp(grid, jounce + h, rack)
        lo = self._interp(grid, jounce - h, rack)
        if not np.isfinite(hi) or not np.isfinite(lo):
            return float("nan")
        return abs(hi - lo) / (2 * h)

    def spring_motion_ratio(self, jounce: float = 0.0, rack: float = 0.0) -> float:
        return self.motion_ratio(self.spring_length, jounce, rack)

    def damper_motion_ratio(self, jounce: float = 0.0, rack: float = 0.0) -> float:
        return self.motion_ratio(self.damper_length, jounce, rack)

    def arb_motion_ratio(self, jounce: float = 0.0, rack: float = 0.0) -> float:
        return self.motion_ratio(self.arb_travel, jounce, rack)

    # -- derived characteristics ---------------------------------------

    @property
    def camber_gain(self) -> float:
        """d(camber)/d(jounce) at design height, rad/m, in absolute SAE sign."""
        h = self.jounce[min(self._zero + 1, len(self.jounce) - 1)] - self.jounce[self._zero]
        if h == 0:
            return 0.0
        return (self.camber_at(self.jounce[self._zero] + h) -
                self.camber_at(self.jounce[self._zero] - h)) / (2 * h)

    @property
    def bump_steer(self) -> float:
        """d(toe)/d(jounce) at design height, rad/m, in absolute SAE sign."""
        h = 0.01
        return (self.toe_at(h) - self.toe_at(-h)) / (2 * h)

    @property
    def half_track_change(self) -> float:
        """d(y)/d(jounce) — the jacking coefficient at design height."""
        return float(self.jacobian(0.0)[1])

    @property
    def recession(self) -> float:
        """d(x)/d(jounce) — longitudinal wheel-centre migration in bump."""
        return float(self.jacobian(0.0)[0])

    @property
    def steer_ratio(self) -> float:
        """d(steer angle)/d(rack travel), rad/m — zero for unsteered stations."""
        if len(self.rack) < 2:
            return 0.0
        per_rack = np.array([np.interp(0.0, self.jounce, self.toe[i]) for i in range(len(self.rack))])
        return float(np.gradient(per_rack, self.rack)[self._mid])

    @property
    def max_residual(self) -> float:
        return float(np.nanmax(self.residual))

    @property
    def max_geometry_error(self) -> float:
        """Worst constraint violation including redundant links that were dropped.

        Large values mean the declared hardpoints do not describe a linkage that
        can actually assemble — usually estimated coordinates, not a solver fault.
        """
        return float(np.nanmax(self.geometry_error))

    def contact_patch(self, radius: float, jounce: float = 0.0, rack: float = 0.0) -> np.ndarray:
        """Tyre contact point in the body frame for a given travel and camber.

        The contact point is the lowest point of a cambered disc of the given
        radius, which sits `r*sin(camber)` inboard of the wheel centre rather
        than directly beneath it.
        """
        wc = self.wheel_center(jounce, rack)
        gamma = self.camber_at(jounce, rack)
        return wc + radius * np.array([0.0, -np.sin(gamma), np.cos(gamma)])

    def contact_jacobian(self, radius: float, jounce: float = 0.0,
                         rack: float = 0.0, h: float = 1e-4) -> np.ndarray:
        """d(contact point)/d(jounce) — lateral term jacks, longitudinal term anti-dives."""
        return (self.contact_patch(radius, jounce + h, rack) -
                self.contact_patch(radius, jounce - h, rack)) / (2 * h)

    def roll_centre_height(self, radius: float) -> float:
        """Kinematic roll-centre height above ground, in metres, positive up.

        Rotating the upright about its instantaneous centre puts the roll centre
        at `h = (dy_contact/dj) * y_contact`, which is the algebraic form of the
        usual front-view construction: draw a line from the contact patch through
        the instantaneous centre and read its height on the vehicle centreline.
        """
        cp = self.contact_patch(radius)
        return float(self.contact_jacobian(radius)[1] * cp[1])
