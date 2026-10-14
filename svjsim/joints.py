"""Joint types of suspension links (SVJ 9.2.3.2): what they mean for the solver.

A link has an **outboard** joint (link to upright, `joint_type`) and an
**inboard** joint (link to chassis, `inboard_joint_type`). Either may be left
out, in which case the spec says converters MUST resolve it from the table in
`infer_joint_types` - the same table `svj.multibody` ships; a test keeps the two
in step. An explicit value always wins.

The solver only ever sees the upright's pose, so what matters about a link is
the constraint it puts on that pose. `plan_link` turns the joint pair into one
of these models:

`distance`   Each inboard pickup keeps a fixed distance from the outboard
             joint: a rod (spherical both ends), or an A-arm (revolute or two
             spherical bushings inboard, spherical outboard), where the two
             distances together are the arc the arm sweeps.
`strut`      A telescoping strut: its axis keeps passing through the top mount.
`carrier`    The upright is rigid with the link, which pivots about the axis
             through its inboard points (trailing, semi-trailing and swing
             arms, a pendulum pin): one freedom, solved as a single angle.
`revolute_pair`
             Revolute at both ends with parallel axes (a parallelogram's
             beams): the upright may only turn about the axis, and the outboard
             joint stays on its arc and in its plane. Four constraints.
`cylinder`   A cylindrical inboard joint on an arm: the arm may also slide
             along its pivot axis, so only the distance from that axis holds.
`pinned`     A joint pinned in space: a link rigid with the chassis ending in a
             ball joint (the ball joint cannot move), or a rigid upright hung
             from a single ball joint (the pivot point cannot move).

SVJ gives hardpoints but no joint axes, so an axis is read from the two inboard
points where there are two, and otherwise assumed lateral (+Y), the usual
direction of a trailing arm or a parallelogram beam. A joint SVJSim cannot turn
into a constraint (a `universal` or `planar` outboard joint, a revolute joint on
a single ball-joint pickup) is reduced to the position constraint it implies,
and the plan says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field

INBOARD_JOINTS = frozenset({"revolute", "spherical", "cylindrical", "fixed", "bushing"})
OUTBOARD_JOINTS = frozenset({"spherical", "revolute", "cylindrical", "translational",
                             "universal", "fixed", "planar", "convel", "hooke"})

# Joints that, as the outboard joint of an ordinary link, only pin the joint's
# position: whatever rotation they leave free or lock is not modelled.
POSITION_ONLY_OUTBOARD = frozenset({"spherical", "convel", "universal", "hooke"})

_STRUT_FAMILY = frozenset({"macpherson", "chapman_strut"})
_ARM_FAMILY = frozenset({"double_wishbone", "macpherson", "chapman_strut", "multi_link", "custom"})
_PIVOT_ARM_FAMILY = frozenset({"trailing_arm", "semi_trailing_arm", "torsion_beam", "parallelogram"})

# Topologies where one arm normally carries the upright rigidly. Used only when
# a link states no joints and the table cannot infer them either (a swing axle's
# arm): the rule SVJSim always used then still applies.
CARRIER_TOPOLOGIES = frozenset({"trailing_arm", "semi_trailing_arm", "torsion_beam",
                                "swing_axle", "pendulum_axle"})
BEAM_TOPOLOGIES = frozenset({"solid_axle", "de_dion"})


def infer_joint_types(system_type: str | None, link_type: str | None) -> tuple[str | None, str | None]:
    """`(outboard, inboard)` for a link that states none; `None` = not inferable."""
    kind = link_type or "arm"
    if kind in ("rod", "torque_rod"):
        return "spherical", "spherical"
    if kind == "strut" and system_type in _STRUT_FAMILY:
        return "translational", "fixed"
    if kind == "arm":
        if system_type in _ARM_FAMILY:
            return "spherical", "revolute"
        if system_type in _PIVOT_ARM_FAMILY:
            return "revolute", "revolute"
    if kind == "pivot" and system_type == "pendulum_axle":
        return "revolute", None
    return None, None


@dataclass
class JointPair:
    outboard: str | None
    inboard: str | None
    outboard_source: str | None      # "explicit" | "inferred" | None
    inboard_source: str | None


def resolve_joints(spec: dict, system_type: str | None, issues: list[str] | None = None) -> JointPair:
    """The effective joints of one link spec, explicit values first."""
    inferred_out, inferred_in = infer_joint_types(system_type, spec.get("type"))
    out, inb = spec.get("joint_type"), spec.get("inboard_joint_type")
    if out is not None and out not in OUTBOARD_JOINTS:
        if issues is not None:
            issues.append(f"joint_type {out!r} is not a known outboard joint - ignored")
        out = None
    if inb is not None and inb not in INBOARD_JOINTS:
        if issues is not None:
            issues.append(f"inboard_joint_type {inb!r} is not a known inboard joint - ignored")
        inb = None
    return JointPair(
        outboard=out or inferred_out, inboard=inb or inferred_in,
        outboard_source="explicit" if out else ("inferred" if inferred_out else None),
        inboard_source="explicit" if inb else ("inferred" if inferred_in else None))


@dataclass
class LinkPlan:
    model: str                                   # see the module docstring
    lateral_axis_assumed: bool = False           # an axis had to be assumed (+Y)
    notes: list[str] = field(default_factory=list)
    demoted_carrier: bool = False                # topology says carrier, the joints say not


def plan_link(*, kind: str, joints: JointPair, inboard_count: int, at_wheel_center: bool,
              system_type: str | None) -> LinkPlan:
    """How a link constrains its upright, from its joints and where it attaches."""
    out, inb = joints.outboard, joints.inboard
    two_points = inboard_count >= 2
    beam = system_type in BEAM_TOPOLOGIES
    topology_carrier = (system_type in CARRIER_TOPOLOGIES and kind == "arm" and at_wheel_center)

    if kind == "strut":
        plan = LinkPlan("strut")
        if out not in (None, "translational", "cylindrical") or inb not in (None, "fixed"):
            plan.notes.append(f"a strut keeps only its axis through the top mount; the "
                              f"{inb or '?'}/{out or '?'} joints are not used beyond that")
        return plan

    if beam:
        # A beam axle is located by the axle body as a whole, so every link of
        # it is a plain distance constraint whatever its joints say.
        return LinkPlan("distance")

    if out is None and inb is None:
        # Nothing stated and nothing to infer: the long-standing topology rule.
        return LinkPlan("carrier" if topology_carrier else "distance",
                        lateral_axis_assumed=topology_carrier and not two_points)

    if kind == "pivot":
        if out == "revolute" and two_points:
            return LinkPlan("carrier")
        return LinkPlan("distance", notes=["a pivot link needs a revolute outboard joint and "
                                           "two inboard points to define its pin axis"])

    if out == "fixed":
        if inb == "fixed":
            return LinkPlan("pinned", notes=["the upright is fixed to the chassis; it cannot move"])
        if inb in ("spherical", "bushing") and not two_points:
            return LinkPlan("pinned", notes=["a rigid upright hung from one ball joint can only "
                                             "swing about it, so the pivot point is pinned"])
        plan = LinkPlan("carrier", lateral_axis_assumed=not two_points)
        if inb == "cylindrical":
            plan.notes.append("a cylindrical inboard joint would also let the arm slide along its "
                              "axis; only the rotation is modelled")
        return plan

    if out == "revolute":
        if at_wheel_center and kind == "arm":
            # Revolute at the hub: the axis is the wheel's own spin axis, so the
            # upright is carried rigidly by the arm as far as K&C goes.
            return LinkPlan("carrier", lateral_axis_assumed=not two_points)
        if inb in (None, "revolute"):
            return LinkPlan("revolute_pair", lateral_axis_assumed=not two_points)
        return LinkPlan("distance", notes=[
            f"a revolute outboard joint behind a {inb} inboard one needs an axis SVJ does not give; "
            "only the distance it implies is held"])

    if out in POSITION_ONLY_OUTBOARD or out is None:
        plan = LinkPlan("distance")
        if out in ("universal", "hooke"):
            plan.notes.append(f"a {out} outboard joint also restricts rotation about two axes "
                              "SVJ does not give; only its position is held")
        if inb == "fixed":
            return LinkPlan("pinned", notes=["a link rigid with the chassis ends in a ball joint "
                                             "that therefore cannot move"])
        if inb == "cylindrical" and two_points:
            return LinkPlan("cylinder", notes=plan.notes)
        if topology_carrier and out is None:
            return LinkPlan("carrier", lateral_axis_assumed=not two_points)
        if topology_carrier and out in POSITION_ONLY_OUTBOARD and kind == "arm":
            # The topology says this arm carries the upright but its joint is a
            # ball joint, which alone does not: whether that is meant is decided
            # once the station's other links are known (see StationKinematics).
            plan.demoted_carrier = True
        return plan

    plan = LinkPlan("distance")
    plan.notes.append(f"a {out} outboard joint on a {kind} is not modelled beyond the distance it "
                      "implies")
    return plan
