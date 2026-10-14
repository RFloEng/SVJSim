"""Where drive torque goes: driven wheels, differentials, centre coupling, lock control.

Engine torque reaches the road in up to three stages, and SVJ describes each:

1. **Centre.** With more than one driven axle the shaft torque is split between
   axles by a fixed `torque_split`, a centre differential (`center_diff`), or a
   torque-limited coupling (`transfer_case.coupling`, SVJ 10.6.1).
2. **Axle.** Each axle's differential shares its torque between its two wheels.
3. **Control.** Any differential may carry `lock_control` (SVJ 10.8.1): a chain
   of stages that modulate its lock from the driving state.

This module holds the parsed description (`Differential`, `CentreSpec`, ...,
built from the document with no dependence on `model.py`) and `DrivePlan`, the
runtime that applies it inside the solver's derivative evaluation.

The spec fixes what each field means but not the dynamics behind it, so where
SVJSim had to choose, the choice is recorded in the build report. They are:

* a `lock` chain is run separately from the static `lock_power` and from the
  static `lock_coast` (the spec's own example only makes sense that way);
* a coupling's primary axle is the one with the larger `torque_split` share,
  the rear when that is silent (`primary_axle` is deferred in the spec);
* the coupling's capacity rises from `ramp_torque` to `max_torque` with the
  speed difference across it, and a `center_diff` given alongside scales it by
  its (possibly modulated) lock; both are torques at the transfer case, so
  they are referred to the wheels through the final drive;
* slip-angle inputs are magnitudes, and `oversteer_factor` is
  `(|a_rear| - |a_front|) / (|a_rear| + |a_front| + 0.01 rad)`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .util import Curve, clean, dig, num

CONTROLLER_INPUTS = ("gear", "brake", "throttle", "speed", "rear_speed_ratio",
                     "oversteer_factor", "slip_angle_front", "slip_angle_rear")
FULL_LOCK_TYPES = frozenset({"locked", "spool"})

# Speed difference (rad/s at the wheels) over which a coupling's capacity
# grows from `ramp_torque` to `max_torque`, and over which it engages at all.
COUPLING_CAPACITY_SPEED = 10.0
COUPLING_ENGAGE_SPEED = 1.0
# A slip-angle floor (rad) so the oversteer indicator stays quiet when the
# car is not sliding.
OVERSTEER_FLOOR = 0.01
# Below this wheel speed (rad/s) an axle speed ratio means nothing.
RATIO_MIN_SPEED = 0.5


# ── Parsed description ────────────────────────────────────────────────

@dataclass
class LockStage:
    input: str
    add: bool                        # False: multiply
    curve: Curve
    tau: float = 0.0                 # low-pass time constant on the output (s)
    lo: float | None = None
    hi: float | None = None
    filter_index: int = -1           # slot in the state's filter block; -1 = none


@dataclass
class LockChain:
    target: str                      # lock | lock_power | lock_coast | preload
    stages: list[LockStage]


@dataclass
class Differential:
    kind: str = "open"
    lock_power: float = 0.0
    lock_coast: float = 0.0
    preload: float = 0.0
    chains: list[LockChain] = field(default_factory=list)

    def static(self) -> tuple[float, float, float]:
        """`(lock_power, lock_coast, preload)` before any control acts."""
        if self.kind in FULL_LOCK_TYPES:
            return 1.0, 1.0, self.preload
        if self.kind == "open":
            return 0.0, 0.0, 0.0
        return self.lock_power, self.lock_coast, self.preload

    @property
    def locks(self) -> bool:
        power, coast, preload = self.static()
        return bool(self.chains) or max(power, coast, preload) > 0.0

    def resolve(self, inputs: dict[str, float], filters: np.ndarray | None,
                rates: np.ndarray | None) -> tuple[float, float, float]:
        """`(lock_power, lock_coast, preload)` for the present driving state.

        Stage outputs go through their low-pass filters when they have one:
        the filtered value is read from `filters` (it is part of the
        integrated state) and its derivative is written into `rates`.
        """
        power, coast, preload = self.static()
        if self.kind in FULL_LOCK_TYPES or self.kind == "open":
            return power, coast, preload
        values = {"lock_power": power, "lock_coast": coast, "preload": preload}
        for chain in self.chains:
            outputs = []
            for stage in chain.stages:
                raw = float(stage.curve.at(inputs[stage.input]))
                if stage.filter_index >= 0 and filters is not None \
                        and stage.filter_index < len(filters):
                    held = float(filters[stage.filter_index])
                    if rates is not None:
                        rates[stage.filter_index] = (raw - held) / stage.tau
                    raw = held
                outputs.append(raw)
            # "lock" drives both locks, each from its own static value.
            for target in (("lock_power", "lock_coast") if chain.target == "lock"
                           else (chain.target,)):
                value = values[target]
                for stage, y in zip(chain.stages, outputs):
                    value = value + y if stage.add else value * y
                    if stage.lo is not None:
                        value = max(value, stage.lo)
                    if stage.hi is not None:
                        value = min(value, stage.hi)
                values[target] = value
        return (float(np.clip(values["lock_power"], 0.0, 1.0)),
                float(np.clip(values["lock_coast"], 0.0, 1.0)),
                max(0.0, values["preload"]))

    def stage_outputs(self, inputs: dict[str, float]) -> dict[int, float]:
        """Raw output of every filtered stage, to start its filter from."""
        return {stage.filter_index: float(stage.curve.at(inputs[stage.input]))
                for chain in self.chains for stage in chain.stages
                if stage.filter_index >= 0}


@dataclass
class CentreSpec:
    """What sits between the front and rear axles of an all-wheel-drive car."""

    kind: str = "fixed"              # fixed | diff | coupling
    split_front: float | None = None  # declared front share of the torque
    diff: Differential | None = None
    ramp_torque: float = 0.0
    max_torque: float = 0.0
    mechanism: str | None = None


# ── Reading the document ──────────────────────────────────────────────

def axle_number(ref: Any) -> int | None:
    """The axle an `axle_ref` (`A2`) or a station name (`RL`, `A3R`) points at."""
    text = str(ref or "").strip().upper()
    if not text:
        return None
    if text[0] == "A" and any(c.isdigit() for c in text):
        return int("".join(c for c in text[1:] if c.isdigit()))
    return {"F": 1, "R": 2}.get(text[0])


def parse_lock_control(raw: Any, counter: list[int], report: Any,
                       label: str) -> list[LockChain]:
    """The chains of a `lock_control` value (one chain or an array of them).

    `counter` hands out filter slots across every differential of the vehicle.
    Stages that read an input SVJSim cannot supply are dropped and reported.
    """
    entries = raw if isinstance(raw, list) else [raw]
    chains: list[LockChain] = []
    for entry in entries:
        entry = clean(entry) if isinstance(entry, dict) else {}
        target = str(entry.get("target", ""))
        if target not in ("lock", "lock_power", "lock_coast", "preload"):
            report.warn(f"{label}: lock_control target {target!r} is not one of "
                        "lock, lock_power, lock_coast, preload - chain ignored")
            continue
        stages: list[LockStage] = []
        for stage in entry.get("stages") or []:
            stage = clean(stage) if isinstance(stage, dict) else {}
            name = str(stage.get("input", ""))
            curve = Curve.from_svj(stage.get("curve"))
            if name not in CONTROLLER_INPUTS:
                report.warn(f"{label}: lock_control input {name!r} is not one SVJSim "
                            "can supply - that stage is skipped")
                continue
            if curve is None or len(curve) < 2:
                report.warn(f"{label}: a lock_control stage on {name!r} has no usable "
                            "curve - skipped")
                continue
            combine = str(stage.get("combine", "multiply"))
            tau = num(stage, "filter_time_constant", default=0.0) or 0.0
            index = -1
            if tau > 0.0:
                index = counter[0]
                counter[0] += 1
            stages.append(LockStage(
                input=name, add=(combine == "add"), curve=curve, tau=tau,
                lo=num(stage, "min", default=None), hi=num(stage, "max", default=None),
                filter_index=index))
        if stages:
            chains.append(LockChain(target=target, stages=stages))
    return chains


def parse_differential(entry: dict[str, Any], counter: list[int], report: Any,
                       label: str) -> Differential:
    entry = clean(entry)
    return Differential(
        kind=str(entry.get("type", "open")),
        lock_power=num(entry, "lock_power", default=0.0) or 0.0,
        lock_coast=num(entry, "lock_coast", default=0.0) or 0.0,
        preload=num(entry, "preload", default=0.0) or 0.0,
        chains=(parse_lock_control(entry["lock_control"], counter, report, label)
                if entry.get("lock_control") else []))


def parse_centre(pt: dict[str, Any], diffs: list[dict[str, Any]], counter: list[int],
                 report: Any) -> CentreSpec | None:
    """The centre device from `transfer_case` and any centre/inter-axle differential."""
    case = clean(pt.get("transfer_case") or {})
    centre_entry = case.get("center_diff")
    if not isinstance(centre_entry, dict):
        centre_entry = next((d for d in diffs
                             if str(d.get("location", "")).lower() in ("center", "inter_axle")),
                            None)
    spec = CentreSpec()
    split = case.get("torque_split")
    if isinstance(split, (list, tuple)) and len(split) == 2 \
            and all(isinstance(v, (int, float)) for v in split) and sum(split) > 0:
        spec.split_front = float(split[0]) / float(split[0] + split[1])
    if isinstance(centre_entry, dict):
        spec.kind = "diff"
        spec.diff = parse_differential(centre_entry, counter, report, "centre differential")
    coupling = case.get("coupling")
    if isinstance(coupling, dict) and num(coupling, "max_torque", default=None):
        spec.kind = "coupling"
        spec.max_torque = float(num(coupling, "max_torque"))
        spec.ramp_torque = min(spec.max_torque, max(0.0, num(coupling, "ramp_torque", default=0.0) or 0.0))
        spec.mechanism = coupling.get("mechanism")
    if spec.kind == "fixed" and spec.split_front is None and not case:
        return None
    if str(case.get("type", "")) == "on_demand" and spec.kind == "fixed"             and spec.split_front is None:
        report.warn("transfer case is on_demand but declares neither a torque_split nor a "
                    "coupling - torque is shared between the driven axles by wheel count")
    return spec


def infer_driven(pt: dict[str, Any], doc: dict[str, Any], stations: Sequence[Any],
                 diffs: list[dict[str, Any]], report: Any) -> list[str]:
    """Which stations the engine drives, from the most to the least explicit source.

    `half_shafts` and `axles[].driven` (already folded into `station.driven`)
    name wheels outright. Failing that the powertrain `layout` (or the
    vehicle's `drive_type`) says which axles are driven, and failing that the
    axles the differentials are located on do. What was used is recorded.
    """
    half_shafts = clean(pt.get("half_shafts") or {})
    named = [s.name for s in stations if s.name in half_shafts]
    if named:
        return named
    named = [s.name for s in stations if s.driven]
    if named:
        return named

    axles = sorted({s.axle for s in stations})
    if not axles:
        return []
    first, last = axles[0], axles[-1]
    layout = str(pt.get("layout") or dig(doc, "vehicle_info", "drive_type", default="")).upper()
    chosen: set[int] | None = {
        "FF": {first}, "FWD": {first},
        "FR": {last}, "MR": {last}, "RR": {last}, "RWD": {last},
        "AWD": set(axles), "4WD": set(axles),
    }.get(layout)
    source = f"the layout ({layout})"
    if chosen is None:
        chosen = set()
        for entry in diffs:
            location = str(entry.get("location", "")).lower()
            if location == "front":
                chosen.add(first)
            elif location == "rear":
                chosen.add(last)
            elif entry.get("axle_ref"):
                number = axle_number(entry["axle_ref"])
                if number in axles:
                    chosen.add(number)
        source = "the axles its differentials are located on"
    if not chosen:
        return []
    names = [s.name for s in stations if s.axle in chosen]
    report.note(f"driven wheels taken from {source}: axle(s) "
                f"{', '.join(str(a) for a in sorted(chosen))} - the file gives no half_shafts "
                "or axles[].driven")
    return names


# ── The runtime ───────────────────────────────────────────────────────

def pair_torques(base_a: float, base_b: float, delta: float, lock: float,
                 preload: float) -> tuple[float, float]:
    """Torque at the two outputs of a limited-slip differential.

    `delta` is output a's speed less output b's, so a positive value moves
    torque from a to b, the slower one. The lock term is limited to the
    smaller nominal share, which keeps the faster output from being pushed
    negative by a lock that is only ever a fraction of what is available.
    """
    transfer = (lock * min(abs(base_a), abs(base_b)) * np.tanh(delta * 2.0)
                + preload * np.tanh(delta * 10.0))
    return base_a - transfer, base_b + transfer


class DrivePlan:
    """A driveline resolved against the solver's station order, ready to evaluate."""

    def __init__(self, driveline: Any, stations: Sequence[Any]):
        self.driveline = driveline
        self.n = len(stations)
        # The centre device sits ahead of the axle differentials, so its torques
        # (a coupling's rating, a centre preload) are referred to the wheels
        # through the final drive; the axle differentials' own are not.
        self.centre_to_wheels = float(driveline.final_drive)
        axles = sorted({s.axle for s in stations})
        self.front_axle, self.rear_axle = axles[0], axles[-1]
        self.front_all = [i for i, s in enumerate(stations) if s.axle == self.front_axle]
        self.rear_all = [i for i, s in enumerate(stations) if s.axle == self.rear_axle]

        driven = set(driveline.driven_stations)
        self.groups: list[tuple[int, list[int], Differential]] = []
        for axle in axles:
            idx = [i for i, s in enumerate(stations) if s.axle == axle and s.name in driven]
            # Left wheel first, so a differential's speed difference has a fixed sign.
            idx.sort(key=lambda i: {"L": 0, "C": 1, "R": 2}.get(stations[i].side, 1))
            if idx:
                diff = driveline.axle_diffs.get(axle, driveline.default_diff)
                self.groups.append((axle, idx, diff))
        self.centre = driveline.centre if len(self.groups) == 2 else None

        self._differentials = [g[2] for g in self.groups]
        if self.centre is not None and self.centre.diff is not None:
            self._differentials.append(self.centre.diff)
        # Slots were numbered while parsing, over every differential in the file
        # (including ones on axles that are not driven), so the count is theirs.
        self.filter_count = int(driveline.filter_count)

    # -- controller inputs -------------------------------------------------

    def inputs(self, state: Any, controls: Any) -> dict[str, float]:
        spin = np.abs(state.spin)
        front, rear = float(np.mean(spin[self.front_all])), float(np.mean(spin[self.rear_all]))
        ratio = rear / front if front > RATIO_MIN_SPEED and rear > RATIO_MIN_SPEED else 1.0
        alpha = np.abs(state.slip_angle)
        a_front, a_rear = float(np.mean(alpha[self.front_all])), float(np.mean(alpha[self.rear_all]))
        return {
            "gear": float(controls.gear), "brake": float(controls.brake),
            "throttle": float(controls.throttle), "speed": float(state.ground_speed),
            "rear_speed_ratio": ratio,
            "oversteer_factor": (a_rear - a_front) / (a_rear + a_front + OVERSTEER_FLOOR),
            "slip_angle_front": a_front, "slip_angle_rear": a_rear,
        }

    def initial_filters(self, inputs: dict[str, float]) -> np.ndarray:
        """Filter states that start at what their stages output, not at zero."""
        out = np.zeros(self.filter_count)
        for diff in {id(d): d for d in self._differentials}.values():
            for index, value in diff.stage_outputs(inputs).items():
                out[index] = value
        return out

    # -- torque ------------------------------------------------------------

    def torques(self, state: Any, controls: Any, shaft: float,
                filters: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        """Torque on every station's wheel (N m) and the derivative of the filters."""
        torque = np.zeros(self.n)
        rates = np.zeros(self.filter_count)
        if not self.groups:
            return torque, rates
        inputs = self.inputs(state, controls)
        spin = state.spin

        axle_torque = self._axle_torques(shaft, spin, inputs, filters, rates)
        for (axle, idx, diff), total in zip(self.groups, axle_torque):
            if len(idx) == 2 and diff.locks:
                power, coast, preload = diff.resolve(inputs, filters, rates)
                lock = power if total >= 0.0 else coast
                a, b = idx
                if lock > 0.0 or preload > 0.0:
                    torque[a], torque[b] = pair_torques(
                        total / 2.0, total / 2.0, float(spin[a] - spin[b]), lock, preload)
                    continue
            for i in idx:
                torque[i] = total / len(idx)
        return torque, rates

    def _axle_torques(self, shaft: float, spin: np.ndarray, inputs: dict[str, float],
                      filters: np.ndarray | None, rates: np.ndarray) -> list[float]:
        wheels = [len(idx) for _, idx, _ in self.groups]
        if self.centre is None:
            # One driven axle takes everything; with several and no centre
            # device the shaft is shared by wheel count.
            return [shaft * count / sum(wheels) for count in wheels]

        centre = self.centre
        front_share = (centre.split_front if centre.split_front is not None
                       else wheels[0] / sum(wheels))
        base = [shaft * front_share, shaft * (1.0 - front_share)]
        speeds = [float(np.mean(spin[idx])) for _, idx, _ in self.groups]
        delta = speeds[0] - speeds[1]            # front less rear

        lock, preload = 1.0, 0.0
        if centre.diff is not None and centre.diff.locks:
            power, coast, preload = centre.diff.resolve(inputs, filters, rates)
            lock = power if shaft >= 0.0 else coast
            preload *= self.centre_to_wheels

        if centre.kind == "coupling":
            primary = 0 if front_share > 0.5 else 1
            sign = 1.0 if primary == 0 else -1.0
            slip = sign * delta                  # primary less secondary
            ramp = centre.ramp_torque * self.centre_to_wheels
            capacity = ramp + (centre.max_torque * self.centre_to_wheels - ramp) * \
                float(np.tanh(abs(slip) / COUPLING_CAPACITY_SPEED))
            carried = float(np.sign(slip) * capacity * lock *
                            np.tanh(abs(slip) / COUPLING_ENGAGE_SPEED))
            carried += preload * float(np.tanh(slip * 10.0))
            if carried * shaft > 0.0:
                carried = float(np.sign(carried) * min(abs(carried), abs(shaft)))
            # The coupling moves torque from the primary axle to the secondary.
            secondary = 1 - primary
            result = [0.0, 0.0]
            result[primary] = shaft - carried
            result[secondary] = carried
            return result
        if centre.kind == "diff" and (lock > 0.0 or preload > 0.0):
            front, rear = pair_torques(base[0], base[1], delta, lock, preload)
            return [front, rear]
        return base
