"""Standard vehicle dynamics tests.

Each maneuver drives the vehicle through a defined procedure and returns a
`Result`: the recorded channels plus the KPIs the test exists to measure. Where
the SVJ file declares a matching `benchmarks` entry, the KPI carries that
comparison too.

The suite covers the four things a vehicle is usually asked to prove — how it
accelerates, how it stops, how it corners, and how it responds to a sudden
input — plus the ISO 3888-2 double lane change, which asks all four at once,
and the NHTSA Fishhook, which asks how close a sudden steering reversal
brings it to lifting two wheels off the road.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from .driver import GearShifter, PathFollower, SpeedController
from .model import SimVehicle
from .solver import Controls, State, VehicleDynamics, step_rk4
from .telemetry import KPI, Recorder, Result, crossing_time

G = 9.80665
KPH = 1.0 / 3.6


# ── Running a maneuver ────────────────────────────────────────────────

def simulate(dynamics: VehicleDynamics, state: State,
             control_fn: Callable[[float, State, float], Controls],
             duration: float, dt: float = 0.002,
             recorder: Recorder | None = None,
             stop: Callable[[float, State], bool] | None = None,
             progress: Callable[[float], None] | None = None) -> State:
    """Step the vehicle forward, recording as it goes.

    `control_fn(time, state, dt)` is the driver; `stop(time, state)` ends the
    run early when the thing being measured has happened; `progress` is called
    with a fraction so a user interface can show something moving.
    """
    time = 0.0
    steps = int(round(duration / dt))
    report_every = max(1, steps // 50)
    for step in range(steps):
        if progress is not None and step % report_every == 0:
            progress(step / max(1, steps))
        controls = control_fn(time, state, dt)
        if recorder is not None:
            derivative, wheels = dynamics.derivatives(state, controls, collect=True)
            recorder.capture(time, state, controls, wheels, derivative)
        if stop is not None and stop(time, state):
            break
        state = step_rk4(dynamics, state, controls, dt)
        time += dt
        if not np.all(np.isfinite(state.data)):
            raise FloatingPointError(
                f"the solution diverged at t = {time:.3f} s — the time step may be too "
                f"large for this vehicle's suspension and tire stiffness")
    return state


class Maneuver:
    """Base class: a named test that produces a `Result`."""

    name = "maneuver"
    label = "Maneuver"
    progress: Callable[[float], None] | None = None

    def run(self, dynamics: VehicleDynamics) -> Result:
        raise NotImplementedError

    def _simulate(self, *args: Any, **kwargs: Any) -> State:
        """`simulate` with this maneuver's progress callback attached."""
        kwargs.setdefault("progress", self.progress)
        return simulate(*args, **kwargs)

    def _result(self, dynamics: VehicleDynamics, recorder: Recorder | None = None) -> Result:
        result = Result(maneuver=self.label, vehicle=dynamics.vehicle.name)
        if recorder is not None:
            result.channels = recorder.finish()
        return result


# ── Straight line ─────────────────────────────────────────────────────

@dataclass
class Acceleration(Maneuver):
    """Full-throttle standing start, with automatic upshifts.

    Measures the numbers a spec sheet quotes: the time to reach given speeds,
    the standing 400 m, and the speed reached by the end of the run.
    """

    name = "acceleration"
    label = "Standing-start acceleration"
    duration: float = 30.0
    dt: float = 0.002
    targets_kph: Sequence[float] = (100.0, 200.0)
    distance_marks: Sequence[float] = (400.0,)
    clutch_release_time: float = 0.8

    def run(self, dynamics: VehicleDynamics) -> Result:
        vehicle = dynamics.vehicle
        state = dynamics.trim(speed=0.0, duration=1.0)
        recorder = Recorder(vehicle.station_names)
        shifter = GearShifter(vehicle)

        launch = self.clutch_release_time

        driven = [i for i, s in enumerate(vehicle.stations)
                  if vehicle.driveline and s.name in vehicle.driveline.driven_stations]

        def control(time: float, st: State, dt: float) -> Controls:
            wheel_speed = float(np.mean([st.spin[i] for i in driven])) if driven else 0.0
            gear, clutch, throttle = shifter.update(wheel_speed, dt)
            if gear == 1 and time < launch:
                # Feed the clutch in rather than dropping it: the engine revs
                # against a slipping plate, which is how a car actually launches.
                clutch = min(clutch, float(np.clip(time / launch, 0.05, 1.0)))
                throttle = 1.0
            return Controls(throttle=throttle, gear=gear, clutch=clutch)

        self._simulate(dynamics, state, control, self.duration, self.dt, recorder)
        result = self._result(dynamics, recorder)
        ch = result.channels
        if not ch:
            return result

        time, speed_kph = ch["time"], ch["speed_kph"]
        distance = np.concatenate([[0.0], np.cumsum(np.diff(time) * ch["speed"][:-1])])
        result.channels["distance"] = distance

        for target in self.targets_kph:
            t = crossing_time(time, speed_kph, target)
            if t is not None:
                result.add_kpi(f"accel_0_{int(target)}kph", f"0–{int(target):d} km/h",
                               t, "s", vehicle)
            else:
                result.notes.append(
                    f"the vehicle did not reach {target:.0f} km/h within {self.duration:.0f} s")

        for mark in self.distance_marks:
            t = crossing_time(time, distance, mark)
            if t is not None:
                result.add_kpi(f"standing_{int(mark)}m", f"Standing {int(mark)} m",
                               t, "s", vehicle)

        top = float(np.max(speed_kph))
        still_rising = bool(speed_kph[-1] > 0.995 * top)
        result.add_kpi("top_speed", "Speed reached", top, "km/h",
                       None if still_rising else vehicle)
        if still_rising:
            result.notes.append(
                f"still accelerating at the end of the run, so {top:.0f} km/h is a floor on "
                f"top speed, not the top speed — extend `duration` to find it")
        result.detail["final_gear"] = shifter.gear
        return result


@dataclass
class Braking(Maneuver):
    """Straight-line stop at full brake pressure from a set speed."""

    name = "braking"
    label = "Straight-line braking"
    entry_speed_kph: float = 100.0
    dt: float = 0.002
    max_duration: float = 15.0

    def run(self, dynamics: VehicleDynamics) -> Result:
        vehicle = dynamics.vehicle
        entry = self.entry_speed_kph * KPH
        state = dynamics.trim(speed=entry, gear=self._gear_for(vehicle, entry), duration=1.5)
        recorder = Recorder(vehicle.station_names)
        start_x = float(state.position[0])

        def control(time: float, st: State, dt: float) -> Controls:
            # A short settling window before the pedal goes down, so the trace
            # shows the undisturbed entry condition.
            brake = 0.0 if time < 0.25 else 1.0
            return Controls(brake=brake, gear=0, clutch=0.0)

        state = self._simulate(dynamics, state, control, self.max_duration, self.dt, recorder,
                         stop=lambda t, st: st.velocity[0] < 0.3)
        result = self._result(dynamics, recorder)
        ch = result.channels
        if not ch:
            return result

        time, speed = ch["time"], ch["speed"]
        moving = speed > 0.3
        distance = ch["x"] - start_x
        stop_index = int(np.flatnonzero(~moving)[0]) if np.any(~moving) else len(time) - 1
        brake_start = crossing_time(time, ch["brake"], 0.5) or 0.0
        start_index = int(np.searchsorted(time, brake_start))

        stopping_distance = float(distance[stop_index] - distance[start_index])
        stopping_time = float(time[stop_index] - brake_start)
        result.add_kpi(f"braking_{int(self.entry_speed_kph)}_0_distance",
                       f"{self.entry_speed_kph:.0f}–0 km/h distance",
                       stopping_distance, "m", vehicle)
        result.add_kpi("braking_mean_deceleration", "Mean deceleration",
                       (speed[start_index] ** 2) / (2 * max(stopping_distance, 1e-3)) / G,
                       "g", vehicle)
        result.add_kpi("braking_peak_deceleration", "Peak deceleration",
                       float(-np.min(ch["ax_g"])), "g", vehicle)
        result.detail["stopping_time"] = stopping_time
        if float(speed[-1]) > 0.3:
            result.notes.append(
                f"still moving at {speed[-1] * 3.6:.1f} km/h when the run ended — "
                f"the distance above is a lower bound")
            result.passed = False
        return result

    @staticmethod
    def _gear_for(vehicle: SimVehicle, speed: float) -> int:
        return max(1, min(3, len(vehicle.driveline.gear_ratios))) if vehicle.driveline else 1


# ── Handling ──────────────────────────────────────────────────────────

@dataclass
class SlowlyIncreasingSteer(Maneuver):
    """ISO 4138 style: hold a speed, wind the steering on gently to the limit.

    A slow ramp keeps the vehicle quasi-steady all the way to the limit, so a
    single run yields both the understeer gradient (from the linear range) and
    the maximum lateral acceleration (from where the curve flattens).
    """

    name = "increasing_steer"
    label = "Slowly increasing steer"
    speed_kph: float = 80.0
    steer_rate_deg_s: float = 12.0
    duration: float = 25.0
    dt: float = 0.002

    def run(self, dynamics: VehicleDynamics) -> Result:
        vehicle = dynamics.vehicle
        speed = self.speed_kph * KPH
        gear = _cruise_gear(vehicle, speed)
        state = dynamics.trim(speed=speed, gear=gear, duration=1.5)
        recorder = Recorder(vehicle.station_names)
        throttle = SpeedController(speed)
        rate = np.radians(self.steer_rate_deg_s)
        max_wheel = vehicle.steering.max_steer_angle * vehicle.steering.overall_ratio

        def control(time: float, st: State, dt: float) -> Controls:
            pedal, brake = throttle.update(st.ground_speed, dt)
            steer = min(rate * max(0.0, time - 1.0), max_wheel)
            return Controls(steering=steer, throttle=pedal, brake=brake, gear=gear)

        self._simulate(dynamics, state, control, self.duration, self.dt, recorder,
                 stop=lambda t, st: abs(st.sideslip) > np.radians(25.0))
        result = self._result(dynamics, recorder)
        ch = result.channels
        if not ch:
            return result

        ay_g = np.abs(ch["ay_g"])
        peak = float(np.max(ay_g))
        result.add_kpi("max_lateral_g", "Maximum lateral acceleration", peak, "g",
                       vehicle, benchmark_id="skidpad_lateral_g")

        gradient, linear_limit = _understeer_gradient(ch, vehicle)
        if gradient is not None:
            result.add_kpi("understeer_gradient", "Understeer gradient",
                           np.degrees(gradient), "deg/g", vehicle)
            result.detail["balance"] = (
                "understeer" if gradient > 0.0005 else
                "oversteer" if gradient < -0.0005 else "neutral")
            result.notes.append(
                f"gradient fitted over the linear range up to {linear_limit:.2f} g; "
                f"the vehicle is {result.detail['balance']}")
        else:
            result.notes.append("not enough linear-range data to fit an understeer gradient")

        result.add_kpi("max_roll_angle", "Body roll at the limit",
                       float(np.degrees(np.max(np.abs(ch["roll"])))), "deg")
        if peak > 0:
            roll_gradient = float(np.degrees(np.max(np.abs(ch["roll"])))) / peak
            result.add_kpi("roll_gradient", "Roll gradient", roll_gradient, "deg/g")
        return result


@dataclass
class ConstantRadius(Maneuver):
    """Skidpad: hold a fixed radius while speed is wound up to the limit."""

    name = "skidpad"
    label = "Constant-radius skidpad"
    radius: float = 30.0
    acceleration_rate: float = 0.35     # m/s of entry speed added per second
    duration: float = 45.0
    dt: float = 0.002
    start_speed_kph: float = 25.0

    def run(self, dynamics: VehicleDynamics) -> Result:
        vehicle = dynamics.vehicle
        speed = self.start_speed_kph * KPH
        gear = _cruise_gear(vehicle, speed)
        state = dynamics.trim(speed=speed, gear=gear, duration=1.5)
        recorder = Recorder(vehicle.station_names)
        throttle = SpeedController(speed)
        # Steering is closed on yaw rate, which is what holding a radius means:
        # the vehicle must turn V/R radians per second.
        steer_pid_gain = vehicle.steering.overall_ratio * 2.5
        command = vehicle.steering.overall_ratio * vehicle.wheelbase / self.radius
        max_wheel = vehicle.steering.max_steer_angle * vehicle.steering.overall_ratio

        def control(time: float, st: State, dt: float) -> Controls:
            nonlocal command
            throttle.target = speed + self.acceleration_rate * max(0.0, time - 2.0)
            pedal, brake = throttle.update(st.ground_speed, dt)
            target_yaw = st.ground_speed / self.radius
            command += steer_pid_gain * (target_yaw - st.omega[2]) * dt
            command = float(np.clip(command, 0.0, max_wheel))
            return Controls(steering=command, throttle=pedal, brake=brake, gear=gear)

        self._simulate(dynamics, state, control, self.duration, self.dt, recorder,
                 stop=lambda t, st: abs(st.sideslip) > np.radians(25.0))
        result = self._result(dynamics, recorder)
        ch = result.channels
        if not ch:
            return result

        # Only count samples where the vehicle was actually on the target radius.
        radius = ch["speed"] / np.maximum(1e-3, np.abs(ch["yaw_rate"]))
        on_line = np.abs(radius - self.radius) < 0.15 * self.radius
        ay = np.abs(ch["ay_g"])[on_line] if np.any(on_line) else np.abs(ch["ay_g"])
        peak = float(np.max(ay)) if len(ay) else 0.0
        result.add_kpi("skidpad_lateral_g", f"Lateral grip at {self.radius:.0f} m radius",
                       peak, "g", vehicle)
        result.detail["radius"] = self.radius
        if not np.any(on_line):
            result.notes.append("the vehicle never settled onto the target radius")
            result.passed = False
        return result


@dataclass
class StepSteer(Maneuver):
    """ISO 7401 step input: how quickly and cleanly the yaw response arrives."""

    name = "step_steer"
    label = "Step steer response"
    speed_kph: float = 80.0
    target_lateral_g: float = 0.4
    duration: float = 6.0
    step_time: float = 1.0
    dt: float = 0.001

    def run(self, dynamics: VehicleDynamics) -> Result:
        vehicle = dynamics.vehicle
        speed = self.speed_kph * KPH
        gear = _cruise_gear(vehicle, speed)
        # Size the step from the linear bicycle model so the test lands near the
        # requested lateral acceleration whatever the vehicle is.
        steer = (self.target_lateral_g * G * vehicle.wheelbase / max(speed ** 2, 1.0)) \
            * vehicle.steering.overall_ratio
        state = dynamics.trim(speed=speed, gear=gear, duration=1.5)
        recorder = Recorder(vehicle.station_names)
        throttle = SpeedController(speed)

        def control(time: float, st: State, dt: float) -> Controls:
            pedal, brake = throttle.update(st.ground_speed, dt)
            # A 0.15 s ramp rather than a true discontinuity: a real steering
            # input cannot be instantaneous, and neither can the integrator.
            fraction = float(np.clip((time - self.step_time) / 0.15, 0.0, 1.0))
            return Controls(steering=steer * fraction, throttle=pedal,
                            brake=brake, gear=gear)

        self._simulate(dynamics, state, control, self.duration, self.dt, recorder)
        result = self._result(dynamics, recorder)
        ch = result.channels
        if not ch:
            return result

        time, yaw_rate = ch["time"], np.abs(ch["yaw_rate"])
        after = time >= self.step_time
        settled = float(np.mean(yaw_rate[time >= time[-1] - 0.5]))
        peak = float(np.max(yaw_rate[after])) if np.any(after) else 0.0

        if settled > 1e-4:
            t90 = crossing_time(time[after], yaw_rate[after], 0.9 * settled)
            if t90 is not None:
                result.add_kpi("yaw_response_time", "Yaw rate 90% response time",
                               t90 - self.step_time, "s", vehicle)
            result.add_kpi("yaw_overshoot", "Yaw rate overshoot",
                           100.0 * (peak / settled - 1.0), "%", vehicle)
            result.add_kpi("steady_yaw_rate", "Steady-state yaw rate",
                           float(np.degrees(settled)), "deg/s")
            lateral = np.abs(ch["ay_g"])
            t90_ay = crossing_time(time[after], lateral[after],
                                   0.9 * float(np.mean(lateral[time >= time[-1] - 0.5])))
            if t90_ay is not None:
                result.add_kpi("lateral_response_time", "Lateral acceleration 90% response",
                               t90_ay - self.step_time, "s")
        else:
            result.notes.append("the vehicle produced no measurable yaw response")
            result.passed = False
        result.detail["steer_input_deg"] = float(np.degrees(steer))
        return result


# ── Obstacle avoidance ────────────────────────────────────────────────

@dataclass
class MooseTest(Maneuver):
    """ISO 3888-2 double lane change, run at rising speed until it fails.

    Geometry follows ISO 3888-2: five sections of 12, 13.5, 11, 12.5 and 12 m,
    gate widths of `1.1 x vehicle width + 0.25` for the entry and offset lanes
    and `1.3 x vehicle width + 0.25` for the exit, with the offset lane placed
    1 m clear of the entry lane. The reported speed is the highest entry speed
    at which the driver model keeps the vehicle inside every gate, so it
    measures the vehicle and this controller together — treat it as comparative
    rather than absolute.
    """

    name = "moose_test"
    label = "Moose test (ISO 3888-2)"
    vehicle_width: float | None = None
    start_speed_kph: float = 50.0
    max_speed_kph: float = 130.0
    resolution_kph: float = 2.0
    dt: float = 0.002

    def geometry(self, width: float) -> dict[str, float]:
        entry_width = 1.1 * width + 0.25
        offset_width = 1.1 * width + 0.25
        exit_width = 1.3 * width + 0.25
        return {
            "entry_width": entry_width,
            "offset_width": offset_width,
            "exit_width": exit_width,
            "offset": 1.0 + 0.5 * (entry_width + offset_width),
            "s1": 12.0, "s2": 13.5, "s3": 11.0, "s4": 12.5, "s5": 12.0,
        }

    def _path(self, g: dict[str, float]) -> Callable[[float], float]:
        x1 = g["s1"]
        x2 = x1 + g["s2"]
        x3 = x2 + g["s3"]
        x4 = x3 + g["s4"]
        offset = g["offset"]

        def path(x: float) -> float:
            if x <= x1:
                return 0.0
            if x <= x2:
                return offset * _smoothstep((x - x1) / (x2 - x1))
            if x <= x3:
                return offset
            if x <= x4:
                return offset * (1.0 - _smoothstep((x - x3) / (x4 - x3)))
            return 0.0
        return path

    def _gates(self, g: dict[str, float]) -> list[tuple[float, float, float, float]]:
        """Each gate as (x_start, x_end, centre, half width)."""
        x1 = g["s1"]
        x2 = x1 + g["s2"]
        x3 = x2 + g["s3"]
        x4 = x3 + g["s4"]
        x5 = x4 + g["s5"]
        return [
            (0.0, x1, 0.0, g["entry_width"] / 2.0),
            (x2, x3, g["offset"], g["offset_width"] / 2.0),
            (x4, x5, 0.0, g["exit_width"] / 2.0),
        ]

    def run(self, dynamics: VehicleDynamics) -> Result:
        vehicle = dynamics.vehicle
        width = self.vehicle_width or _vehicle_width(vehicle)
        geometry = self.geometry(width)
        path = self._path(geometry)
        gates = self._gates(geometry)
        course_end = sum(geometry[k] for k in ("s1", "s2", "s3", "s4", "s5"))

        attempts: list[tuple[float, bool, float]] = []

        def attempt(speed_kph: float) -> tuple[bool, Result]:
            ok, result, clearance = self._attempt(dynamics, speed_kph, path, gates,
                                                  course_end, width)
            attempts.append((speed_kph, ok, clearance))
            return ok, result

        # Bisect rather than walk the speed up: each attempt is a full
        # simulation, so halving the interval beats stepping through it.
        low_ok, low_result = attempt(self.start_speed_kph)
        if not low_ok:
            best, failed_at = (self.start_speed_kph, low_result), self.start_speed_kph
        else:
            high_ok, high_result = attempt(self.max_speed_kph)
            if high_ok:
                best, failed_at = (self.max_speed_kph, high_result), None
            else:
                low, high = self.start_speed_kph, self.max_speed_kph
                best, failed_at = (low, low_result), high
                while high - low > self.resolution_kph:
                    middle = 0.5 * (low + high)
                    ok, result = attempt(middle)
                    if ok:
                        low, best = middle, (middle, result)
                    else:
                        high, failed_at = middle, middle

        speed, result = best
        result.detail["attempts"] = [
            {"speed_kph": s, "passed": bool(p), "min_clearance_m": c}
            for s, p, c in sorted(attempts)]
        result.detail["geometry"] = geometry
        result.passed = any(p for _, p, _ in attempts)

        if result.passed:
            result.add_kpi("moose_test_speed", "Maximum entry speed through the gates",
                           speed, "km/h", vehicle)
            result.notes.append(
                f"found by bisection over {len(attempts)} runs"
                + (f", failing at {failed_at:.1f} km/h" if failed_at is not None else
                   f"; cleared the {self.max_speed_kph:.0f} km/h search limit"))
        else:
            result.notes.append(
                f"could not clear the gates even at {self.start_speed_kph:.0f} km/h — "
                f"the driver model may need retuning for this vehicle")
        return result

    def _attempt(self, dynamics: VehicleDynamics, speed_kph: float,
                 path: Callable[[float], float],
                 gates: Sequence[tuple[float, float, float, float]],
                 course_end: float, width: float) -> tuple[bool, Result, float]:
        vehicle = dynamics.vehicle
        speed = speed_kph * KPH
        gear = _cruise_gear(vehicle, speed)
        state = dynamics.trim(speed=speed, gear=gear, duration=1.5)
        # Start the run one preview distance before the first gate.
        state.position[0] = -max(15.0, speed * 0.8)
        recorder = Recorder(vehicle.station_names)
        follower = PathFollower(path, vehicle)
        throttle = SpeedController(speed)
        half_width = width / 2.0
        clearance = [float("inf")]
        inside = [True]

        def control(time: float, st: State, dt: float) -> Controls:
            steer = follower.update(st, dt)
            pedal, brake = throttle.update(st.ground_speed, dt)
            x, y = float(st.position[0]), float(st.position[1])
            for start, end, centre, half in gates:
                if start <= x <= end:
                    margin = half - abs(y - centre) - half_width
                    clearance[0] = min(clearance[0], margin)
                    if margin < 0:
                        inside[0] = False
            return Controls(steering=steer, throttle=pedal, brake=brake, gear=gear)

        duration = (course_end + 40.0) / max(speed, 5.0) + 2.0
        self._simulate(dynamics, state, control, duration, self.dt, recorder,
                 stop=lambda t, st: st.position[0] > course_end + 5.0
                 or not inside[0] or abs(st.sideslip) > np.radians(35.0))

        result = self._result(dynamics, recorder)
        result.detail["entry_speed_kph"] = speed_kph
        result.detail["path"] = [[x, path(x)] for x in np.linspace(0, course_end, 120)]
        result.detail["gates"] = [list(g) for g in gates]
        margin = clearance[0] if np.isfinite(clearance[0]) else -1.0
        return bool(inside[0]), result, float(margin)


# ── Rollover propensity ──────────────────────────────────────────────

@dataclass
class Fishhook(Maneuver):
    """NHTSA Fishhook — a rapid steering reversal that rates rollover propensity.

    Reproduces NHTSA's "Roll Rate Feedback" Default Procedure (the one behind
    its NCAP dynamic rollover-resistance rating). An internal Slowly
    Increasing Steer (SIS) sub-test first finds the steering-wheel angle that
    produces 0.3 g of lateral acceleration at the test speed, ramping the
    wheel at 13.5 deg/s. The Fishhook itself then ramps the wheel at
    720 deg/s to 6.5x that angle, holds until the vehicle's own roll rate has
    decayed back through 1.5 deg/s — timing the reversal to the vehicle's
    actual roll dynamics (near its roll-angle peak) rather than a fixed
    delay, which is what makes the Default Procedure comparable across
    vehicles with different roll resonance — reverses at the same rate to
    the same angle the other way, and holds there for 3 s. The pass/fail
    criterion NHTSA actually uses is two-wheel lift: both wheels on one side
    losing contact with the road at the same instant.
    """

    name = "fishhook"
    label = "Fishhook (NHTSA rollover propensity)"
    speed_kph: float = 80.5              # 50 mph, the NHTSA test speed
    sis_rate_deg_s: float = 13.5
    sis_target_g: float = 0.3
    sis_duration: float = 6.0
    steering_scalar: float = 6.5
    ramp_rate_deg_s: float = 720.0
    roll_rate_threshold_deg_s: float = 1.5
    # Not an NHTSA number: a real test times the reversal off the roll rate's
    # return through 1.5 deg/s *after* its excursion, which on a real car is
    # unambiguous. In a clean simulation roll rate starts at exactly 0 deg/s,
    # so "below 1.5 deg/s" is trivially true before any roll has developed —
    # this arms the check only once the roll rate has clearly moved, so the
    # trigger is the return from a real peak, not the starting condition.
    roll_rate_arm_deg_s: float = 4.0
    hold_after_reversal: float = 3.0
    max_dwell: float = 3.0               # safety cap if roll rate never settles
    direction: float = 1.0               # sign of the initial steer
    dt: float = 0.001

    def run(self, dynamics: VehicleDynamics) -> Result:
        vehicle = dynamics.vehicle
        speed = self.speed_kph * KPH
        gear = _cruise_gear(vehicle, speed)
        max_wheel = vehicle.steering.max_steer_angle * vehicle.steering.overall_ratio
        sign = 1.0 if self.direction >= 0 else -1.0

        # -- SIS sub-test: find the steering-wheel angle that gives 0.3 g --
        sis_state = dynamics.trim(speed=speed, gear=gear, duration=1.5)
        sis_recorder = Recorder(vehicle.station_names)
        sis_throttle = SpeedController(speed)
        sis_rate = np.radians(self.sis_rate_deg_s)

        def sis_control(time: float, st: State, dt: float) -> Controls:
            pedal, brake = sis_throttle.update(st.ground_speed, dt)
            steer = sign * min(sis_rate * time, max_wheel)
            return Controls(steering=float(steer), throttle=pedal, brake=brake, gear=gear)

        self._simulate(dynamics, sis_state, sis_control, self.sis_duration, self.dt, sis_recorder,
                 stop=lambda t, st: abs(st.sideslip) > np.radians(25.0))
        sis_ch = sis_recorder.finish()

        result = self._result(dynamics)
        if not sis_ch or float(np.max(np.abs(sis_ch["ay_g"]))) < self.sis_target_g:
            result.notes.append(
                "the Slowly Increasing Steer sub-test never reached the 0.3 g "
                "reference lateral acceleration — the vehicle is too grip-limited "
                "at this speed to define a Fishhook steering angle")
            result.passed = False
            return result

        t_cross = crossing_time(sis_ch["time"], np.abs(sis_ch["ay_g"]), self.sis_target_g) or 0.0
        sis_wheel_angle = sis_rate * t_cross
        target = min(self.steering_scalar * sis_wheel_angle, max_wheel)
        clamped = self.steering_scalar * sis_wheel_angle > max_wheel

        # -- Fishhook proper: ramp, hold for roll rate to decay, reverse, hold --
        ramp_rate = np.radians(self.ramp_rate_deg_s)
        roll_rate_threshold = np.radians(self.roll_rate_threshold_deg_s)
        roll_rate_arm = np.radians(self.roll_rate_arm_deg_s)
        ramp1_duration = target / ramp_rate
        ramp2_duration = 2.0 * target / ramp_rate
        total_duration = (ramp1_duration + self.max_dwell + ramp2_duration
                          + self.hold_after_reversal + 0.5)

        state = dynamics.trim(speed=speed, gear=gear, duration=1.5)
        recorder = Recorder(vehicle.station_names)
        throttle = SpeedController(speed)
        phase = {"name": "ramp1", "t0": 0.0, "armed": False}

        def control(time: float, st: State, dt: float) -> Controls:
            pedal, brake = throttle.update(st.ground_speed, dt)
            elapsed = time - phase["t0"]
            name = phase["name"]
            if name == "ramp1":
                steer = sign * min(ramp_rate * elapsed, target)
                if elapsed >= ramp1_duration:
                    phase["name"], phase["t0"] = "hold1", time
            elif name == "hold1":
                steer = sign * target
                roll_rate = abs(float(st.omega[0]))
                if roll_rate > roll_rate_arm:
                    phase["armed"] = True
                if ((phase["armed"] and roll_rate <= roll_rate_threshold)
                        or elapsed >= self.max_dwell):
                    phase["name"], phase["t0"] = "ramp2", time
            elif name == "ramp2":
                steer = sign * (target - min(ramp_rate * elapsed, 2.0 * target))
                if elapsed >= ramp2_duration:
                    phase["name"], phase["t0"] = "hold2", time
            else:
                steer = -sign * target
            return Controls(steering=float(steer), throttle=pedal, brake=brake, gear=gear)

        self._simulate(dynamics, state, control, total_duration, self.dt, recorder,
                 stop=lambda t, st: phase["name"] == "hold2"
                 and (t - phase["t0"]) >= self.hold_after_reversal)
        result = self._result(dynamics, recorder)
        result.detail["sis_wheel_angle_deg"] = float(np.degrees(sis_wheel_angle))
        result.detail["target_wheel_angle_deg"] = float(np.degrees(target))
        if clamped:
            result.notes.append(
                "6.5x the SIS steering angle exceeds the steering system's own "
                "travel — clamped to full lock, so this run under-represents "
                "the Default Procedure's severity")
        ch = result.channels
        if not ch:
            return result

        left = [s.name for s in vehicle.stations if s.side == "L"]
        right = [s.name for s in vehicle.stations if s.side == "R"]
        static_load = float(ch["Fz_total"][0]) if len(ch["Fz_total"]) else 0.0

        def side_load(names: list[str]) -> float | None:
            if not names or static_load <= 0:
                return None
            total = sum(ch[f"Fz_{n}"] for n in names)
            return float(np.min(total))

        left_min, right_min = side_load(left), side_load(right)
        lift_threshold = 0.02 * static_load     # near-zero combined load on a side
        lifted_sides = [side for side, load in (("left", left_min), ("right", right_min))
                        if load is not None and load < lift_threshold]

        fractions = [load / static_load for load in (left_min, right_min)
                    if load is not None]
        if fractions:
            result.add_kpi("min_side_load_fraction",
                           "Lowest one-side wheel load (fraction of static)",
                           min(fractions), "fraction")
        result.add_kpi("max_roll_angle", "Peak body roll",
                       float(np.degrees(np.max(np.abs(ch["roll"])))), "deg")
        result.detail["two_wheel_lift"] = bool(lifted_sides)
        if lifted_sides:
            result.notes.append(
                f"two wheels lifted together on the {' and '.join(lifted_sides)} "
                f"side (combined load fell under {lift_threshold:.0f} N) — "
                "NHTSA's own rollover-resistance failure criterion")
            result.passed = False
        return result


# ── Combined grip ─────────────────────────────────────────────────────

@dataclass
class FrictionCircle(Maneuver):
    """Maps the combined longitudinal/lateral grip envelope — the g-g diagram.

    Runs a sweep of steering angles, each held while throttle or brake is
    applied at a series of levels, and records the lateral and longitudinal
    acceleration reached. The convex outline of those points is the envelope
    the vehicle can actually use.
    """

    name = "friction_circle"
    label = "Friction circle (g-g diagram)"
    speed_kph: float = 80.0
    steer_steps: int = 7
    pedal_steps: int = 5
    hold: float = 1.6
    dt: float = 0.004

    def run(self, dynamics: VehicleDynamics) -> Result:
        vehicle = dynamics.vehicle
        speed = self.speed_kph * KPH
        gear = _cruise_gear(vehicle, speed)
        base = dynamics.trim(speed=speed, gear=gear, duration=1.5)
        max_wheel = vehicle.steering.max_steer_angle * vehicle.steering.overall_ratio

        points: list[tuple[float, float]] = []
        steers = np.linspace(0.0, 1.0, self.steer_steps) ** 1.5
        pedals = np.linspace(-1.0, 1.0, self.pedal_steps)

        for steer_fraction in steers:
            # Enough lock to reach the limit at this speed, without asking for
            # a angle the steering cannot physically reach.
            limit_steer = min(max_wheel, 2.5 * vehicle.steering.overall_ratio
                              * vehicle.wheelbase * G / max(speed ** 2, 1.0))
            steer = steer_fraction * limit_steer
            for pedal in pedals:
                state = base.copy()
                throttle = max(0.0, pedal)
                brake = max(0.0, -pedal)

                def control(time: float, st: State, dt: float,
                            _s=steer, _t=throttle, _b=brake) -> Controls:
                    ramp = float(np.clip(time / 0.3, 0.0, 1.0))
                    return Controls(steering=_s * ramp, throttle=_t * ramp,
                                    brake=_b * ramp, gear=gear, clutch=1.0)

                recorder = Recorder(vehicle.station_names)
                try:
                    self._simulate(dynamics, state, control, self.hold, self.dt, recorder,
                             stop=lambda t, st: abs(st.sideslip) > np.radians(30.0))
                except FloatingPointError:
                    continue
                ch = recorder.finish()
                if not ch:
                    continue
                window = ch["time"] >= ch["time"][-1] - 0.35
                ay = float(np.mean(ch["ay_g"][window]))
                ax = float(np.mean(ch["ax_g"][window]))
                points.append((ay, ax))
                points.append((-ay, ax))       # the vehicle is symmetric

        result = self._result(dynamics)
        result.detail["points"] = points
        if not points:
            result.notes.append("no usable operating points were reached")
            result.passed = False
            return result

        lateral = [abs(p[0]) for p in points]
        longitudinal = [p[1] for p in points]
        result.add_kpi("max_lateral_g", "Maximum lateral acceleration",
                       max(lateral), "g", vehicle, benchmark_id="skidpad_lateral_g")
        result.add_kpi("max_acceleration_g", "Maximum longitudinal acceleration",
                       max(longitudinal), "g")
        result.add_kpi("max_braking_g", "Maximum deceleration",
                       abs(min(longitudinal)), "g")
        result.detail["speed_kph"] = self.speed_kph
        return result


# ── Helpers ───────────────────────────────────────────────────────────

def _smoothstep(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def _vehicle_width(vehicle: SimVehicle) -> float:
    ys = [abs(float(s.position0[1])) for s in vehicle.stations]
    return 2.0 * max(ys) if ys else 1.8


def _cruise_gear(vehicle: SimVehicle, speed: float) -> int:
    """Highest gear that keeps the engine between idle and its power peak."""
    if vehicle.driveline is None or vehicle.engine is None:
        return 1
    radius = vehicle.stations[0].tire.rolling_radius
    best, best_rpm = 1, 0.0
    for gear in range(1, len(vehicle.driveline.gear_ratios) + 1):
        rpm = speed / max(0.05, radius) * vehicle.driveline.ratio(gear) * 60.0 / (2 * np.pi)
        if vehicle.engine.idle_rpm * 1.3 <= rpm <= vehicle.engine.max_rpm * 0.92:
            if rpm > best_rpm or best_rpm == 0.0:
                best, best_rpm = gear, rpm
    return best


def _understeer_gradient(channels: dict[str, np.ndarray],
                         vehicle: SimVehicle) -> tuple[float | None, float]:
    """Fit K from steer angle against lateral acceleration in the linear range.

    Uses the classic definition: the road wheel angle needed above the
    geometric Ackermann angle, per unit of lateral acceleration. Positive is
    understeer.
    """
    ay = np.abs(channels["ay_g"]) * G
    steer = np.abs(channels["steering"]) / max(1e-6, vehicle.steering.overall_ratio)
    speed = channels["speed"]
    valid = (ay > 0.5) & (ay < 0.4 * G) & (speed > 1.0)
    if np.count_nonzero(valid) < 20:
        return None, 0.0
    ackermann = vehicle.wheelbase * ay[valid] / np.maximum(speed[valid] ** 2, 1.0)
    excess = steer[valid] - ackermann
    gradient = float(np.polyfit(ay[valid] / G, excess, 1)[0])
    return gradient, float(np.max(ay[valid]) / G)


STANDARD_SUITE: list[type[Maneuver]] = [
    Acceleration, Braking, SlowlyIncreasingSteer, ConstantRadius,
    StepSteer, FrictionCircle, MooseTest,
]
