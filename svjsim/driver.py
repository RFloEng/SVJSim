"""Driver models: the closed loops that make a maneuver repeatable.

An automated test needs something to hold a speed, follow a path, or change
gear — the parts a human would do without thinking. These are deliberately
simple and transparent: a test result should say something about the vehicle,
not about a clever controller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .model import SimVehicle
from .solver import Controls, State


@dataclass
class PID:
    kp: float
    ki: float = 0.0
    kd: float = 0.0
    limits: tuple[float, float] = (-1e9, 1e9)
    integral: float = 0.0
    previous: float | None = None

    def reset(self) -> None:
        self.integral = 0.0
        self.previous = None

    def __call__(self, error: float, dt: float) -> float:
        derivative = 0.0 if self.previous is None else (error - self.previous) / max(dt, 1e-9)
        self.previous = error
        raw = self.kp * error + self.ki * self.integral + self.kd * derivative
        output = float(np.clip(raw, *self.limits))
        # Only accumulate while the output is free to move, so the integrator
        # cannot wind up against a clipped throttle or brake.
        if output == raw:
            self.integral += error * dt
        return output


class SpeedController:
    """Holds a target speed using throttle and brake, never both at once."""

    def __init__(self, target: float, kp: float = 0.6, ki: float = 0.35):
        self.target = target
        self.pid = PID(kp=kp, ki=ki, limits=(-1.0, 1.0))

    def update(self, speed: float, dt: float) -> tuple[float, float]:
        demand = self.pid(self.target - speed, dt)
        return (max(0.0, demand), max(0.0, -demand) * 0.5)


class GearShifter:
    """Automatic shifting with a torque cut through the change.

    Shift decisions are made on the engine speed each gear *would* give at the
    current wheel speed, not on the measured engine speed. During a change the
    clutch is open, so a measured engine speed just runs to the limiter against
    no load and triggers another upshift, and the box walks itself up through
    several gears in a row.
    """

    def __init__(self, vehicle: SimVehicle, upshift_fraction: float = 0.97,
                 downshift_fraction: float = 0.45, hold_time: float = 0.8):
        self.vehicle = vehicle
        self.gear = 1
        self.timer = 0.0
        self.hold = 0.0
        self.hold_time = hold_time
        self.upshift = upshift_fraction
        self.downshift = downshift_fraction

    @property
    def top_gear(self) -> int:
        return len(self.vehicle.driveline.gear_ratios) if self.vehicle.driveline else 1

    def rpm_in(self, gear: int, wheel_speed: float) -> float:
        if self.vehicle.driveline is None:
            return 0.0
        return abs(wheel_speed) * self.vehicle.driveline.ratio(gear) * 60.0 / (2 * np.pi)

    def update(self, wheel_speed: float, dt: float) -> tuple[int, float, float]:
        """Returns the gear, the clutch engagement, and a throttle multiplier."""
        engine = self.vehicle.engine
        if engine is None or self.vehicle.driveline is None:
            return self.gear, 1.0, 1.0
        self.hold = max(0.0, self.hold - dt)
        if self.timer > 0.0:
            self.timer = max(0.0, self.timer - dt)
            # Clutch out and throttle closed: cutting torque is what makes a
            # shift cost time, and it stops the engine hitting the limiter.
            return self.gear, 0.0, 0.0
        if self.hold > 0.0:
            return self.gear, 1.0, 1.0

        rpm = self.rpm_in(self.gear, wheel_speed)
        if (rpm > engine.max_rpm * self.upshift and self.gear < self.top_gear
                and self.rpm_in(self.gear + 1, wheel_speed) > engine.idle_rpm * 1.15):
            return self._change(1)
        # Only drop a gear if the lower one would not immediately be over the
        # upshift point again, which is what stops hunting on a close-ratio box.
        if (rpm < engine.max_rpm * self.downshift and self.gear > 1
                and self.rpm_in(self.gear - 1, wheel_speed) < engine.max_rpm * self.upshift):
            return self._change(-1)
        return self.gear, 1.0, 1.0

    def _change(self, direction: int) -> tuple[int, float, float]:
        self.gear += direction
        self.timer = self.vehicle.driveline.shift_time
        self.hold = self.hold_time
        return self.gear, 0.0, 0.0


class PathFollower:
    """Preview-point steering controller, the way a driver aims at the road ahead.

    The controller looks a fixed time ahead of the vehicle, measures how far the
    path has moved sideways by then, and steers in proportion. Preview time
    rather than preview distance keeps the gain sensible across speeds.

    The default preview is deliberately short. A long preview makes the
    controller anticipate a lane change and start drifting towards it while the
    vehicle is still in the gate before it — which is precisely what a lane
    change test forbids, and it fails the test on the entry gate rather than on
    anything the vehicle did.
    """

    def __init__(self, path: Callable[[float], float], vehicle: SimVehicle,
                 preview_time: float = 0.25, gain: float = 1.2, damping: float = 0.05):
        self.path = path
        self.vehicle = vehicle
        self.preview_time = preview_time
        self.gain = gain
        self.damping = damping
        self.max_wheel_angle = vehicle.steering.overall_ratio * \
            vehicle.steering.max_steer_angle
        self.previous_error = 0.0

    def update(self, state: State, dt: float) -> float:
        """Steering wheel angle in radians."""
        x, y = float(state.position[0]), float(state.position[1])
        yaw = float(state.angles[2])
        speed = max(1.0, state.ground_speed)
        preview = speed * self.preview_time

        ahead_x = x + preview * np.cos(yaw)
        ahead_y = y + preview * np.sin(yaw)
        error = self.path(ahead_x) - ahead_y
        rate = (error - self.previous_error) / max(dt, 1e-9)
        self.previous_error = error

        # Convert a lateral offset over the preview distance into a road wheel
        # angle, then through the steering ratio to a wheel-rim angle.
        curvature = 2.0 * error / max(1.0, preview ** 2)
        road_wheel = self.gain * np.arctan(curvature * self.vehicle.wheelbase) \
            + self.damping * rate / speed
        command = road_wheel * self.vehicle.steering.overall_ratio
        return float(np.clip(command, -self.max_wheel_angle, self.max_wheel_angle))


@dataclass
class ScriptedDriver:
    """Open-loop inputs as functions of time, for tests with no feedback."""

    steering: Callable[[float], float] = lambda t: 0.0
    throttle: Callable[[float], float] = lambda t: 0.0
    brake: Callable[[float], float] = lambda t: 0.0
    gear: int = 1

    def update(self, time: float, state: State, dt: float) -> Controls:
        return Controls(steering=self.steering(time), throttle=self.throttle(time),
                        brake=self.brake(time), gear=self.gear)
