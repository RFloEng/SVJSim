"""Driver-in-the-loop: a fixed-step physics loop running in a background thread.

The maneuvers in `maneuvers.py` drive the vehicle open-loop or through a scripted
controller and run as fast as Python allows, which is the right shape for a test.
Driving needs the opposite: the vehicle steps at a fixed rate tied to the clock,
control input arrives asynchronously from a gamepad or keyboard, and a viewer
reads out whatever the latest state is, whenever it wants it. `RealtimeSession`
is that: one thread stepping physics, one lock guarding the handoff, no queueing
and no backpressure — a viewer that misses a frame just reads the next one.

Stepped with RK4 (`step_rk4`) at 100 Hz rather than the cheaper semi-implicit
Euler (`step_euler`) at 500 Hz — see `DEFAULT_DT` for why: Euler was not
accurate enough at an affordable step size for this vehicle's suspension
dynamics, and produced a parked car that visibly rocked and crept for many
seconds. RK4 costs four evaluations per step instead of one, but at a lower
rate that still fits comfortably inside the real-time budget, with margin
against the GIL contention of sharing the process with the HTTP server.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .driver import GearShifter
from .model import SimVehicle
from .solver import Controls, Environment, State, VehicleDynamics, _rk4_constrained, step_rk4

# RK4 at 100 Hz, not semi-implicit Euler at 500 Hz as an earlier version used.
# This vehicle's coupled heave/pitch dynamics have a mode too fast for Euler
# (even the symplectic form) to resolve accurately at 500 Hz — a parked car
# would rock and slowly drift for many seconds, because the discretisation
# itself was injecting oscillatory error rather than the physics being
# unstable (confirmed by re-running the same standstill case at successively
# smaller dt: the drift shrank in proportion, and disappeared with RK4).
# RK4's own accuracy has margin to spare well below 100 Hz — dropping from
# 200 Hz to 100 Hz made no measurable difference to that standstill residual —
# so 100 Hz was chosen for wall-clock headroom, not accuracy: this loop shares
# the process with the HTTP server's request-handling threads, and the GIL
# contention from many small `/api/dil/*` requests per second measurably ate
# into the margin a bare-loop timing test (no server) does not see.
DEFAULT_DT = 1.0 / 100.0
G = 9.80665

# `VehicleDynamics.trim` defaults to 1.5 s of simulated time at RK4's four
# evaluations per step — around 5 s of wall clock, which is fine for an
# offline maneuver but an unacceptable freeze on "start driving" or "reset".
# `_quick_settle` below replaces it for realtime use: good enough that the
# car does not visibly lurch, not good enough for measuring a KPI from.
SETTLE_DURATION = 0.35
SETTLE_DT = 0.002
# Damping applied to heave, roll/pitch rate and jounce rate on every settle
# step. `trim()` only damps this hard for its first half and lets the
# suspension's own damping do the rest — fine over its 1.5 s budget, but a
# 0.35 s settle does not have time to wait out an underdamped pitch mode (the
# ride frequencies here are order 1-2 Hz, so 0.35 s is under one period).
# Continuous strong damping converges in well under 0.1 s regardless of the
# vehicle's own damping ratio, at the cost of not being physically faithful —
# acceptable for an interactive "start driving" moment, not for a KPI.
SETTLE_DAMPING = 0.80

# Below this speed, with the driver asking for no throttle, the clutch is held
# open rather than fully engaged. Without it the idle governor (which adds
# whatever torque it takes to hold idle rpm, see `solver._engine_derivative`)
# creeps the car forward far harder than a real idling engine would, since a
# real one would simply bog down under that load instead of holding rpm.
CREEP_SPEED_THRESHOLD = 0.5

# Auto-clutch launch ramp (see its use in `_build_controls`): how far the
# engine and driven-wheel speed may differ (rad/s, at the engine) before a
# launch is considered still in progress, how long engagement takes to reach
# full from a standing start, and how much engagement to start with — enough
# to move the car, not so much that the ramp is pointless.
LAUNCH_SLIP_THRESHOLD = 5.0
LAUNCH_RAMP_TIME = 0.8
LAUNCH_START_ENGAGEMENT = 0.15

# How far simulated time may trail wall time before the viewer is told the
# session is running behind real time. A little lag is normal (thread
# scheduling jitter, an occasional slow HTTP request); this is set well above
# that noise floor so the warning means something when it does appear.
REALTIME_LAG_WARNING = 0.25


def _quick_settle(dynamics: VehicleDynamics, speed: float, gear: int,
                  duration: float = SETTLE_DURATION, dt: float = SETTLE_DT) -> State:
    """Settle the vehicle onto its suspension in well under a second.

    Reuses `_rk4_constrained` — the same RK4 stepper `VehicleDynamics.trim`
    itself steps with, neutral, clutch out, x/y/yaw held fixed — rather than a
    from-scratch integrator. That choice is deliberate: an early version of
    this used semi-implicit Euler for speed, and it diverged. This vehicle's
    heave/pitch coupling has a lightly-damped mode around 15-20 Hz, and for a
    mode with a near-imaginary eigenvalue, Euler amplifies it at *any* step
    size — RK4's much larger stability region does not.

    What differs from `trim()` is the damping schedule: `trim()` only damps
    hard for its first half and leaves the suspension's own damping to finish
    the job over its full 1.5 s budget. A 0.35 s settle does not have time to
    wait out an under-a-second-period oscillation on its own, so this damps
    on every step instead — not physically faithful, but this settle exists to
    make "start driving" and "reset" feel instant, not to produce a clean
    result for measuring a KPI from.
    """
    state = dynamics.initial_state(speed, gear)
    controls = Controls(gear=0, throttle=0.0, clutch=0.0)
    frozen = (0, 1, 5, 6, 7, 11)      # x, y, yaw, and their accelerations
    for _ in range(int(duration / dt)):
        state = _rk4_constrained(dynamics, state, controls, dt, frozen)
        state.velocity[0] = speed
        state.velocity[2] *= SETTLE_DAMPING
        state.omega[0:2] *= SETTLE_DAMPING
        state.jounce_rate[:] *= SETTLE_DAMPING
    return state


@dataclass
class InputState:
    """Latest control input, written by the input source and read by the loop."""

    steering: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    manual_gear: int | None = None      # None = let the auto-shifter choose
    handbrake: bool = False


@dataclass
class SessionStats:
    """How well the real-time loop is keeping up, for the viewer to display."""

    sim_time: float = 0.0
    wall_time: float = 0.0
    steps: int = 0
    behind_schedule: bool = False       # the loop had to skip ahead at least once
    step_dt: float = DEFAULT_DT


class RealtimeSession:
    """Runs a `SimVehicle` continuously at a fixed rate in a background thread.

    Not reentrant across sessions on the same vehicle: build a fresh
    `VehicleDynamics` per session so two sessions never share solver state.
    """

    def __init__(self, vehicle: SimVehicle, environment: Environment | None = None,
                 dt: float = DEFAULT_DT):
        self.vehicle = vehicle
        self.dynamics = VehicleDynamics(vehicle, environment or Environment())
        self.dt = dt
        self.shifter = GearShifter(vehicle)
        self.auto_shift = True

        self._lock = threading.Lock()
        self._input = InputState()
        self._state = self.dynamics.initial_state()
        self._stats = SessionStats(step_dt=dt)
        self._thread: threading.Thread | None = None
        self._running = False
        self._reset_request: tuple[float, int] | None = None
        # The controls actually applied on the last physics step. `snapshot()`
        # reuses these rather than calling `_build_controls` again, which would
        # advance the gear-shift state machine a second time per real step.
        self._last_controls = Controls(gear=1)
        self._launch_elapsed = 0.0     # progress through the auto-clutch launch ramp

    # -- lifecycle -------------------------------------------------------

    def start(self, speed: float = 0.0, gear: int = 1) -> None:
        if self._running:
            return
        state = _quick_settle(self.dynamics, speed, gear)
        with self._lock:
            self._state = state
            self._stats = SessionStats(step_dt=self.dt)
            self.shifter.gear = max(1, gear)
            self._launch_elapsed = 0.0
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    @property
    def running(self) -> bool:
        return self._running

    def reset(self, speed: float = 0.0, gear: int = 1) -> None:
        """Re-trim the vehicle without stopping the loop (after a spin, say)."""
        self._reset_request = (speed, gear)

    # -- input -------------------------------------------------------------

    def set_input(self, *, steering: float | None = None, throttle: float | None = None,
                  brake: float | None = None, manual_gear: int | None = ...,
                  handbrake: bool | None = None) -> None:
        """Update whichever control channels are given; the rest hold their value.

        `manual_gear` uses `...` as "leave unchanged" since `None` is itself a
        meaningful value (auto-shift). Pass `None` explicitly to hand shifting
        back to the automatic gearbox.
        """
        with self._lock:
            if steering is not None:
                self._input.steering = float(np.clip(steering, -1.0, 1.0))
            if throttle is not None:
                self._input.throttle = float(np.clip(throttle, 0.0, 1.0))
            if brake is not None:
                self._input.brake = float(np.clip(brake, 0.0, 1.0))
            if manual_gear is not ...:
                self._input.manual_gear = manual_gear
                self.auto_shift = manual_gear is None
            if handbrake is not None:
                self._input.handbrake = handbrake

    def shift(self, delta: int) -> None:
        """Manual paddle shift: +1 up, -1 down. Switches off auto-shift."""
        with self._lock:
            top = self.shifter.top_gear
            current = self._input.manual_gear or self.shifter.gear
            self._input.manual_gear = int(np.clip(current + delta, 1, top))
            self.auto_shift = False

    # -- the loop ------------------------------------------------------------

    def _run(self) -> None:
        max_wheel = (self.vehicle.steering.max_steer_angle
                    * self.vehicle.steering.overall_ratio)
        next_tick = time.perf_counter()
        start_wall = next_tick
        sim_time = 0.0
        steps = 0

        while self._running:
            if self._reset_request is not None:
                speed, gear = self._reset_request
                self._reset_request = None
                state = _quick_settle(self.dynamics, speed, gear)
                with self._lock:
                    self._state = state
                    self.shifter.gear = max(1, gear)
                    self._launch_elapsed = 0.0
                start_wall = time.perf_counter()
                sim_time = 0.0
                steps = 0
                next_tick = start_wall

            controls = self._build_controls(max_wheel)
            with self._lock:
                state = self._state
            state = step_rk4(self.dynamics, state, controls, self.dt)
            sim_time += self.dt
            steps += 1

            if not np.all(np.isfinite(state.data)):
                # A diverged run is not recoverable; hold the last good state so
                # the viewer shows where it went wrong instead of NaN geometry.
                self._running = False
                break

            wall_time = time.perf_counter() - start_wall
            with self._lock:
                self._state = state
                self._last_controls = controls
                # `behind_schedule` reflects the CURRENT lag (wall time ahead of
                # simulated time) against a threshold, not "has ever missed a
                # single step" — the loop deliberately does not burst-catch-up
                # after a stall (see below), so a one-off hiccup would otherwise
                # leave this flag stuck on for the rest of the session even
                # after the loop is keeping up fine again.
                self._stats = SessionStats(
                    sim_time=sim_time, wall_time=wall_time, steps=steps,
                    behind_schedule=(wall_time - sim_time) > REALTIME_LAG_WARNING,
                    step_dt=self.dt)

            next_tick += self.dt
            now = time.perf_counter()
            sleep_time = next_tick - now
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_tick = now      # do not try to burst-catch-up after a stall

    def _build_controls(self, max_wheel: float) -> Controls:
        with self._lock:
            inp = InputState(**vars(self._input))
            state = self._state

        gear = self.shifter.gear
        clutch = 1.0
        wheel_speed = self._driven_wheel_speed(state)
        if inp.manual_gear is not None:
            gear = inp.manual_gear
        else:
            gear, clutch, _ = self.shifter.update(wheel_speed, self.dt)

        if (inp.throttle <= 1e-3 and abs(wheel_speed) < CREEP_SPEED_THRESHOLD
                and abs(state.velocity[0]) < CREEP_SPEED_THRESHOLD):
            clutch = 0.0
            self._launch_elapsed = 0.0
        elif inp.manual_gear is None and gear > 0:
            # Auto-clutch launch ramp: this interface has no clutch pedal, and
            # the shifter's default is full engagement whenever it is not
            # mid-shift. `_clutch_torque` (solver.py) transmits up to
            # `clutch_capacity` — several hundred Nm — the moment there is any
            # slip, regardless of what the engine can produce; dumped straight
            # onto the large speed mismatch of a dead-stop launch, that drags
            # the engine down towards stall instead of the car moving off,
            # exactly like instantly releasing a clutch pedal would.
            #
            # A momentary "match the engine's current torque" cap was tried
            # first and rejected: it traps the car at a fixed, unmoving creep
            # speed — the engine can't rev up because the clutch is always
            # pulling it down by design, and the clutch can't feed in more
            # because it is capped to whatever that suppressed engine is
            # doing, a self-reinforcing equilibrium. A time-based ramp instead
            # (mirroring `Acceleration`'s validated launch, `maneuvers.py`)
            # feeds engagement in smoothly over LAUNCH_RAMP_TIME regardless of
            # how slip evolves, which is what lets both rpm and road speed
            # climb together the way a real clutch release does.
            ratio = self.vehicle.driveline.ratio(gear) if self.vehicle.driveline else 0.0
            input_speed = wheel_speed * ratio
            launching = abs(float(state.engine_speed) - input_speed) > LAUNCH_SLIP_THRESHOLD
            if launching and inp.throttle > 0.05:
                self._launch_elapsed = min(LAUNCH_RAMP_TIME, self._launch_elapsed + self.dt)
            else:
                self._launch_elapsed = LAUNCH_RAMP_TIME       # fully engaged once caught up
            ramp = LAUNCH_START_ENGAGEMENT + (1.0 - LAUNCH_START_ENGAGEMENT) * (
                self._launch_elapsed / LAUNCH_RAMP_TIME)
            clutch = min(clutch, ramp)

        brake = inp.brake
        if inp.handbrake:
            # A handbrake only ever locks the rear axle; approximate that by
            # applying full brake pressure with the front bias zeroed out.
            brake = max(brake, 1.0)

        return Controls(steering=inp.steering * max_wheel, throttle=inp.throttle,
                        brake=brake, gear=gear, clutch=clutch)

    def _driven_wheel_speed(self, state: State) -> float:
        driveline = self.vehicle.driveline
        if driveline is None or not driveline.driven_stations:
            return 0.0
        indices = [i for i, s in enumerate(self.vehicle.stations)
                  if s.name in driveline.driven_stations]
        return float(np.mean([state.spin[i] for i in indices])) if indices else 0.0

    # -- reading state -----------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Everything a viewer needs to render one frame and update a HUD.

        Read-only: reuses the controls already applied on the last physics step
        rather than rebuilding them, so calling this does not itself advance the
        gear-shift state machine or consume a manual-shift request.
        """
        with self._lock:
            state = self._state
            stats = self._stats
            controls = self._last_controls
            gear = controls.gear

        derivative, wheels = self.dynamics.derivatives(state, controls, collect=True)
        ay_g = float(derivative[7] + state.velocity[0] * state.omega[2]) / G
        ax_g = float(derivative[6] - state.velocity[1] * state.omega[2]) / G

        return {
            "running": self._running,
            "sim_time": round(stats.sim_time, 4),
            "wall_time": round(stats.wall_time, 4),
            "behind_schedule": stats.behind_schedule,
            "position": [round(float(v), 5) for v in state.position],
            "angles": [round(float(v), 6) for v in state.angles],
            "speed_kph": round(float(state.ground_speed) * 3.6, 2),
            "yaw_rate": round(float(state.yaw_rate), 5),
            "sideslip": round(float(state.sideslip), 5),
            "ay_g": round(ay_g, 4),
            "ax_g": round(ax_g, 4),
            "engine_rpm": round(float(state.engine_speed) * 60.0 / (2 * np.pi), 1),
            "gear": gear,
            "auto_shift": self.auto_shift,
            "jounce": {s.name: round(float(state.jounce[i]), 5)
                      for i, s in enumerate(self.vehicle.stations)},
            "rack": round(self.vehicle.steering.rack_from_wheel(controls.steering), 5),
            "wheels": [{
                "name": w.name, "Fz": round(w.Fz, 1), "slip_angle": round(w.slip_angle, 5),
                "slip_ratio": round(w.slip_ratio, 5), "camber": round(w.camber, 5),
                "steer": round(w.steer, 5), "load_ratio": round(w.load_ratio, 3),
                "grounded": w.grounded,
            } for w in wheels],
        }
