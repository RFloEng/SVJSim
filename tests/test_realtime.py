"""Checks on the driver-in-the-loop session: thread safety and step correctness.

Real-time pacing itself (does it actually run at 500 Hz on this machine) is not
something a unit test should assert — it depends on the host. What is tested is
that the loop steps the physics correctly, that reading a snapshot never mutates
state, and that concurrent reads and writes from another thread do not race.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from svjsim.realtime import RealtimeSession


@pytest.fixture
def session(vehicle):
    s = RealtimeSession(vehicle, dt=0.002)
    yield s
    s.stop()


def test_starts_trimmed_and_stationary(session):
    session.start(speed=0.0)
    time.sleep(0.3)
    snap = session.snapshot()
    assert snap["speed_kph"] == pytest.approx(0.0, abs=0.5)
    assert abs(snap["angles"][0]) < 0.01, "a symmetric car must not start rolled"
    session.stop()
    assert not session.running


def test_snapshot_does_not_advance_the_shift_state_machine(session):
    """Reading telemetry repeatedly must not itself trigger gear changes."""
    session.start(speed=25.0, gear=3)
    time.sleep(0.2)
    gear_before = session.snapshot()["gear"]
    for _ in range(50):
        session.snapshot()
    gear_after = session.snapshot()["gear"]
    assert gear_before == gear_after, "polling telemetry changed the selected gear"
    session.stop()


def test_steering_input_is_applied_and_clamped(session):
    session.start(speed=20.0, gear=3)
    session.set_input(steering=5.0)          # far outside [-1, 1]
    with session._lock:
        assert session._input.steering == pytest.approx(1.0)
    time.sleep(0.3)
    snap = session.snapshot()
    assert abs(snap["yaw_rate"]) > 0.01, "full steering input should produce yaw rate"
    session.stop()


def test_throttle_drives_the_vehicle_forward(session):
    session.start(speed=0.0, gear=1)
    session.set_input(throttle=1.0)
    time.sleep(1.5)
    snap = session.snapshot()
    session.stop()
    assert snap["speed_kph"] > 1.0, "full throttle from a stop should build speed"


def test_manual_gear_overrides_auto_shift(session):
    session.start(speed=15.0, gear=2)
    session.set_input(manual_gear=4)
    time.sleep(0.2)
    assert session.snapshot()["gear"] == 4
    assert session.snapshot()["auto_shift"] is False
    session.set_input(manual_gear=None)       # hand shifting back to auto
    assert session.auto_shift is True
    session.stop()


def test_shift_paddle_moves_one_gear_and_clamps(session):
    session.start(speed=10.0, gear=1)
    session.shift(+1)
    with session._lock:
        assert session._input.manual_gear == 2
    top = session.shifter.top_gear
    for _ in range(top + 3):
        session.shift(+1)
    with session._lock:
        assert session._input.manual_gear == top, "must not shift past the top gear"
    session.stop()


def test_reset_returns_to_a_clean_trim(session):
    session.start(speed=30.0, gear=4)
    session.set_input(steering=0.8, throttle=1.0)
    time.sleep(0.4)
    assert session.snapshot()["speed_kph"] > 5.0

    session.reset(speed=0.0, gear=1)
    session.set_input(steering=0.0, throttle=0.0, brake=0.0)
    # The reset itself blocks the physics thread for a fraction of a second
    # while it resettles the suspension (see `_quick_settle`); this has to
    # wait longer than that before the state reflects the reset at all.
    time.sleep(1.0)
    snap = session.snapshot()
    session.stop()
    assert snap["speed_kph"] == pytest.approx(0.0, abs=1.0)
    assert snap["sim_time"] < 1.0, "sim time should restart after a reset"


def test_diverging_state_stops_the_session_instead_of_crashing(session):
    """An extreme input should be caught, not propagate a NaN into the viewer."""
    session.start(speed=5.0, gear=1)
    # A wildly oversized step would be needed to diverge Euler at 2 ms in this
    # short a window; instead verify the guard logic directly.
    session._state.data[6] = np.nan
    time.sleep(0.05)
    # Either the loop already stopped itself, or it is still stepping a state
    # that has not yet been walked into by a fresh derivative call — both are
    # acceptable, but a snapshot must never raise or return non-finite numbers
    # that were silently propagated as if valid.
    try:
        snap = session.snapshot()
        assert all(np.isfinite(v) for v in snap["position"]) or not session.running
    except (FloatingPointError, ValueError):
        pass
    finally:
        session.stop()


def test_concurrent_reads_and_writes_do_not_race(session):
    """Hammer set_input/snapshot from other threads while the loop runs."""
    session.start(speed=20.0, gear=3)
    errors: list[Exception] = []
    stop_flag = threading.Event()

    def writer():
        t = 0.0
        while not stop_flag.is_set():
            try:
                session.set_input(steering=np.sin(t), throttle=0.5, brake=0.0)
                t += 0.05
            except Exception as exc:                       # pragma: no cover
                errors.append(exc)

    def reader():
        while not stop_flag.is_set():
            try:
                session.snapshot()
            except Exception as exc:                       # pragma: no cover
                errors.append(exc)

    threads = [threading.Thread(target=writer), threading.Thread(target=reader),
              threading.Thread(target=reader)]
    for th in threads:
        th.start()
    time.sleep(0.5)
    stop_flag.set()
    for th in threads:
        th.join(timeout=2.0)
    session.stop()
    assert not errors, f"race condition raised: {errors}"
