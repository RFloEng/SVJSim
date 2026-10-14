"""Recording a run and summarising it into KPIs.

A maneuver produces two things: continuous channels, and a handful of numbers
that answer the question the test was asking. Where the SVJ file declares a
`benchmarks` entry (§20b) for the same KPI, the result carries the comparison
with it, so a run says not just what the model did but whether it matches what
the car is known to do.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .model import SimVehicle
from .solver import State, WheelOutput


@dataclass
class KPI:
    """One summary number from a run, optionally checked against a benchmark."""

    key: str
    label: str
    value: float
    unit: str
    benchmark: float | None = None
    benchmark_type: str | None = None
    benchmark_source: str | None = None
    tolerance: float = 0.10          # fractional, for the pass/fail verdict

    @property
    def error(self) -> float | None:
        if self.benchmark in (None, 0):
            return None
        return self.value / self.benchmark - 1.0

    @property
    def verdict(self) -> str:
        err = self.error
        if err is None:
            return "no benchmark"
        return "match" if abs(err) <= self.tolerance else "off"

    def formatted(self) -> str:
        return f"{self.value:.3g} {self.unit}"


class Recorder:
    """Collects per-step channels during a run."""

    def __init__(self, stations: Sequence[str]):
        self.stations = list(stations)
        self._rows: list[dict[str, float]] = []

    def capture(self, time: float, state: State, controls: Any,
                wheels: Iterable[WheelOutput], derivative: np.ndarray) -> None:
        n = state.n
        speed = state.ground_speed
        row: dict[str, float] = {
            "time": time,
            "x": float(state.position[0]),
            "y": float(state.position[1]),
            "z": float(state.position[2]),
            "speed": speed,
            "speed_kph": speed * 3.6,
            "vx": float(state.velocity[0]),
            "vy": float(state.velocity[1]),
            "roll": float(state.angles[0]),
            "pitch": float(state.angles[1]),
            "yaw": float(state.angles[2]),
            "yaw_rate": float(state.omega[2]),
            "sideslip": state.sideslip,
            "engine_rpm": state.engine_speed * 60.0 / (2 * np.pi),
            "steering": float(getattr(controls, "steering", 0.0)),
            "throttle": float(getattr(controls, "throttle", 0.0)),
            "brake": float(getattr(controls, "brake", 0.0)),
            "gear": float(getattr(controls, "gear", 0)),
            # Accelerations of the body reference point, including the rotation
            # of the frame — this is what an accelerometer in the car reads.
            "ax": float(derivative[6] - state.velocity[1] * state.omega[2]),
            "ay": float(derivative[7] + state.velocity[0] * state.omega[2]),
            "az": float(derivative[8]),
        }
        row["ax_g"] = row["ax"] / 9.80665
        row["ay_g"] = row["ay"] / 9.80665

        total_load = 0.0
        for i, wheel in enumerate(wheels):
            p = wheel.name
            row[f"Fz_{p}"] = wheel.Fz
            row[f"Fx_{p}"] = wheel.Fx
            row[f"Fy_{p}"] = wheel.Fy
            row[f"slip_angle_{p}"] = wheel.slip_angle
            row[f"slip_ratio_{p}"] = wheel.slip_ratio
            row[f"camber_{p}"] = wheel.camber
            row[f"steer_{p}"] = wheel.steer
            row[f"jounce_{p}"] = float(state.jounce[i])
            row[f"spin_{p}"] = float(state.spin[i])
            row[f"drive_torque_{p}"] = wheel.drive_torque
            row[f"grip_used_{p}"] = wheel.load_ratio
            total_load += wheel.Fz
        row["Fz_total"] = total_load
        self._rows.append(row)

    def finish(self) -> dict[str, np.ndarray]:
        if not self._rows:
            return {}
        keys = self._rows[0].keys()
        return {k: np.array([r.get(k, np.nan) for r in self._rows]) for k in keys}


@dataclass
class Result:
    """The outcome of one maneuver."""

    maneuver: str
    vehicle: str
    channels: dict[str, np.ndarray] = field(default_factory=dict)
    kpis: list[KPI] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    passed: bool = True
    detail: dict[str, Any] = field(default_factory=dict)

    def kpi(self, key: str) -> KPI | None:
        return next((k for k in self.kpis if k.key == key), None)

    def add_kpi(self, key: str, label: str, value: float, unit: str,
                vehicle: SimVehicle | None = None, benchmark_id: str | None = None,
                tolerance: float = 0.10) -> KPI:
        entry = KPI(key=key, label=label, value=float(value), unit=unit, tolerance=tolerance)
        if vehicle is not None:
            declared = vehicle.benchmark(benchmark_id or key)
            if declared is not None:
                entry.benchmark = float(declared.get("value"))
                entry.benchmark_type = str(declared.get("type", ""))
                entry.benchmark_source = str(declared.get("source", "") or "")
        self.kpis.append(entry)
        return entry

    @property
    def duration(self) -> float:
        t = self.channels.get("time")
        return float(t[-1]) if t is not None and len(t) else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "maneuver": self.maneuver,
            "vehicle": self.vehicle,
            "passed": self.passed,
            "notes": self.notes,
            "kpis": [
                {"key": k.key, "label": k.label, "value": k.value, "unit": k.unit,
                 "benchmark": k.benchmark, "benchmark_type": k.benchmark_type,
                 "benchmark_source": k.benchmark_source, "verdict": k.verdict}
                for k in self.kpis
            ],
            "detail": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                       for k, v in self.detail.items()},
        }

    def write_csv(self, path: str | Path, rate_hz: float = 100.0) -> Path:
        """Write the channels as CSV, decimated to `rate_hz`.

        The solver runs at 500 Hz or faster, which makes a multi-megabyte file
        out of a thirty-second run for no analytical gain. Pass `rate_hz=0` to
        keep every sample.
        """
        path = Path(path)
        keys = list(self.channels.keys())
        if not keys:
            path.write_text("", encoding="utf-8")
            return path
        rows = len(self.channels[keys[0]])
        stride = 1
        time = self.channels.get("time")
        if rate_hz > 0 and time is not None and rows > 2:
            dt = float(time[1] - time[0])
            if dt > 0:
                stride = max(1, int(round(1.0 / (rate_hz * dt))))
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(",".join(keys) + "\n")
            for i in range(0, rows, stride):
                f.write(",".join(f"{self.channels[k][i]:.6g}" for k in keys) + "\n")
        return path

    def write_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


def crossing_time(time: np.ndarray, signal: np.ndarray, threshold: float,
                  rising: bool = True) -> float | None:
    """Time at which `signal` first crosses `threshold`, interpolated.

    Returns None when the threshold is never reached, which is a meaningful
    answer — a car that never gets to 100 km/h has no 0-100 time.
    """
    if len(time) < 2:
        return None
    above = signal >= threshold if rising else signal <= threshold
    indices = np.flatnonzero(above)
    if len(indices) == 0:
        return None
    i = int(indices[0])
    if i == 0:
        return float(time[0])
    y0, y1 = float(signal[i - 1]), float(signal[i])
    if y1 == y0:
        return float(time[i])
    fraction = (threshold - y0) / (y1 - y0)
    return float(time[i - 1] + fraction * (time[i] - time[i - 1]))
