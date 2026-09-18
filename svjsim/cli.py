"""Command line interface.

    svjsim gui     [car.svj.json]            local web app: 3D viewer, setup editor, tests
    svjsim info    car.svj.json              what the model looks like and where it came from
    svjsim kc      car.svj.json              suspension kinematics and compliance summary
    svjsim run     car.svj.json --out report run the test suite and write an HTML report
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from . import maneuvers as mv
from . import __version__
from .kinematics import to_svj_camber
from .model import SimVehicle, build_vehicle
from .report import write_report
from .solver import Environment, Road, VehicleDynamics
from .telemetry import Result

TESTS = {
    "acceleration": mv.Acceleration,
    "braking": mv.Braking,
    "steer": mv.SlowlyIncreasingSteer,
    "skidpad": mv.ConstantRadius,
    "step": mv.StepSteer,
    "friction-circle": mv.FrictionCircle,
    "moose": mv.MooseTest,
    "fishhook": mv.Fishhook,
}
DEFAULT_TESTS = ["acceleration", "braking", "steer", "step"]


def load_vehicle(path: str, prefer_tire: str | None = None,
                 validate: bool = False) -> SimVehicle:
    from svj.loader import load
    vehicle = load(path, validate_on_load=validate)
    return build_vehicle(vehicle, prefer_tire=prefer_tire)


def _number(value: float | None, spec: str = ".2f") -> str:
    """Format a number, showing an em dash where there is nothing to report."""
    if value is None or not np.isfinite(value):
        return "—"
    return format(value, spec)


def _print_report(sim: SimVehicle) -> None:
    report = sim.report
    print(f"\nModel confidence: {report.confidence}")
    for title, items in (("Warnings", report.warnings),
                         ("Estimates", report.estimates),
                         ("Notes", report.notes)):
        if items:
            print(f"\n{title}:")
            for message in items:
                print(f"  - {message}")


def cmd_info(args: argparse.Namespace) -> int:
    sim = load_vehicle(args.file, args.tire, args.validate)
    print(f"{sim.name}")
    print(f"  mass            {sim.total_mass:,.0f} kg "
          f"({sim.sprung_mass:,.0f} sprung + {sim.unsprung_mass:,.0f} unsprung)")
    print(f"  wheelbase       {sim.wheelbase:.3f} m, {len(sim.stations)} stations")
    print(f"  centre of mass  x {sim.total_cg[0]:+.3f}  y {sim.total_cg[1]:+.3f}  "
          f"z {sim.total_cg[2]:+.3f} m")
    print(f"  sprung inertia  Ixx {sim.sprung_inertia[0, 0]:,.0f}  "
          f"Iyy {sim.sprung_inertia[1, 1]:,.0f}  Izz {sim.sprung_inertia[2, 2]:,.0f} kg·m²")
    print(f"  steering        {sim.steering.overall_ratio:.1f}:1, "
          f"{np.degrees(sim.steering.max_steer_angle):.1f}° of lock")
    if sim.engine:
        peak = float(np.max(sim.engine.torque_curve.y))
        print(f"  engine          {peak:.0f} Nm peak, {sim.engine.max_rpm:,.0f} rpm limit")
    if sim.driveline:
        print(f"  driveline       {len(sim.driveline.gear_ratios)} gears, final drive "
              f"{sim.driveline.final_drive:.3f}, driving "
              f"{', '.join(sim.driveline.driven_stations) or 'nothing'}")

    print("\n  station   load(N)  rate(kN/m)  ride(Hz)  damping  camber°   tire")
    for s in sim.stations:
        freq, zeta = sim.ride_frequency(s), sim.damping_ratio(s)
        camber = np.degrees(to_svj_camber(s.static_camber, float(s.position0[1])))
        print(f"  {s.name:8s} {s.static_load:8.0f} {s.spring.rate / 1000:11.1f} "
              f"{_number(freq):>9s} {_number(zeta):>8s} {camber:+8.2f}   "
              f"{s.tire.kind}/{s.tire.source}")

    if sim.benchmarks:
        print("\n  declared benchmarks:")
        for entry in sim.benchmarks:
            print(f"    {entry.get('id')}: {entry.get('value')} {entry.get('unit')} "
                  f"({entry.get('type')})")
    _print_report(sim)
    return 0


def cmd_kc(args: argparse.Namespace) -> int:
    sim = load_vehicle(args.file, args.tire, args.validate)
    print(f"{sim.name} — suspension kinematics\n")
    print("  station  topology          travel(mm)   camber gain   bump steer   roll centre   "
          "motion ratio")
    print("                              lo    hi     °/100mm      °/100mm         mm         "
          "spring damper")
    for s in sim.stations:
        kc = s.kc
        y = float(s.position0[1])
        gain = np.degrees(to_svj_camber(s.camber_gain, y)) * 0.1
        bump = np.degrees(kc.bump_steer) * 0.1
        rc = s.roll_centre_height()
        print(f"  {s.name:8s} {kc.topology:16s} {kc.jounce[0] * 1000:5.0f} "
              f"{kc.jounce[-1] * 1000:5.0f} {gain:+12.2f} {bump:+12.2f} "
              f"{_number(rc * 1000 if rc is not None else None, '+.0f'):>12s}   "
              f"{_number(kc.spring_motion_ratio(), '.3f'):>6s} "
              f"{_number(kc.damper_motion_ratio(), '.3f'):>6s}")
        if kc.max_geometry_error > 1e-3:
            print(f"           geometry does not close by {kc.max_geometry_error * 1000:.1f} mm "
                  f"— the hardpoints describe a linkage that cannot assemble")
    _print_report(sim)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    sim = load_vehicle(args.file, args.tire, args.validate)
    environment = Environment(road=Road(friction=args.friction))
    dynamics = VehicleDynamics(sim, environment)

    names = args.tests or DEFAULT_TESTS
    if "all" in names:
        names = list(TESTS)
    unknown = [n for n in names if n not in TESTS]
    if unknown:
        print(f"unknown test(s): {', '.join(unknown)}\n"
              f"available: {', '.join(TESTS)}, or 'all'", file=sys.stderr)
        return 2

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{sim.name} — running {len(names)} test(s) "
          f"(model confidence: {sim.report.confidence})\n")

    results: list[Result] = []
    for name in names:
        maneuver = TESTS[name]()
        started = time.time()
        print(f"  {maneuver.label} ... ", end="", flush=True)
        try:
            result = maneuver.run(dynamics)
        except (FloatingPointError, ValueError) as exc:
            print(f"failed: {exc}")
            failed = Result(maneuver=maneuver.label, vehicle=sim.name, passed=False)
            failed.notes.append(str(exc))
            results.append(failed)
            continue
        print(f"{time.time() - started:.1f}s")
        for kpi in result.kpis:
            verdict = ""
            if kpi.benchmark is not None:
                verdict = (f"   vs benchmark {kpi.benchmark:.4g} {kpi.unit} "
                           f"({(kpi.error or 0) * 100:+.1f}%)")
            print(f"      {kpi.label:44s} {kpi.value:10.4g} {kpi.unit}{verdict}")
        for note in result.notes:
            print(f"      note: {note}")
        results.append(result)
        if args.csv:
            result.write_csv(out / f"{maneuver.name}.csv")

    path = write_report(sim, results, out / "report.html")
    print(f"\nReport written to {path}")
    return 0


def cmd_gui(args: argparse.Namespace) -> int:
    from .server import serve
    serve(port=args.port, path=args.file, open_browser=not args.no_browser,
          search_root=args.library)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="svjsim",
        description="Vehicle dynamics simulator for SVJ (Standard Vehicle JSON) files.")
    parser.add_argument("--version", action="version",
                        version=f"svjsim {__version__}")
    parser.add_argument("--tire", choices=["pacejka", "tmeasy", "brush"],
                        help="force a tire model where the file carries several")
    parser.add_argument("--validate", action="store_true",
                        help="validate the file against the SVJ schema before loading")
    sub = parser.add_subparsers(dest="command", required=True)

    info = sub.add_parser("info", help="summarise the model and its provenance")
    info.add_argument("file")
    info.set_defaults(func=cmd_info)

    kc = sub.add_parser("kc", help="report suspension kinematics")
    kc.add_argument("file")
    kc.set_defaults(func=cmd_kc)

    run = sub.add_parser("run", help="run maneuvers and write an HTML report")
    run.add_argument("file")
    run.add_argument("--out", default="svjsim-report", help="output directory")
    run.add_argument("--tests", nargs="+", metavar="TEST",
                     help=f"tests to run: {', '.join(TESTS)}, or 'all' "
                          f"(default: {', '.join(DEFAULT_TESTS)})")
    run.add_argument("--friction", type=float, default=1.0,
                     help="road friction scaling (1.0 = dry asphalt)")
    run.add_argument("--csv", action="store_true", help="also write channel CSVs")
    run.set_defaults(func=cmd_run)

    gui = sub.add_parser("gui", help="open the local web app: 3D viewer, setup editor, tests")
    gui.add_argument("file", nargs="?", help="SVJ file to load on startup")
    gui.add_argument("--port", type=int, default=8765)
    gui.add_argument("--no-browser", action="store_true",
                     help="do not open a browser window automatically")
    gui.add_argument("--library", help="extra folder to scan for .svj.json files")
    gui.set_defaults(func=cmd_gui)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    # Reports carry degree signs and unit symbols; Windows consoles still
    # default to a legacy code page that cannot encode them.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass

    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        print(f"file not found: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"cannot simulate this file: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
