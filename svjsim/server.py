"""Local web app: load a vehicle, change its setup, run tests, watch the result.

A plain `http.server` on localhost with a JSON API and a single-page front end.
No web framework, no build step — the same shape as the SVJ viewer, which also
means it starts instantly and has nothing to keep up to date.

The important design decision is that **setup edits are SVJ override patches**.
Moving a slider does not mutate the model; it builds an RFC 7396 merge patch
(SVJ §3.4), applies it to the base document, and rebuilds the vehicle from
scratch. So anything the GUI can do is reproducible from a file, exportable as a
`.svj-override.json`, and identical to what the command line would produce.
"""

from __future__ import annotations

import glob
import json
import mimetypes
import re
import threading
import time
import traceback
import uuid
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PureWindowsPath
from typing import Any, Callable
from urllib.parse import parse_qs, quote, unquote, urlparse

import numpy as np

from . import maneuvers as mv
from .kinematics import to_svj_camber, to_svj_toe
from .model import SimVehicle, build_vehicle, static_alignment
from .realtime import RealtimeSession
from .solver import Environment, Road, VehicleDynamics
from .telemetry import Result
from .util import dig, merge_patch, num, point3, vec3

APP_HTML = Path(__file__).with_name("webapp.html")
MAX_ASSET_BYTES = 400 * 2**20          # the biggest hand-picked mesh accepted

TESTS = {
    "acceleration": ("Acceleration", mv.Acceleration),
    "braking": ("Braking", mv.Braking),
    "steer": ("Increasing steer", mv.SlowlyIncreasingSteer),
    "skidpad": ("Skidpad", mv.ConstantRadius),
    "step": ("Step steer", mv.StepSteer),
    "friction-circle": ("Friction circle", mv.FrictionCircle),
    "moose": ("Moose test", mv.MooseTest),
    "fishhook": ("Fishhook (rollover)", mv.Fishhook),
}

TEST_OPTIONS: dict[str, list[dict[str, Any]]] = {
    "braking": [{"id": "entry_speed_kph", "label": "Entry speed", "unit": "km/h",
                 "value": 100.0, "min": 30.0, "max": 250.0, "step": 5.0}],
    "steer": [{"id": "speed_kph", "label": "Speed", "unit": "km/h",
               "value": 80.0, "min": 30.0, "max": 200.0, "step": 5.0}],
    "skidpad": [{"id": "radius", "label": "Radius", "unit": "m",
                 "value": 30.0, "min": 10.0, "max": 120.0, "step": 5.0}],
    "step": [{"id": "speed_kph", "label": "Speed", "unit": "km/h",
              "value": 80.0, "min": 30.0, "max": 200.0, "step": 5.0},
             {"id": "target_lateral_g", "label": "Target lateral", "unit": "g",
              "value": 0.4, "min": 0.1, "max": 0.9, "step": 0.05}],
    "friction-circle": [{"id": "speed_kph", "label": "Speed", "unit": "km/h",
                         "value": 80.0, "min": 30.0, "max": 200.0, "step": 5.0}],
    "moose": [{"id": "start_speed_kph", "label": "Lowest speed", "unit": "km/h",
               "value": 50.0, "min": 30.0, "max": 100.0, "step": 5.0},
              {"id": "max_speed_kph", "label": "Highest speed", "unit": "km/h",
               "value": 130.0, "min": 60.0, "max": 200.0, "step": 5.0}],
    "acceleration": [{"id": "duration", "label": "Run time", "unit": "s",
                      "value": 30.0, "min": 5.0, "max": 90.0, "step": 5.0}],
    "fishhook": [{"id": "speed_kph", "label": "Speed", "unit": "km/h",
                 "value": 80.5, "min": 40.0, "max": 130.0, "step": 5.0}],
}


# ── Setup fields ──────────────────────────────────────────────────────

def _axle_stations(document: dict[str, Any]) -> dict[int, list[str]]:
    from .model import _station_axle
    axles: dict[int, list[str]] = {}
    for name in (document.get("suspension") or {}):
        axles.setdefault(_station_axle(name), []).append(name)
    return {k: sorted(v) for k, v in sorted(axles.items())}


def _axle_label(index: int, count: int) -> str:
    if count == 2:
        return "Front" if index == 1 else "Rear"
    return f"Axle {index}"


def setup_fields(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Describe the editable setup, with the values the document currently has.

    Each field says which stations it writes to; `build_patch` turns a set of
    values back into an SVJ merge patch.
    """
    axles = _axle_stations(document)
    fields: list[dict[str, Any]] = []

    for index, stations in axles.items():
        label = _axle_label(index, len(axles))
        first = stations[0]
        corner = dig(document, "suspension", first, default={}) or {}

        rate = num(corner, "spring", "rate", default=25000.0) or 25000.0
        fields.append({"id": f"spring_rate_{index}", "group": label,
                       "label": "Spring rate", "unit": "N/m", "value": rate,
                       "min": round(rate * 0.3), "max": round(rate * 2.5),
                       "step": max(100.0, round(rate * 0.02)), "stations": stations,
                       "kind": "spring_rate"})

        fields.append({"id": f"damper_scale_{index}", "group": label,
                       "label": "Damper scale", "unit": "x", "value": 1.0,
                       "min": 0.3, "max": 2.5, "step": 0.05, "stations": stations,
                       "kind": "damper_scale"})

        arb = num(corner, "arb", "wheel_rate_equiv", default=None)
        if arb is not None:
            fields.append({"id": f"arb_rate_{index}", "group": label,
                           "label": "Anti-roll bar", "unit": "N/m", "value": arb,
                           "min": 0.0, "max": round(max(arb, 1000.0) * 3),
                           "step": 250.0, "stations": stations, "kind": "arb_rate"})

        camber = num(corner, "alignment", "camber", default=0.0) or 0.0
        fields.append({"id": f"camber_{index}", "group": label, "label": "Camber",
                       "unit": "deg", "value": round(np.degrees(camber), 3),
                       "min": -6.0, "max": 2.0, "step": 0.1, "stations": stations,
                       "kind": "camber"})

        toe = num(corner, "alignment", "toe", default=0.0) or 0.0
        fields.append({"id": f"toe_{index}", "group": label, "label": "Toe (+ is in)",
                       "unit": "deg", "value": round(np.degrees(toe), 3),
                       "min": -1.0, "max": 1.0, "step": 0.02, "stations": stations,
                       "kind": "toe"})

        pressure = num(corner, "tire", "pressure", default=None)
        if pressure is None:
            set_ref = dig(corner, "tire", "set_ref", default="")
            pressure = num(document, "tires", "sets", set_ref, "pressure",
                           default=220000.0) or 220000.0
        fields.append({"id": f"pressure_{index}", "group": label, "label": "Tire pressure",
                       "unit": "bar", "value": round(pressure / 1e5, 2),
                       "min": 1.0, "max": 9.0, "step": 0.05, "stations": stations,
                       "kind": "pressure"})

    mass = num(document, "chassis", "mass_total", default=1400.0) or 1400.0
    cg = dig(document, "chassis", "center_of_gravity", default=[-1.2, 0.0, -0.5])
    fields += [
        {"id": "mass_total", "group": "Vehicle", "label": "Total mass", "unit": "kg",
         "value": mass, "min": round(mass * 0.6), "max": round(mass * 1.6),
         "step": 5.0, "kind": "mass"},
        {"id": "cg_x", "group": "Vehicle", "label": "CG behind axle 1", "unit": "m",
         "value": round(abs(float(cg[0])), 3), "min": 0.1,
         "max": round(abs(float(cg[0])) * 2 + 0.5, 2), "step": 0.01, "kind": "cg_x"},
        {"id": "cg_z", "group": "Vehicle", "label": "CG height", "unit": "m",
         "value": round(abs(float(cg[2])), 3), "min": 0.15, "max": 1.6,
         "step": 0.01, "kind": "cg_z"},
    ]

    bias = num(document, "brakes", "bias", default=None)
    if bias is not None:
        fields.append({"id": "brake_bias", "group": "Vehicle", "label": "Brake bias front",
                       "unit": "-", "value": bias, "min": 0.3, "max": 0.9,
                       "step": 0.01, "kind": "brake_bias"})

    aero = dig(document, "aerodynamics", "coefficients", default=None)
    if aero:
        fields += [
            {"id": "aero_cd", "group": "Aerodynamics", "label": "Drag Cd", "unit": "-",
             "value": num(aero, "Cd", default=0.33), "min": 0.1, "max": 1.5,
             "step": 0.01, "kind": "aero_cd"},
            {"id": "aero_cl_front", "group": "Aerodynamics",
             "label": "Lift front (− is downforce)", "unit": "-",
             "value": num(aero, "Cl_front", default=0.0) or 0.0,
             "min": -3.0, "max": 1.0, "step": 0.01, "kind": "aero_cl_front"},
            {"id": "aero_cl_rear", "group": "Aerodynamics",
             "label": "Lift rear (− is downforce)", "unit": "-",
             "value": num(aero, "Cl_rear", default=0.0) or 0.0,
             "min": -3.0, "max": 1.0, "step": 0.01, "kind": "aero_cl_rear"},
        ]
    return fields


def build_patch(document: dict[str, Any], values: dict[str, float]) -> dict[str, Any]:
    """Turn a set of setup values into an SVJ merge patch against `document`."""
    fields = {f["id"]: f for f in setup_fields(document)}
    patch: dict[str, Any] = {}

    def corner_patch(station: str) -> dict[str, Any]:
        return patch.setdefault("suspension", {}).setdefault(station, {})

    for key, raw in values.items():
        field_spec = fields.get(key)
        if field_spec is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value == field_spec["value"]:
            continue                      # unchanged, keep it out of the patch
        kind = field_spec["kind"]
        stations = field_spec.get("stations", [])

        if kind == "spring_rate":
            for s in stations:
                corner_patch(s).setdefault("spring", {})["rate"] = value
        elif kind == "arb_rate":
            for s in stations:
                corner_patch(s).setdefault("arb", {})["wheel_rate_equiv"] = value
        elif kind == "camber":
            for s in stations:
                corner_patch(s).setdefault("alignment", {})["camber"] = float(
                    np.radians(value))
        elif kind == "toe":
            for s in stations:
                corner_patch(s).setdefault("alignment", {})["toe"] = float(
                    np.radians(value))
        elif kind == "pressure":
            for s in stations:
                corner_patch(s).setdefault("tire", {})["pressure"] = value * 1e5
        elif kind == "damper_scale":
            # A merge patch replaces values, so a scale has to be written out as
            # the scaled curve. That keeps the exported override self-contained.
            for s in stations:
                damper = dig(document, "suspension", s, "damper", default={}) or {}
                scaled = {}
                for curve in ("bump_curve", "rebound_curve"):
                    points = damper.get(curve)
                    if isinstance(points, list):
                        scaled[curve] = [[p[0], round(p[1] * value, 3)]
                                         for p in points if len(p) >= 2]
                if scaled:
                    corner_patch(s).setdefault("damper", {}).update(scaled)
        elif kind == "mass":
            patch.setdefault("chassis", {})["mass_total"] = value
        elif kind in ("cg_x", "cg_z"):
            cg = list(dig(document, "chassis", "center_of_gravity",
                          default=[-1.2, 0.0, -0.5]))
            cg[0 if kind == "cg_x" else 2] = -abs(value)
            patch.setdefault("chassis", {})["center_of_gravity"] = cg
        elif kind == "brake_bias":
            patch.setdefault("brakes", {})["bias"] = value
        elif kind.startswith("aero_"):
            key_name = {"aero_cd": "Cd", "aero_cl_front": "Cl_front",
                        "aero_cl_rear": "Cl_rear"}[kind]
            patch.setdefault("aerodynamics", {}).setdefault(
                "coefficients", {})[key_name] = value
    return patch


# ── glTF visual binding layer (SVJ §22) ────────────────────────────────
#
# A `visual` block may sit on any part of the file (v0.99.2, §22.3): the
# chassis and mass bodies, every piece of a suspension corner, steering,
# powertrain and aero. Rather than hardcode the table, `find_visual_bindings`
# walks the whole document and classifies each binding by where it sits,
# because that decides how the part moves:
#
#   body      fixed to the sprung body, authored in the vehicle frame
#             (chassis, mass bodies, steering, powertrain, aero, and the
#             parts §22.6 gives no way to articulate: springs, dampers, ARBs,
#             and links left at `rigid`)
#   station   follows one station's solved upright pose (upright, axle body,
#             caliper), with `spin` for the parts that also turn with the
#             wheel (wheel, dual wheels, disc)
#   link      `link_between_points` on a suspension link: inboard end fixed
#             to the body, outboard end on an upright hardpoint, live
#   segment   `link_between_points` on a driveshaft: joint to joint, static
#   other     found, but there is no defined way to place it; reported
#
# Anything the spec calls an error (a placement on a carrier that can't take
# it, a missing `mesh_ref` with several meshes declared, a node bound twice)
# is carried as an `issue` on the binding so the viewer can show it, instead
# of being silently dropped.

_PLACEMENTS = ("rigid", "link_between_points")
_MESH_AXES = ("+x", "-x", "+y", "-y", "+z", "-z")
_CORNER_PART = {
    "": "upright",                         # v0.97 corner-level binding
    ".topology.upright": "upright",
    ".topology.axle_body": "axle_body",
    ".wheel": "wheel",
    ".brake.disc": "disc",
    ".brake.caliper": "caliper",
    ".spring": "spring",
    ".damper": "damper",
    ".arb": "arb",
}
_SPINNING = {"wheel", "disc"}
_BODY_FIXED_CORNER_PARTS = {"spring", "damper", "arb"}


def find_visual_bindings(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Every `visual` binding in the document, with how it attaches.

    One entry per binding: `{mesh_ref, node, path, kind, part, placement, ...}`
    (kinds as described above). `station` names the station for corner parts;
    `link` bindings carry `inboard` (the inboard end, already reduced per
    `from_point`), `outboard` (the upright hardpoint's key) and `mesh_axis` /
    `scale_to_length`; `segment` bindings carry `from`/`to`. A binding the file
    gets wrong carries an `issue` string.
    """
    meshes = [m.get("id") for m in (dig(doc, "assets", "meshes", default=[]) or [])
              if isinstance(m, dict) and isinstance(m.get("id"), str)]
    found: list[dict[str, Any]] = []
    seen: dict[tuple[str, str], dict[str, Any]] = {}

    def link_ends(carrier: dict[str, Any], visual: dict[str, Any],
                  entry: dict[str, Any]) -> bool:
        """Fill in a link binding's two ends; False if the link doesn't have them."""
        points = [q for q in (point3(p) for p in (carrier.get("inboard_points") or []))
                  if q is not None]
        outboard = str(carrier.get("outboard_ref", "")).split(".")[-1]
        if not points or not outboard:
            entry["issue"] = "link_between_points needs inboard_points and an outboard_ref"
            return False
        index = visual.get("from_point")
        if isinstance(index, int) and not isinstance(index, bool):
            if 0 <= index < len(points):
                inboard = points[index]
            else:
                entry["issue"] = (f"from_point {index} is out of range for "
                                  f"{len(points)} inboard point(s); using their centroid")
                inboard = np.mean(points, axis=0).tolist()
        else:
            inboard = np.mean(points, axis=0).tolist()
        entry.update(inboard=inboard, outboard=outboard, name=carrier.get("name"))
        return True

    def classify(path: str, carrier: dict[str, Any], visual: dict[str, Any],
                 entry: dict[str, Any]) -> None:
        placement = entry["placement"]
        between = placement == "link_between_points"

        if path == ".chassis":
            entry.update(kind="body", part="chassis")
        elif re.match(r"^\.chassis\.mass_bodies\[\d+\]$", path):
            entry.update(kind="body", part="mass_body", body_id=carrier.get("id"))
        elif match := re.match(r"^\.suspension\.([^.\[]+)(.*)$", path):
            station, rest = match.group(1), match.group(2)
            entry["station"] = station
            if re.match(r"^\.topology\.links\[\d+\]$", rest):
                entry["part"] = "link"
                if between and link_ends(carrier, visual, entry):
                    entry["kind"] = "link"
                else:
                    # §22.6: a rigid link is authored in the vehicle frame and
                    # not moved, so it stays where the design pose put it.
                    entry["kind"] = "body"
                    between = False
            elif re.match(r"^\.wheel\.positions\[\d+\]$", rest):
                entry.update(kind="station", part="wheel", spin=True,
                             label=carrier.get("label"))
            elif rest in _CORNER_PART:
                part = _CORNER_PART[rest]
                entry["part"] = part
                entry["legacy_corner"] = rest == ""
                if part in _BODY_FIXED_CORNER_PARTS:
                    entry["kind"] = "body"
                else:
                    entry.update(kind="station", spin=part in _SPINNING)
            else:
                entry.update(kind="other", part="unknown")
        elif path == ".steering" or path.startswith(".steering."):
            entry.update(kind="body", part="steering")
        elif re.match(r"^\.powertrain\.driveshafts\[\d+\]$", path):
            entry["part"] = "driveshaft"
            ends = [point3(carrier.get(key)) for key in ("joint_front", "joint_rear")]
            if between and all(end is not None for end in ends):
                entry.update(kind="segment", **{"from": ends[0], "to": ends[1]})
            else:
                if between:
                    entry["issue"] = "link_between_points needs joint_front/joint_rear positions"
                entry["kind"] = "body"
                between = False
        elif path.startswith(".powertrain."):
            entry.update(kind="body", part="powertrain")
        elif re.match(r"^\.aerodynamics\.components\[\d+\]$", path):
            entry.update(kind="body", part="aero")
        else:
            entry.update(kind="other", part="unknown")

        if between and entry["kind"] not in ("link", "segment") and "issue" not in entry:
            # §22.7 rule 5. Shown as rigid, which is what a parser that doesn't
            # know `placement` would do anyway (§22.8).
            entry["issue"] = "link_between_points is only valid on a link or a driveshaft; shown as rigid"
        if entry["kind"] in ("link", "segment"):
            axis = visual.get("mesh_axis", "+x")
            entry["mesh_axis"] = axis if axis in _MESH_AXES else "+x"
            entry["scale_to_length"] = bool(visual.get("scale_to_length", False))

    def walk(obj: Any, path: str) -> None:
        if isinstance(obj, dict):
            visual = obj.get("visual")
            if isinstance(visual, dict) and isinstance(visual.get("node"), str):
                mesh_ref = visual.get("mesh_ref")
                placement = visual.get("placement", "rigid")
                entry: dict[str, Any] = {
                    "mesh_ref": mesh_ref, "node": visual["node"], "path": path,
                    "placement": placement if placement in _PLACEMENTS else "rigid",
                }
                if not isinstance(mesh_ref, str):
                    # §22.3: may be omitted when exactly one mesh is declared.
                    if len(meshes) == 1:
                        entry["mesh_ref"] = meshes[0]
                    else:
                        entry["issue"] = (f"no mesh_ref, and {len(meshes)} meshes are declared "
                                          "so it can't be inferred")
                classify(path, obj, visual, entry)
                key = (str(entry["mesh_ref"]), entry["node"])
                if key in seen:
                    first = seen[key]
                    # §22.7 rule 4: one axle part bound from both of its stations
                    # is correct; anything else bound twice is not. Drawn once.
                    entry["duplicate_of"] = first["path"]
                    if not (entry.get("station") and first.get("station")
                            and entry.get("part") == first.get("part")
                            and entry.get("part") == "axle_body"):
                        entry.setdefault("issue", f"node already bound at {first['path']}")
                else:
                    seen[key] = entry
                found.append(entry)
            for key, value in obj.items():
                if key != "visual":
                    walk(value, f"{path}.{key}")
        elif isinstance(obj, list):
            for index, value in enumerate(obj):
                walk(value, f"{path}[{index}]")

    walk(doc, "")
    return found


def resolve_assets(doc: dict[str, Any], source_dir: Path | None,
                   uploaded: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """The `assets.meshes` manifest, each entry resolved to a fetchable URL.

    `uri` is relative to the SVJ file itself (§22.2), so resolving it needs to
    know where that file lives on disk. That is known for a file opened from
    the picker or the command line, and found for a "Load file…" / drag-and-drop
    one when an identical file turns up under the search roots (see
    `Handler._locate_source`). When it is still unknown, a mesh the user
    picked by hand (`uploaded`, keyed by asset id) stands in for it.

    An entry that cannot be served says why in `reason`, rather than just
    being absent: the file still reports what it *wants* to show, and the
    viewer can tell the user what to do about it.
    """
    uploaded = uploaded or {}
    entries = []
    for mesh in (dig(doc, "assets", "meshes", default=[]) or []):
        if not isinstance(mesh, dict) or not isinstance(mesh.get("id"), str):
            continue
        uri = str(mesh.get("uri", ""))
        entry = {"id": mesh["id"], "uri": uri, "description": mesh.get("description"),
                 "url": None, "available": False, "reason": None, "source": None}
        upload = uploaded.get(mesh["id"])
        if upload is not None:
            entry.update(url=f"/api/asset?id={quote(mesh['id'], safe='')}&v={upload['version']}",
                         available=True, source=f"picked file {upload['name']}")
        elif source_dir is None:
            entry["reason"] = ("the folder this vehicle came from is unknown, so the mesh "
                               "path can't be resolved")
        elif not uri:
            entry["reason"] = "the asset has no uri"
        else:
            resolved = _safe_resolve(source_dir, uri)
            if resolved is None:
                entry["reason"] = (f"'{uri}' is not a usable mesh path (it must be relative "
                                   f"and end in .glb or .gltf)")
            elif not resolved.is_file():
                entry["reason"] = f"no file at {resolved}"
            else:
                entry.update(url=f"/api/asset?uri={quote(uri, safe='')}", available=True,
                             source=str(resolved))
        entries.append(entry)
    return entries


# Model and texture files are the only thing /api/asset will ever serve.
_ASSET_EXTENSIONS = frozenset({".glb", ".gltf", ".bin", ".png", ".jpg", ".jpeg",
                               ".webp", ".ktx2"})


def _safe_resolve(root: Path, relative: str) -> Path | None:
    """`root / relative` for a mesh `uri` (§22.2), or None if it isn't servable.

    The spec requires `uri` to be relative, so absolute paths, drive letters
    and URLs are refused. Climbing out of `root` with `..` is deliberately
    *allowed*: real files do exactly that (`../meshes/car.glb`, the meshes
    sitting in a sibling folder), and refusing it made those meshes
    unloadable. What keeps that from becoming a
    way to read arbitrary files is the extension allow-list — only model and
    texture files are ever served, wherever they sit.
    """
    if not relative or "://" in relative:
        return None
    windows = PureWindowsPath(relative)
    if windows.is_absolute() or windows.drive or relative.startswith(("/", "\\")):
        return None
    try:
        candidate = (root / relative).resolve()
    except (OSError, ValueError):
        return None
    if candidate.suffix.lower() not in _ASSET_EXTENSIONS:
        return None
    return candidate


# ── Describing a built vehicle to the front end ───────────────────────

def _station_geometry(vehicle: SimVehicle, station: Any,
                      base_document: dict[str, Any] | None = None) -> dict[str, Any]:
    corner = dig(vehicle.source, "suspension", station.name, default={}) or {}
    topology = dig(corner, "topology", default={}) or {}
    # A point is either `[x, y, z]` or a v0.96 marker `{position, orientation}`;
    # the solver reads both (`vec3`), so the viewer must too, or a link whose
    # points are markers simply vanishes from the 3D view.
    hardpoints = {k: q for k, q in
                  ((k, point3(v)) for k, v in
                   (dig(topology, "upright", "hardpoints", default={}) or {}).items())
                  if q is not None}
    links = []
    for link in topology.get("links") or []:
        inboard = [q for q in (point3(p) for p in (link.get("inboard_points") or []))
                   if q is not None]
        ref = str(link.get("outboard_ref", "")).split(".")[-1]
        if inboard and ref in hardpoints:
            links.append({"name": link.get("name", "link"),
                          "type": link.get("type", "arm"),
                          "inboard": inboard, "outboard": ref})

    kc = station.kc
    tire = station.tire
    base_camber, base_toe = (static_alignment(base_document, station.name,
                                              float(station.position0[1]))
                             if base_document else (station.static_camber, station.static_toe))
    return {
        "name": station.name, "axle": station.axle, "side": station.side,
        "steered": bool(station.steered), "driven": bool(station.driven),
        "topology": kc.topology,
        "hardpoints": hardpoints, "links": links,
        "wheel_center0": [float(v) for v in station.position0],
        "wheel": {"radius": float(tire.unloaded_radius),
                  "width": float(tire.dims.section_width),
                  "count": int(station.wheel_count)},
        "jounce": [float(v) for v in kc.jounce],
        "rack": [float(v) for v in kc.rack],
        "pose_d": kc.pose_d.round(6).tolist(),
        "pose_theta": kc.pose_theta.round(6).tolist(),
        # The wheel the physics actually runs has the static alignment added
        # on top of what the hardpoints solve to (`Station.resolve`:
        # static + solved - reference). `pose_theta` alone is just the solved
        # part, so a viewer drawing only that never shows a camber or toe
        # change — the sliders moved the tyre maths and nothing on screen.
        "camber_offset": float(station.static_camber - station.camber_ref),
        "toe_offset": float(station.static_toe - station.toe_ref),
        # A mesh already shows the file's own alignment, so what it needs is
        # only how far the setup has moved from that, not the absolute figure
        # the procedural wheel (which *is* the model) uses.
        "mesh_camber_delta": float(station.static_camber - base_camber),
        "mesh_toe_delta": float(station.static_toe - base_toe),
        "static_load": float(station.static_load),
    }


def describe(vehicle: SimVehicle, patch: dict[str, Any],
             document: dict[str, Any], source_dir: Path | None = None,
             uploaded_assets: dict[str, dict[str, Any]] | None = None,
             base_document: dict[str, Any] | None = None) -> dict[str, Any]:
    stations = []
    for s in vehicle.stations:
        rc = s.roll_centre_height()
        freq, zeta = vehicle.ride_frequency(s), vehicle.damping_ratio(s)
        stations.append({
            "name": s.name, "axle": s.axle, "side": s.side,
            "static_load": round(float(s.static_load), 1),
            "spring_rate": round(float(s.spring.rate), 1),
            "ride_frequency": None if not np.isfinite(freq) else round(float(freq), 3),
            "damping_ratio": None if not np.isfinite(zeta) else round(float(zeta), 3),
            "camber": round(float(np.degrees(to_svj_camber(
                s.static_camber, float(s.position0[1])))), 3),
            "toe": round(float(np.degrees(to_svj_toe(
                s.static_toe, float(s.position0[1])))), 3),
            "camber_gain": round(float(np.degrees(to_svj_camber(
                s.camber_gain, float(s.position0[1]))) * 0.1), 3),
            "bump_steer": round(float(np.degrees(s.kc.bump_steer) * 0.1), 3),
            "roll_centre": None if rc is None else round(float(rc) * 1000, 1),
            "motion_ratio": (None if not np.isfinite(s.kc.spring_motion_ratio())
                             else round(float(s.kc.spring_motion_ratio()), 3)),
            "tire": f"{s.tire.kind}/{s.tire.source}",
            "tire_estimated": s.tire.source == "estimated",
            "steered": bool(s.steered), "driven": bool(s.driven),
        })

    return {
        "name": vehicle.name,
        "info": {
            "mass_total": round(float(vehicle.total_mass), 1),
            "sprung_mass": round(float(vehicle.sprung_mass), 1),
            "unsprung_mass": round(float(vehicle.unsprung_mass), 1),
            "wheelbase": round(float(vehicle.wheelbase), 3),
            "cg": [round(float(v), 4) for v in vehicle.total_cg],
            "station_count": len(vehicle.stations),
            "steering_ratio": round(float(vehicle.steering.overall_ratio), 2),
            "max_steer_deg": round(float(np.degrees(
                vehicle.steering.max_steer_angle)), 1),
            "gears": len(vehicle.driveline.gear_ratios) if vehicle.driveline else 0,
            "driven": vehicle.driveline.driven_stations if vehicle.driveline else [],
            # The viewer needs these to turn a recorded steering wheel angle back
            # into the rack travel that moves the tie rods.
            "rack_per_wheel_angle": round(float(
                vehicle.steering.rack_per_wheel_angle), 6),
            "max_rack": round(float(vehicle.steering.max_rack), 6),
        },
        "stations": stations,
        "geometry": [_station_geometry(vehicle, s, base_document) for s in vehicle.stations],
        "report": {"confidence": vehicle.report.confidence,
                   "warnings": vehicle.report.warnings,
                   "estimates": vehicle.report.estimates,
                   "notes": vehicle.report.notes},
        "benchmarks": vehicle.benchmarks,
        "validation": vehicle.validation,
        "setup": setup_fields(document),
        "patch": patch,
        "tests": [{"id": k, "label": v[0], "options": TEST_OPTIONS.get(k, [])}
                  for k, v in TESTS.items()],
        # glTF visual binding layer (SVJ §22) — see find_visual_bindings/
        # resolve_assets for what these actually mean.
        "assets": resolve_assets(document, source_dir, uploaded_assets),
        "visual_bindings": find_visual_bindings(document),
        "coordinate_system": dig(document, "_metadata", "coordinate_system", default=None),
    }


def result_payload(result: Result, max_points: int = 1200) -> dict[str, Any]:
    channels = {}
    for key, values in result.channels.items():
        array = np.asarray(values, dtype=float)
        stride = max(1, len(array) // max_points)
        trimmed = array[::stride]
        channels[key] = [None if not np.isfinite(v) else round(float(v), 5)
                         for v in trimmed]
    payload = result.to_dict()
    payload["channels"] = channels
    return payload


# ── Session ───────────────────────────────────────────────────────────

@dataclass
class RunRecord:
    test: str
    status: str = "running"
    progress: float = 0.0
    payload: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class Session:
    base_document: dict[str, Any] | None = None
    source_name: str = ""
    source_dir: Path | None = None    # None when the file's folder could not be found
    # Meshes the user picked by hand, keyed by asset id: {name, data, version}.
    uploaded_assets: dict[str, dict[str, Any]] = field(default_factory=dict)
    patch: dict[str, Any] = field(default_factory=dict)
    vehicle: SimVehicle | None = None
    friction: float = 1.0
    runs: dict[str, RunRecord] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    realtime: RealtimeSession | None = None

    def load(self, document: dict[str, Any], name: str,
             source_dir: Path | None = None) -> dict[str, Any]:
        self.stop_drive()
        self.base_document = document
        self.source_name = name
        self.source_dir = source_dir
        self.uploaded_assets = {}        # a picked mesh belongs to one vehicle
        self.patch = {}
        return self.rebuild()

    def rebuild(self) -> dict[str, Any]:
        if self.base_document is None:
            raise ValueError("no vehicle has been loaded yet")
        # A setup change invalidates any driving session: it was built against
        # the vehicle before the change, and driving through a rebuild would
        # either use stale physics or silently teleport the car.
        self.stop_drive()
        merged = merge_patch(self.base_document, self.patch)
        self.vehicle = build_vehicle(merged)
        return describe(self.vehicle, self.patch, merged, self.source_dir,
                        self.uploaded_assets, self.base_document)

    def attach_asset(self, asset_id: str, name: str, data: bytes) -> dict[str, Any]:
        """Use `data` as the mesh for `asset_id`, in place of whatever file the
        vehicle's own `uri` points to. A rebuild (setup edits) keeps it."""
        if self.base_document is None:
            raise ValueError("no vehicle has been loaded yet")
        known = {m.get("id") for m in (dig(self.base_document, "assets", "meshes",
                                           default=[]) or []) if isinstance(m, dict)}
        if asset_id not in known:
            raise ValueError(f"this vehicle has no mesh asset '{asset_id}'")
        # A file that lists the same mesh under two ids (a chassis and a wheel
        # set that are one .glb) means one picked file, not two.
        meshes = [m for m in (dig(self.base_document, "assets", "meshes", default=[]) or [])
                  if isinstance(m, dict)]
        uri = next((m.get("uri") for m in meshes if m.get("id") == asset_id), None)
        for mesh in meshes:
            if mesh.get("id") == asset_id or (uri and mesh.get("uri") == uri):
                version = self.uploaded_assets.get(mesh["id"], {}).get("version", 0) + 1
                self.uploaded_assets[mesh["id"]] = {"name": name, "data": data,
                                                    "version": version}
        return self.rebuild()

    def apply_setup(self, values: dict[str, float]) -> dict[str, Any]:
        if self.base_document is None:
            raise ValueError("no vehicle has been loaded yet")
        self.patch = build_patch(self.base_document, values)
        return self.rebuild()

    # -- driver-in-the-loop ----------------------------------------------

    def start_drive(self, speed_kph: float, gear: int) -> None:
        if self.vehicle is None:
            raise ValueError("no vehicle has been loaded yet")
        self.stop_drive()
        self.realtime = RealtimeSession(self.vehicle, Environment(road=Road(friction=self.friction)))
        self.realtime.start(speed=speed_kph / 3.6, gear=gear)

    def stop_drive(self) -> None:
        if self.realtime is not None:
            self.realtime.stop()
            self.realtime = None

    def require_drive(self) -> RealtimeSession:
        if self.realtime is None or not self.realtime.running:
            raise ValueError("no driving session is running — start one first")
        return self.realtime

    def override_document(self) -> dict[str, Any]:
        """The current setup as a standalone SVJ override file (§3.4)."""
        version = str(dig(self.base_document, "_metadata", "version", default="0.99.1"))
        return {
            "_metadata": {
                "specification": "SVJ-OVERRIDE",
                "version": version if version in ("0.98", "0.99", "0.99.1") else "0.99.1",
                "base": f"./{self.source_name}",
                "description": "Setup change made in SVJSim",
            },
            "patch": self.patch,
        }

    def start_run(self, test: str, options: dict[str, Any]) -> str:
        if self.vehicle is None:
            raise ValueError("no vehicle has been loaded yet")
        if test not in TESTS:
            raise ValueError(f"unknown test '{test}'")
        run_id = uuid.uuid4().hex[:12]
        record = RunRecord(test=test)
        self.runs[run_id] = record
        vehicle, friction = self.vehicle, self.friction

        def worker() -> None:
            try:
                maneuver = TESTS[test][1](**_clean_options(test, options))
                maneuver.progress = lambda f: setattr(record, "progress", float(f))
                dynamics = VehicleDynamics(vehicle, Environment(road=Road(friction=friction)))
                result = maneuver.run(dynamics)
                record.payload = result_payload(result)
                record.status = "done"
                record.progress = 1.0
            except Exception as exc:                  # a failed run is a result too
                record.status = "failed"
                record.error = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()

        threading.Thread(target=worker, daemon=True).start()
        return run_id


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _clean_options(test: str, options: dict[str, Any]) -> dict[str, Any]:
    allowed = {o["id"] for o in TEST_OPTIONS.get(test, [])}
    cleaned: dict[str, Any] = {}
    for key, value in (options or {}).items():
        if key in allowed:
            try:
                cleaned[key] = float(value)
            except (TypeError, ValueError):
                continue
    return cleaned


# ── HTTP ──────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    session: Session
    search_roots: list[Path] = []

    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        pass                                   # the console belongs to the user

    # -- helpers -------------------------------------------------------

    def _send(self, payload: Any, status: int = 200,
              content_type: str = "application/json") -> None:
        body = (json.dumps(payload).encode("utf-8")
                if content_type == "application/json" else payload)
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _fail(self, exc: Exception, status: int = 400) -> None:
        self._send({"error": f"{type(exc).__name__}: {exc}"}, status)

    _ASSET_CONTENT_TYPES = {".glb": "model/gltf-binary", ".gltf": "model/gltf+json"}

    def _content_type(self, filename: str) -> str:
        suffix = Path(filename).suffix.lower()
        return (self._ASSET_CONTENT_TYPES.get(suffix)
                or mimetypes.guess_type(filename)[0] or "application/octet-stream")

    def _serve_asset(self, uri: str, asset_id: str = "") -> None:
        if asset_id:                              # a mesh the user picked by hand
            upload = self.session.uploaded_assets.get(asset_id)
            if upload is None:
                self._send({"error": "no mesh was picked for that asset"}, 404)
            else:
                self._send(upload["data"], content_type=self._content_type(upload["name"]))
            return
        # A file whose folder is unknown has nothing to resolve a uri against.
        if not uri or self.session.source_dir is None:
            self._send({"error": "no asset folder for this session"}, 404)
            return
        resolved = _safe_resolve(self.session.source_dir, unquote(uri))
        if resolved is None or not resolved.is_file():
            self._send({"error": "asset not found"}, 404)
            return
        self._send(resolved.read_bytes(), content_type=self._content_type(resolved.name))

    def _locate_source(self, name: str, document: dict[str, Any]) -> Path | None:
        """Folder of the file `document` was read from, if it can be found.

        Only an exact match counts: a file of that name under the search roots
        that parses to *this very document*. A same-named but different file
        (an edited copy, another car's `vehicle.svj.json`) is not the same
        file, and borrowing its folder could load the wrong car's mesh.
        """
        if not name:
            return None
        deadline = time.monotonic() + 6.0         # the roots can be large trees
        inspected = 0
        for root in self.search_roots:
            if not root.exists():
                continue
            for path in root.rglob(glob.escape(name)):
                if time.monotonic() > deadline or inspected >= 200:
                    return None
                if not path.is_file():
                    continue
                inspected += 1
                try:
                    if json.loads(path.read_text(encoding="utf-8")) == document:
                        return path.parent
                except (OSError, ValueError):
                    continue
        return None

    # -- routes --------------------------------------------------------

    def do_GET(self) -> None:                  # noqa: N802
        route = urlparse(self.path)
        query = parse_qs(route.query)
        try:
            if route.path in ("/", "/index.html"):
                self._send(APP_HTML.read_bytes(), content_type="text/html; charset=utf-8")
            elif route.path == "/api/examples":
                self._send({"files": self._examples()})
            elif route.path == "/api/session":
                # Lets the page pick up a vehicle the server was started with.
                if self.session.vehicle is None:
                    self._send({})
                else:
                    merged = merge_patch(self.session.base_document, self.session.patch)
                    self._send(describe(self.session.vehicle, self.session.patch, merged,
                                        self.session.source_dir, self.session.uploaded_assets,
                                        self.session.base_document))
            elif route.path == "/api/run":
                run_id = (query.get("id") or [""])[0]
                record = self.session.runs.get(run_id)
                if record is None:
                    self._send({"error": "unknown run"}, 404)
                    return
                self._send({"status": record.status, "progress": record.progress,
                            "error": record.error, "result": record.payload})
            elif route.path == "/api/override":
                self._send(self.session.override_document())
            elif route.path == "/api/dil/state":
                self._send(self.session.require_drive().snapshot())
            elif route.path == "/api/asset":
                self._serve_asset((query.get("uri") or [""])[0], (query.get("id") or [""])[0])
            else:
                self._send({"error": "not found"}, 404)
        except ValueError as exc:
            self._fail(exc)
        except Exception as exc:
            self._fail(exc, 500)

    def do_POST(self) -> None:                 # noqa: N802
        route = urlparse(self.path)
        try:
            if route.path == "/api/asset":
                # A mesh file picked by hand: the body is the raw bytes, not JSON.
                query = parse_qs(route.query)
                length = int(self.headers.get("Content-Length") or 0)
                if length <= 0 or length > MAX_ASSET_BYTES:
                    raise ValueError(f"a mesh must be between 1 byte and "
                                     f"{MAX_ASSET_BYTES // 2**20} MB")
                self._send(self.session.attach_asset(
                    (query.get("id") or [""])[0], (query.get("name") or ["mesh.glb"])[0],
                    self.rfile.read(length)))
                return
            body = self._body()
            if route.path == "/api/load":
                document = body.get("document")
                if isinstance(document, str):
                    document = json.loads(document)
                name = body.get("name", "vehicle.svj.json")
                # The browser gives us the text and name of a picked or dropped
                # file but never its path; find the real file so its meshes
                # (relative to it) can still be resolved.
                self._send(self.session.load(document, name,
                                             source_dir=self._locate_source(name, document)))
            elif route.path == "/api/open":
                path = Path(body.get("path", ""))
                document = json.loads(path.read_text(encoding="utf-8"))
                self._send(self.session.load(document, path.name, source_dir=path.parent))
            elif route.path == "/api/setup":
                self.session.friction = float(body.get("friction", 1.0) or 1.0)
                self._send(self.session.apply_setup(body.get("values") or {}))
            elif route.path == "/api/run":
                run_id = self.session.start_run(body.get("test", ""),
                                                body.get("options") or {})
                self._send({"id": run_id})
            elif route.path == "/api/dil/start":
                self.session.start_drive(float(body.get("speed_kph", 0.0) or 0.0),
                                         int(body.get("gear", 1) or 1))
                self._send(self.session.require_drive().snapshot())
            elif route.path == "/api/dil/stop":
                self.session.stop_drive()
                self._send({"stopped": True})
            elif route.path == "/api/dil/input":
                drive = self.session.require_drive()
                manual_gear = body.get("manual_gear", ...)
                drive.set_input(
                    steering=_optional_float(body.get("steering")),
                    throttle=_optional_float(body.get("throttle")),
                    brake=_optional_float(body.get("brake")),
                    manual_gear=(manual_gear if manual_gear is ...
                                 else (int(manual_gear) if manual_gear is not None else None)),
                    handbrake=body.get("handbrake"))
                self._send({"ok": True})
            elif route.path == "/api/dil/shift":
                self.session.require_drive().shift(int(body.get("delta", 0) or 0))
                self._send({"ok": True})
            elif route.path == "/api/dil/reset":
                self.session.require_drive().reset(
                    float(body.get("speed_kph", 0.0) or 0.0), int(body.get("gear", 1) or 1))
                self._send({"ok": True})
            else:
                self._send({"error": "not found"}, 404)
        except (ValueError, KeyError, json.JSONDecodeError, OSError) as exc:
            self._fail(exc)
        except Exception as exc:
            self._fail(exc, 500)

    def _examples(self) -> list[dict[str, str]]:
        found: list[dict[str, str]] = []
        seen: set[str] = set()
        for root in self.search_roots:
            if not root.exists():
                continue
            for path in sorted(root.rglob("*.svj.json")):
                key = str(path.resolve())
                if key in seen:
                    continue
                seen.add(key)
                found.append({"name": path.name, "path": key,
                              "folder": path.parent.name})
                if len(found) >= 200:
                    return found
        return found


def default_search_roots(extra: Path | None = None) -> list[Path]:
    """Places to look for SVJ files, so the picker is useful out of the box."""
    roots = []
    if extra:
        roots.append(extra)
    here = Path(__file__).resolve().parent.parent
    roots += [here, here.parent]
    return [r for r in roots if r.exists()]


def serve(port: int = 8765, path: str | Path | None = None,
          open_browser: bool = True, search_root: str | Path | None = None) -> None:
    """Start the local web app and block until interrupted."""
    session = Session()
    if path:
        source = Path(path)
        session.load(json.loads(source.read_text(encoding="utf-8")), source.name,
                     source_dir=source.parent)

    Handler.session = session
    Handler.search_roots = default_search_roots(
        Path(search_root) if search_root else None)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"SVJSim is running at {url}")
    print("Press Ctrl+C to stop.")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        session.stop_drive()          # a driving thread must not outlive the server
        server.server_close()
