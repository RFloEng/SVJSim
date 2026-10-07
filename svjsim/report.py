"""Self-contained HTML report for a set of maneuver results.

One file, no server, Plotly from a CDN — the same shape as the SVJ viewer and
library browser, so a report can be opened anywhere or mailed to someone.

The report leads with provenance rather than results. A number produced from a
synthesised tire and an estimated inertia tensor is not the same claim as one
produced from measured data, and the reader should see which they are looking at
before they see the charts.
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .model import SimVehicle
from .telemetry import Result

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"

_CSS = """
:root {
  --bg: #12141a; --panel: #1a1d25; --line: #2a2f3a; --text: #e6e8ee;
  --muted: #98a0b3; --accent: #5aa9e6; --good: #58c08a; --warn: #e0b341;
  --bad: #e06c5f; --est: #b98ce0;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
  font: 15px/1.55 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
.wrap { max-width: 1180px; margin: 0 auto; padding: 32px 20px 80px; }
h1 { font-size: 26px; margin: 0 0 4px; letter-spacing: -0.01em; }
h2 { font-size: 19px; margin: 40px 0 14px; padding-bottom: 8px;
  border-bottom: 1px solid var(--line); }
h3 { font-size: 15px; margin: 22px 0 10px; color: var(--muted);
  text-transform: uppercase; letter-spacing: 0.07em; }
.sub { color: var(--muted); margin: 0 0 22px; }
.panel { background: var(--panel); border: 1px solid var(--line);
  border-radius: 10px; padding: 18px 20px; margin: 14px 0; }
.grid { display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); }
.stat { background: var(--panel); border: 1px solid var(--line);
  border-radius: 10px; padding: 14px 16px; }
.stat .k { color: var(--muted); font-size: 12px; text-transform: uppercase;
  letter-spacing: 0.06em; }
.stat .v { font-size: 23px; font-variant-numeric: tabular-nums; margin-top: 4px; }
.stat .n { color: var(--muted); font-size: 12px; margin-top: 4px; }
table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 9px 10px; border-bottom: 1px solid var(--line); }
th { color: var(--muted); font-weight: 600; font-size: 12px;
  text-transform: uppercase; letter-spacing: 0.06em; }
td.num, th.num { text-align: right; }
.tag { display: inline-block; padding: 2px 9px; border-radius: 999px;
  font-size: 12px; font-weight: 600; }
.tag.good { background: rgba(88,192,138,.16); color: var(--good); }
.tag.warn { background: rgba(224,179,65,.16); color: var(--warn); }
.tag.bad { background: rgba(224,108,95,.16); color: var(--bad); }
.tag.est { background: rgba(185,140,224,.16); color: var(--est); }
.tag.none { background: rgba(152,160,179,.14); color: var(--muted); }
ul.msgs { margin: 8px 0 0; padding-left: 20px; }
ul.msgs li { margin: 5px 0; color: var(--muted); }
ul.msgs li b { color: var(--text); font-weight: 600; }
.chart { height: 380px; margin: 8px 0 4px; }
.note { color: var(--muted); font-size: 13px; margin: 8px 0 0; }
footer { color: var(--muted); font-size: 12px; margin-top: 50px;
  border-top: 1px solid var(--line); padding-top: 16px; }
"""

_LAYOUT = {
    "paper_bgcolor": "#1a1d25", "plot_bgcolor": "#1a1d25",
    "font": {"color": "#e6e8ee", "size": 12},
    "margin": {"l": 62, "r": 24, "t": 30, "b": 48},
    "xaxis": {"gridcolor": "#2a2f3a", "zerolinecolor": "#39404e"},
    "yaxis": {"gridcolor": "#2a2f3a", "zerolinecolor": "#39404e"},
    "legend": {"orientation": "h", "y": -0.2},
    "colorway": ["#5aa9e6", "#58c08a", "#e0b341", "#e06c5f", "#b98ce0", "#6fd4d4"],
}


def _esc(text: Any) -> str:
    return html.escape(str(text))


def _num(value: float | None, spec: str = ".2f") -> str:
    """Format a number, or an em dash where there is nothing meaningful to show."""
    if value is None or not np.isfinite(value):
        return "—"
    return format(value, spec)


# Charts are read by eye, so a few thousand points is plenty. Embedding every
# solver sample would make a multi-megabyte HTML file that scrolls badly.
MAX_POINTS = 1500


def _stride(length: int) -> int:
    return max(1, length // MAX_POINTS)


def _series(channels: dict[str, np.ndarray], key: str, scale: float = 1.0) -> list | None:
    data = channels.get(key)
    if data is None:
        return None
    values = np.asarray(data, dtype=float)[::_stride(len(data))] * scale
    return [None if not np.isfinite(v) else round(float(v), 5) for v in values]


def _chart(div_id: str, traces: list[dict], title: str, xtitle: str,
           ytitle: str, extra: dict | None = None) -> tuple[str, str]:
    layout = json.loads(json.dumps(_LAYOUT))
    layout["title"] = {"text": title, "font": {"size": 14}}
    layout["xaxis"]["title"] = xtitle
    layout["yaxis"]["title"] = ytitle
    if extra:
        for key, value in extra.items():
            if isinstance(value, dict) and isinstance(layout.get(key), dict):
                layout[key].update(value)
            else:
                layout[key] = value
    markup = f'<div class="chart" id="{div_id}"></div>'
    script = (f'Plotly.newPlot("{div_id}", {json.dumps(traces)}, '
              f'{json.dumps(layout)}, {{displayModeBar:false, responsive:true}});')
    return markup, script


def _line(channels: dict[str, np.ndarray], x: str, y: str, name: str,
          yscale: float = 1.0, xscale: float = 1.0) -> dict | None:
    xs, ys = _series(channels, x, xscale), _series(channels, y, yscale)
    if xs is None or ys is None:
        return None
    return {"x": xs, "y": ys, "name": name, "type": "scatter", "mode": "lines"}


# ── Sections ──────────────────────────────────────────────────────────

def _vehicle_section(vehicle: SimVehicle) -> str:
    report = vehicle.report
    confidence = report.confidence
    tag = {"high": "good", "medium": "warn", "low": "bad"}[confidence]
    estimated_tires = sum(1 for s in vehicle.stations if s.tire.source == "estimated")

    stats = [
        ("Total mass", f"{vehicle.total_mass:,.0f} kg",
         f"{vehicle.sprung_mass:,.0f} sprung / {vehicle.unsprung_mass:,.0f} unsprung"),
        ("Wheelbase", f"{vehicle.wheelbase:.3f} m",
         f"{len(vehicle.stations)} wheel stations"),
        ("Centre of gravity", f"{abs(vehicle.total_cg[0]):.3f} m behind axle 1",
         f"{abs(vehicle.total_cg[2]) * 1000:.0f} mm above ground"),
        ("Model confidence", confidence,
         f"{len(report.warnings)} warnings, {len(report.estimates)} estimates"),
    ]
    cards = "".join(
        f'<div class="stat"><div class="k">{_esc(k)}</div>'
        f'<div class="v">{_esc(v)}</div><div class="n">{_esc(n)}</div></div>'
        for k, v, n in stats)

    rows = []
    for s in vehicle.stations:
        freq = vehicle.ride_frequency(s)
        zeta = vehicle.damping_ratio(s)
        source = ('<span class="tag est">estimated</span>' if s.tire.source == "estimated"
                  else f'<span class="tag none">{_esc(s.tire.kind)}</span>')
        rc = s.roll_centre_height()
        rows.append(
            f"<tr><td>{_esc(s.name)}</td>"
            f"<td class='num'>{s.static_load:,.0f}</td>"
            f"<td class='num'>{s.spring.rate / 1000:.1f}</td>"
            f"<td class='num'>{_num(freq, '.2f')}</td>"
            f"<td class='num'>{_num(zeta, '.2f')}</td>"
            f"<td class='num'>{np.degrees(s.static_camber) * (-1 if s.side == 'L' else 1):+.2f}</td>"
            f"<td class='num'>{_num(np.degrees(s.camber_gain) * 0.1, '+.2f')}</td>"
            f"<td class='num'>{_num(rc * 1000 if rc is not None else None, '+.0f')}</td>"
            f"<td>{source}</td></tr>")

    def messages(title: str, items: Sequence[str], css: str) -> str:
        if not items:
            return ""
        entries = "".join(f"<li>{_esc(m)}</li>" for m in items)
        return (f'<h3><span class="tag {css}">{_esc(title)}</span></h3>'
                f'<ul class="msgs">{entries}</ul>')

    return f"""
<h2>Vehicle</h2>
<div class="grid">{cards}</div>
<div class="panel">
<table>
<thead><tr><th>Station</th><th class="num">Static load (N)</th>
<th class="num">Wheel rate (kN/m)</th><th class="num">Ride freq (Hz)</th>
<th class="num">Damping</th><th class="num">Camber (°)</th>
<th class="num">Camber gain (°/100mm)</th><th class="num">Roll centre (mm)</th>
<th>Tire model</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<p class="note">Camber is shown in SVJ's relative-to-centreline sign, where negative leans
the wheel top inboard. Roll centre height is derived from the solved contact-patch path,
not read from the file. A rigid beam axle shows no camber gain in bump, because pure bump
does not roll the axle; its roll centre is still shown, solved from both sides' links and
locating linkage (Panhard rod, Watt's link...) together rather than from one wheel's own
travel.</p>
</div>
<h2>Where this model came from</h2>
<div class="panel">
<p class="sub">Confidence <span class="tag {tag}">{_esc(confidence)}</span>
&nbsp;·&nbsp; {estimated_tires} of {len(vehicle.stations)} tires were synthesised from
dimensions rather than fitted from data.</p>
{messages('Warnings', report.warnings, 'bad')}
{messages('Estimates', report.estimates, 'est')}
{messages('Notes', report.notes, 'none')}
</div>
"""


def _kpi_table(results: Sequence[Result]) -> str:
    rows = []
    for result in results:
        for kpi in result.kpis:
            if kpi.benchmark is None:
                verdict = '<span class="tag none">no benchmark</span>'
                benchmark = "—"
            else:
                css = "good" if kpi.verdict == "match" else "warn"
                error = kpi.error or 0.0
                verdict = f'<span class="tag {css}">{error * 100:+.1f}%</span>'
                benchmark = f"{kpi.benchmark:.4g} {_esc(kpi.unit)}"
                if kpi.benchmark_type:
                    benchmark += f' <span class="tag none">{_esc(kpi.benchmark_type)}</span>'
            rows.append(
                f"<tr><td>{_esc(result.maneuver)}</td><td>{_esc(kpi.label)}</td>"
                f"<td class='num'>{kpi.value:.4g} {_esc(kpi.unit)}</td>"
                f"<td class='num'>{benchmark}</td><td class='num'>{verdict}</td></tr>")
    if not rows:
        return ""
    return f"""
<h2>Results</h2>
<div class="panel">
<table><thead><tr><th>Test</th><th>Measure</th><th class="num">Simulated</th>
<th class="num">Declared benchmark</th><th class="num">Agreement</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<p class="note">Benchmarks come from the SVJ file's <code>benchmarks</code> block (§20b).
Agreement is the simulated value against the declared one.</p>
</div>
"""


def _maneuver_section(result: Result, index: int) -> tuple[str, str]:
    channels = result.channels
    charts: list[tuple[str, str]] = []
    prefix = f"c{index}"

    if result.detail.get("points"):
        points = result.detail["points"]
        charts.append(_chart(
            f"{prefix}gg",
            [{"x": [round(p[0], 4) for p in points], "y": [round(p[1], 4) for p in points],
              "mode": "markers", "type": "scatter", "name": "operating points",
              "marker": {"size": 7, "color": "#5aa9e6"}}],
            "Combined grip envelope", "Lateral acceleration (g)",
            "Longitudinal acceleration (g)",
            {"yaxis": {"scaleanchor": "x", "scaleratio": 1}}))

    if result.detail.get("path") and "y" in channels:
        path = result.detail["path"]
        traces = [{"x": [round(p[0], 3) for p in path], "y": [round(p[1], 3) for p in path],
                   "name": "reference path", "type": "scatter", "mode": "lines",
                   "line": {"dash": "dash", "color": "#98a0b3"}}]
        driven = _line(channels, "x", "y", "vehicle path")
        if driven:
            traces.append(driven)
        for start, end, centre, half in result.detail.get("gates", []):
            for edge in (centre - half, centre + half):
                traces.append({"x": [start, end], "y": [edge, edge], "type": "scatter",
                               "mode": "lines", "showlegend": False,
                               "line": {"color": "#e06c5f", "width": 2}})
        charts.append(_chart(f"{prefix}path", traces, "Course and path taken",
                             "Distance along course (m)", "Lateral position (m)",
                             {"yaxis": {"scaleanchor": "x", "scaleratio": 1}}))

    if channels:
        speed = _line(channels, "time", "speed_kph", "speed (km/h)")
        if speed and result.detail.get("points") is None:
            traces = [speed]
            rpm = _line(channels, "time", "engine_rpm", "engine (rpm/100)", yscale=0.01)
            if rpm:
                traces.append(rpm)
            charts.append(_chart(f"{prefix}speed", traces, "Speed", "Time (s)", "km/h"))

        accel = [t for t in (_line(channels, "time", "ay_g", "lateral (g)"),
                             _line(channels, "time", "ax_g", "longitudinal (g)")) if t]
        if accel:
            charts.append(_chart(f"{prefix}acc", accel, "Acceleration",
                                 "Time (s)", "g"))

        yaw = [t for t in (
            _line(channels, "time", "yaw_rate", "yaw rate (deg/s)", yscale=180 / np.pi),
            _line(channels, "time", "roll", "roll (deg)", yscale=180 / np.pi),
            _line(channels, "time", "sideslip", "sideslip (deg)", yscale=180 / np.pi)) if t]
        if yaw:
            charts.append(_chart(f"{prefix}att", yaw, "Attitude", "Time (s)", "deg"))

        loads = [t for t in (_line(channels, "time", f"Fz_{name}", name)
                             for name in _station_names(channels)) if t]
        if loads:
            charts.append(_chart(f"{prefix}fz", loads, "Vertical tire load",
                                 "Time (s)", "N"))

        grip = [t for t in (_line(channels, "time", f"grip_used_{name}", name)
                            for name in _station_names(channels)) if t]
        if grip:
            charts.append(_chart(f"{prefix}grip", grip,
                                 "Fraction of available grip in use", "Time (s)", "—"))

    kpi_rows = "".join(
        f"<tr><td>{_esc(k.label)}</td><td class='num'>{k.value:.4g} {_esc(k.unit)}</td></tr>"
        for k in result.kpis)
    notes = "".join(f"<li>{_esc(n)}</li>" for n in result.notes)
    status = ('<span class="tag good">completed</span>' if result.passed
              else '<span class="tag bad">did not complete</span>')

    body = f"""
<h2>{_esc(result.maneuver)} {status}</h2>
<div class="panel">
{f'<table><tbody>{kpi_rows}</tbody></table>' if kpi_rows else ''}
{f'<ul class="msgs">{notes}</ul>' if notes else ''}
{''.join(markup for markup, _ in charts)}
</div>
"""
    return body, "\n".join(script for _, script in charts)


def _station_names(channels: dict[str, np.ndarray]) -> list[str]:
    return [k[3:] for k in channels if k.startswith("Fz_") and k != "Fz_total"]


# ── Entry point ───────────────────────────────────────────────────────

def write_report(vehicle: SimVehicle, results: Sequence[Result],
                 path: str | Path, title: str | None = None) -> Path:
    """Render an offline HTML report and return where it was written."""
    path = Path(path)
    sections, scripts = [], []
    for index, result in enumerate(results):
        body, script = _maneuver_section(result, index)
        sections.append(body)
        scripts.append(script)

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    heading = title or f"{vehicle.name} — simulation report"
    validation = vehicle.validation.get("status") if vehicle.validation else None
    validation_line = (f' · file validation status: <b>{_esc(validation)}</b>'
                       if validation else
                       ' · the file records no correlation status (SVJ §20a)')

    document = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(heading)}</title>
<script src="{PLOTLY_CDN}"></script>
<style>{_CSS}</style></head>
<body><div class="wrap">
<h1>{_esc(heading)}</h1>
<p class="sub">SVJSim · {len(results)} test{'s' if len(results) != 1 else ''} ·
generated {generated}{validation_line}</p>
{_vehicle_section(vehicle)}
{_kpi_table(results)}
{''.join(sections)}
<footer>Produced by SVJSim from a Standard Vehicle JSON file. Simulated results are only
as good as the data behind them — read the provenance section above before quoting any
number here.</footer>
</div>
<script>{chr(10).join(scripts)}</script>
</body></html>"""
    path.write_text(document, encoding="utf-8")
    return path
