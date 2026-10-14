# SVJSim

[![Tests](https://github.com/RFloEng/SVJSim/actions/workflows/tests.yml/badge.svg)](https://github.com/RFloEng/SVJSim/actions/workflows/tests.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**An open-source vehicle dynamics simulator that runs on [SVJ](https://github.com/RFloEng/SVJ-standard-vehicle-json) files.**

SVJ describes what a car *is* — hardpoints, Pacejka coefficients, damper curves,
LSD ramp angles, inertia tensors. SVJSim is the thing that finds out what the car
*does*. Import a vehicle, change it, and put it through the tests an engineer
would run on a real one: a friction circle, a moose test, a skidpad, a braking
stop from 100 km/h.

The usual open-source answer to "how does this car handle?" is a game physics
engine, where suspension geometry is a spring rate and a roll-centre height you
type in. SVJ is one of the very few formats that carries complete linkage
geometry, so SVJSim does the thing that data makes possible: it **solves the
suspension**. Roll centres, bump steer, camber gain, motion ratios, anti-dive and
anti-squat are all computed from the hardpoints, not entered as tuning constants.

Nothing here is specific to four wheels. SVJ v0.99 covers trucks, trailers,
dual wheels and any number of axles, and so does the solver.

---

## Status

Early but working. The physics core, the automated test bench, the local web
app and driver-in-the-loop are all in place and validated against a real car.

Run on the Mazda MX-5 ND2 reference file that ships with the SVJ standard:

| Measure | SVJSim | Real ND2 (181 hp) |
|---|---|---|
| 0–100 km/h | 6.01 s | ~6.5 s |
| Standing 400 m | 14.2 s | ~14.8 s |
| 100–0 km/h braking | 34.4 m | ~34–36 m |
| Maximum lateral acceleration | 1.01 g | ~0.95 g |
| Understeer gradient | +0.73 °/g | mild understeer |
| Roll gradient | 3.96 °/g | typical for the class |

None of these were tuned to match. They fall out of the file.

---

## Quick start

```bash
pip install -e .
```

```bash
svjsim gui car.svj.json
```

On Windows, `start_gui.bat` does both: the first run creates a `.venv` and
installs SVJSim into it (needs internet and git), after that it just starts the
app. Drag a `.svj.json` file onto it, or pass one as an argument, to open that
vehicle straight away.

Opens a local web app: a 3D view of the vehicle built from its own hardpoints, a
setup panel, the test suite, and telemetry. Drop any `.svj.json` onto the page to
load it.

**Real meshes, shown next to the theoretical geometry, not instead of it.** If
the SVJ file has a `visual` binding to a glTF asset (SVJ §22), the 3D view loads
and draws it alongside the procedural placeholder — the placeholder ghosts to a
dim wireframe rather than disappearing, so you can see exactly where a real
mesh and the physics model's own hardpoints agree or don't. Every part SVJ
v0.99.2 can bind is placed by what it is: uprights and calipers follow the
solved suspension, wheels and discs also spin, links bound with
`link_between_points` stretch between their live hardpoints, and the body,
steering, powertrain and aero parts ride on the chassis. If the mesh file sits
somewhere the app can't find, *Load .glb…* (or dropping it with the JSON)
supplies it.

**A driveline that follows the file.** Engine torque goes through a centre
stage (fixed split, centre differential, or a torque-limited coupling) and one
differential per axle, and differential locks can be modulated by gear, pedals,
speed, axle speed ratio and slip angles (SVJ `lock_control`), so an on-demand
all-wheel-drive car actually behaves like one.

**Tire forces and friction circles, live.** A checkbox over the 3D view draws,
at each wheel's contact patch, a ring sized to the tire's currently available
grip, an arrow for the horizontal force it's actually using, and a vertical
arrow for its load — the arrow reaching the ring *is* the friction-circle
limit, shown where the force acts rather than on a separate chart, green
under the limit and red once the tire is actually slipping. The same
checkbox draws the vehicle's velocity and net force at its centre of
gravity. Everything is filtered, not raw — real per-step tire-force noise
would otherwise read as jitter at the frame rate of an arrow. Works
scrubbing a finished run or driving live.

**Setup edits are SVJ override files.** Moving a slider does not mutate the
model — it builds an RFC 7396 merge patch (SVJ §3.4), applies it to the base
document and rebuilds the vehicle from scratch. *Export override* downloads that
as a `.svj-override.json` which validates with the standard's own
`tools/validate_override.py`, so a setup found in the GUI is a real file you can
keep, share, diff, or feed back through the command line.

*Set baseline* keeps a run so the next one shows the difference on every KPI —
change the anti-roll bar, run again, read the delta.

**Drive it.** The *Drive* tab starts a live session: `W`/`↑` throttle, `S`/`↓`
brake, `A`/`D` or `←`/`→` steer, `Space` handbrake, `Q`/`E` to shift, `R` to
reset — or plug in a gamepad or wheel, which is picked up automatically the
moment you move it. The 3D view switches to a chase camera and shows the
suspension articulating live. Physics runs at 100 Hz in a background thread
independent of the browser's frame rate, so driving stays smooth even if a
frame or two of rendering is slow.

```bash
svjsim info car.svj.json
```

```bash
svjsim run car.svj.json --tests all --out report --csv
```

That writes a self-contained `report/report.html` — open it in any browser, no
server needed.

```python
from svj.loader import load
from svjsim import build_vehicle, VehicleDynamics
from svjsim.maneuvers import SlowlyIncreasingSteer

sim = build_vehicle(load("car.svj.json", validate_on_load=False))
result = SlowlyIncreasingSteer(speed_kph=80).run(VehicleDynamics(sim))
print(result.kpi("max_lateral_g").value)
```

---

## The tests

| Test | What it measures |
|---|---|
| `acceleration` | Standing start with automatic shifts: 0–100 km/h, standing 400 m, top speed |
| `braking` | Full-pressure stop from 100 km/h: distance, mean and peak deceleration |
| `steer` | ISO 4138 slowly increasing steer: maximum lateral g, understeer gradient, roll gradient |
| `skidpad` | Constant-radius cornering to the limit |
| `step` | ISO 7401 step steer: yaw response time and overshoot |
| `friction-circle` | Combined grip envelope — the g-g diagram |
| `moose` | ISO 3888-2 double lane change, run at rising speed until the vehicle leaves the gates |
| `fishhook` | NHTSA rollover propensity: rapid steering reversal, checks for two-wheel lift |

Where the SVJ file declares a `benchmarks` entry (§20b) for the same KPI, the
report shows the simulated value against it, so a run answers not just what the
model did but whether it agrees with what the car is known to do.

---

## How it works

**1. Kinematics preprocessing.** For each wheel station, the link constraints are
solved by damped least-squares Newton over a sweep of wheel travel and steering
rack travel. The result is a K&C table: wheel-centre path, camber, toe, and
spring, damper and ARB motion ratios, all measured from the geometry.

Two things fall out of this that most simulators cannot do. The first is that
the solver reports when a file's hardpoints describe a linkage that cannot
physically assemble. The second is that redundant links are handled: SVJ's own
multi-link guidance suggests two A-arms plus three rods, which is seven
constraints on a five-constraint linkage and would lock the suspension solid —
SVJSim picks a maximal independent set and says what it dropped.

**2. Tire models.** Pacejka Magic Formula (MF 5.2 and 6.2, pure and combined
slip), TMeasy, and a brush model, behind one interface. Most SVJ files in the
wild carry tire *dimensions* but no fitted force model, so there is also a
synthesis path that builds a plausible TMeasy tire from size, pressure and
service class — always marked `estimated`, never presented as measured.

**3. Equations of motion.** `6 + N` degrees of freedom — six for the sprung mass,
one wheel travel per station — assembled from per-body velocity Jacobians rather
than hand-derived load-transfer formulas:

```
M(q) u̇ = Q_external(q, u) − Q_bias(q, u)
```

Because a tire force is applied at the contact patch and projected onto the
travel coordinate through `∂p/∂j` from the K&C tables, jacking, anti-dive and
anti-squat are consequences of the geometry rather than parameters. Wheel spin,
slip relaxation, the clutch and the engine are separate states. Integration is
RK4 at 1–2 ms.

**4. Provenance, everywhere.** Every assumption, substitution and inconsistency
found while building the model is recorded and surfaced — in `svjsim info`, and
at the top of the HTML report, above the results. A number computed from a
synthesised tire and an estimated inertia tensor is not the same claim as one
computed from measured data, and the report never lets you forget which you are
reading.

---

## Conventions

The internal reference frame **is** the SVJ frame — SAE J670, X forward, Y right,
**Z down**, origin at the centre of axle 1 at ground level in the design
attitude. Hardpoints are used without conversion. Gravity acts along +Z.

`jounce` is suspension compression, positive when the wheel moves up towards the
body, measured from the design position the hardpoints describe. Spring force is
expressed as *static load + rate × jounce*, so the vehicle stands at its design
attitude at rest without depending on free-length and preload values that are
usually missing.

Camber and toe are handled internally as absolute SAE rotations and converted to
SVJ's relative-to-centreline signs (§9.6) for reporting.

---

## Requirements

Python 3.11+, NumPy, and the `svj` parser from the SVJ standard repository.
Plotly is loaded from a CDN by the HTML report and is not a Python dependency.

---

## Repository layout

```
svjsim/
  kinematics.py   suspension solver and K&C tables
  tires.py        Pacejka, TMeasy, brush, and tire synthesis
  model.py        SVJ document -> solver-ready vehicle, with the build report
  solver.py       state, equations of motion, integrators
  driver.py       speed control, gear shifting, path following
  maneuvers.py    the test suite
  realtime.py     driver-in-the-loop: fixed-step physics in a background thread
  telemetry.py    channel recording and KPIs
  report.py       self-contained HTML report
  server.py       local web app: JSON API over stdlib http.server
  webapp.html     the front end: three.js 3D view, Plotly telemetry, live driving
  cli.py          command line interface
tests/            unit tests, including hand-checked geometry cases
```

---

## Contributing

Bug reports, data-quality findings, and PRs are welcome — see
[CONTRIBUTING.md](CONTRIBUTING.md) for how to get set up and what a good PR
looks like. [DEVELOPMENT_BRIEF.md](DEVELOPMENT_BRIEF.md) has the
architecture and the reasoning behind the harder design decisions.

## Licence

[Apache-2.0](LICENSE), the same as the SVJ standard.
