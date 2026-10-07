# SVJSim — Development Brief

For anyone extending this. Current state is **v0.5.0**: the physics core, the
automated test bench, the local web app, driver-in-the-loop, glTF visual mesh
binding (SVJ §22), and axle-level kinematics for solid-axle/de-Dion beam
axles all work and are validated against a real car. This covers the
architecture, the decisions that are load-bearing, and what is worth doing
next.

## 1. Scope

SVJSim simulates SVJ vehicles. It does not parse simulator formats (that is the
converters), does not validate against the schema (that is `svj-py`), and does
not render 3-D geometry (that is the viewer). It reads an SVJ document, solves
what the vehicle does, and reports it.

Two constraints drive the architecture:

1. **The data is the model.** Anything derivable from the file is derived, not
   asked for. Roll centres, motion ratios, bump steer and anti-effects come from
   the hardpoints. Where something genuinely cannot be derived, it is estimated
   *and recorded as an estimate*, never quietly defaulted.
2. **Stations, not corners.** SVJ v0.99 covers trucks, trailers, dual wheels and
   any number of axles. Nothing in the solver may assume four wheels.

## 2. Architecture

```
svj document
   │
   ├─ kinematics.py   per station: solve the linkage over travel and rack sweeps
   │                  → KCTable (wheel-centre path, camber, toe, motion ratios)
   │
   ├─ tires.py        per tire set: Pacejka / TMeasy / brush / synthesised
   │                  → TireModel with one forces() interface
   │
   └─ model.py        assemble: masses, static loads, springs, dampers, brakes,
                      aero, powertrain, steering  → SimVehicle + BuildReport
                              │
                              ▼
                      solver.py    VehicleDynamics.derivatives()
                              │    M(q) u̇ = Q_ext − Q_bias, assembled from Jacobians
                              ▼
      driver.py ──────► maneuvers.py ──────► telemetry.py ──────► report.py
   (speed, gears, path)   (the tests)      (channels, KPIs)      (HTML)
                                │
                                ├──────► server.py + webapp.html
                                │        (JSON API, 3D viewer, setup editor)
                                │
                                └──────► realtime.py ──────► server.py /api/dil/*
                                         (RealtimeSession,       (input in, state
                                          background thread)      out, polled live)
```

### Driver-in-the-loop

`realtime.py`'s `RealtimeSession` runs its own `VehicleDynamics` in a dedicated
background thread, stepping at a fixed rate (`DEFAULT_DT`) independent of
anything the HTTP server or browser are doing. One `threading.Lock` guards the
handoff in both directions: `set_input(...)` writes the latest control values,
`snapshot()` reads the latest state — no queue, no backpressure, a viewer that
misses a poll just reads the next one. The browser drives this with plain HTTP:
`POST /api/dil/input` on every rendered frame (fire-and-forget), `GET
/api/dil/state` on every rendered frame (awaited, feeds `Viewer.updateLive`).
There is deliberately no WebSocket — HTTP round trips on localhost are well
under a millisecond, and a plain `http.server` handler is enough.

Two things in this module exist only because the naive version of a real-time
loop breaks in ways that do not show up in an offline maneuver:

- **`_quick_settle`** — `VehicleDynamics.trim`'s default 1.5 s / RK4 settle
  costs about 5 s of wall clock, fine before a scripted maneuver, an
  unacceptable freeze on an interactive "start driving" click. It replaces
  `trim`'s first-half-only damping with continuous damping over a much
  shorter window, trading physical faithfulness (nobody is measuring a KPI
  off this state) for speed.
- **RK4 at 100 Hz, not Euler at 500 Hz** — see item 7 in §3. Get this wrong
  and the symptom is a parked car that quietly rocks and creeps forward on
  its own over several seconds, which looks like a physics bug and is
  actually a discretisation one.

### The web app

`server.py` is a stdlib `http.server` with a JSON API; `webapp.html` is the whole
front end. The load-bearing decision is that **a setup edit is an SVJ override
patch**: `build_patch` turns slider values into an RFC 7396 merge patch,
`Session.rebuild` applies it to the base document and calls `build_vehicle`
again. Nothing mutates a built model in place. That gives three things for free —
the GUI cannot drift from the command line, any setup is exportable as a real
`.svj-override.json`, and "reset" is just an empty patch.

Adding a setup control means adding an entry to `setup_fields` and a branch in
`build_patch`; the front end renders whatever the server describes.

The 3D viewer articulates the linkage from `KCTable.pose_d` / `pose_theta`,
interpolated over travel and rack, so what is drawn is the pose the solver
actually used. SVJ is Z-down and three.js is Y-up; `toThree` converts a
*point*. A *rotation* is a different, sharper problem: a body's children
(wheels, links, chassis box) already have their own positions expressed in
three-space via `toThree`, so rotating the assembly correctly means
conjugating the whole matrix by that same change of basis
(`R_three = M · R_svj · M⁻¹`, `bodyQuaternion`/`conjugateIntoThree`) —
converting the rotation's columns through `toThree` one at a time looks
plausible but bakes in a spurious rotation even at identity attitude.

When a body carries a `visual` binding (SVJ §22), the viewer loads the
referenced glTF node via `GLTFLoader` and draws it *alongside* the
procedural placeholder, never instead of it — ghosting the placeholder to a
dim wireframe — so a real mesh asset and the physics model's own theoretical
geometry stay simultaneously visible and any disagreement between them is
visible rather than papered over. A station's mesh is moved only by the
*incremental* jounce/steer motion since the design pose, not snapped to the
theoretical wheel centre, so a wrongly-placed hardpoint in the asset actually
shows up instead of being silently corrected. `server.py` resolves
`assets.meshes[].uri` against the SVJ file's own folder (`GET /api/asset`);
a vehicle loaded via drag-and-drop has no such folder, so its meshes are
reported unavailable rather than failing the whole load.

### The equations of motion

Generalized coordinates are the six sprung-mass freedoms plus one wheel-travel
coordinate per station. `VehicleDynamics.derivatives` assembles

```
M = m_s·Jsᵀ Js + Jωᵀ I Jω + Σ m_u,i·Ju,iᵀ Ju,i
Q = Σ Jᵀ F   (gravity, tire forces at the contact patch, aero)
```

and solves `M u̇ = Q − bias`. Wheel spin, slip relaxation, clutch and engine are
integrated alongside as separate first-order states.

The important consequence: a tire force is applied at the **contact patch** and
projected onto the travel coordinate through `∂p/∂j` from the K&C table. That
single term is what produces jacking, anti-dive and anti-squat, so there are no
roll-centre heights or anti-percentages anywhere in the code.

### Extension points

| To add | Where | Notes |
|---|---|---|
| A tire model | `tires.py` | Subclass `TireModel`, implement `forces()`, register in `build_tire` |
| A suspension topology | `kinematics.py` | Add link constraints, or add to `CARRIER_TOPOLOGIES` / `BEAM_TOPOLOGIES` (a rigid-beam type joins `AxleKinematics`'s coupled solve; a single-arm carrier joins `StationKinematics`'s revolute mode) |
| A maneuver | `maneuvers.py` | Subclass `Maneuver`, use `simulate()`, return a `Result`; add to `STANDARD_SUITE` and `cli.TESTS` |
| A driver behaviour | `driver.py` | Controllers are plain callables over `(state, dt)` |
| A road surface | `solver.py` | `Road` takes `height_fn` and `friction_fn` |
| A setup control | `server.py` | Add to `setup_fields` and a branch in `build_patch` |
| A driving input channel | `realtime.py` + `webapp.html` | Add a field to `InputState`, a case in `RealtimeSession.set_input`, and read it in the browser's input loop |

## 3. Decisions not to change without thinking

1. **The body frame is the SVJ frame.** SAE J670, Z down, origin at axle 1 on
   the ground. Hardpoints are used unconverted. Changing this would introduce a
   whole class of sign bugs for no gain.
2. **Jounce is compression, measured from the design position.** Spring force is
   `static load + rate × jounce`, so the vehicle stands at design attitude at
   rest without needing free length and preload, which are usually missing.
3. **The spring carries the sprung load only.** The tire holds the unsprung mass
   up directly. Getting this wrong makes the car sit ~15 mm low and was the
   first real bug found in the solver.
4. **Tire data is mirrored on left-hand stations.** Fitted coefficients carry the
   handedness of the wheel they were measured on; applying them unmirrored to
   both sides makes a symmetric car pull sideways while standing still.
5. **Low-speed contact-patch damping is two separate terms, not one.**
   `slip_damping` (`SPIN_DAMPING_TAU`, faded out over `V_DAMP_BLEND` = 5 m/s)
   damps the wheel-spin equation's nearly-undamped ~80 Hz mode at standstill,
   sized from a wheel's own small rotational inertia. `body_damping`
   (`BODY_DAMPING_TAU`, faded out over the much narrower `V_BODY_DAMP_BLEND` =
   0.15 m/s) is sized from the *whole vehicle's* mass and stands in for the
   static friction the tire model otherwise lacks at v≈0, so a car does not
   creep under a tiny sustained force (gravity's own component along a
   slightly pitched equilibrium attitude is enough). **Do not let these two
   share a fade-out width.** They did, briefly, at `body_damping`'s
   introduction: at the wide window every wheel — including undriven ones —
   dragged with several thousand newtons of numerical friction all the way up
   to 18 km/h, swamping the driven wheels' real tractive force so thoroughly
   that a standing-start launch could barely reach walking pace. See
   `test_launch_from_a_standstill_actually_builds_speed` in `test_solver.py`,
   which exists specifically because no other test happened to start below
   that threshold and so none of them caught it.
6. **The clutch is a friction plate, not a rigid link.** A rigid connection drags
   the engine to zero rpm at a standing start, where an extrapolated torque
   curve goes negative and the car will not pull away.
7. **Real-time stepping uses RK4 at 100 Hz, not Euler at 500 Hz.** Even the
   properly symplectic form of semi-implicit Euler (rates advanced first,
   positions from the updated rates — `step_euler` was fixed to actually do
   this and previously did not) is not accurate enough at an affordable step
   size for this vehicle's coupled heave/pitch dynamics: a parked car would
   visibly rock and creep for many seconds. Confirmed to be a discretisation
   problem rather than an unstable-physics one by shrinking dt and watching
   the drift shrink in proportion until RK4 made it vanish. RK4's four
   evaluations per step cost about the same per simulated second at 100 Hz as
   Euler's one did at 500 Hz, with margin to spare against the GIL contention
   of sharing a process with the HTTP server — see `svjsim/realtime.py`.
8. **Everything assumed is recorded.** `BuildReport` is not a debug aid; it is
   part of the output, and the HTML report leads with it.

## 4. Known limitations

- **Real-time driving uses HTTP polling, not push.** The browser posts input
  and polls state at roughly its own render rate; there is no WebSocket or
  Server-Sent Events channel. This works well on localhost (sub-millisecond
  round trips) and keeps `server.py` a plain stdlib `http.server`, but it is
  not the shape you would want over a real network.
- **No force feedback.** The wheel/gamepad input path is read-only; nothing
  is sent back to a wheel's motors.
- **Solid-axle/de-Dion beam axles (`solid_axle`, `de_dion`) are properly
  solved**, as of `AxleKinematics` (`kinematics.py`): both sides' links and
  the shared `axle_body` are solved as one rigid body (six freedoms, two
  driven by each side's own commanded travel, the rest by whichever links
  the file declares — trailing links, a Panhard rod, a Watt's link...), so
  camber, toe/bump-steer and the roll centre all come from the real coupled
  geometry instead of a per-side approximation. `torsion_beam`,
  `parallelogram` and `pendulum_axle` are not covered by this yet — see
  `BEAM_TOPOLOGIES` — nor is a *steered* beam axle (no bundled example
  combines the two; it would need a genuinely 3D table indexed by jounce,
  partner jounce and rack together).
- **`suspension_couplings` (SVJ §23.5) are still ignored.** Walking beams,
  equalizer rockers, trunnion springs and shared air/hydraulic circuits are
  read but not simulated, so multi-axle load sharing is wrong on vehicles
  that use them — a different problem from the axle kinematics above (this
  one is mass/spring load-sharing *between* axles, via a rocker, beam or
  fluid circuit, not the rigid-body kinematics *within* one axle).
- **No compliance.** Tier 2 `compliance_summary` and Tier 3 bushings are parsed
  but not applied; all joints are rigid.
- **No thermal models.** Tire, brake and engine thermal data is read but not
  integrated, so there is no fade and no tire warm-up.
- **Trailers have no hitch.** SVJ does not describe coupling points yet (§23.8),
  so a trailer is simulated standing on its own axles.
- **Aero maps are only partly used.** `Cd_vs_yaw` and `Cs_vs_yaw` are applied;
  ride-height and pitch maps, components and active systems are not.

## 5. Next, roughly in order

1. **`suspension_couplings`** (walking beams, equalizer rockers, trunnion
   springs, pneumatic/hydraulic circuits) — the other half of "trustworthy
   truck and trailer examples" alongside the axle kinematics above; a
   mass/spring load-sharing problem, not a kinematic one, so it wants its own
   design rather than extending `AxleKinematics`.
2. **Compliance**, at least Tier 2 — it changes limit balance noticeably.
3. **Thermal models** for tires and brakes, so long runs and fade behave.
4. **Force feedback** for a connected wheel, once there is a wheel worth
   testing it against.
5. **The car editor**, building setup-editing UI beyond the numeric sliders
   the Setup panel already has — moving hardpoints, adding aero components.

## 6. Testing

```bash
python -m pytest tests/ -q
```

`tests/conftest.py` builds a complete symmetric vehicle in code, so tests do not
depend on the SVJ repository being checked out. The kinematics tests check the
solver against geometry worked out by hand — the roll-centre test reproduces the
front-view graphical construction. The solver tests check the properties that
catch sign errors: symmetry, load transfer direction, camber thrust cancelling,
and grip staying inside what the tires can make.

Run the suite against the real corpus too — every example in the SVJ repository
should at least build:

```bash
for f in ../SVJ-standard-vehicle-json_repo/repo/examples/*.svj.json; do
  python -m svjsim.cli info "$f" > /dev/null || echo "FAILED: $f"
done
```
