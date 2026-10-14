# SVJSim Changelog

## Unreleased — SVJ v0.99.3: differential control, joint types, and a driveline that follows the file

- **`lock_control` (SVJ §10.8.1) is simulated.** Chains of stages (input → curve
  → multiply/add, with a low-pass `filter_time_constant` and `min`/`max`
  clamps) modulate `lock`, `lock_power`, `lock_coast` or `preload` from gear,
  brake, throttle, speed, axle speed ratio, an oversteer indicator and axle
  slip angles; an array gives separate power and coast chains. The filters are
  integrated with the vehicle state, so they follow `exp(-t/τ)` exactly.
- **`transfer_case.coupling` and `center_diff` are simulated.** Before, the
  shaft torque was shared equally between every driven wheel, whatever the
  file said. Now there is a centre stage (fixed split, centre differential, or
  a torque-limited coupling whose capacity rises from `ramp_torque` to
  `max_torque`) followed by one differential per axle.
- **Fixed: cars that declare no `half_shafts` had no driven wheels.** Seven
  bundled examples (RWD, FWD and 4WD) could not accelerate; the driven wheels
  now come from the layout / `drive_type`, then from where the differentials
  sit, and the build report says which.
- **Fixed: trucks ran with a final drive of 1.** The inter-axle differential,
  which has no final drive, was taken for the axle's.
- **Fixed: marker-form points (`{position, orientation}`) vanished from the
  3D view.** The solver read them, the viewer's geometry did not.
- `svjsim info` shows the centre device and lock control; telemetry gains
  `drive_torque_<station>` channels. MX-5 results are unchanged (0–100 km/h
  6.01 s, standing 400 m 14.21 s).

- **`joint_type` and `inboard_joint_type` (SVJ §9.2.3.2) decide how a link
  constrains its upright.** Explicit values win; missing ones come from the
  spec's inference table (kept in step with `svj.multibody` by a test). The
  common combinations behave as before: a rod, or an A-arm with a revolute
  inboard and ball outboard, holds its distances; a strut keeps its axis
  through the top mount. What changes, and only for files that state or imply
  something else: a `fixed` outboard joint makes the arm carry the upright
  rigidly whichever hardpoint it attaches at (a trailing arm bolted at an axle
  seat, previously solved as a lone distance and left under-located); a
  `pivot` link with a revolute outboard turns the upright about its pin
  (pendulum axles, likewise under-located before); revolute joints at both
  ends with parallel axes keep a parallelogram's housing level and move its
  wheel on the beams' arc; a `cylindrical` inboard joint lets the arm slide
  along its axis; a link rigid with the chassis pins its ball joint. A ball
  joint stated on a trailing arm that nothing else locates is overruled, with a
  warning, rather than leaving the upright free. Joint axes are not in SVJ: they
  come from the two inboard points, otherwise a lateral axis is assumed, and
  joints SVJSim cannot turn into a constraint (`universal`, `planar`, ...) are
  reduced to the position they imply, all reported in the build notes. Of the
  26 bundled examples, three change (the two trailers and the parallelogram
  truck); the other 23, including the MX-5, are identical.
## Unreleased — Vehicle meshes that load, SVJ v0.99.2 binding, alignment you can see

- **Static camber and toe now show in the 3D view.** The server used to send
  only the solved suspension motion, so a camber or toe change reached the tyre
  maths but never the picture. The viewer now builds the wheel axis the way the
  solver does (static alignment plus solved motion), and a mesh bound to a
  station follows alignment changes made since the file's own setup.
- **Mesh pose is a rotation about the wheel centre**, not about the body
  origin. Harmless for a millimetre of jounce, wrong for a visible camber change.
- **Loading the `.glb` that belongs to a `.svj.json`.** `../meshes/x.glb` style
  URIs (meshes kept in a sibling folder) now resolve, served only for glTF/image
  extensions; absolute paths, drive letters and URLs are still refused. Load
  file and drag-and-drop locate the file's folder when it is unambiguous, a
  *Load .glb…* button (or a `.glb` dropped with the JSON) supplies it when the
  browser can't, and the viewport shows per-asset status with the reason when a
  mesh doesn't load.
- **Meshes that don't follow the SVJ naming.** When a file has no `SVJ::` nodes
  the whole model is shown as the chassis visual (unplaced tyre/disc/caliper
  parts left out). Y-up glTF written over a bare `SAE_J670` label is detected,
  and which end is the front is read from part names (`bumper_f`, `bonnet`,
  `plate_b`…), falling back to the SVJ §22.4 default. The guess is reported, and
  an Axes selector, opacity slider and optional *Fit to wheels* override it.
- **SVJ v0.99.2 visual binding: every bound part placed by what it is.**
  v0.99.2 lets `visual` sit on links, wheels, dual wheels, discs, calipers,
  springs, dampers, ARBs, steering, powertrain units, driveshafts and aero
  parts. Before, every binding under a corner was treated as the upright and
  only the last one per station moved; steering, powertrain and aero bindings
  were dropped. Now uprights, axle bodies and calipers follow the solved
  upright; wheels and discs also spin; `placement: "link_between_points"`
  links sit between their inboard point and the live outboard hardpoint
  (`mesh_axis`, `from_point`, `scale_to_length` honoured); driveshafts sit
  joint to joint; everything else rides on the body. A binding without
  `mesh_ref` uses the file's only mesh. Spec errors (placement on the wrong
  carrier, a node bound twice, `from_point` out of range) are shown in the
  viewport; one axle part bound from both its stations is drawn once and not
  flagged. Checked against the spec's 45-binding
  `skeleton_visual_binding_dw_corner` example with synthetic meshes: links on
  their hardpoints to nanometres at the design pose, stretched tie rods
  exactly on them through a step-steer run.
- **Fixed: mass-body meshes were offset twice.** The body's `position` was
  added to a mesh already authored in the vehicle frame (§22.6 `rigid`).
- **A nested glTF node keeps its parents' transforms.** A node cloned on its
  own used to lose them, so a part under a moved group landed in the wrong
  place.
- **One set of mesh axes per vehicle**, decided from the chassis's asset:
  §22.4 declares the convention per file, and a small asset (a steering
  column, four calipers) can't be read from its shape.

## v0.5.0 — Axle-level kinematics for beam axles

- **`solid_axle`/`de_dion` beam axles are now properly solved, not
  approximated.** Previously each side of a rigid axle was solved as an
  independent, necessarily under-constrained upright (per `BEAM_TOPOLOGIES`'s
  own docstring), with only camber patched up afterward from a
  `(jounce_L − jounce_R) / track` small-angle formula — toe was hard-set to
  static, and roll centre height returned `None` ("not derivable from one
  wheel"), because neither actually is derivable from one side alone. New
  `AxleKinematics` (`kinematics.py`) solves the axle as the single rigid body
  it physically is: six degrees of freedom, two driven directly by each
  side's own commanded travel, the other four (lateral/longitudinal position,
  pitch, yaw) from whichever links the file declares — trailing links, a
  Panhard rod, a Watt's link, a leaf spring — selected for independence the
  same way an ordinary corner's five are, just four instead of five. Camber
  and toe fall out for free from the one solved pose (no more small-angle
  approximation, and bump-steer-through-roll is now real instead of zero),
  and roll centre height is genuinely computed — a small roll is sampled
  either side of design and the resulting motion gives the axle's own
  instantaneous centre of rotation, the whole-axle generalization of the
  single-corner front-view graphical construction. Verified against the
  classic hand construction (a Panhard-rod axle's roll centre sits on the
  rod's own line, at the height it crosses the vehicle centreline) to
  sub-millimetre precision, and against two real bundled example files
  (`skeleton_4wd_solid_axle`, `alfa_romeo_75_de_dion`) building and running
  cleanly with plausible roll-centre numbers. `torsion_beam`, `parallelogram`
  and `pendulum_axle` aren't covered by this yet, nor is a steered beam axle
  (no bundled example combines the two, and it would need a genuinely 3D
  table). `suspension_couplings` (§23.5 — walking beams, equalizer rockers,
  shared air/hydraulic circuits) is a separate, still-unstarted problem: mass
  and spring load-sharing *between* axles, not kinematics *within* one.
- `axle_body.mass` now contributes to unsprung mass (split 50/50 between the
  axle's two corners) when SVJSim has to estimate unsprung mass itself, per
  SVJ §9.2.4's "MUST be included" rule — previously ignored entirely, which
  meant an estimated solid-axle vehicle was silently missing its whole axle
  housing's worth of unsprung mass (often 100+ kg).

- Prepared the repository for public release: `.gitignore`, a GitHub Actions
  test workflow, `CONTRIBUTING.md`, issue templates, and `svjsim --version`.
- **Tire force arrows and friction-circle rings are now filtered.** The raw
  per-step Fx/Fy has real high-frequency content from contact-patch dynamics
  that is physically genuine but reads as jitter at the frame rate of an
  arrow. A one-pole low-pass (`FORCE_FILTER_TAU`, 0.06 s) is applied to
  Fx, Fy and the derived available-grip radius *before* they're drawn —
  measured as roughly a 3x reduction in frame-to-frame force noise on the
  Acceleration test's rear wheels. For a finished run this is precomputed
  once (`filterChannels`, alongside `integrateSpinAngles`); for live driving
  it's the same filter run one sample at a time (`liveFilter`), since there
  is no future to precompute from yet.
- **Speed and net force at the CG.** Same checkbox as the per-wheel overlay,
  now drawn at the vehicle's centre of gravity too: a blue arrow for the
  body-frame velocity (so sideslip is directly visible — it points away from
  the car's own heading during a slide, not always straight ahead) and an
  orange arrow for the net force (vehicle mass times the filtered
  acceleration, i.e. what an accelerometer at the CG reads, times mass —
  not just the sum of tire forces, so it also reflects aero drag and the
  rest of what `ax`/`ay` already account for).
- **Vertical load (Fz) drawn at each wheel, and the friction-circle color is
  now a threshold, not a gradient.** A third arrow per wheel, straight up
  from the contact patch, sized through the same `FORCE_SCALE` as the
  horizontal one — so a glance across ring, horizontal arrow and vertical
  arrow shows how loaded a corner is, how much grip that buys it, and how
  much of that grip it's using, e.g. front load visibly dropping and rear
  rising under hard acceleration. The ring itself used to shade continuously
  from green to red as usage rose, which read as orange-ish most of the
  time; it's now green under the limit and red at or past it (actually
  slipping), with only a narrow 92–100% ramp between the two so filtered
  data sitting right at the limit doesn't flicker.
- **3D viewer: wheels now actually spin.** `articulateStations` set the
  wheel's camber/steer tilt but never its roll angle — `setFromUnitVectors`
  gives the minimal rotation between two vectors, which has no defined twist
  about the resulting axis, so a wheel's rotation was never applied at all.
  Composing a separate spin quaternion (rotate about local Y first, then
  reorient that axis into place) fixes it, but a plain cylinder looks
  identical at every angle regardless, so a radial marker was added to each
  wheel to make the rotation visible. The wheel-spin *state* the physics
  actually tracks is angular velocity, not an angle (there is no rotation
  state to record — it has no effect on the dynamics), so a maneuver replay
  integrates the recorded `spin_<station>` channel once when the run loads,
  and live driving integrates it incrementally per poll, keyed on the
  session's own sim clock.
- **3D viewer: the ground plane now has a visible fixed reference.** It was
  already parented to the scene rather than the car (so it was never
  literally moving), but a perfectly repeating grid gives the eye nothing to
  anchor on, so camera/car motion against it was hard to read — the same
  problem the wheels had. Two colored lines through the world origin
  (forward/lateral) fix it the same way the wheel spoke does.
- **Wheel spin was rotating backwards.** `state.spin`'s sign was chosen in
  `solver.py` purely so `spin * r_eff` reads as a forward contact-patch
  speed in the slip-ratio maths — before this viewer existed, it was never
  used as a literal rotation, so nothing had verified it also matched the
  right-hand rotation sense about the geometric spin axis. It didn't;
  negated at the one point it becomes a `THREE.Quaternion`.
- **Maneuver replay now shows the car actually moving.** It still holds the
  car at the origin (so the camera stays framed on it and the suspension is
  easy to watch — see the entry above), but that used to mean the ground
  never appeared to move either, since nothing else in the scene reflected
  the car's true recorded trajectory. Now the car also carries its real yaw
  (so you can see it turn), and the ground — both the grid and the two axis
  lines — carries the inverse of the car's true x/y/yaw, so it visibly
  slides and turns under the car by exactly as much as the car really moved.
  The grid's offset is wrapped to a single cell so a patch that looks
  identical at any multiple of its own size can scroll indefinitely without
  either the car outrunning it or the grid needing to be unreasonably large;
  the axis lines use the true, unwrapped offset, since they're a landmark at
  a specific point, not a repeating pattern.
- **3D viewer: real-time tire forces and friction circles.** A "Tire forces
  & friction circle" checkbox over the viewport draws, at each wheel's
  ground contact point, a ring sized to the tire's currently available grip
  (`mu * Fz`) and an arrow for the actual (Fx, Fy) force it's using — the
  arrow tip reaching or crossing the ring is exactly the friction-circle
  limit, drawn where the force actually acts instead of on a separate chart.
  Colored green well within the limit, sliding to red at and beyond it.
  Works both scrubbing a finished run and while driving live; the live path
  needed `Fx`/`Fy` added to `RealtimeSession.snapshot()`'s per-wheel data,
  which previously only carried `Fz` and the slip/camber/steer angles.

## v0.4.0 — glTF visual meshes (SVJ §22)

- `svjsim gui`'s 3D view now loads and displays a vehicle's real glTF meshes
  when the SVJ file declares them (`assets.meshes` + a `visual` block on
  `chassis` and/or a `suspension.<corner>` entry, per §22). The mesh is drawn
  **alongside** the existing procedural placeholder geometry, never instead
  of it, and the placeholder ghosts to a dim wireframe once a mesh is shown
  for the same body — so a real asset and the physics model's own theoretical
  hardpoints stay simultaneously visible and any discrepancy between them is
  something you can actually see, not something the viewer papers over.
- New backend pieces (`svjsim/server.py`): `find_visual_bindings()` walks the
  document structurally for `visual` blocks (found in practice sitting
  directly on `.chassis` and `.suspension.<corner>`, not nested under
  `.wheel`/`.topology.upright` as the spec's naming table alone might
  suggest); `resolve_assets()` resolves each `assets.meshes[].uri` against
  the SVJ file's own folder; a new `GET /api/asset` route serves the mesh
  bytes, path-checked the same way `_safe_resolve` already guards
  `/api/open`. A vehicle opened via `svjsim gui file.svj.json` or the picker
  now carries its folder through to mesh resolution; one loaded via
  drag-and-drop has no known folder, so its assets are correctly reported
  `available: false` with a console warning rather than failing the load.
- **Two coordinate-frame bugs found and fixed** while wiring this up, both
  the kind that only show up once you actually look at the render rather
  than trust the algebra of one hand-checked case (see `bodyQuaternion` in
  the v0.3.1 entry below for the earlier instance of the same lesson):
  - three.js's `GLTFLoader` silently sanitizes every glTF node name on
    import (`THREE.PropertyBinding.sanitizeNodeName` strips `.:[]/` and
    spaces, since node names double as `AnimationClip` track-path
    components) — so SVJ's own `SVJ::category::id` node-naming convention
    (§22.5) never survives a glTF round-trip unchanged. `attachVisualMeshes`
    now looks a node up by its literal name first, falling back to the
    sanitized form.
  - A station's visual mesh must **not** be repositioned to the theoretical
    wheel-centre point the way the procedural wheel is. A mesh node's own
    authored position already places it in the shared vehicle-frame asset —
    the real example files bind the chassis *and* all four uprights to one
    shared file (`corvette_c3.glb`), modeled together like a real assembly.
    Overwriting that with `wheel_center0` would force every mesh to coincide
    with the theoretical point by construction, making it impossible to ever
    see a real hardpoint-authoring error — exactly backwards for a feature
    whose purpose is showing discrepancies. Fixed by applying only the
    *incremental* jounce/steer motion since the design pose (computed once
    via `poseAt(g, 0, 0)`), leaving whatever absolute offset the asset
    already has fully visible while still tracking suspension articulation.
- Procedural wheels got a pass of visual detail (lip and sidewall tori, a
  darker rim, a hub sphere) borrowed from the reference SVJ viewer
  (`svj_viewer_v4.2.html`), partly for its own sake and partly so the
  ghosted-wireframe fallback still reads as a wheel when no mesh is bound.
- New maneuver: **`fishhook`**, NHTSA's Fishhook rollover-propensity test
  (the "Roll Rate Feedback" Default Procedure behind its NCAP dynamic
  rollover rating). An internal Slowly Increasing Steer sub-test finds the
  steering-wheel angle giving 0.3 g at the test speed (13.5 deg/s ramp);
  the Fishhook proper ramps to 6.5x that angle at 720 deg/s, holds until the
  vehicle's own roll rate decays back through 1.5 deg/s — timing the
  reversal to the vehicle's actual roll dynamics rather than a fixed delay
  — reverses to the same angle the other way, and holds. Reports two-wheel
  lift (both wheels on one side losing contact at once), NHTSA's own
  failure criterion, plus peak roll angle. The existing suite already had
  ISO 4138 (both the constant-radius and increasing-steer methods, as
  `skidpad`/`steer`), ISO 7401 step response (`step`) and ISO 3888-2
  (`moose`) — Fishhook was the one standardized test genuinely missing:
  nothing else in the bench says anything about rollover risk, which
  matters given SVJSim's multi-axle/truck support.

## v0.3.1 — Fixed the car rendering on its end

- **Fixed:** the 3D view's body orientation (`Viewer.bodyQuaternion` in
  `webapp.html`) rendered the vehicle rotated ~90°, standing on end instead of
  lying flat, in *both* the static and driving views ever since the viewer was
  generalised to full 6-DOF for driver-in-the-loop. The rotation matrix's
  three columns were each converted through `toThree()` individually, which
  looks reasonable but is wrong: the body's children (wheels, links, chassis
  box) already have their positions expressed in three-space via `toThree()`,
  so rotating them correctly requires conjugating the whole matrix by the
  SVJ→three change of basis (`R_three = M · R_svj · M⁻¹`), not converting
  columns one at a time — the column-wise version bakes in a spurious
  rotation even at roll = pitch = yaw = 0. Caught from a screenshot after the
  user asked "why is the car perpendicular to the floor plane?" — a reminder
  to actually look at a render after a coordinate-frame change, not just
  check that the maths of one isolated case works out.

## v0.3.0 — Driver-in-the-loop

- `svjsim gui`'s **Drive** tab: start/stop a live session, drive with WASD or
  arrow keys or a connected gamepad/wheel, watch it in the same 3D view (now a
  full 6-DOF chase camera instead of the maneuver-replay "car stays in
  place" view), reset after a spin, shift manually or leave it on auto.
- `svjsim/realtime.py` — `RealtimeSession`: a fixed-step physics loop in a
  background thread, decoupled from HTTP request/response timing. Thread-safe
  control input and state snapshots via one lock, no queueing — a viewer that
  misses a frame just reads the next one.
- Stepped with **RK4 at 100 Hz**, not the cheaper semi-implicit Euler at
  500 Hz an early version used. Discovered why the hard way: a parked car
  would slowly rock and creep forward over several seconds under Euler,
  which turned out to be discretisation error, not unstable physics — the
  vehicle's coupled heave/pitch dynamics have a mode Euler cannot resolve
  accurately at an affordable step size (its amplification factor exceeds 1
  for any oscillatory mode, at any step size), confirmed by shrinking dt and
  watching the drift shrink in proportion until it vanished with RK4. RK4 at
  a lower rate costs about the same per second and has ample margin against
  the GIL contention of sharing a process with the HTTP server.
- **`step_euler` fixed to be genuinely semi-implicit** (rates first, then
  positions from the updated rates) rather than one-shot forward Euler,
  which is unconditionally unstable for any oscillatory mode — still not
  accurate enough for this vehicle at 500 Hz (hence the RK4 switch above),
  but correct in its own right for anyone stepping with it directly.
- **Found and fixed a physics bug this work surfaced**: low-speed contact-
  patch damping (added to stop a parked car creeping under gravity's
  component along its own slightly-pitched equilibrium attitude) shared its
  fade-out distance with an unrelated wheel-spin damping term sized for a
  much smaller effective mass. At the shared 18 km/h fade-out width, every
  wheel — including undriven ones — dragged with several thousand newtons of
  numerical friction, comfortably swamping the driven wheels' actual
  tractive force. A standing-start launch could barely reach walking pace.
  Fixed with a separate, much narrower fade-out (`V_BODY_DAMP_BLEND`) sized
  to when it is actually needed: the last few centimetres per second before
  standstill. Added `test_launch_from_a_standstill_actually_builds_speed` as
  a regression guard — every existing throttle test happened to start above
  the old bug's threshold, so none of them had caught it.
- A quick, RK4-based settle (`_quick_settle`) replaces `VehicleDynamics.trim`
  for "start driving" and "reset" — the same 1.5 s / RK4 settle used before a
  maneuver costs about 5 s of wall clock, an unacceptable freeze for an
  interactive button; damping continuously rather than only for trim's first
  half converges in well under a second.
- An auto-clutch launch ramp for the no-clutch-pedal driving interface,
  mirroring `Acceleration`'s validated scripted launch.
- Setup edits and vehicle reloads now stop any running drive session — driving
  through a stale `VehicleDynamics` after a rebuild was never a sound state.

## v0.2.0 — Local web app

- `svjsim gui` serves a single-page app from the standard library's
  `http.server`: no framework, no build step, starts instantly.
- 3D viewer built with three.js from the vehicle's own data — suspension links
  move through the solved upright pose, so the geometry on screen is the
  geometry being simulated, not a stylised drawing. Works for any number of
  stations; an 8×4 truck renders the same way a roadster does.
- Setup panel over spring rates, damper scaling, anti-roll bars, camber, toe,
  tire pressure, mass, CG, brake bias and aerodynamics — grouped per axle, built
  from whatever the loaded file actually contains.
- **Setup edits are SVJ override patches.** A change builds an RFC 7396 merge
  patch (§3.4), applies it to the base document and rebuilds the vehicle.
  *Export override* writes a `.svj-override.json` that passes the standard's own
  `tools/validate_override.py`.
- Run any maneuver from the browser with live progress, then scrub the result
  through the 3D view with a playback bar and a live telemetry readout.
- *Set baseline* keeps a run so the next one reports the change on every KPI.
- Charts, the wheel-station table and the provenance report are shown alongside.
- `KCTable` now carries the full upright pose per operating point, which is what
  lets the viewer articulate the linkage.
- Maneuvers report progress through an optional callback.

## v0.1.0 — Physics core and automated test bench

First working release. Loads an SVJ vehicle, solves its suspension, and runs it
through a standard test suite, producing an offline HTML report.

### Suspension kinematics

- Constraint solver for the SVJ link topologies, by damped least-squares Newton
  on the upright pose. Covers `double_wishbone`, `macpherson`, `chapman_strut`,
  `multi_link`, `solid_axle`, `de_dion` and `custom` in link mode, and
  `trailing_arm`, `semi_trailing_arm`, `torsion_beam`, `swing_axle` and
  `pendulum_axle` as single-degree-of-freedom revolute joints.
- K&C tables over wheel travel and steering rack travel: wheel-centre path,
  camber, toe, and spring, damper and ARB motion ratios.
- Derived characteristics computed from the geometry rather than declared:
  camber gain, bump steer, half-track change, wheel-centre recession, and the
  roll-centre height from the solved contact-patch path.
- Redundant links are resolved by selecting a maximal independent constraint
  set, so an over-defined topology reports what was dropped instead of locking
  the suspension solid.
- The travel and rack envelopes are probed and clipped to what the linkage can
  actually reach, which surfaces hardpoints that cannot assemble.

### Tires

- Pacejka Magic Formula (MF 5.2 and 6.2): pure lateral, pure longitudinal,
  combined slip via the MF weighting functions, self-aligning and overturning
  moments, rolling resistance. Falls back to a friction ellipse when a file has
  no `combined` coefficient group.
- TMeasy, including the SVJ §11.10.3 load-sensitivity derivatives.
- Brush model with a load-dependent contact patch length.
- Tire synthesis from size, pressure and service class for the many SVJ files
  that carry dimensions but no fitted force model. Always marked `estimated`.

### Vehicle model

- Mass decomposition into sprung and unsprung parts, with the sprung inertia
  taken from `mass_bodies` where present and otherwise derived from the vehicle
  tensor.
- Static wheel loads from a rigid body on linear springs, generalised to any
  number of stations; trailers skip longitudinal balance because their king-pin
  load is not yet described by SVJ (§23.8).
- Aerodynamics, hydraulic brakes (SVJ §12.6 force chain), engine, clutch,
  gearbox and limited-slip differential.
- A build report recording every assumption, estimate and inconsistency, with an
  overall confidence rating.

### Solver

- `6 + N` degree-of-freedom equations of motion assembled from per-body velocity
  Jacobians, so jacking, anti-dive and anti-squat follow from the geometry
  instead of being parameters, and any number of wheel stations works unchanged.
- Inertial-frame ground contact with a cambered rigid tire disc.
- Slip relaxation states, low-speed contact-patch damping, a friction clutch and
  an engine with an idle governor.
- RK4 and semi-implicit Euler integrators, and a constrained trim that settles
  the vehicle onto its suspension before a maneuver begins.

### Tests and reporting

- Maneuvers: standing-start acceleration, straight-line braking, slowly
  increasing steer, constant-radius skidpad, step steer, friction circle, and
  the ISO 3888-2 moose test located by bisection on entry speed.
- KPIs compared against the file's `benchmarks` block (SVJ §20b).
- Self-contained HTML report with Plotly, leading with provenance.
- `svjsim info`, `svjsim kc` and `svjsim run` command line entry points.

### Validated against

The Mazda MX-5 ND2 reference file from the SVJ standard: 0–100 km/h in 6.46 s
(real ~6.5 s), standing 400 m in 14.5 s (~14.8 s), 100–0 km/h in 34.4 m
(~34–36 m), 1.01 g maximum lateral acceleration (~0.95 g), mild understeer. No
parameter was tuned to reach these.
