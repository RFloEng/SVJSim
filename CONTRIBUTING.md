# Contributing to SVJSim

## Setup

```bash
git clone https://github.com/RFloEng/SVJSim.git
cd SVJSim
pip install -e ".[dev]"
python -m pytest tests/ -q
```

No SVJ files ship with this repo — the test suite builds a minimal vehicle in
code (`tests/conftest.py`) so it never depends on the SVJ standard repo being
checked out next to it. To try SVJSim against a real car, grab a `.svj.json`
from the [SVJ standard repo's examples](https://github.com/RFloEng/SVJ-standard-vehicle-json/tree/main/examples)
or its Mazda MX-5 template.

## Before you start on something non-trivial

Open an issue first for anything beyond a small fix — a new maneuver, a
tire/solver change, a GUI feature. It's a quick way to agree on the approach
before you put time into it. [DEVELOPMENT_BRIEF.md](DEVELOPMENT_BRIEF.md) has
the architecture and the reasoning behind the harder design decisions
(the K&C solver, the equations of motion, the coordinate-frame conventions);
read it before touching `kinematics.py` or `solver.py`.

## Code style

- No type-hint-free code: this is a `from __future__ import annotations`,
  fully-annotated codebase. Match it.
- Comments explain *why*, not *what* — a non-obvious constraint, a subtle
  invariant, the reason behind a magic number. If the code is
  self-explanatory, leave it uncommented.
- Don't guess physical constants, ISO/NHTSA procedure numbers, or SVJ field
  semantics — this is an engineering tool; a plausible-looking wrong number
  is worse than an obvious gap. Cite where a standard's numbers came from.
- Prefer small, focused changes over drive-by refactors.

## Tests

Every solver or kinematics change needs a test. `tests/` has hand-checked
geometry cases (`test_kinematics.py`), integrator/dynamics regressions
(`test_solver.py`), the driver-in-the-loop realtime session
(`test_realtime.py`), and tire model checks (`test_tires.py`). If you fix a
bug, add the regression test that would have caught it — see
`test_launch_from_a_standstill_actually_builds_speed` in `test_solver.py` for
the pattern: reproduce the symptom, fix it, keep the test.

For a GUI or 3D-viewer change, run `svjsim gui` and actually look at it
before opening a PR — `webapp.html` has no automated test coverage.

## Pull requests

- Keep the diff focused on one change.
- Update `CHANGELOG.md` under a new `## Unreleased` heading (or the next
  version, if you're also bumping it) — say what changed and, if it's a fix,
  what was actually wrong.
- `python -m pytest tests/ -q` should pass before you open the PR.

## Reporting bugs / data-quality issues in an SVJ file

If SVJSim's output disagrees with a real vehicle and you've ruled out a
solver bug, it may be the input data — see `svjsim info car.svj.json` for
what got estimated vs. measured. Data-quality issues in the SVJ standard's
own example files belong on the
[SVJ standard repo](https://github.com/RFloEng/SVJ-standard-vehicle-json),
not here.
