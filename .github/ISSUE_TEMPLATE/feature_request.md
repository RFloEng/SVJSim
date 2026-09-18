---
name: Feature request
about: A maneuver, tire model, solver capability, or GUI feature SVJSim is missing
title: ""
labels: enhancement
---

**What's missing**

**Why it matters**
What real vehicle-dynamics question does this let you answer that you can't
today?

**If it's a standardized test (ISO/SAE/NHTSA/etc.)**
Link the standard and, if you know them, the key numeric parameters (speeds,
rates, thresholds). SVJSim's existing tests are implemented against the
actual published procedures, not approximations from memory — a PR adding
one should cite where its numbers came from.

**Rough shape of the change**
Which part of the codebase this likely touches (`maneuvers.py`, a new tire
model in `tires.py`, the K&C solver, the GUI), if you have a guess.
