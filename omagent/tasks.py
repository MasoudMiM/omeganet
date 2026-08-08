"""Escalating benchmark tasks for LLM-assisted OpenModelica modeling.

Each task pairs a natural-language prompt with an auto-gradable quantitative
verifier — the format the OSMC benchmark effort (OpenModelica issue #15385)
asks for. Difficulty tiers:

1. pure-equation modeling (saturated for frontier models)
2. MSL component composition (exercises class lookup / connect semantics)
3. hybrid dynamics with events
4. verifier-driven design (requirement given, parameter NOT given)
5. multi-domain MSL composition

Transcripts from running this ladder — successes and failures alike — are
benchmark seed material.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .loop import Verifier
from .results import (all_of, expect_bounds, expect_dips_below,
                      expect_final, expect_value_at)


@dataclass(frozen=True)
class Task:
    id: str
    tier: int
    prompt: str
    verifier: Verifier
    simulate_options: dict = field(default_factory=dict)
    notes: str = ""
    requires_msl: bool = False


def _t(**kw) -> Task:
    return Task(**kw)


TASKS: list[Task] = [
    _t(
        id="msd_equations",
        tier=1,
        prompt=(
            "A translational mass-spring-damper: mass m = 1 kg, spring "
            "stiffness c = 100 N/m, damping d = 1 N.s/m (damping ratio 0.05). "
            "The mass starts displaced at x = 0.1 m with zero velocity and is "
            "released (no external force). Model it directly with equations "
            "(der(x), der(v)); name the position x and the velocity v."),
        simulate_options={"stopTime": 10.0, "outputFormat": "csv"},
        verifier=all_of(
            expect_bounds("x", lo=-0.105, hi=0.105),
            expect_final("x", 0.0, atol=0.06, rtol=0.0)),
        notes="Tier-1 baseline; frontier models pass first try."),

    _t(
        id="msl_oscillator",
        tier=2,
        prompt=(
            "The same oscillator built from Modelica Standard Library "
            "components: use Modelica.Mechanics.Translational.Components "
            "(a Mass of 1 kg, a Spring of c = 100 N/m, a Damper of "
            "d = 1 N.s/m in parallel with the spring, anchored to a Fixed "
            "flange). The mass starts at position 0.1 m (use s(start=0.1, "
            "fixed=true) on the mass, v starting at 0), spring unstretched "
            "length zero. Expose top-level variables `Real x = mass.s;` and "
            "`Real v = mass.v;` where `mass` is the Mass component."),
        simulate_options={"stopTime": 10.0, "outputFormat": "csv"},
        verifier=all_of(
            expect_bounds("x", lo=-0.105, hi=0.105),
            expect_final("x", 0.0, atol=0.06, rtol=0.0)),
        notes="Exercises MSL class lookup and connect(); where ModiGen saw "
              "hallucinated class names.",
        requires_msl=True),

    _t(
        id="bouncing_ball",
        tier=3,
        prompt=(
            "A bouncing ball: height h starts at 1.0 m (fixed), velocity v "
            "starts at 0, gravity g = 9.81 m/s2, coefficient of restitution "
            "e = 0.7. On ground contact (h <= 0 while falling) the velocity "
            "reverses with factor e using a when-equation and reinit. Name "
            "the height h and velocity v."),
        simulate_options={"stopTime": 5.0, "outputFormat": "csv"},
        verifier=all_of(
            expect_bounds("h", lo=-0.02, hi=1.02),
            expect_bounds("h", lo=-0.02, hi=0.12, after=4.0)),
        notes="Hybrid dynamics: events, reinit, potential chattering near "
              "rest — exercises runtime feedback."),

    _t(
        id="tune_damping",
        tier=4,
        prompt=(
            "A translational mass-spring-damper with m = 1 kg and "
            "c = 100 N/m, released from x = 0.1 m at zero velocity. CHOOSE "
            "the damping coefficient d yourself such that the response is "
            "underdamped (it must visibly overshoot through zero at least "
            "once) but |x| stays below 0.005 m for all t >= 2 s. Model with "
            "direct equations; name position x, velocity v, and make d a "
            "parameter with your chosen value."),
        simulate_options={"stopTime": 5.0, "outputFormat": "csv"},
        verifier=all_of(
            expect_bounds("x", lo=-0.105, hi=0.105),
            expect_dips_below("x", -1e-4),   # underdamped: overshoots zero
            expect_bounds("x", lo=-0.005, hi=0.005, after=2.0)),
        notes="Design task: the requirement is given, the parameter is not. "
              "Verifier complaints must drive tuning, not just repair. "
              "(zeta must land in roughly (0, 0.3]; e.g. d around 3-6.)"),

    _t(
        id="dc_motor",
        tier=5,
        prompt=(
            "A DC motor drive built from Modelica Standard Library "
            "components: a 12 V constant voltage source, armature resistance "
            "R = 1 Ohm and inductance L = 1e-3 H, an EMF with machine "
            "constant k = 0.05 N.m/A, driving an Inertia of J = 1e-3 kg.m2 "
            "with no load and no friction. Ground the electrical circuit. "
            "Expose the shaft speed as a top-level variable "
            "`Real w = inertia.w;` where `inertia` is the Inertia component."),
        simulate_options={"stopTime": 2.0, "outputFormat": "csv"},
        verifier=all_of(
            # steady state: w = V/k = 240 rad/s (no friction)
            expect_final("w", 240.0, atol=0.0, rtol=0.02),
            expect_bounds("w", lo=-1.0, hi=245.0)),
        notes="Multi-domain (electrical + rotational) MSL composition.",
        requires_msl=True),
]

TASKS_BY_ID: dict[str, Task] = {t.id: t for t in TASKS}


def get_tasks(ids: Optional[list[str]] = None,
              max_tier: Optional[int] = None) -> list[Task]:
    tasks = TASKS
    if ids:
        unknown = [i for i in ids if i not in TASKS_BY_ID]
        if unknown:
            raise KeyError(f"unknown task id(s): {unknown}; "
                           f"available: {list(TASKS_BY_ID)}")
        tasks = [TASKS_BY_ID[i] for i in ids]
    if max_tier is not None:
        tasks = [t for t in tasks if t.tier <= max_tier]
    return tasks
