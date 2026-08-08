"""First live run: mass-spring-damper with quantitative verification.

Usage:
    pip install -e ".[omc,results,llm]"
    export ANTHROPIC_API_KEY=...
    python examples/first_run.py

The task is deliberately quantitative: the verifier checks physics
(settling toward equilibrium, bounded overshoot), so a model that merely
compiles and runs is not enough — complaints feed back into the fix loop.
"""

import json
import pathlib
import sys

from omagent import (
    AgentLoop, OMSession, all_of, expect_bounds, expect_final,
)
from omagent.llm import ClaudeLLM

TASK = (
    "A translational mass-spring-damper: mass m = 1 kg, spring stiffness "
    "c = 100 N/m, damping chosen for a damping ratio of 0.05 (i.e. d = 1 N.s/m). "
    "The mass starts displaced at x = 0.1 m with zero velocity and is released "
    "(no external force). Model it directly with equations (der(x), der(v)); "
    "name the position variable x and the velocity v."
)

# Physics acceptance: underdamped oscillation decays toward 0; with zeta=0.05
# the envelope after 10 s (~16 periods) is well under 0.06 m, and |x| can
# never exceed the initial displacement.
verifier = all_of(
    expect_bounds("x", lo=-0.105, hi=0.105),
    expect_final("x", 0.0, atol=0.06, rtol=0.0),
)


def main() -> int:
    session = OMSession()
    print(f"omc {session.get_version()}")
    llm = ClaudeLLM()

    loop = AgentLoop(
        session, llm, max_attempts=4,
        simulate_options={"stopTime": 10.0, "outputFormat": "csv"},
        verifier=verifier,
    )
    result = loop.run(TASK)

    print(f"\nsuccess: {result.success}  model: {result.model_name}")
    for a in result.attempts:
        print(f"  attempt {a.n}: stage={a.stage}"
              + (f"  ({len(a.diagnostics)} diagnostics)" if a.failed else ""))
        if a.complaint:
            print(f"    verifier: {a.complaint}")

    out = pathlib.Path("first_run_transcript.json")
    out.write_text(json.dumps({
        "task": TASK,
        "success": result.success,
        "attempts": [
            {"n": a.n, "stage": a.stage, "complaint": a.complaint,
             "diagnostics": [d.brief() for d in a.diagnostics]}
            for a in result.attempts],
        "llm_transcript": llm.transcript,
    }, indent=2))
    print(f"\ntranscript saved to {out}  (benchmark seed material!)")

    if result.success and result.final_code:
        print("\n--- final model ---\n" + result.final_code)
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
