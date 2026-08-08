"""Run the escalating benchmark ladder against a live LLM and real omc.

Usage:
    export ANTHROPIC_API_KEY=...
    python examples/run_ladder.py                 # all 5 tiers
    python examples/run_ladder.py --max-tier 3    # first three rungs
    python examples/run_ladder.py --tasks msl_oscillator dc_motor
    python examples/run_ladder.py --model claude-opus-4-8

Each task gets a fresh omc session and fresh LLM. Transcripts land in
transcripts/<task>.json plus transcripts/summary.json.
"""

import argparse
import sys

from omagent import OMSession
from omagent.llm import ClaudeLLM
from omagent.runner import run_ladder


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--max-tier", type=int, default=None)
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--max-attempts", type=int, default=4)
    ap.add_argument("--out", default="transcripts")
    args = ap.parse_args()

    report = run_ladder(
        session_factory=OMSession,
        llm_factory=lambda: ClaudeLLM(model=args.model),
        task_ids=args.tasks,
        max_tier=args.max_tier,
        out_dir=args.out,
        max_attempts=args.max_attempts,
        verbose=True,
    )
    print(f"\n{report['passed']}/{report['total']} tasks passed "
          f"-> {args.out}/summary.json")
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    sys.exit(main())
