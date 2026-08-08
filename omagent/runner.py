"""Run the benchmark task ladder and persist transcripts.

Each task gets a fresh session and a fresh LLM (factories), so tasks cannot
contaminate each other through loaded classes or conversation state.
"""

from __future__ import annotations

import json
import pathlib
import time
from typing import Callable, Optional

from .loop import AgentLoop, LLM
from .session import OMSession
from .tasks import get_tasks


def run_ladder(
    session_factory: Callable[[], OMSession],
    llm_factory: Callable[[], LLM],
    task_ids: Optional[list[str]] = None,
    max_tier: Optional[int] = None,
    out_dir: str = "transcripts",
    max_attempts: int = 4,
    verbose: bool = False,
) -> dict:
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = []

    for task in get_tasks(task_ids, max_tier):
        if verbose:
            print(f"[tier {task.tier}] {task.id} ...", flush=True)
        llm = llm_factory()
        session = session_factory()
        t0 = time.time()

        env_failure = None
        if task.requires_msl:
            msl = session.load_msl()
            if not msl.success:
                env_failure = [d.brief() for d in msl.diagnostics] or \
                    ["loadModel(Modelica) returned false"]

        if env_failure is not None:
            res = None
            elapsed = time.time() - t0
        else:
            loop = AgentLoop(
                session, llm, max_attempts=max_attempts,
                simulate_options=dict(task.simulate_options),
                verifier=task.verifier)
            res = loop.run(task.prompt)
            elapsed = time.time() - t0

        record = {
            "task_id": task.id,
            "tier": task.tier,
            "prompt": task.prompt,
            "notes": task.notes,
            "success": bool(res and res.success),
            "model_name": res.model_name if res else None,
            "elapsed_s": round(elapsed, 2),
            "environment_failure": env_failure,
            "attempts": [
                {"n": a.n, "stage": a.stage, "complaint": a.complaint,
                 "diagnostics": [d.brief() for d in a.diagnostics],
                 "code": a.code}
                for a in res.attempts] if res else [],
            "llm_transcript": getattr(llm, "transcript", None),
        }
        (out / f"{task.id}.json").write_text(json.dumps(record, indent=2))

        summary_row = {
            "task": task.id,
            "tier": task.tier,
            "success": record["success"],
            "attempts": len(res.attempts) if res else 0,
            "final_stage": ("environment" if env_failure is not None else
                            (res.attempts[-1].stage if res and res.attempts
                             else None)),
            "elapsed_s": record["elapsed_s"],
        }
        results.append(summary_row)
        if verbose:
            if env_failure is not None:
                print("    FAIL (environment) — task not attempted:")
                for line in env_failure:
                    print(f"      {line}")
            else:
                mark = "PASS" if record["success"] else "FAIL"
                print(f"    {mark} in {len(res.attempts)} attempt(s), "
                      f"{elapsed:.1f}s, final stage: {summary_row['final_stage']}")

    report = {"results": results,
              "passed": sum(r["success"] for r in results),
              "total": len(results)}
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    return report
