"""Tests for the task ladder registry and runner (fake LLM + fake omc)."""

import json

import pytest

from omagent.session import OMSession
from omagent.tasks import TASKS, TASKS_BY_ID, get_tasks
from omagent.runner import run_ladder

from tests.fakes import FakeOMC
from tests.test_loop import ScriptedLLM, GOOD, CHECK_OK, SIM_OK


class TestRegistry:
    def test_ids_unique_and_tiers_ordered(self):
        ids = [t.id for t in TASKS]
        assert len(ids) == len(set(ids))
        assert [t.tier for t in TASKS] == sorted(t.tier for t in TASKS)

    def test_every_task_complete(self):
        for t in TASKS:
            assert t.prompt and t.verifier and t.simulate_options
            assert "stopTime" in t.simulate_options
            assert t.simulate_options.get("outputFormat") == "csv", \
                "ladder must not require scipy"

    def test_get_tasks_by_id_and_tier(self):
        assert [t.id for t in get_tasks(["dc_motor", "msd_equations"])] == \
            ["dc_motor", "msd_equations"]
        assert all(t.tier <= 2 for t in get_tasks(max_tier=2))
        with pytest.raises(KeyError):
            get_tasks(["nope"])


def _passing_session(tmp_path):
    """Fake omc whose simulate writes a real CSV the verifiers accept."""
    csv_path = tmp_path / "res.csv"
    csv_path.write_text(
        "time,x,v\n" + "".join(
            f"{t/10},{0.1 * (0.0 if t > 5 else 1.0):.3f},0\n"
            for t in range(0, 101)))
    # x = 0.1 until t=0.5 then 0 — satisfies msd_equations verifier
    sim = dict(SIM_OK)
    sim["resultFile"] = str(csv_path)
    return OMSession(FakeOMC({
        "loadString(": [True],
        "checkModel(": [CHECK_OK],
        "simulate(": [sim],
        "getErrorString()": ['""'] * 3,
    }))


def test_run_ladder_single_task(tmp_path):
    out = tmp_path / "transcripts"
    report = run_ladder(
        session_factory=lambda: _passing_session(tmp_path),
        llm_factory=lambda: ScriptedLLM([GOOD]),
        task_ids=["msd_equations"],
        out_dir=str(out),
        max_attempts=2,
    )
    assert report["results"][0]["task"] == "msd_equations"
    assert report["results"][0]["success"] is True
    assert report["results"][0]["attempts"] == 1
    saved = json.loads((out / "msd_equations.json").read_text())
    assert saved["task_id"] == "msd_equations"
    assert saved["attempts"][0]["code"] == GOOD
    assert saved["llm_transcript"] is None   # ScriptedLLM records none; optional
    assert (out / "summary.json").exists()


def test_run_ladder_records_failure(tmp_path):
    fail_session = OMSession(FakeOMC({
        "loadString(": [True, True],
        "getErrorString()": [
            '"Error: Class Foo not found in scope M.\n"'] * 2,
    }))
    report = run_ladder(
        session_factory=lambda: fail_session,
        llm_factory=lambda: ScriptedLLM([GOOD, GOOD]),
        task_ids=["msd_equations"],
        out_dir=str(tmp_path / "t"),
        max_attempts=2,
    )
    r = report["results"][0]
    assert r["success"] is False and r["attempts"] == 2
    assert r["final_stage"] == "load"


class TestMSLPreload:
    def test_msl_tasks_flagged(self):
        assert TASKS_BY_ID["msl_oscillator"].requires_msl
        assert TASKS_BY_ID["dc_motor"].requires_msl
        assert not TASKS_BY_ID["msd_equations"].requires_msl
        assert not TASKS_BY_ID["tune_damping"].requires_msl

    def test_runner_loads_msl_when_required(self, tmp_path):
        fake = FakeOMC({
            "loadModel(Modelica)": [True],
            "loadString(": [True],
            "checkModel(": [CHECK_OK],
            "simulate(": [dict(SIM_OK)],
            "getErrorString()": ['""'] * 4,
        })
        run_ladder(
            session_factory=lambda: OMSession(fake),
            llm_factory=lambda: ScriptedLLM([GOOD]),
            task_ids=["msl_oscillator"],
            out_dir=str(tmp_path / "t"), max_attempts=1)
        assert any(s.startswith("loadModel(Modelica)") for s in fake.sent)
        # MSL must be loaded BEFORE the model string
        assert fake.sent.index("loadModel(Modelica)") < \
            next(i for i, s in enumerate(fake.sent) if s.startswith("loadString("))

    def test_runner_skips_msl_when_not_required(self, tmp_path):
        fake = FakeOMC({
            "loadString(": [True],
            "checkModel(": [CHECK_OK],
            "simulate(": [dict(SIM_OK)],
            "getErrorString()": ['""'] * 3,
        })
        run_ladder(
            session_factory=lambda: OMSession(fake),
            llm_factory=lambda: ScriptedLLM([GOOD]),
            task_ids=["msd_equations"],
            out_dir=str(tmp_path / "t"), max_attempts=1)
        assert not any(s.startswith("loadModel(") for s in fake.sent)

    def test_msl_load_failure_is_environment_error(self, tmp_path):
        fake = FakeOMC({
            "loadModel(Modelica)": [False],
            "getErrorString()": ['"Error: Failed to load package Modelica.\n"'],
        })
        report = run_ladder(
            session_factory=lambda: OMSession(fake),
            llm_factory=lambda: ScriptedLLM([GOOD]),
            task_ids=["msl_oscillator"],
            out_dir=str(tmp_path / "t"), max_attempts=1)
        r = report["results"][0]
        assert r["success"] is False
        assert r["final_stage"] == "environment"
        assert r["attempts"] == 0


def test_env_failure_verbose_does_not_crash(tmp_path, capsys):
    fake = FakeOMC({
        "loadModel(Modelica)": [False],
        "getErrorString()": ['"Error: Failed to load package Modelica.\n"'],
    })
    report = run_ladder(
        session_factory=lambda: OMSession(fake),
        llm_factory=lambda: ScriptedLLM([GOOD]),
        task_ids=["msl_oscillator"],
        out_dir=str(tmp_path / "t"), max_attempts=1, verbose=True)
    out = capsys.readouterr().out
    assert "environment" in out.lower()
    assert "Failed to load package Modelica" in out
    assert report["results"][0]["final_stage"] == "environment"
