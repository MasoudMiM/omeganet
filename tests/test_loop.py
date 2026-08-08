"""Agent-loop tests: scripted fake LLM + fake omc backend, no network, no omc."""

import pytest

from omagent.loop import AgentLoop, extract_code, extract_model_name
from omagent.session import OMSession

from tests.fakes import FakeOMC

GOOD = "model T Real x(start=0, fixed=true); equation der(x)=1; end T;"
BAD = "model T Real x(start=0, fixed=true) equation der(x)=1; end T;"  # missing ;

SYNTAX_ERR = '"[<interactive>:1:36-1:44:writable] Error: Missing token: SEMICOLON\n"'
CHECK_OK = '"Check of T completed successfully.\nClass T has 1 equation(s) and 1 variable(s).\n"'
SIM_OK = {"resultFile": "/tmp/T_res.mat",
          "messages": "LOG_SUCCESS | info | The simulation finished successfully.\n"}
SIM_FAIL = {"resultFile": "",
            "messages": "Simulation execution failed for model: T\n"
                        "LOG_STDOUT | error | division by zero at time 0.5\n"}


class ScriptedLLM:
    """Returns queued outputs; records every (task, prev_code, error_summary) call."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def propose(self, task, previous_code, error_summary):
        self.calls.append((task, previous_code, error_summary))
        return self.outputs.pop(0)


def test_success_first_try():
    fake = FakeOMC({
        "loadString(": [True],
        "checkModel(": [CHECK_OK],
        "simulate(": [SIM_OK],
        "getErrorString()": ['""', '""', '""'],
    })
    llm = ScriptedLLM([GOOD])
    result = AgentLoop(OMSession(fake), llm).run("integrator model")
    assert result.success
    assert result.model_name == "T"
    assert len(result.attempts) == 1
    assert result.attempts[0].stage == "ok"
    # first call is a fresh generation: no previous code / errors
    assert llm.calls[0][1] is None and llm.calls[0][2] is None


def test_retry_after_syntax_error_then_succeed():
    fake = FakeOMC({
        "loadString(": [True, True],
        "checkModel(": [CHECK_OK],
        "simulate(": [SIM_OK],
        "getErrorString()": [SYNTAX_ERR, '""', '""', '""'],
    })
    llm = ScriptedLLM([BAD, GOOD])
    result = AgentLoop(OMSession(fake), llm).run("integrator model")
    assert result.success
    assert [a.stage for a in result.attempts] == ["load", "ok"]
    # the fix call received the failing code and a digest mentioning the error
    task, prev, err = llm.calls[1]
    assert prev == BAD
    assert "SEMICOLON" in err and "load" in err


def test_gives_up_after_max_attempts():
    fake = FakeOMC({
        "loadString(": [True, True],
        "getErrorString()": [SYNTAX_ERR, SYNTAX_ERR],
    })
    llm = ScriptedLLM([BAD, BAD])
    result = AgentLoop(OMSession(fake), llm, max_attempts=2).run("integrator")
    assert not result.success
    assert len(result.attempts) == 2
    assert all(a.stage == "load" for a in result.attempts)
    assert result.final_code == BAD


def test_runtime_failure_feeds_back_then_succeeds():
    fake = FakeOMC({
        "loadString(": [True, True],
        "checkModel(": [CHECK_OK, CHECK_OK],
        "simulate(": [SIM_FAIL, SIM_OK],
        "getErrorString()": ['""'] * 6,
    })
    llm = ScriptedLLM([GOOD, GOOD])
    result = AgentLoop(OMSession(fake), llm).run("integrator")
    assert result.success
    assert [a.stage for a in result.attempts] == ["simulate", "ok"]
    assert "division by zero" in llm.calls[1][2]


def test_verifier_rejection_feeds_back():
    fake = FakeOMC({
        "loadString(": [True, True],
        "checkModel(": [CHECK_OK, CHECK_OK],
        "simulate(": [SIM_OK, SIM_OK],
        "getErrorString()": ['""'] * 6,
    })
    complaints = ["final value of x should be 1.0, got 0.0", None]

    def verifier(sim):
        return complaints.pop(0)

    llm = ScriptedLLM([GOOD, GOOD])
    result = AgentLoop(OMSession(fake), llm, verifier=verifier).run("integrator")
    assert result.success
    assert [a.stage for a in result.attempts] == ["verify", "ok"]
    assert "final value of x" in llm.calls[1][2]


def test_missing_model_name_is_reported_to_llm():
    fake = FakeOMC({
        "loadString(": [True],
        "checkModel(": [CHECK_OK],
        "simulate(": [SIM_OK],
        "getErrorString()": ['""'] * 3,
    })
    llm = ScriptedLLM(["Real x = 1;", GOOD])  # first output has no model keyword
    result = AgentLoop(OMSession(fake), llm).run("integrator")
    assert result.success
    assert "model name" in llm.calls[1][2].lower()


def test_explicit_model_name_overrides_extraction():
    fake = FakeOMC({
        "loadString(": [True],
        "checkModel(": [CHECK_OK],
        "simulate(": [SIM_OK],
        "getErrorString()": ['""'] * 3,
    })
    result = AgentLoop(OMSession(fake), ScriptedLLM([GOOD])).run("x", model_name="T")
    assert result.success
    assert any(s.startswith("checkModel(T") for s in fake.sent)


def test_max_attempts_validation():
    with pytest.raises(ValueError):
        AgentLoop(OMSession(FakeOMC({})), ScriptedLLM([]), max_attempts=0)


class TestExtractHelpers:
    def test_fenced_block(self):
        assert extract_code("Here you go:\n```modelica\nmodel A end A;\n```") == \
            "model A end A;"

    def test_bare_code_passthrough(self):
        assert extract_code("  model A end A;  ") == "model A end A;"

    def test_model_name(self):
        assert extract_model_name("// comment\nmodel Pump2_a\n end Pump2_a;") == "Pump2_a"
        assert extract_model_name("block B end B;") == "B"
        assert extract_model_name("Real x;") is None


@pytest.mark.integration
def test_loop_end_to_end_with_real_omc():
    """Full loop against live omc: first proposal is broken, second is fixed.

    Uses a scripted LLM so no API key is needed — validates that real omc
    diagnostics flow through feedback and the loop recovers.
    """
    pytest.importorskip("OMPython")
    import shutil
    if not shutil.which("omc"):
        pytest.skip("omc binary not on PATH")

    llm = ScriptedLLM([BAD, GOOD])
    result = AgentLoop(OMSession(), llm, max_attempts=3).run(
        "unit integrator", model_name="T")
    assert result.success, [a.stage for a in result.attempts]
    assert result.attempts[0].failed          # real omc rejected the broken code
    assert llm.calls[1][2] is not None        # feedback reached the second call
    assert result.sim_result.value["resultFile"]


def test_loop_recovers_when_backend_raises_on_bad_code():
    """Simulates newer OMPython raising OMCSessionException on load errors."""
    from tests.test_session import RaisingOMC, OMPY_STYLE_EXC
    fake = RaisingOMC({
        "loadString(": [OMPY_STYLE_EXC, True],
        "checkModel(": [CHECK_OK],
        "simulate(": [SIM_OK],
        "getErrorString()": ['""'] * 4,
    })
    llm = ScriptedLLM([BAD, GOOD])
    result = AgentLoop(OMSession(fake), llm).run("integrator", model_name="T")
    assert result.success
    assert [a.stage for a in result.attempts] == ["load", "ok"]
    assert "SEMICOLON" in llm.calls[1][2]


LOOKUP_ERR_EMF = ('"[/x.mo:5:3-5:60:writable] Error: Class '
                  "Modelica.Electrical.Analog.Basic.EMF not found in scope "
                  'DCMotorDrive.\n"')


def test_lookup_feedback_includes_omc_suggestions():
    """On a lookup failure, feedback must list close matches from omc's own
    getClassNames of the parent package — the renamed-class escape hatch."""
    fake = FakeOMC({
        "loadString(": [True, True],
        "getClassNames(Modelica.Electrical.Analog.Basic": [
            ("Ground", "Resistor", "Inductor", "RotationalEMF",
             "TranslationalEMF", "Capacitor")],
        "checkModel(": [CHECK_OK],
        "simulate(": [SIM_OK],
        # one extra reply: class_names() drains lookup noise after the query
        "getErrorString()": [LOOKUP_ERR_EMF, '""', '""', '""', '""'],
    })
    llm = ScriptedLLM([GOOD, GOOD])
    result = AgentLoop(OMSession(fake), llm).run("dc motor", model_name="T")
    assert result.success
    fb = llm.calls[1][2]
    assert "RotationalEMF" in fb
    assert "Modelica.Electrical.Analog.Basic" in fb


def test_lookup_feedback_survives_missing_parent_package():
    fake = FakeOMC({
        "loadString(": [True, True],
        "getClassNames(": [()],       # omc knows nothing about the package
        "checkModel(": [CHECK_OK],
        "simulate(": [SIM_OK],
        "getErrorString()": ['"Error: Class Foo.Bar not found in scope M.\n"',
                             '""', '""', '""', '""'],
    })
    llm = ScriptedLLM([GOOD, GOOD])
    result = AgentLoop(OMSession(fake), llm).run("x", model_name="T")
    assert result.success                     # no crash, plain feedback
    assert "not found" in llm.calls[1][2]


def test_dotless_class_name_no_suggestion_lookup():
    fake = FakeOMC({
        "loadString(": [True, True],
        "checkModel(": [CHECK_OK],
        "simulate(": [SIM_OK],
        "getErrorString()": ['"Error: Class Foo not found in scope M.\n"',
                             '""', '""', '""'],
    })
    llm = ScriptedLLM([GOOD, GOOD])
    result = AgentLoop(OMSession(fake), llm).run("x", model_name="T")
    assert result.success
    assert not any(s.startswith("getClassNames(") for s in fake.sent)
