"""Tests for OMSession using a fake backend that replays recorded omc replies."""

import pytest

from omagent.session import OMSession, OpResult
from omagent.errors import Kind, Severity
from tests.fakes import FakeOMC




def test_load_string_success():
    fake = FakeOMC({
        "loadString(": [True],
        "getErrorString()": ['""'],
    })
    s = OMSession(backend=fake)
    res = s.load_string("model M Real x = 1; end M;")
    assert res.success
    assert res.diagnostics == []
    # the code must be escaped and quoted
    assert fake.sent[0].startswith('loadString("model M')


def test_load_string_true_but_errors_means_failure():
    fake = FakeOMC({
        "loadString(": [True],
        "getErrorString()": [
            '"[<interactive>:1:30-1:30:writable] Error: Missing token: SEMICOLON\n"'
        ],
    })
    res = OMSession(backend=fake).load_string("model M Real x = 1 end M;")
    assert not res.success
    assert res.errors[0].kind == Kind.SYNTAX


def test_check_model_success():
    fake = FakeOMC({
        "checkModel(": ['"Check of M completed successfully.\nClass M has 2 equation(s) and 2 variable(s).\n"'],
        "getErrorString()": ['""'],
    })
    res = OMSession(backend=fake).check_model("M")
    assert res.success
    assert "completed successfully" in res.value


def test_check_model_failure_parses_return_text():
    fake = FakeOMC({
        "checkModel(": ['"Error: An independent subset of the model has imbalanced number of equations (1) and variables (2).\n"'],
        "getErrorString()": ['""'],
    })
    res = OMSession(backend=fake).check_model("M")
    assert not res.success
    assert any(d.kind == Kind.BALANCE for d in res.diagnostics)


def test_simulate_success():
    fake = FakeOMC({
        "simulate(": [{
            "resultFile": "/tmp/M_res.mat",
            "simulationOptions": "startTime = 0.0, stopTime = 1.0",
            "messages": "LOG_SUCCESS | info | The simulation finished successfully.\n",
            "timeSimulation": 0.01,
        }],
        "getErrorString()": ['""'],
    })
    s = OMSession(backend=fake)
    res = s.simulate("M", stopTime=1.0)
    assert res.success
    assert res.value["resultFile"].endswith("_res.mat")
    assert "stopTime=1.0" in fake.sent[0]


def test_simulate_runtime_failure():
    fake = FakeOMC({
        "simulate(": [{
            "resultFile": "",
            "messages": (
                "Simulation execution failed for model: Tank\n"
                "LOG_STDOUT | error | division by zero at time 1.2\n"),
        }],
        "getErrorString()": ['""'],
    })
    res = OMSession(backend=fake).simulate("Tank")
    assert not res.success
    kinds = {d.kind for d in res.errors}
    assert Kind.RUNTIME in kinds
    assert any("division by zero" in d.message for d in res.errors)


def test_simulate_string_option_quoted():
    fake = FakeOMC({
        "simulate(": [{"resultFile": "/tmp/r.mat", "messages": ""}],
        "getErrorString()": ['""'],
    })
    OMSession(backend=fake).simulate("M", method="dassl")
    assert 'method="dassl"' in fake.sent[0]


def test_opresult_errors_property():
    from omagent.errors import Diagnostic
    r = OpResult("x", False, diagnostics=[
        Diagnostic(Severity.WARNING, "w"), Diagnostic(Severity.ERROR, "e")])
    assert [d.message for d in r.errors] == ["e"]


@pytest.mark.integration
def test_against_real_omc():
    """Runs only when a real omc + OMPython are available."""
    pytest.importorskip("OMPython")
    import shutil
    if not shutil.which("omc"):
        pytest.skip("omc binary not on PATH")
    s = OMSession()
    assert s.get_version()
    res = s.load_string("model T Real x(start=0, fixed=true); equation der(x)=1; end T;")
    assert res.success
    sim = s.simulate("T", stopTime=1.0)
    assert sim.success


class RaisingOMC:
    """Mimics newer OMPython: raises on expressions whose scripted reply is an
    Exception instance; otherwise behaves like FakeOMC."""

    def __init__(self, script):
        self.script = {k: list(v) for k, v in script.items()}
        self.sent = []

    def sendExpression(self, expr):
        self.sent.append(expr)
        for prefix, replies in self.script.items():
            if expr.startswith(prefix):
                reply = replies.pop(0)
                if isinstance(reply, Exception):
                    raise reply
                return reply
        raise AssertionError(f"unexpected expression: {expr!r}")


OMPY_STYLE_EXC = RuntimeError(
    "[OMC log for 'sendExpression(loadString(\"model T ...\"), True)']: "
    "[syntax:error:2] Missing token: SEMICOLON")


def test_load_string_backend_exception_becomes_failed_result():
    fake = RaisingOMC({
        "loadString(": [OMPY_STYLE_EXC],
        "getErrorString()": ['""'],
    })
    res = OMSession(backend=fake).load_string("model T ... end T;")
    assert not res.success
    assert res.errors, "exception must surface as error diagnostics"
    assert res.errors[0].kind == Kind.SYNTAX
    assert "SEMICOLON" in res.errors[0].message


def test_simulate_backend_exception_becomes_failed_result():
    fake = RaisingOMC({
        "simulate(": [RuntimeError(
            "[OMC log for 'sendExpression(simulate(T), True)']: "
            "[simulation:error:7] Simulation stopped.")],
        "getErrorString()": ['""'],
    })
    res = OMSession(backend=fake).simulate("T")
    assert not res.success
    assert any(d.kind == Kind.RUNTIME for d in res.errors)


def test_drain_failure_does_not_mask_op_result():
    """Even if getErrorString() itself raises, ops must return, not throw."""
    fake = RaisingOMC({
        "loadString(": [True],
        "getErrorString()": [RuntimeError("connection lost")],
    })
    res = OMSession(backend=fake).load_string("model M end M;")
    assert not res.success
    assert any("connection lost" in d.message for d in res.diagnostics)


def test_class_names():
    fake = FakeOMC({
        "getClassNames(Modelica.Electrical.Analog.Basic": [
            ("Ground", "Resistor", "RotationalEMF", "TranslationalEMF")],
    })
    names = OMSession(backend=fake).class_names("Modelica.Electrical.Analog.Basic")
    assert "RotationalEMF" in names


def test_class_names_failure_returns_empty():
    fake = RaisingOMC({
        "getClassNames(": [RuntimeError("boom")],
    })
    assert OMSession(backend=fake).class_names("No.Such.Pkg") == []
