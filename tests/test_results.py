"""Tests for omagent.results: result-file reading and quantitative verifiers.

The .mat fixtures are synthetic but byte-faithful to what OpenModelica writes:
MATLAB v4 files with Aclass/name/dataInfo/data_1/data_2 matrices, in both
binTrans (transposed, OM default) and binNormal layouts, including negated
alias variables (negative dataInfo index).
"""

import csv
import math

import pytest

from omagent.results import (
    SimulationResult, expect_bounds, expect_final, expect_value_at, load_result,
)
from omagent.session import OpResult

try:
    import numpy as np
    from scipy.io import savemat
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

needs_scipy = pytest.mark.skipif(
    not HAS_SCIPY, reason="scipy required for .mat fixtures/reading "
                          "(pip install omagent[results])")


# ---------------------------------------------------------------- fixtures --
def write_csv(path, times, series: dict):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time"] + list(series))
        for i, t in enumerate(times):
            w.writerow([t] + [series[k][i] for k in series])


def _char_matrix(strings):
    """Column-padded char matrix like Dymola/OM writes (rows = entries)."""
    width = max(len(s) for s in strings)
    return np.array([list(s.ljust(width)) for s in strings])


def write_dymola_mat(path, times, series: dict, params: dict, binTrans=True,
                     alias: dict | None = None):
    """Create an OpenModelica-style result file.

    series: continuous variables (in data_2); params: constants (in data_1);
    alias: name -> (target_name, sign) stored via dataInfo only.
    """
    alias = alias or {}
    names = ["time"] + list(params) + list(series) + list(alias)
    ncont = len(series)

    # dataInfo rows: [which data matrix, signed 1-based column, interp, extrap]
    # data_1 layout: 2 rows [start, stop]; columns: time + params
    info = [[2, 1, 0, -1]]                                   # time -> data_2 col 1
    info += [[1, 2 + i, 0, 0] for i in range(len(params))]
    info += [[2, 2 + i, 0, -1] for i in range(ncont)]
    cont_names = list(series)
    for tgt, sign in alias.values():
        col = 2 + cont_names.index(tgt)
        info.append([2, sign * col, 0, -1])
    dataInfo = np.array(info, dtype=np.int32)

    d1 = np.array([[times[0]] + [params[k] for k in params],
                   [times[-1]] + [params[k] for k in params]], dtype=float)
    d2_cols = [np.asarray(times, dtype=float)] + \
              [np.asarray(series[k], dtype=float) for k in series]
    d2 = np.column_stack(d2_cols)

    # NOTE: scipy's MATLAB-v4 writer re-wraps char arrays row-major on
    # round-trip, scrambling transposed char matrices. Real OM files store
    # them fine; for fixtures we keep `name` row-oriented and rely on the
    # reader's adaptive orientation detection (which real files need anyway
    # for the square-matrix ambiguity). Numeric matrices transpose faithfully.
    name_m = _char_matrix(names)
    aclass = _char_matrix(["Atrajectory", "1.1", "",
                           "binTrans" if binTrans else "binNormal"])
    if binTrans:
        dataInfo, d1, d2 = dataInfo.T, d1.T, d2.T
    savemat(path, {"Aclass": aclass, "name": name_m, "dataInfo": dataInfo,
                   "data_1": d1, "data_2": d2}, format="4")


TIMES = [0.0, 0.5, 1.0, 1.5, 2.0]
X = [0.0, 0.5, 1.0, 1.5, 2.0]          # x = t
V = [1.0, 1.0, 1.0, 1.0, 1.0]          # der(x)


# ------------------------------------------------------------------- tests --
class TestCSV:
    def test_roundtrip(self, tmp_path):
        p = tmp_path / "r.csv"
        write_csv(p, TIMES, {"x": X, "v": V})
        r = load_result(str(p))
        assert set(r.names()) == {"time", "x", "v"}
        t, x = r.series("x")
        assert t == TIMES and x == X
        assert r.final("v") == 1.0

    def test_quoted_headers(self, tmp_path):
        p = tmp_path / "r.csv"
        p.write_text('"time","x"\n0,0\n1,2\n')
        r = load_result(str(p))
        assert r.final("x") == 2.0


@needs_scipy
class TestMat:
    @pytest.mark.parametrize("binTrans", [True, False])
    def test_layouts(self, tmp_path, binTrans):
        p = tmp_path / "r.mat"
        write_dymola_mat(p, TIMES, {"x": X, "v": V}, {"k": 3.0},
                         binTrans=binTrans)
        r = load_result(str(p))
        assert "x" in r.names() and "k" in r.names()
        t, x = r.series("x")
        assert t == pytest.approx(TIMES) and x == pytest.approx(X)

    def test_parameter_expands_to_constant_series(self, tmp_path):
        p = tmp_path / "r.mat"
        write_dymola_mat(p, TIMES, {"x": X}, {"k": 3.0})
        t, k = load_result(str(p)).series("k")
        assert t == pytest.approx(TIMES)
        assert k == pytest.approx([3.0] * len(TIMES))

    def test_negated_alias(self, tmp_path):
        p = tmp_path / "r.mat"
        write_dymola_mat(p, TIMES, {"x": X}, {}, alias={"minus_x": ("x", -1)})
        _, mx = load_result(str(p)).series("minus_x")
        assert mx == pytest.approx([-v for v in X])

    def test_unknown_variable_raises(self, tmp_path):
        p = tmp_path / "r.mat"
        write_dymola_mat(p, TIMES, {"x": X}, {})
        with pytest.raises(KeyError):
            load_result(str(p)).series("nope")


class TestInterpolation:
    def test_at_exact_and_between(self, tmp_path):
        p = tmp_path / "r.csv"
        write_csv(p, TIMES, {"x": X})
        r = load_result(str(p))
        assert r.at("x", 1.0) == pytest.approx(1.0)
        assert r.at("x", 0.75) == pytest.approx(0.75)   # linear interp

    def test_at_out_of_range_raises(self, tmp_path):
        p = tmp_path / "r.csv"
        write_csv(p, TIMES, {"x": X})
        with pytest.raises(ValueError):
            load_result(str(p)).at("x", 99.0)


def _sim_opresult(result_file: str) -> OpResult:
    return OpResult("simulate", True,
                    {"resultFile": result_file, "messages": ""})


class TestVerifiers:
    def test_expect_final_pass_and_fail(self, tmp_path):
        p = tmp_path / "r.csv"
        write_csv(p, TIMES, {"x": X})
        sim = _sim_opresult(str(p))
        assert expect_final("x", 2.0, atol=1e-9)(sim) is None
        complaint = expect_final("x", 5.0, atol=1e-9)(sim)
        assert complaint and "x" in complaint and "5.0" in complaint

    def test_expect_value_at(self, tmp_path):
        p = tmp_path / "r.csv"
        write_csv(p, TIMES, {"x": X})
        sim = _sim_opresult(str(p))
        assert expect_value_at("x", t=0.5, expected=0.5, atol=1e-9)(sim) is None
        assert expect_value_at("x", t=0.5, expected=9.9, atol=1e-9)(sim)

    def test_expect_bounds(self, tmp_path):
        p = tmp_path / "r.csv"
        write_csv(p, TIMES, {"x": X})
        sim = _sim_opresult(str(p))
        assert expect_bounds("x", lo=-0.1, hi=2.1)(sim) is None
        msg = expect_bounds("x", lo=0.0, hi=1.0)(sim)
        assert msg and "exceed" in msg.lower()

    def test_missing_result_file_is_a_complaint_not_a_crash(self):
        sim = _sim_opresult("/nonexistent/file.csv")
        msg = expect_final("x", 1.0)(sim)
        assert msg and "result" in msg.lower()

    def test_all_of_combines(self, tmp_path):
        from omagent.results import all_of
        p = tmp_path / "r.csv"
        write_csv(p, TIMES, {"x": X})
        sim = _sim_opresult(str(p))
        v = all_of(expect_final("x", 2.0), expect_bounds("x", lo=-1, hi=3))
        assert v(sim) is None
        v2 = all_of(expect_final("x", 2.0), expect_final("x", 7.0))
        assert "7.0" in v2(sim)


def _real_session():
    pytest.importorskip("OMPython")
    import shutil
    if not shutil.which("omc"):
        pytest.skip("omc binary not on PATH")
    from omagent.session import OMSession
    s = OMSession()
    assert s.load_string(
        "model T Real x(start=0, fixed=true); equation der(x)=1; end T;").success
    return s


@pytest.mark.integration
def test_verifier_against_real_omc_csv():
    """Real omc, CSV output — verifier leg with zero extra dependencies."""
    sim = _real_session().simulate("T", stopTime=2.0, outputFormat="csv")
    assert sim.success
    assert expect_final("x", 2.0, atol=1e-4)(sim) is None
    assert expect_value_at("x", t=1.0, expected=1.0, atol=1e-3)(sim) is None


@needs_scipy
@pytest.mark.integration
def test_verifier_against_real_omc_mat():
    """Real omc, default .mat output — validates the Dymola-format reader
    against a genuine OpenModelica-written file."""
    sim = _real_session().simulate("T", stopTime=2.0)
    assert sim.success
    assert expect_final("x", 2.0, atol=1e-4)(sim) is None


class TestBoundsAfter:
    """expect_bounds(after=...) — needed for settling-time criteria."""

    def test_bounds_only_after_time(self, tmp_path):
        # x rings to 0.8 early, settles to |x|<0.01 after t=1.0
        times = [0.0, 0.5, 1.0, 1.5, 2.0]
        x = [0.8, -0.4, 0.009, -0.005, 0.002]
        p = tmp_path / "r.csv"
        write_csv(p, times, {"x": x})
        sim = _sim_opresult(str(p))
        # full-window check fails, after-window check passes
        assert expect_bounds("x", lo=-0.01, hi=0.01)(sim)
        assert expect_bounds("x", lo=-0.01, hi=0.01, after=1.0)(sim) is None

    def test_after_window_violation_reports_window(self, tmp_path):
        times = [0.0, 1.0, 2.0]
        p = tmp_path / "r.csv"
        write_csv(p, times, {"x": [0.0, 0.0, 5.0]})
        msg = expect_bounds("x", lo=-1, hi=1, after=0.5)(_sim_opresult(str(p)))
        assert msg and "t >= 0.5" in msg

    def test_after_beyond_range_is_complaint(self, tmp_path):
        times = [0.0, 1.0]
        p = tmp_path / "r.csv"
        write_csv(p, times, {"x": [0.0, 0.0]})
        msg = expect_bounds("x", lo=-1, hi=1, after=5.0)(_sim_opresult(str(p)))
        assert msg and "no samples" in msg


class TestExpectDipsBelow:
    def test_pass_and_fail(self, tmp_path):
        from omagent.results import expect_dips_below
        p = tmp_path / "r.csv"
        write_csv(p, [0, 1, 2], {"x": [0.1, -0.03, 0.01]})
        sim = _sim_opresult(str(p))
        assert expect_dips_below("x", -0.01)(sim) is None
        msg = expect_dips_below("x", -0.05)(sim)
        assert msg and "-0.05" in msg and "minimum" in msg

    def test_overdamped_never_crosses(self, tmp_path):
        from omagent.results import expect_dips_below
        p = tmp_path / "r.csv"
        write_csv(p, [0, 1, 2], {"x": [0.1, 0.05, 0.02]})
        assert expect_dips_below("x", 0.0)(_sim_opresult(str(p)))
