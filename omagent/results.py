"""Simulation result reading and quantitative trajectory verification.

Two supported result formats:

- CSV (``simulate(..., outputFormat="csv")``) — zero dependencies.
- OpenModelica's default Dymola-format MATLAB v4 ``.mat`` — requires scipy
  (install extra: ``pip install omagent[results]``). Handles both binTrans
  (OM default, transposed) and binNormal layouts, parameters stored in
  ``data_1``, and negated alias variables (negative dataInfo index).

Verifier factories (``expect_final``, ``expect_value_at``, ``expect_bounds``,
``all_of``) return callables matching the ``omagent.loop.Verifier`` protocol:
they read the result file referenced by a simulate ``OpResult`` and return
``None`` when satisfied or a human/LLM-readable complaint string when not —
so quantitative acceptance criteria feed straight back into the fix loop.
"""

from __future__ import annotations

import bisect
import csv
import os
from typing import Callable, Optional

from .session import OpResult


class SimulationResult:
    """Uniform access to a simulation trajectory, whatever the file format."""

    def __init__(self, times: list[float], series: dict[str, list[float]],
                 params: dict[str, float]):
        self._times = times
        self._series = series          # continuous variables, aligned with times
        self._params = params          # constants (parameters / data_1)

    # -- introspection ----------------------------------------------------
    def names(self) -> list[str]:
        return ["time"] + list(self._params) + list(self._series)

    # -- access -----------------------------------------------------------
    def series(self, var: str) -> tuple[list[float], list[float]]:
        """Return (times, values); parameters expand to a constant series."""
        if var == "time":
            return self._times, self._times
        if var in self._series:
            return self._times, self._series[var]
        if var in self._params:
            return self._times, [self._params[var]] * len(self._times)
        raise KeyError(
            f"variable {var!r} not in result; available: {self.names()}")

    def final(self, var: str) -> float:
        return self.series(var)[1][-1]

    def at(self, var: str, t: float) -> float:
        """Linearly interpolated value of `var` at time `t`."""
        times, vals = self.series(var)
        if not times or t < times[0] or t > times[-1]:
            raise ValueError(
                f"t={t} outside simulated range [{times[0]}, {times[-1]}]")
        i = bisect.bisect_left(times, t)
        if i < len(times) and times[i] == t:
            return vals[i]
        t0, t1 = times[i - 1], times[i]
        w = (t - t0) / (t1 - t0)
        return vals[i - 1] * (1 - w) + vals[i] * w


# ---------------------------------------------------------------- loaders --
def _load_csv(path: str) -> SimulationResult:
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header = [h.strip().strip('"') for h in rows[0]]
    cols: dict[str, list[float]] = {h: [] for h in header}
    for row in rows[1:]:
        if not row:
            continue
        for h, cell in zip(header, row):
            cols[h].append(float(cell))
    times = cols.pop("time")
    return SimulationResult(times, cols, {})


def _load_mat(path: str) -> SimulationResult:
    try:
        from scipy.io import loadmat
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Reading .mat results requires scipy: pip install omagent[results] "
            "(or simulate with outputFormat=\"csv\")") from exc

    raw = loadmat(path, chars_as_strings=False)
    aclass = _char_rows(raw["Aclass"]) + _char_rows(raw["Aclass"].T)
    trans = any("binTrans" in r for r in aclass)

    # dataInfo anchors everything: it must be (nvars, 4). Apply the binTrans
    # rule, then correct by shape — scipy's v4 round-trip and real files do
    # not always agree on orientation, especially for char matrices.
    info = raw["dataInfo"]
    if trans:
        info = info.T
    if info.shape[1] != 4 and info.shape[0] == 4:
        info = info.T
    nvars = info.shape[0]

    names = _pick_names(raw["name"], nvars, trans)

    d1 = raw["data_1"]
    if trans:
        d1 = d1.T
    if d1.shape[0] != 2 and d1.shape[1] == 2:   # expect rows = [start, stop]
        d1 = d1.T

    d2 = raw["data_2"]
    if trans:
        d2 = d2.T
    exp_cols = max((abs(int(r[1])) for r in info if int(r[0]) == 2), default=1)
    if d2.shape[1] != exp_cols and d2.shape[0] == exp_cols:
        d2 = d2.T

    times = [float(v) for v in d2[:, 0]]

    series: dict[str, list[float]] = {}
    params: dict[str, float] = {}
    for name, row in zip(names, info):
        which, col = int(row[0]), int(row[1])
        sign = -1.0 if col < 0 else 1.0
        idx = abs(col) - 1
        if name == "time" or which == 0:
            continue
        if which == 1:                       # constant: data_1, start row
            params[name] = sign * float(d1[0, idx])
        elif which == 2 and idx > 0:         # trajectory: data_2 (col 0 = time)
            series[name] = [sign * float(v) for v in d2[:, idx]]
    return SimulationResult(times, series, params)


def _char_rows(mat) -> list[str]:
    return ["".join(str(c) for c in row).rstrip("\x00 ") for row in mat]


def _pick_names(name_mat, nvars: int, trans: bool) -> list[str]:
    """Choose the orientation of the char `name` matrix.

    The spec says binTrans stores it transposed, but scipy's v4 handling of
    char matrices doesn't always round-trip orientation, and a square matrix
    is inherently ambiguous. Anchor on the variable count from dataInfo and
    require the mandatory 'time' entry; fall back to the spec orientation.
    """
    rows = _char_rows(name_mat)
    cols = _char_rows(name_mat.T)
    spec_first = [cols, rows] if trans else [rows, cols]
    for cand in spec_first:
        if len(cand) == nvars and "time" in cand:
            return cand
    for cand in spec_first:
        if len(cand) == nvars:
            return cand
    return spec_first[0]


def load_result(path: str) -> SimulationResult:
    if path.endswith(".csv"):
        return _load_csv(path)
    return _load_mat(path)


# -------------------------------------------------------------- verifiers --
Verifier = Callable[[OpResult], Optional[str]]


def _open_result(sim: OpResult) -> tuple[Optional[SimulationResult], Optional[str]]:
    path = ""
    if isinstance(sim.value, dict):
        path = str(sim.value.get("resultFile", ""))
    if not path or not os.path.exists(path):
        return None, f"no readable result file (got {path!r}); the simulation may not have produced output"
    try:
        return load_result(path), None
    except Exception as exc:
        return None, f"could not read result file {path!r}: {exc}"


def _check(sim: OpResult, fn: Callable[[SimulationResult], Optional[str]]) -> Optional[str]:
    res, err = _open_result(sim)
    if err:
        return err
    try:
        return fn(res)
    except KeyError as exc:
        return str(exc).strip('"\'')
    except ValueError as exc:
        return str(exc)


def expect_final(var: str, expected: float, *, atol: float = 1e-6,
                 rtol: float = 1e-3) -> Verifier:
    """Final value of `var` must equal `expected` within tolerances."""
    def verify(sim: OpResult) -> Optional[str]:
        def fn(r: SimulationResult):
            got = r.final(var)
            if abs(got - expected) <= atol + rtol * abs(expected):
                return None
            return (f"final value of {var} is {got:.6g}, expected {expected} "
                    f"(atol={atol}, rtol={rtol})")
        return _check(sim, fn)
    return verify


def expect_value_at(var: str, *, t: float, expected: float,
                    atol: float = 1e-6, rtol: float = 1e-3) -> Verifier:
    """Interpolated value of `var` at time `t` must equal `expected`."""
    def verify(sim: OpResult) -> Optional[str]:
        def fn(r: SimulationResult):
            got = r.at(var, t)
            if abs(got - expected) <= atol + rtol * abs(expected):
                return None
            return (f"{var} at t={t} is {got:.6g}, expected {expected} "
                    f"(atol={atol}, rtol={rtol})")
        return _check(sim, fn)
    return verify


def expect_bounds(var: str, *, lo: float = float("-inf"),
                  hi: float = float("inf"), after: float = float("-inf")) -> Verifier:
    """Trajectory of `var` must stay within [lo, hi], optionally only for
    t >= `after` (settling-time criteria)."""
    def verify(sim: OpResult) -> Optional[str]:
        def fn(r: SimulationResult):
            times, vals = r.series(var)
            window = [v for t, v in zip(times, vals) if t >= after]
            if not window:
                return (f"no samples at t >= {after} for {var} "
                        f"(simulated range ends at t={times[-1]})")
            vmin, vmax = min(window), max(window)
            if vmin < lo or vmax > hi:
                where = f" for t >= {after}" if after != float("-inf") else ""
                return (f"{var} exceeds bounds [{lo}, {hi}]{where}: "
                        f"observed range [{vmin:.6g}, {vmax:.6g}]")
            return None
        return _check(sim, fn)
    return verify


def all_of(*verifiers: Verifier) -> Verifier:
    """Combine verifiers; returns the concatenation of all complaints, or None."""
    def verify(sim: OpResult) -> Optional[str]:
        complaints = [c for c in (v(sim) for v in verifiers) if c]
        return "; ".join(complaints) if complaints else None
    return verify


def expect_dips_below(var: str, threshold: float) -> Verifier:
    """`var` must reach a value <= `threshold` at least once (e.g. to require
    an underdamped response that visibly overshoots through zero)."""
    def verify(sim: OpResult) -> Optional[str]:
        def fn(r: SimulationResult):
            _, vals = r.series(var)
            vmin = min(vals)
            if vmin <= threshold:
                return None
            return (f"{var} never dips to {threshold}: observed minimum "
                    f"is {vmin:.6g} (response may be overdamped)")
        return _check(sim, fn)
    return verify
