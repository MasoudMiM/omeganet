"""Tests for the omc diagnostic parser, using recorded omc output formats."""

from omagent.errors import (
    Diagnostic, Kind, Severity,
    classify, parse_error_string, parse_simulation_messages, summarize_for_llm,
)

LOOKUP_ERR = (
    '"[/home/user/Circuit.mo:12:3-12:34:writable] Error: '
    "Class Modelica.Electrical.Analog.Basic.Resistr not found in scope Circuit "
    '(looking for a class or component).\n"'
)

SYNTAX_ERR = (
    "[<interactive>:4:1-4:1:writable] Error: Missing token: SEMICOLON\n"
    "[<interactive>:5:3-5:9:writable] Error: Parse error: unexpected token 'model'\n"
)

BALANCE_ERR = (
    "Error: An independent subset of the model has imbalanced number of "
    "equations (3) and variables (4).\n"
    "variable Real x;\nvariable Real y;\n"
)

MIXED = (
    "[/tmp/M.mo:2:3-2:20:writable] Warning: Connector p is not balanced: "
    "The number of potential variables (1) is not equal to the number of flow variables (0).\n"
    "[/tmp/M.mo:8:5-8:30:writable] Error: Type mismatch in binding v = \"abc\", "
    "expected subtype of Real, got type String.\n"
    "Notification: From here on, errors are ignored.\n"
)

SIM_MESSAGES_FAIL = (
    "Simulation execution failed for model: Tank\n"
    "LOG_STDOUT        | error   | division by zero at time 1.20034\n"
    "LOG_ASSERT        | error   | The following assertion has been violated at time 1.2\n"
    "LOG_STDOUT        | info    | The simulation finished with errors.\n"
)

SIM_MESSAGES_OK = "LOG_SUCCESS | info | The simulation finished successfully.\n"


class TestParseErrorString:
    def test_empty_and_quote_only(self):
        assert parse_error_string("") == []
        assert parse_error_string('""') == []
        assert parse_error_string("   ") == []

    def test_located_lookup_error(self):
        diags = parse_error_string(LOOKUP_ERR)
        assert len(diags) == 1
        d = diags[0]
        assert d.severity == Severity.ERROR
        assert d.kind == Kind.LOOKUP
        assert d.file == "/home/user/Circuit.mo"
        assert (d.line_start, d.col_start, d.line_end, d.col_end) == (12, 3, 12, 34)
        assert "Resistr not found" in d.message

    def test_multiple_syntax_errors_split(self):
        diags = parse_error_string(SYNTAX_ERR)
        assert len(diags) == 2
        assert all(d.kind == Kind.SYNTAX for d in diags)
        assert all(d.severity == Severity.ERROR for d in diags)

    def test_bare_balance_error_with_continuation_lines(self):
        diags = parse_error_string(BALANCE_ERR)
        assert len(diags) == 1
        d = diags[0]
        assert d.kind == Kind.BALANCE
        assert d.file is None
        assert "variable Real y;" in d.message  # continuation kept with record

    def test_mixed_severities(self):
        diags = parse_error_string(MIXED)
        assert [d.severity for d in diags] == [
            Severity.WARNING, Severity.ERROR, Severity.NOTIFICATION]
        assert diags[0].kind == Kind.CONNECT
        assert diags[1].kind == Kind.TYPE

    def test_windows_style_path(self):
        raw = ('[C:\\Users\\me\\Model.mo:3:1-3:10:writable] '
               "Error: Variable q not found in scope Model.\n")
        d = parse_error_string(raw)[0]
        assert d.file == "C:\\Users\\me\\Model.mo"
        assert d.line_start == 3
        assert d.kind == Kind.LOOKUP


class TestClassify:
    def test_priority_syntax_over_lookup(self):
        assert classify("Parse error: Class X not found") == Kind.SYNTAX

    def test_initialization(self):
        msg = "The initial conditions are not fully specified."
        assert classify(msg) == Kind.INITIALIZATION

    def test_runtime(self):
        assert classify("Simulation execution failed for model: X") == Kind.RUNTIME

    def test_unknown_is_other(self):
        assert classify("Something completely different happened") == Kind.OTHER


class TestParseSimulationMessages:
    def test_failure_log(self):
        diags = parse_simulation_messages(SIM_MESSAGES_FAIL)
        errs = [d for d in diags if d.severity == Severity.ERROR]
        assert len(errs) == 3  # failed-line + division by zero + assert
        assert any(d.kind == Kind.RUNTIME for d in errs)
        assert any("division by zero" in d.message for d in errs)

    def test_success_log_has_no_errors(self):
        diags = parse_simulation_messages(SIM_MESSAGES_OK)
        assert all(d.severity != Severity.ERROR for d in diags)

    def test_empty(self):
        assert parse_simulation_messages("") == []


class TestSummarize:
    def test_dedup_and_order_errors_first(self):
        diags = parse_error_string(MIXED) + parse_error_string(MIXED)
        text = summarize_for_llm(diags)
        # Deduplicated: each brief appears once; errors listed before warnings.
        assert text.count("Type mismatch") == 1
        assert text.index("Type mismatch") < text.index("Connector p")
        assert "Notification" not in text

    def test_no_issues(self):
        assert summarize_for_llm([]) == "No errors or warnings."

    def test_limit(self):
        diags = [Diagnostic(Severity.ERROR, f"e{i}") for i in range(30)]
        text = summarize_for_llm(diags, limit=5)
        assert "more suppressed" in text


class TestParseOMPythonException:
    """Newer OMPython raises OMCSessionException instead of returning; its
    message embeds omc's log as: [kind:level:id] message"""

    def test_syntax_error_format(self):
        from omagent.errors import parse_ompython_exception
        msg = ("[OMC log for 'sendExpression(loadString(\"model T ... end T;\"), True)']: "
               "[syntax:error:2] Missing token: SEMICOLON")
        diags = parse_ompython_exception(msg)
        assert len(diags) == 1
        assert diags[0].severity == Severity.ERROR
        assert diags[0].kind == Kind.SYNTAX
        assert "SEMICOLON" in diags[0].message

    def test_translation_error_classified_by_message(self):
        from omagent.errors import parse_ompython_exception
        msg = ("[OMC log for 'sendExpression(x, True)']: "
               "[translation:error:1] Class Modelica.Foo not found in scope M.")
        diags = parse_ompython_exception(msg)
        assert diags[0].kind == Kind.LOOKUP

    def test_simulation_kind_maps_to_runtime(self):
        from omagent.errors import parse_ompython_exception
        msg = ("[OMC log for 'sendExpression(simulate(M), True)']: "
               "[simulation:error:7] The simulation stopped unexpectedly."
               )
        diags = parse_ompython_exception(msg)
        assert diags[0].kind == Kind.RUNTIME

    def test_unstructured_exception_message_still_captured(self):
        from omagent.errors import parse_ompython_exception
        diags = parse_ompython_exception("No connection with OMC (timeout=10.0).")
        assert len(diags) == 1
        assert diags[0].severity == Severity.ERROR
        assert "No connection" in diags[0].message
