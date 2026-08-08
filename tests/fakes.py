"""Shared test doubles."""


class FakeOMC:
    """Replays canned responses; records every expression sent."""

    def __init__(self, script: dict[str, list]):
        # script maps a prefix of the expression -> list of successive replies
        self.script = {k: list(v) for k, v in script.items()}
        self.sent: list[str] = []

    def sendExpression(self, expr: str):
        self.sent.append(expr)
        for prefix, replies in self.script.items():
            if expr.startswith(prefix):
                if not replies:
                    raise AssertionError(f"no reply left for {expr!r}")
                return replies.pop(0)
        raise AssertionError(f"unexpected expression: {expr!r}")
