"""Tests for omagent.llm — adapter logic with a mocked Anthropic client."""

import pytest

from omagent.llm import ClaudeLLM, SYSTEM_PROMPT


class FakeMessage:
    def __init__(self, text):
        self.content = [type("Block", (), {"type": "text", "text": text})()]


class FakeMessages:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeMessage(self.outputs.pop(0))


class FakeClient:
    def __init__(self, outputs):
        self.messages = FakeMessages(outputs)


def make_llm(outputs, **kw):
    return ClaudeLLM(client=FakeClient(outputs), **kw)


def test_fresh_generation_prompt():
    llm = make_llm(["model M end M;"])
    out = llm.propose("a unit integrator", None, None)
    assert out == "model M end M;"
    call = llm.client.messages.calls[0]
    user = call["messages"][0]["content"]
    assert "unit integrator" in user
    assert "previous" not in user.lower()          # no fix framing on first call
    assert call["system"] == SYSTEM_PROMPT
    assert call["messages"][0]["role"] == "user"


def test_fix_prompt_contains_code_and_errors():
    llm = make_llm(["model M end M;"])
    llm.propose("integrator", "model Bad end Bad;", "[error/syntax] Missing token")
    user = llm.client.messages.calls[0]["messages"][0]["content"]
    assert "model Bad end Bad;" in user
    assert "Missing token" in user
    assert "integrator" in user                    # task restated for context


def test_model_and_tokens_configurable():
    llm = make_llm(["x"], model="claude-opus-4-8", max_tokens=1234)
    llm.propose("t", None, None)
    call = llm.client.messages.calls[0]
    assert call["model"] == "claude-opus-4-8"
    assert call["max_tokens"] == 1234


def test_multiple_text_blocks_joined():
    llm = make_llm([""])
    fm = FakeMessage("a")
    fm.content.append(type("B", (), {"type": "text", "text": "b"})())
    llm.client.messages.outputs = []
    llm.client.messages.create = lambda **kw: fm
    assert llm.propose("t", None, None) == "a\nb"


def test_transcript_records_rounds():
    llm = make_llm(["one", "two"])
    llm.propose("t", None, None)
    llm.propose("t", "code", "errs")
    assert len(llm.transcript) == 2
    assert llm.transcript[0]["error_summary"] is None
    assert llm.transcript[1]["error_summary"] == "errs"
    assert llm.transcript[1]["response"] == "two"


def test_missing_anthropic_dependency_message():
    with pytest.raises(ImportError, match=r"omagent\[llm\]"):
        ClaudeLLM(client=None, _force_import_error=True)


class TestOpenAICompatLLM:
    """Adapter for OpenAI-compatible endpoints (Ollama, LM Studio, vLLM...)."""

    def _make(self, replies, **kw):
        from omagent.llm import OpenAICompatLLM
        calls = []

        def transport(url, payload, headers):
            calls.append({"url": url, "payload": payload, "headers": headers})
            return {"choices": [{"message": {"content": replies.pop(0)}}]}

        llm = OpenAICompatLLM(model="qwen2.5-coder", transport=transport, **kw)
        return llm, calls

    def test_fresh_and_fix_prompts_shared_with_claude(self):
        llm, calls = self._make(["model M end M;", "model M end M;"])
        llm.propose("integrator", None, None)
        llm.propose("integrator", "bad code", "[error] boom")
        from omagent.llm import SYSTEM_PROMPT
        p0, p1 = calls[0]["payload"], calls[1]["payload"]
        assert p0["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
        assert "integrator" in p0["messages"][1]["content"]
        assert "bad code" in p1["messages"][1]["content"]
        assert "[error] boom" in p1["messages"][1]["content"]
        assert p0["model"] == "qwen2.5-coder"

    def test_default_base_url_is_ollama(self):
        llm, calls = self._make(["x"])
        llm.propose("t", None, None)
        assert calls[0]["url"].startswith("http://localhost:11434/v1")
        assert calls[0]["url"].endswith("/chat/completions")

    def test_api_key_header_only_when_given(self):
        llm, calls = self._make(["x"])
        llm.propose("t", None, None)
        assert "Authorization" not in calls[0]["headers"]
        llm2, calls2 = self._make(["x"], api_key="sk-local")
        llm2.propose("t", None, None)
        assert calls2[0]["headers"]["Authorization"] == "Bearer sk-local"

    def test_transcript_recorded(self):
        llm, _ = self._make(["one"])
        llm.propose("t", None, None)
        assert llm.transcript[0]["response"] == "one"

    def test_malformed_response_raises_clearly(self):
        from omagent.llm import OpenAICompatLLM
        llm = OpenAICompatLLM(model="m", transport=lambda u, p, h: {"error": "nope"})
        with pytest.raises(RuntimeError, match="unexpected response"):
            llm.propose("t", None, None)
