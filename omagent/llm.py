"""LLM adapters implementing the ``omagent.loop.LLM`` protocol.

Currently ships an Anthropic adapter (``ClaudeLLM``). Any other provider can
be used by implementing the one-method ``propose`` protocol; nothing in the
loop depends on this module.

Install: ``pip install omagent[llm]`` and set ``ANTHROPIC_API_KEY``.
"""

from __future__ import annotations

from typing import Any, Optional

SYSTEM_PROMPT = (
    "You are an expert Modelica engineer producing models for OpenModelica.\n"
    "Rules:\n"
    "- Return ONE complete, self-contained Modelica model, and nothing else: "
    "no prose before or after. A ```modelica fence is acceptable.\n"
    "- The model must simulate in plain OpenModelica. Prefer the Modelica "
    "Standard Library; never invent classes.\n"
    "- Give every state a start value with fixed=true unless the task says "
    "otherwise, so initialization is deterministic.\n"
    "- Use SI units and add unit attributes where natural.\n"
    "- When shown a failed attempt and its errors, return the corrected FULL "
    "model, not a diff or fragment."
)

_FRESH_TEMPLATE = """Task: write a Modelica model for the following.

{task}

Return the complete model."""

_FIX_TEMPLATE = """Task: {task}

The model below failed. Fix it and return the complete corrected model.

Failed model:
```modelica
{code}
```

Diagnostics:
{errors}"""


class ClaudeLLM:
    """Anthropic-backed implementation of the LLM protocol.

    Records every round in ``self.transcript`` — these (task, code, errors,
    response) tuples are exactly the material worth keeping as benchmark
    seeds and paper data.
    """

    def __init__(self, model: str = "claude-sonnet-4-6", max_tokens: int = 3000,
                 client: Optional[Any] = None, _force_import_error: bool = False):
        if client is None or _force_import_error:
            try:
                if _force_import_error:
                    raise ImportError
                import anthropic
                client = anthropic.Anthropic()
            except ImportError as exc:
                raise ImportError(
                    "ClaudeLLM requires the anthropic package: "
                    "pip install omagent[llm]") from exc
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.transcript: list[dict] = []

    def propose(self, task: str, previous_code: Optional[str],
                error_summary: Optional[str]) -> str:
        user = _build_user_prompt(task, previous_code, error_summary)

        msg = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user}],
        )
        text = "\n".join(
            block.text for block in msg.content
            if getattr(block, "type", "") == "text")
        self.transcript.append({
            "task": task,
            "previous_code": previous_code,
            "error_summary": error_summary,
            "response": text,
        })
        return text


def _build_user_prompt(task: str, previous_code: Optional[str],
                       error_summary: Optional[str]) -> str:
    if previous_code is None:
        return _FRESH_TEMPLATE.format(task=task)
    return _FIX_TEMPLATE.format(
        task=task, code=previous_code,
        errors=error_summary or "(no diagnostics captured)")


def _default_transport(url: str, payload: dict, headers: dict) -> dict:
    """POST JSON via stdlib; kept injectable for tests."""
    import json
    import urllib.request
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req) as resp:  # pragma: no cover
        return json.loads(resp.read().decode())  # pragma: no cover


class OpenAICompatLLM:
    """Adapter for any OpenAI-compatible chat-completions endpoint.

    Works with local model servers out of the box — Ollama (the default
    base_url), LM Studio, llama.cpp server, vLLM — and with any hosted
    service speaking the same API. No extra dependencies (stdlib HTTP).

    Example (Ollama):
        llm = OpenAICompatLLM(model="qwen2.5-coder:14b")
    Example (LM Studio):
        llm = OpenAICompatLLM(model="loaded-model",
                              base_url="http://localhost:1234/v1")
    """

    def __init__(self, model: str,
                 base_url: str = "http://localhost:11434/v1",
                 api_key: Optional[str] = None,
                 max_tokens: int = 3000,
                 temperature: float = 0.2,
                 transport=None):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._transport = transport or _default_transport
        self.transcript: list[dict] = []

    def propose(self, task: str, previous_code: Optional[str],
                error_summary: Optional[str]) -> str:
        user = _build_user_prompt(task, previous_code, error_summary)
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = self._transport(
            f"{self.base_url}/chat/completions",
            {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
            },
            headers)
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise RuntimeError(
                f"unexpected response from {self.base_url}: {str(data)[:300]}")
        self.transcript.append({
            "task": task, "previous_code": previous_code,
            "error_summary": error_summary, "response": text,
        })
        return text
