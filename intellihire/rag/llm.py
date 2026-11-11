"""LLM via a secure client API: Anthropic (Claude) or any OpenAI-compatible
endpoint (OpenAI, Groq, Together, a local Ollama / LM Studio server, ...).

Keys are read from the environment only and never logged or sent to the browser.
Both providers implement the same small interface, including a bounded
tool-calling loop used by the agent.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from collections.abc import Callable

from intellihire.config import get_settings

log = logging.getLogger(__name__)


class LLMError(Exception):
    pass


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict  # JSON schema


ToolExecutor = Callable[[str, dict], str]


class LLMClient:
    provider = "none"
    model = ""

    def complete(self, system: str, user: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def run_tools(self, system: str, messages: list[dict], tools: list[ToolSpec], execute: ToolExecutor,
                  max_steps: int) -> tuple[str, list[dict]]:  # pragma: no cover - interface
        raise NotImplementedError


def _safe_args(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw or "{}")
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        return {}


class AnthropicClient(LLMClient):
    provider = "anthropic"

    def __init__(self, api_key: str, model: str, max_tokens: int):
        import anthropic

        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=api_key, max_retries=2, timeout=90)
        self.model = model
        self._max_tokens = max_tokens

    def _create(self, **kwargs):
        try:
            return self._client.messages.create(model=self.model, max_tokens=self._max_tokens, **kwargs)
        except self._anthropic.APIError as exc:
            raise LLMError(f"Anthropic API error: {exc}") from exc

    @staticmethod
    def _text(resp) -> str:
        return "\n".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()

    def complete(self, system: str, user: str) -> str:
        resp = self._create(system=system, messages=[{"role": "user", "content": user}])
        return self._text(resp)

    def run_tools(self, system, messages, tools, execute, max_steps):
        tool_defs = [{"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools]
        msgs = [{"role": m["role"], "content": m["content"]} for m in messages]
        trace: list[dict] = []
        for step in range(max_steps + 1):
            final_step = step == max_steps
            resp = self._create(
                system=system, messages=msgs, tools=tool_defs,
                tool_choice={"type": "none"} if final_step else {"type": "auto"},
            )
            tool_uses = [b for b in resp.content if getattr(b, "type", "") == "tool_use"]
            if resp.stop_reason != "tool_use" or not tool_uses or final_step:
                return self._text(resp) or "I could not produce an answer.", trace
            assistant_blocks = []
            for b in resp.content:
                if b.type == "text" and b.text:
                    assistant_blocks.append({"type": "text", "text": b.text})
                elif b.type == "tool_use":
                    assistant_blocks.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
            msgs.append({"role": "assistant", "content": assistant_blocks})
            results = []
            for b in tool_uses:
                args = _safe_args(b.input)
                output = execute(b.name, args)
                trace.append({"tool": b.name, "args": args, "result": output[:4000]})
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": output})
            msgs.append({"role": "user", "content": results})
        return "I could not complete the request within the step limit.", trace  # pragma: no cover


class OpenAICompatibleClient(LLMClient):
    provider = "openai"

    def __init__(self, api_key: str, model: str, max_tokens: int, base_url: str = ""):
        import openai

        self._openai = openai
        self._client = openai.OpenAI(api_key=api_key or "not-needed", base_url=base_url or None,
                                     max_retries=2, timeout=90)
        self.model = model
        self._max_tokens = max_tokens

    def _create(self, **kwargs):
        try:
            return self._client.chat.completions.create(
                model=self.model, max_tokens=self._max_tokens, temperature=0.2, **kwargs
            )
        except self._openai.OpenAIError as exc:
            raise LLMError(f"OpenAI-compatible API error: {exc}") from exc

    def complete(self, system: str, user: str) -> str:
        resp = self._create(messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        return (resp.choices[0].message.content or "").strip()

    def run_tools(self, system, messages, tools, execute, max_steps):
        tool_defs = [
            {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
            for t in tools
        ]
        msgs = [{"role": "system", "content": system}] + [
            {"role": m["role"], "content": m["content"]} for m in messages
        ]
        trace: list[dict] = []
        for step in range(max_steps + 1):
            final_step = step == max_steps
            resp = self._create(messages=msgs, tools=tool_defs, tool_choice="none" if final_step else "auto")
            msg = resp.choices[0].message
            calls = msg.tool_calls or []
            if not calls or final_step:
                return (msg.content or "").strip() or "I could not produce an answer.", trace
            msgs.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
                    for c in calls
                ],
            })
            for c in calls:
                args = _safe_args(c.function.arguments)
                output = execute(c.function.name, args)
                trace.append({"tool": c.function.name, "args": args, "result": output[:4000]})
                msgs.append({"role": "tool", "tool_call_id": c.id, "content": output})
        return "I could not complete the request within the step limit.", trace  # pragma: no cover


@lru_cache(maxsize=1)
def get_llm() -> LLMClient | None:
    """Return the configured client, or None to run the offline (no-LLM) mode."""
    s = get_settings()
    if not s.llm_enabled:
        return None
    try:
        if s.llm_provider == "anthropic":
            return AnthropicClient(s.anthropic_api_key, s.llm_model, s.llm_max_tokens)
        if s.llm_provider == "openai":
            return OpenAICompatibleClient(s.openai_api_key, s.llm_model, s.llm_max_tokens, s.openai_base_url)
    except ImportError as exc:
        log.warning("LLM SDK not installed: %s", exc)
    return None


def llm_status() -> str:
    client = get_llm()
    if client is None:
        return "offline mode (no LLM configured)"
    return f"{client.provider}: {client.model}"
