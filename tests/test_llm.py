"""Provider adapters tested against fake SDK clients (no network, no API key).

The fakes enforce the protocol rules that real APIs reject requests for:
Anthropic - tool_result blocks must answer the previous assistant tool_use ids;
OpenAI    - every assistant tool_call id must be answered by a role=tool message."""
from __future__ import annotations

import json
from types import SimpleNamespace as NS

from intellihire.rag.llm import AnthropicClient, OpenAICompatibleClient, ToolSpec

TOOLS = [ToolSpec("lookup", "Look something up", {"type": "object", "properties": {"q": {"type": "string"}}})]


class FakeAnthropicMessages:
    def __init__(self):
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        msgs = kw["messages"]
        if len(self.calls) == 1:
            assert msgs[-1]["role"] == "user" and kw["tool_choice"] == {"type": "auto"}
            return NS(stop_reason="tool_use", content=[
                NS(type="text", text="Let me check."),
                NS(type="tool_use", id="tu_1", name="lookup", input={"q": "python"}),
            ])
        assistant, results = msgs[-2], msgs[-1]
        assert assistant["role"] == "assistant" and assistant["content"][1]["id"] == "tu_1"
        assert results["content"][0]["tool_use_id"] == "tu_1"
        assert json.loads(results["content"][0]["content"]) == {"found": "python"}
        return NS(stop_reason="end_turn", content=[NS(type="text", text="Answer [E1]")])


class FakeOpenAICompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        msgs = kw["messages"]
        assert msgs[0]["role"] == "system"
        if len(self.calls) == 1:
            call = NS(id="call_1", function=NS(name="lookup", arguments='{"q": "sql"}'))
            return NS(choices=[NS(message=NS(content=None, tool_calls=[call]))])
        assert msgs[-2]["tool_calls"][0]["id"] == "call_1"
        assert msgs[-1] == {"role": "tool", "tool_call_id": "call_1", "content": json.dumps({"found": "sql"})}
        return NS(choices=[NS(message=NS(content="Done", tool_calls=None))])


def execute(name, args):
    assert name == "lookup"
    return json.dumps({"found": args["q"]})


def test_anthropic_tool_loop():
    client = AnthropicClient.__new__(AnthropicClient)
    client._client = NS(messages=FakeAnthropicMessages())
    client._anthropic = NS(APIError=RuntimeError)
    client.model, client._max_tokens = "test-model", 100
    text, trace = client.run_tools("sys", [{"role": "user", "content": "hi"}], TOOLS, execute, max_steps=4)
    assert text == "Answer [E1]"
    assert trace == [{"tool": "lookup", "args": {"q": "python"}, "result": '{"found": "python"}'}]
    first = client._client.messages.calls[0]
    assert first["tools"][0]["input_schema"]["type"] == "object" and first["system"] == "sys"


def test_openai_tool_loop():
    client = OpenAICompatibleClient.__new__(OpenAICompatibleClient)
    client._client = NS(chat=NS(completions=FakeOpenAICompletions()))
    client._openai = NS(OpenAIError=RuntimeError)
    client.model, client._max_tokens = "test-model", 100
    text, trace = client.run_tools("sys", [{"role": "user", "content": "hi"}], TOOLS, execute, max_steps=4)
    assert text == "Done" and trace[0]["args"] == {"q": "sql"}
    assert client._client.chat.completions.calls[0]["tools"][0]["function"]["name"] == "lookup"


def test_step_limit_forces_final_answer():
    class Looping(FakeAnthropicMessages):
        def create(self, **kw):
            self.calls.append(kw)
            if kw["tool_choice"] == {"type": "none"}:
                return NS(stop_reason="end_turn", content=[NS(type="text", text="final")])
            return NS(stop_reason="tool_use", content=[NS(type="tool_use", id=f"t{len(self.calls)}", name="lookup",
                                                          input={"q": "x"})])

    client = AnthropicClient.__new__(AnthropicClient)
    client._client = NS(messages=Looping())
    client._anthropic = NS(APIError=RuntimeError)
    client.model, client._max_tokens = "m", 10
    text, trace = client.run_tools("sys", [{"role": "user", "content": "hi"}], TOOLS, execute, max_steps=2)
    assert text == "final" and len(trace) == 2


def test_agent_history_alternates():
    from intellihire.agent.agent import RecruiterAgent

    hist = [{"role": "assistant", "content": "a0"}, {"role": "user", "content": "u1"}, {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a1"}, {"role": "user", "content": "pending"}]
    out = RecruiterAgent._history(hist)
    assert [m["role"] for m in out] == ["user", "assistant"]
    assert out[0]["content"] == "u1\n\nu2"
