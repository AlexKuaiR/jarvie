"""Fake models for worker tests: they call the provided tools the way Claude would.

FakeAgent runs a script(prompt, tools_by_name) that calls tools and returns the final
text, so tests exercise the real tool functions and database writes with no API calls.
"""

import json

import anthropic
import httpx2

from jarvie.llm import RunResult, Usage


class FakeAgent:
    def __init__(self, script, cost: float = 0.10):
        self.script = script
        self.cost = cost
        self.calls: list[dict] = []

    def run(self, system, prompt, tools, web_search=0, web_fetch=0, max_iterations=25):
        by_name = {fn.__name__: fn for fn in tools}
        self.calls.append(
            {
                "system": system,
                "prompt": prompt,
                "tools": list(by_name),
                "web_search": web_search,
                "web_fetch": web_fetch,
            }
        )
        text = self.script(prompt, by_name) or "done"
        return RunResult(
            text=text, usage=Usage(1000, 100, cost_usd=self.cost), stop_reason="end_turn"
        )


class FakeWriter:
    def __init__(self, make, cost: float = 0.01):
        self.make = make  # (prompt, schema) -> schema instance
        self.cost = cost
        self.calls: list[str] = []

    def write(self, system, prompt, schema):
        self.calls.append(prompt)
        return self.make(prompt, schema), Usage(500, 200, cost_usd=self.cost)


def mock_anthropic(responses: list[dict], seen: list | None = None) -> anthropic.Anthropic:
    """A real SDK client whose HTTP layer replays canned Messages API responses."""
    queue = list(responses)

    def handler(request):
        if seen is not None:
            seen.append({"headers": dict(request.headers), "body": json.loads(request.content)})
        return httpx2.Response(200, json=queue.pop(0))

    transport = httpx2.MockTransport(handler)
    return anthropic.Anthropic(
        api_key="sk-test",
        http_client=anthropic.DefaultHttpxClient(transport=transport),
        max_retries=0,
    )


def message(content: list[dict], stop: str = "end_turn", model: str = "claude-opus-5-5", **usage):
    return {
        "id": "msg",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {"input_tokens": 1000, "output_tokens": 100, **usage},
    }
