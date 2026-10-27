"""Claude models for the background work: the research agent and the writer.

The voice bot keeps using OpenAI (bot.py). These models only run inside the worker
process, never in the voice pipeline. Everything goes through two small interfaces,
AgentModel and WriterModel, so tests use fakes that never call the API or cost money.
"""

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from loguru import logger
from pydantic import BaseModel

RESEARCH_MODEL = os.getenv("JARVIE_RESEARCH_MODEL", "claude-opus-5-5")
WRITER_MODEL = os.getenv("JARVIE_WRITER_MODEL", "claude-sonnet-5-5")

# USD per million tokens: input, output, cache read, cache write (5-minute). Estimates for
# the per-job cost caps; the Anthropic console is the source of truth for billing.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00, 0.20, 5.00),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20, 2.50),
    "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125),
}
WEB_SEARCH_USD = 10.00 / 1000  # per search; web fetch has no charge beyond its tokens


class LLMError(Exception):
    """The model call failed (no key, network, refusal, ...); the message says how."""


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    web_searches: int = 0
    cost_usd: float = 0.0

    def add(self, other: "Usage") -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))


def usage_from(model: str, u) -> Usage:
    """Turn an API usage object into tokens and an estimated dollar cost."""
    p_in, p_out, p_read, p_write = PRICES.get(model, PRICES["claude-opus-5-5"])
    searches = 0
    server = getattr(u, "server_tool_use", None)
    if server is not None:
        searches = getattr(server, "web_search_requests", 0) or 0
    read = getattr(u, "cache_read_input_tokens", 0) or 0
    write = getattr(u, "cache_creation_input_tokens", 0) or 0
    cost = (
        u.input_tokens * p_in + u.output_tokens * p_out + read * p_read + write * p_write
    ) / 1e6 + searches * WEB_SEARCH_USD
    return Usage(u.input_tokens, u.output_tokens, read, write, searches, cost)


@dataclass
class RunResult:
    text: str  # the model's final words (a summary; real results go through tools)
    usage: Usage
    stop_reason: str | None = None


class AgentModel(Protocol):
    def run(
        self,
        system: list[dict],
        prompt: str,
        tools: list[Callable],
        web_search: int = 0,
        web_fetch: int = 0,
        max_iterations: int = 25,
    ) -> RunResult:
        """Run a tool-using loop. tools are plain Python functions with docstrings;
        web_search/web_fetch are how many server-side searches/fetches are allowed."""
        ...


class WriterModel(Protocol):
    def write(
        self, system: list[dict], prompt: str, schema: type[BaseModel]
    ) -> tuple[BaseModel, Usage]:
        """One structured response validated against `schema`."""
        ...


def cached_system(stable: str, extra: str = "") -> list[dict]:
    """System prompt blocks with the stable part (resume, rubric) marked for caching.

    Repeated jobs then reuse the cached prefix at a tenth of the input price. Keep
    anything that changes per job (dates, ids) in `extra` or the prompt, after it.
    """
    blocks = [{"type": "text", "text": stable, "cache_control": {"type": "ephemeral"}}]
    if extra:
        blocks.append({"type": "text", "text": extra})
    return blocks


def _client():
    import anthropic

    # Retries 429/5xx/connection errors itself (default 2); a long agent turn can take
    # minutes, so allow a generous timeout per request.
    return anthropic.Anthropic(timeout=600.0, max_retries=3)


NO_KEY = "Anthropic API key missing or invalid (set ANTHROPIC_API_KEY in server/.env)"


def _no_credentials(e: TypeError) -> bool:
    # With no key at all, the SDK raises TypeError before sending anything.
    return "authentication method" in str(e)


class ClaudeAgent:
    """AgentModel on Claude via the SDK's tool runner, with server-side web tools."""

    def __init__(self, model: str = RESEARCH_MODEL, effort: str = "high", client=None):
        self.model = model
        self.effort = effort
        self._client = client  # injectable for tests

    def run(self, system, prompt, tools, web_search=0, web_fetch=0, max_iterations=25):
        import anthropic
        from anthropic import beta_tool

        client = self._client or _client()
        tool_list: list = [beta_tool(fn) for fn in tools]
        if web_search:
            tool_list.append(
                {"type": "web_search_20260209", "name": "web_search", "max_uses": web_search}
            )
        if web_fetch:
            tool_list.append(
                {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": web_fetch}
            )
        total = Usage()
        last = None
        try:
            runner = client.beta.messages.tool_runner(
                model=self.model,
                max_tokens=16000,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                tools=tool_list,
                thinking={"type": "adaptive"},
                output_config={"effort": self.effort},
                max_iterations=max_iterations,
                # If a safety classifier declines, the API retries on a fallback model.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
            for message in runner:
                last = message
                total.add(usage_from(self.model, message.usage))
        except anthropic.AuthenticationError as e:
            raise LLMError(NO_KEY) from e
        except TypeError as e:
            if _no_credentials(e):
                raise LLMError(NO_KEY) from e
            raise
        except anthropic.RateLimitError as e:
            raise LLMError("Anthropic rate limit; try again later") from e
        except anthropic.APIConnectionError as e:
            raise LLMError("couldn't reach the Anthropic API") from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"Anthropic API error {e.status_code}: {e.message}") from e
        if last is None:
            raise LLMError("the model returned nothing")
        if last.stop_reason == "refusal":
            raise LLMError("the model declined this request")
        text = "".join(b.text for b in last.content if getattr(b, "type", "") == "text")
        logger.info(
            "agent {}: {} in / {} out tokens, {} searches, ${:.3f}",
            self.model,
            total.input_tokens,
            total.output_tokens,
            total.web_searches,
            total.cost_usd,
        )
        return RunResult(text=text, usage=total, stop_reason=last.stop_reason)


class ClaudeWriter:
    """WriterModel on Claude with structured outputs (validated against a pydantic model)."""

    def __init__(self, model: str = WRITER_MODEL, effort: str = "medium", client=None):
        self.model = model
        self.effort = effort
        self._client = client

    def write(self, system, prompt, schema):
        import anthropic

        client = self._client or _client()
        try:
            response = client.messages.parse(
                model=self.model,
                max_tokens=8000,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                output_format=schema,
                output_config={"effort": self.effort},
            )
        except anthropic.AuthenticationError as e:
            raise LLMError(NO_KEY) from e
        except TypeError as e:
            if _no_credentials(e):
                raise LLMError(NO_KEY) from e
            raise
        except anthropic.APIConnectionError as e:
            raise LLMError("couldn't reach the Anthropic API") from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"Anthropic API error {e.status_code}: {e.message}") from e
        if response.stop_reason == "refusal":
            raise LLMError("the model declined this request")
        if response.parsed_output is None:
            raise LLMError("the model's answer didn't match the expected format")
        return response.parsed_output, usage_from(self.model, response.usage)


BATCH_DISCOUNT = 0.5  # the Message Batches API bills at half price


def write_many(
    writer: "ClaudeWriter",
    system: list[dict],
    prompts: list[str],
    schema: type[BaseModel],
    poll_seconds: float = 30.0,
    timeout_seconds: float = 3 * 3600,
    sleep=None,
) -> tuple[list[tuple[BaseModel | None, str | None]], Usage]:
    """Many structured responses through the Batches API (half price, usually minutes).

    Returns one (result, error) pair per prompt, in prompt order, plus total usage.
    Batches finish in any order, so results are matched by custom_id, not position.
    """
    import time

    import anthropic
    from anthropic import transform_schema
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    sleep = sleep or time.sleep
    client = writer._client or _client()
    fmt = {"type": "json_schema", "schema": transform_schema(schema)}
    try:
        batch = client.messages.batches.create(
            requests=[
                Request(
                    custom_id=f"p{i}",
                    params=MessageCreateParamsNonStreaming(
                        model=writer.model,
                        max_tokens=8000,
                        system=system,
                        messages=[{"role": "user", "content": prompt}],
                        output_config={"format": fmt, "effort": writer.effort},
                    ),
                )
                for i, prompt in enumerate(prompts)
            ]
        )
        waited = 0.0
        while batch.processing_status != "ended":
            if waited >= timeout_seconds:
                raise LLMError(f"batch {batch.id} still running after {timeout_seconds:.0f}s")
            sleep(poll_seconds)
            waited += poll_seconds
            batch = client.messages.batches.retrieve(batch.id)
        results = list(client.messages.batches.results(batch.id))
    except anthropic.AuthenticationError as e:
        raise LLMError(NO_KEY) from e
    except TypeError as e:
        if _no_credentials(e):
            raise LLMError(NO_KEY) from e
        raise
    except anthropic.APIStatusError as e:
        raise LLMError(f"Anthropic API error {e.status_code}: {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise LLMError("couldn't reach the Anthropic API") from e

    out: list[tuple[BaseModel | None, str | None]] = [(None, "missing from batch")] * len(prompts)
    total = Usage()
    for r in results:
        i = int(r.custom_id[1:])
        if r.result.type != "succeeded":
            out[i] = (None, f"batch request {r.result.type}")
            continue
        msg = r.result.message
        u = usage_from(writer.model, msg.usage)
        u.cost_usd *= BATCH_DISCOUNT
        total.add(u)
        if msg.stop_reason == "refusal":
            out[i] = (None, "the model declined this request")
            continue
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        try:
            out[i] = (schema.model_validate_json(text), None)
        except ValueError as e:
            out[i] = (None, f"bad format: {e}")
    return out, total
