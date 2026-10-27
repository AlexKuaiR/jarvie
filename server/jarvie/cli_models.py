"""Models that run on your subscriptions (Claude Pro, ChatGPT Plus) instead of the API.

Each job runs the vendor's own command-line tool once, non-interactively:
    claude -p ...      Claude Code, signed in with your Claude plan
    codex exec ...     Codex CLI, signed in with your ChatGPT plan
Both are given web search only (no shell, no file edits) and must answer with JSON
that matches a schema we pass.

The CLIs can't call Jarvie's Python tools mid-run, so CLIAgent turns the tools into
a schema: the model returns the calls it wants to make, and Jarvie runs them through
the same validated functions the API agent uses. If any call fails validation, the
model gets one round to fix those calls. Jobs therefore work the same on any backend.

Pick backends per job type in career_profile.toml under [models]; see make_agent().
"""

import inspect
import json
import re
import subprocess
import tempfile
import types
import typing
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from loguru import logger
from pydantic import BaseModel, ValidationError

from .llm import LLMError, RunResult, Usage


class SubscriptionLimit(LLMError):
    """A plan's usage limit is reached; the job should wait and retry, not fail."""


# What the CLIs print when a plan's limit is hit (wording varies by version).
LIMIT_WORDS = re.compile(
    r"usage limit|rate[ _-]?limit|limit reached|limit will reset|quota|too many requests|\b429\b",
    re.I,
)


class Backend(Protocol):
    name: str

    def complete(self, system: str, prompt: str, schema: dict, web: bool) -> dict:
        """One run that returns a JSON object matching `schema`."""
        ...


def _run(cmd: list[str], stdin: str, timeout: float, runner, cwd: str):
    try:
        return runner(cmd, input=stdin, capture_output=True, text=True, timeout=timeout, cwd=cwd)
    except FileNotFoundError as e:
        raise LLMError(f"{cmd[0]} isn't installed or isn't on PATH") from e
    except subprocess.TimeoutExpired as e:
        raise LLMError(f"{cmd[0]} took longer than {timeout:.0f}s") from e


def _failure(tool: str, text: str) -> LLMError:
    text = (text or "").strip()[-600:]
    if LIMIT_WORDS.search(text):
        return SubscriptionLimit(f"{tool}: plan usage limit reached ({text[-200:]})")
    if re.search(r"not logged in|log ?in|authenticat|unauthori[sz]ed|401", text, re.I):
        return LLMError(f"{tool} isn't signed in; run `{tool}` once in a terminal and log in")
    return LLMError(f"{tool} failed: {text or 'no output'}")


class ClaudeCode:
    """Claude Code in print mode, on the Claude plan you're signed in with.

    Not --bare: bare mode skips your subscription login and needs an API key. Runs in
    an empty temporary folder with no MCP servers and only the web tools allowed, so it
    can't touch the repo, and the session isn't saved.
    """

    name = "claude-code"

    def __init__(
        self, model: str = "sonnet", binary: str = "claude", timeout: float = 900, runner=None
    ):
        self.model = model
        self.binary = binary
        self.timeout = timeout
        self.runner = runner or subprocess.run  # injectable for tests

    def complete(self, system, prompt, schema, web):
        tools = "WebSearch,WebFetch" if web else ""
        cmd = [
            self.binary,
            "-p",
            "--output-format",
            "json",
            "--json-schema",
            json.dumps(schema),
            "--system-prompt",
            system,
            "--model",
            self.model,
            "--tools",
            tools,
            "--permission-mode",
            "dontAsk",
            "--strict-mcp-config",
            "--no-session-persistence",
        ]
        if web:
            cmd += ["--allowedTools", tools]
        with tempfile.TemporaryDirectory() as cwd:
            proc = _run(cmd, prompt, self.timeout, self.runner, cwd)
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError:
            raise _failure(self.binary, proc.stderr or proc.stdout) from None
        if data.get("is_error") or "structured_output" not in data:
            raise _failure(self.binary, str(data.get("result") or data.get("subtype") or data))
        return data["structured_output"]


class Codex:
    """Codex CLI (`codex exec`), on the ChatGPT plan you're signed in with.

    Read-only sandbox, never asks for approval, ignores your Codex config (sign-in
    still works), and keeps no session. Web search is Codex's built-in live search.
    """

    name = "codex"

    def __init__(
        self, model: str | None = None, binary: str = "codex", timeout: float = 900, runner=None
    ):
        self.model = model  # None = your Codex default
        self.binary = binary
        self.timeout = timeout
        self.runner = runner or subprocess.run

    def complete(self, system, prompt, schema, web):
        with tempfile.TemporaryDirectory() as cwd:
            schema_file = Path(cwd, "schema.json")
            schema_file.write_text(json.dumps(schema))
            out_file = Path(cwd, "answer.json")
            cmd = [self.binary, *(["--search"] if web else []), "exec"]
            cmd += [
                "--skip-git-repo-check",
                "--ephemeral",
                "--ignore-user-config",
                "--sandbox",
                "read-only",
                "-c",
                'approval_policy="never"',
                "--color",
                "never",
                "--output-schema",
                str(schema_file),
                "-o",
                str(out_file),
                "-C",
                cwd,
            ]
            if self.model:
                cmd += ["-m", self.model]
            # Codex has no system-prompt flag, so the instructions lead the prompt.
            stdin = f"<instructions>\n{system}\n</instructions>\n\n{prompt}"
            proc = _run([*cmd, "-"], stdin, self.timeout, self.runner, cwd)
            answer = out_file.read_text() if out_file.exists() else ""
        if proc.returncode != 0 or not answer.strip():
            raise _failure(self.binary, proc.stderr or proc.stdout)
        try:
            return json.loads(answer)
        except json.JSONDecodeError:
            raise LLMError(f"{self.binary} returned something that isn't JSON") from None


class Fallback:
    """Try backends in order; move to the next one only when a plan limit is hit."""

    def __init__(self, backends: list[Backend]):
        self.backends = backends
        self.name = ",".join(b.name for b in backends)

    def complete(self, system, prompt, schema, web):
        last: SubscriptionLimit | None = None
        for b in self.backends:
            try:
                return b.complete(system, prompt, schema, web)
            except SubscriptionLimit as e:
                logger.warning("{} hit its plan limit; trying the next backend", b.name)
                last = e
        raise last or SubscriptionLimit("no backends configured")


# --- turning Python tools and pydantic models into strict JSON schemas ---------------
# Strict = every property required (optional ones nullable) and no extra keys, which
# ChatGPT's structured outputs require and Claude accepts.

PAIRS = "__pairs__"  # marks a dict[str, str] sent as a list of {key, value}


def _schema_for(hint) -> dict:
    origin, args = typing.get_origin(hint), typing.get_args(hint)
    if origin in (typing.Union, types.UnionType):
        inner = [a for a in args if a is not type(None)]
        base = (
            _schema_for(inner[0]) if len(inner) == 1 else {"anyOf": [_schema_for(a) for a in inner]}
        )
        return {"anyOf": [base, {"type": "null"}]} if type(None) in args else base
    if origin is list:
        return {"type": "array", "items": _schema_for(args[0] if args else str)}
    if origin is dict or hint is dict:
        pair = {
            "type": "object",
            "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
            "required": ["key", "value"],
            "additionalProperties": False,
        }
        return {"type": "array", "items": pair, PAIRS: True}
    simple = {str: "string", int: "integer", float: "number", bool: "boolean"}
    if hint in simple:
        return {"type": simple[hint]}
    raise TypeError(f"no JSON schema for {hint!r}")


def _arg_docs(fn: Callable) -> dict[str, str]:
    """name -> description from a Google-style docstring's Args: section."""
    docs, current = {}, None
    in_args = False
    for line in (inspect.getdoc(fn) or "").splitlines():
        if line.strip() == "Args:":
            in_args = True
            continue
        if in_args:
            m = re.match(r"\s{2,}(\w+):\s*(.*)", line)
            if m and len(line) - len(line.lstrip()) <= 4:
                current = m.group(1)
                docs[current] = m.group(2)
            elif line.strip() and current:
                docs[current] += " " + line.strip()
            elif not line.strip():
                in_args = False
    return docs


def tool_schema(fn: Callable) -> dict:
    hints = typing.get_type_hints(fn)
    docs = _arg_docs(fn)
    props = {}
    for name, param in inspect.signature(fn).parameters.items():
        s = _schema_for(hints.get(name, str))
        if param.default is not inspect.Parameter.empty and "anyOf" not in s:
            s = {"anyOf": [s, {"type": "null"}]}  # optional: send null to use the default
        if name in docs:
            s = {**s, "description": docs[name]}
        props[name] = s
    return {
        "type": "object",
        "properties": {
            "tool": {"type": "string", "enum": [fn.__name__]},
            "args": {
                "type": "object",
                "properties": props,
                "required": list(props),
                "additionalProperties": False,
            },
        },
        "required": ["tool", "args"],
        "additionalProperties": False,
    }


def _clean(schema):
    """Drop our private markers before sending a schema to a CLI."""
    if isinstance(schema, dict):
        return {k: _clean(v) for k, v in schema.items() if k != PAIRS}
    if isinstance(schema, list):
        return [_clean(v) for v in schema]
    return schema


def calls_schema(tools: list[Callable]) -> dict:
    props: dict = {"summary": {"type": "string", "description": "One or two sentences."}}
    if tools:
        props["calls"] = {
            "type": "array",
            "items": {"anyOf": [tool_schema(fn) for fn in tools]},
        }
    return _clean(
        {
            "type": "object",
            "properties": props,
            "required": list(props),
            "additionalProperties": False,
        }
    )


def strict_schema(model: type[BaseModel]) -> dict:
    """A pydantic model's JSON schema with refs inlined and strict objects."""
    raw = model.model_json_schema()
    defs = raw.pop("$defs", {})

    def walk(node):
        if isinstance(node, list):
            return [walk(n) for n in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            return walk(defs[node["$ref"].split("/")[-1]])
        node = {k: walk(v) for k, v in node.items() if k not in ("title", "default")}
        if node.get("type") == "object" and "properties" in node:
            node["required"] = list(node["properties"])
            node["additionalProperties"] = False
        return node

    return walk(raw)


def _to_python(value, schema: dict):
    if schema.get(PAIRS) and isinstance(value, list):
        return {str(p.get("key")): str(p.get("value")) for p in value if isinstance(p, dict)}
    for option in schema.get("anyOf", []):
        if option.get(PAIRS):
            return _to_python(value, option)
    return value


PROTOCOL = """\
## How to answer
The tools listed below are Jarvie's, and they run after you finish: don't try to call
them. Put each call you want, in order, in "calls": the tool's name and all its
arguments (null for optional ones you don't use). Jarvie then runs and validates each
one. Write a short "summary" too. Don't invent facts to fill arguments: use null or
"unknown". (Your own built-in tools, like web search when it's allowed above, you use
normally while working, before you answer.)

## Tools
"""


class CLIAgent:
    """AgentModel over a subscription CLI. See the module docstring for how tools work."""

    def __init__(self, backend: Backend, repair_rounds: int = 1):
        self.backend = backend
        self.repair_rounds = repair_rounds

    def _execute(self, calls: list[dict], tools: dict[str, Callable]) -> list[tuple[dict, str]]:
        done = []
        for c in calls or []:
            fn = tools.get(c.get("tool"))
            if fn is None:
                done.append((c, f"error: there's no tool named {c.get('tool')!r}"))
                continue
            schemas = tool_schema(fn)["properties"]["args"]["properties"]
            args = {
                k: _to_python(v, schemas.get(k, {}))
                for k, v in (c.get("args") or {}).items()
                if v is not None and k in schemas
            }
            try:
                done.append((c, str(fn(**args))))
            except TypeError as e:  # e.g. a required argument left null
                done.append((c, f"error: bad arguments: {e}"))
        return done

    def run(self, system, prompt, tools, web_search=0, web_fetch=0, max_iterations=25):
        by_name = {fn.__name__: fn for fn in tools}
        system_text = "\n\n".join(b["text"] for b in system)
        web = bool(web_search or web_fetch)
        docs = "\n".join(f"- {fn.__name__}: {inspect.getdoc(fn) or ''}" for fn in tools)
        limits = (
            f"Use your web search tool as needed (about {web_search} searches at most) and "
            f"read up to {web_fetch} pages."
            if web
            else "Don't browse the web for this; use only what's given."
        )
        task = f"{prompt}\n\n{limits}\n\n{PROTOCOL}{docs or '- none'}"
        schema = calls_schema(tools)
        out = self.backend.complete(system_text, task, schema, web)
        results = self._execute(out.get("calls", []), by_name)
        summary = out.get("summary", "")
        for _ in range(self.repair_rounds):
            failed = [(c, r) for c, r in results if r.startswith("error")]
            if not failed:
                break
            retry = (
                f"{task}\n\n## Your previous answer\nThese calls failed validation:\n"
                + "\n".join(f"- {json.dumps(c)} -> {r}" for c, r in failed)
                + "\nReturn corrected versions of only those calls (or none, if they can't "
                "be fixed from what you know). The other calls already succeeded."
            )
            out = self.backend.complete(system_text, retry, schema, web)
            results = [r for r in results if not r[1].startswith("error")]
            results += self._execute(out.get("calls", []), by_name)
            summary = out.get("summary", summary)
        log = "; ".join(r for _, r in results)
        logger.info("{}: {} tool calls ({})", self.backend.name, len(results), log[:300])
        # Subscription runs cost nothing per call, so the job's dollar cost stays 0.
        return RunResult(text=f"{summary}\n{log}".strip(), usage=Usage(), stop_reason="end_turn")


class CLIWriter:
    """WriterModel over a subscription CLI: one structured answer, validated."""

    def __init__(self, backend: Backend):
        self.backend = backend

    def write(self, system, prompt, schema):
        system_text = "\n\n".join(b["text"] for b in system)
        out = self.backend.complete(system_text, prompt, strict_schema(schema), web=False)
        try:
            return schema.model_validate(out), Usage()
        except ValidationError as e:
            raise LLMError(f"{self.backend.name} returned a malformed answer: {e}") from e


# --- choosing backends -------------------------------------------------------------
# A spec is "backend[:model]", or several joined by "," to fall back when a plan's
# limit is hit: "codex", "claude-code:opus", "codex,claude-code:sonnet", "api".

DEFAULT_SPECS = {
    "research": "codex",  # scoring, searches, people finding: frequent, spares Claude
    "documents": "claude-code:opus",  # cover letters and essays: few, quality first
    "writer": "claude-code:sonnet",  # outreach emails
}
BACKENDS = {"claude-code": ClaudeCode, "codex": Codex}


def parse_spec(spec: str) -> list[tuple[str, str | None]]:
    parts = []
    for item in spec.split(","):
        name, _, model = item.strip().partition(":")
        if name not in (*BACKENDS, "api"):
            raise ValueError(f"unknown model backend {name!r}; use {', '.join([*BACKENDS, 'api'])}")
        parts.append((name, model or None))
    if any(n == "api" for n, _ in parts) and len(parts) > 1:
        raise ValueError("'api' can't be part of a fallback list")
    return parts


def _backend(spec: str) -> Backend:
    made = []
    for name, model in parse_spec(spec):
        cls = BACKENDS[name]
        made.append(cls(model=model) if model else cls())
    return made[0] if len(made) == 1 else Fallback(made)


def make_agent(spec: str):
    """An AgentModel for a spec ("api" = the pay-as-you-go ClaudeAgent)."""
    [(first, model)] = parse_spec(spec)[:1]
    if first == "api":
        from .llm import RESEARCH_MODEL, ClaudeAgent

        return ClaudeAgent(model=model or RESEARCH_MODEL)
    return CLIAgent(_backend(spec))


def make_writer(spec: str):
    [(first, model)] = parse_spec(spec)[:1]
    if first == "api":
        from .llm import WRITER_MODEL, ClaudeWriter

        return ClaudeWriter(model=model or WRITER_MODEL)
    return CLIWriter(_backend(spec))
