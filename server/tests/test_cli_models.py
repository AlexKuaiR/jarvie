"""Subscription backends (Claude Code, Codex CLI), with a fake runner: no real CLI runs."""

import json
import subprocess
import time
from pathlib import Path

import pytest
from career_fakes import write_profile

from jarvie import cli_models, config, jobs, worker
from jarvie.career import postings, profile, research_agent
from jarvie.cli_models import (
    ClaudeCode,
    CLIAgent,
    CLIWriter,
    Codex,
    Fallback,
    SubscriptionLimit,
    calls_schema,
    parse_spec,
    strict_schema,
    tool_schema,
)
from jarvie.llm import LLMError, cached_system
from jarvie.outreach import contacts
from jarvie.outreach.research import make_save_contact
from jarvie.outreach.writer import DraftEmail

pytestmark = pytest.mark.usefixtures("temp_db")


class FakeRunner:
    """Stands in for subprocess.run: records each command and replies in turn."""

    def __init__(self, *replies):
        self.replies = list(replies)  # (returncode, stdout, stderr, codex_answer)
        self.calls: list[dict] = []

    def __call__(self, cmd, input, capture_output, text, timeout, cwd):
        code, out, err, answer = self.replies.pop(0)
        self.calls.append({"cmd": cmd, "stdin": input, "cwd": cwd})
        if answer is not None and "-o" in cmd:
            Path(cmd[cmd.index("-o") + 1]).write_text(answer)
        if "--output-schema" in cmd:
            self.calls[-1]["schema"] = json.loads(
                Path(cmd[cmd.index("--output-schema") + 1]).read_text()
            )
        return subprocess.CompletedProcess(cmd, code, out, err)


def claude_reply(structured=None, **extra):
    data = {"type": "result", "is_error": False, **extra}
    if structured is not None:
        data["structured_output"] = structured
    return (0, json.dumps(data), "", None)


class Scripted:
    """A Backend that returns canned answers, for CLIAgent tests."""

    name = "scripted"

    def __init__(self, *answers):
        self.answers = list(answers)
        self.seen: list[dict] = []

    def complete(self, system, prompt, schema, web):
        self.seen.append({"system": system, "prompt": prompt, "schema": schema, "web": web})
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def save_tool():
    return research_agent.make_save_posting(research_agent.SaveLog(), None, "web", ("swe",))


# --- schemas -----------------------------------------------------------------------


def walk(node):
    yield node
    if isinstance(node, dict):
        for v in node.values():
            yield from walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from walk(v)


def test_tool_schema_is_strict_with_nullable_optionals():
    s = tool_schema(save_tool())
    args = s["properties"]["args"]
    assert s["properties"]["tool"]["enum"] == ["save_posting"]
    assert set(args["required"]) == set(args["properties"])  # every key required...
    assert args["properties"]["company"]["type"] == "string"
    assert {"type": "null"} in args["properties"]["prestige"]["anyOf"]  # ...optionals nullable
    assert args["properties"]["skills"]["anyOf"][0]["items"] == {"type": "string"}
    assert "1-5 per the rubric" in args["properties"]["prestige"]["description"]
    for node in walk(calls_schema([save_tool()])):
        if isinstance(node, dict) and node.get("type") == "object":
            assert node["additionalProperties"] is False
            assert set(node["required"]) == set(node["properties"])


def test_writer_schema_inlines_refs():
    s = strict_schema(DraftEmail)
    assert "$ref" not in json.dumps(s) and "$defs" not in s
    item = s["properties"]["personalization_used"]["items"]
    assert item["required"] == ["feature_key", "how_used"] and item["additionalProperties"] is False


# --- CLIAgent: tool calls come back as JSON and run through the real tools ------------


def test_agent_runs_returned_calls_through_the_validated_tool():
    pid = postings.save(company="CACI", title="SWE Intern", source="web", url="https://c/1")
    backend = Scripted(
        {
            "summary": "scored it",
            "calls": [
                {
                    "tool": "save_posting",
                    "args": {
                        "company": "CACI",
                        "title": "SWE Intern",
                        "posting_id": pid,
                        "prestige": 3,
                        "fit_score": 0.6,
                        "skills": None,
                    },
                },
            ],
        }
    )
    result = CLIAgent(backend).run(cached_system("rubric"), "score it", [save_tool()], web_search=4)
    assert postings.get(pid)["prestige"] == 3 and postings.get(pid)["fit_score"] == 0.6
    assert "saved posting" in result.text and result.usage.cost_usd == 0
    seen = backend.seen[0]
    assert seen["system"] == "rubric" and seen["web"] is True
    assert "save_posting" in seen["prompt"] and "about 4 searches" in seen["prompt"]


def test_failed_calls_get_one_repair_round():
    pid = postings.save(company="A", title="B", source="web", url="https://a/1")
    bad = {
        "tool": "save_posting",
        "args": {"company": "A", "title": "B", "posting_id": pid, "prestige": 9},
    }
    good = {
        "tool": "save_posting",
        "args": {"company": "A", "title": "B", "posting_id": pid, "prestige": 4},
    }
    backend = Scripted({"summary": "", "calls": [bad]}, {"summary": "fixed", "calls": [good]})
    CLIAgent(backend).run(cached_system("s"), "go", [save_tool()])
    assert postings.get(pid)["prestige"] == 4
    assert "prestige must be 1-5" in backend.seen[1]["prompt"]
    assert len(backend.seen) == 2  # one repair, no more


def test_unknown_tools_and_missing_arguments_are_errors_not_crashes():
    backend = Scripted(
        {
            "summary": "",
            "calls": [
                {"tool": "delete_everything", "args": {}},
                {"tool": "save_posting", "args": {"company": "A"}},
            ],
        },
        {"summary": "gave up", "calls": []},
    )
    result = CLIAgent(backend).run(cached_system("s"), "go", [save_tool()])
    assert (
        "no tool named" in backend.seen[1]["prompt"]
        and "bad arguments" in backend.seen[1]["prompt"]
    )
    assert "gave up" in result.text


def test_dict_arguments_travel_as_key_value_pairs():
    log = type("Log", (), {"saved": [], "emails_found": 0, "finder_error": None})()
    tool = make_save_contact(log, None, None, 5)
    features = tool_schema(tool)["properties"]["args"]["properties"]["features"]
    assert features["anyOf"][0]["items"]["properties"].keys() == {"key", "value"}
    backend = Scripted(
        {
            "summary": "",
            "calls": [
                {
                    "tool": "save_contact",
                    "args": {
                        "name": "Dana Lee",
                        "evidence_urls": ["https://acme.bio/team"],
                        "features": [{"key": "team", "value": "university recruiting"}],
                    },
                }
            ],
        }
    )
    CLIAgent(backend).run(cached_system("s"), "go", [tool])
    [c] = contacts.search()
    assert c["features"] == {"team": "university recruiting"}


def test_writer_validates_the_answer():
    good = {
        "subject": "Hi",
        "body": "Hello",
        "personalization_used": [{"feature_key": "school", "how_used": "x"}],
    }
    draft, usage = CLIWriter(Scripted(good)).write(cached_system("s"), "write", DraftEmail)
    assert draft.subject == "Hi" and usage.cost_usd == 0
    with pytest.raises(LLMError, match="malformed"):
        CLIWriter(Scripted({"subject": "Hi"})).write(cached_system("s"), "write", DraftEmail)


# --- the two CLIs ------------------------------------------------------------------


def test_claude_code_command_and_answer():
    runner = FakeRunner(claude_reply({"summary": "ok"}), claude_reply({"summary": "ok"}))
    cc = ClaudeCode(model="opus", runner=runner)
    assert cc.complete("SYS", "PROMPT", {"type": "object"}, web=False) == {"summary": "ok"}
    cmd = runner.calls[0]["cmd"]
    assert cmd[:2] == ["claude", "-p"] and "--bare" not in cmd  # bare would skip your login
    assert cmd[cmd.index("--model") + 1] == "opus"
    assert cmd[cmd.index("--system-prompt") + 1] == "SYS" and runner.calls[0]["stdin"] == "PROMPT"
    assert cmd[cmd.index("--tools") + 1] == "" and "--allowedTools" not in cmd  # no tools at all
    assert "--no-session-persistence" in cmd and "--strict-mcp-config" in cmd
    assert runner.calls[0]["cwd"] != str(Path.cwd())  # an empty temp folder, not the repo
    cc.complete("SYS", "PROMPT", {"type": "object"}, web=True)
    cmd = runner.calls[1]["cmd"]
    assert cmd[cmd.index("--tools") + 1] == "WebSearch,WebFetch"
    assert cmd[cmd.index("--allowedTools") + 1] == "WebSearch,WebFetch"


@pytest.mark.parametrize(
    "reply, error, message",
    [
        (
            claude_reply(None, is_error=True, result="Claude AI usage limit reached|1760000000"),
            SubscriptionLimit,
            "limit",
        ),
        (
            claude_reply(None, is_error=True, result="Not logged in · Please run /login"),
            LLMError,
            "signed in",
        ),
        ((1, "", "boom", None), LLMError, "boom"),
    ],
)
def test_claude_code_failures(reply, error, message):
    with pytest.raises(error, match=message):
        ClaudeCode(runner=FakeRunner(reply)).complete("s", "p", {}, web=False)


def test_codex_command_and_answer():
    runner = FakeRunner((0, "", "", json.dumps({"summary": "ok"})), (0, "", "", '{"summary": "x"}'))
    codex = Codex(runner=runner)
    schema = calls_schema([save_tool()])
    assert codex.complete("SYS", "PROMPT", schema, web=True) == {"summary": "ok"}
    call = runner.calls[0]
    cmd = call["cmd"]
    assert cmd[:3] == ["codex", "--search", "exec"]  # live web search is a top-level flag
    assert cmd[cmd.index("--sandbox") + 1] == "read-only"
    assert (
        'approval_policy="never"' in cmd and "--ephemeral" in cmd and "--ignore-user-config" in cmd
    )
    assert call["schema"] == schema and cmd[-1] == "-"
    assert call["stdin"].startswith("<instructions>\nSYS") and call["stdin"].endswith("PROMPT")
    assert "-m" not in cmd  # your Codex default model
    Codex(model="gpt-5.5", runner=runner).complete("s", "p", {}, web=False)
    cmd = runner.calls[1]["cmd"]
    assert cmd[:2] == ["codex", "exec"] and cmd[cmd.index("-m") + 1] == "gpt-5.5"


def test_codex_failures():
    limited = FakeRunner((1, "", "ERROR: You've hit your usage limit. Try again in 2 hours.", None))
    with pytest.raises(SubscriptionLimit):
        Codex(runner=limited).complete("s", "p", {}, web=False)
    with pytest.raises(LLMError, match="isn't JSON"):
        Codex(runner=FakeRunner((0, "", "", "sure! here you go"))).complete("s", "p", {}, web=False)

    def missing(*args, **kwargs):
        raise FileNotFoundError("codex")

    with pytest.raises(LLMError, match="isn't installed"):
        Codex(runner=missing).complete("s", "p", {}, web=False)


def test_fallback_moves_on_only_for_plan_limits():
    both = Fallback([Scripted(SubscriptionLimit("codex limit")), Scripted({"summary": "claude"})])
    assert both.complete("s", "p", {}, False) == {"summary": "claude"}
    broken = Fallback([Scripted(LLMError("bad schema")), Scripted({"summary": "never"})])
    with pytest.raises(LLMError, match="bad schema"):
        broken.complete("s", "p", {}, False)
    with pytest.raises(SubscriptionLimit):
        Fallback([Scripted(SubscriptionLimit("a")), Scripted(SubscriptionLimit("b"))]).complete(
            "s", "p", {}, False
        )


# --- choosing backends ---------------------------------------------------------------


def test_specs():
    assert parse_spec("codex") == [("codex", None)]
    assert parse_spec("codex, claude-code:sonnet") == [("codex", None), ("claude-code", "sonnet")]
    with pytest.raises(ValueError, match="unknown"):
        parse_spec("gemini")
    with pytest.raises(ValueError, match="fallback"):
        parse_spec("api,codex")


def test_profile_models_default_and_validate(tmp_path):
    path = write_profile(tmp_path)
    assert profile.load(path).models == cli_models.DEFAULT_SPECS
    path.write_text(
        path.read_text().replace(
            "[watchlist]", '[models]\nresearch = "claude-code:sonnet"\n[watchlist]'
        )
    )
    assert profile.load(path).models["research"] == "claude-code:sonnet"
    path.write_text(
        path.read_text().replace('research = "claude-code:sonnet"', 'research = "gemini"')
    )
    with pytest.raises(profile.ProfileError, match="research"):
        profile.load(path)


def test_worker_builds_each_role_from_the_profile(tmp_path, monkeypatch):
    built = []

    class Recorder:
        def __init__(self, model=None, **kwargs):
            built.append((type(self).__name__, model))
            self.name = "recorder"

    monkeypatch.setattr(cli_models, "BACKENDS", {"claude-code": Recorder, "codex": Recorder})
    path = write_profile(tmp_path)
    monkeypatch.setattr(config, "CAREER_PROFILE", path)
    ctx = worker.WorkerContext()
    assert isinstance(ctx.get_agent("research"), CLIAgent)
    ctx.get_agent("documents")
    ctx.get_agent("research")  # cached, not rebuilt
    assert isinstance(ctx.get_writer(), CLIWriter)
    assert built == [("Recorder", None), ("Recorder", "opus"), ("Recorder", "sonnet")]


# --- plan limits in the worker -------------------------------------------------------


class LimitedAgent:
    def run(self, *args, **kwargs):
        raise SubscriptionLimit("codex: plan usage limit reached")


def test_a_plan_limit_requeues_the_job_and_pauses_ai(tmp_path, monkeypatch):
    path = write_profile(tmp_path)
    prof = profile.load(path)
    pid = postings.save(company="A", title="SWE Intern", source="web", url="https://a/1")
    ai_job = jobs.enqueue("score_posting", {"posting_ids": [pid]})
    free_job = jobs.enqueue("find_email", {"name": "Dana Lee", "domain": "acme.bio"})

    class Finder:
        def find(self, first, last, domain):
            return None

    ctx = worker.WorkerContext(
        agent=LimitedAgent(), profile_loader=lambda: prof, email_finder=Finder()
    )
    worker.run_pending(ctx)
    assert jobs.get(ai_job)["status"] == "queued"  # back in line, not failed
    assert jobs.get(free_job)["status"] in ("done", "failed")  # free jobs keep running
    assert ctx.ai_paused_until > time.time() + 60
    worker.run_pending(ctx)
    assert jobs.get(ai_job)["status"] == "queued"  # still paused
    ctx.ai_paused_until = 0
    ctx.agent = None
    monkeypatch.setattr(ctx, "get_agent", lambda role="research": LimitedAgent())
    worker.run_pending(ctx)
    assert jobs.get(ai_job)["status"] == "queued"  # tried again once the pause ended


# --- comparing backends --------------------------------------------------------------


def test_compare_records_scores_without_touching_postings(tmp_path):
    from fastapi.testclient import TestClient

    from jarvie.career import compare
    from jarvie.dashboard import create_app

    prof = profile.load(write_profile(tmp_path))
    a = postings.save(
        company="CACI", title="SWE Intern", source="web", url="https://c/1", raw_text="Java"
    )
    b = postings.save(
        company="Acme", title="ML Intern", source="web", url="https://a/1", raw_text="PyTorch"
    )

    def scores(fit):
        return {
            "summary": "",
            "calls": [
                {
                    "tool": "save_posting",
                    "args": {
                        "company": "x",
                        "title": "x",
                        "posting_id": pid,
                        "fit_score": fit,
                        "fit_reasons": f"fit {fit}",
                    },
                }
                for pid in (a, b)
            ],
        }

    agents = {
        "codex": CLIAgent(Scripted(scores(0.4))),
        "claude-code:sonnet": CLIAgent(Scripted(scores(0.8))),
    }
    items = compare.pick(10)
    assert {p["id"] for p in items} == {a, b}
    run_id = compare.run(prof, ["codex", "claude-code:sonnet"], items, agents)
    t = compare.table(run_id)
    assert t["backends"] == ["codex", "claude-code:sonnet"]
    row = next(r for r in t["rows"] if r["posting"]["id"] == a)
    assert row["by_backend"]["codex"]["fit_score"] == 0.4
    assert row["by_backend"]["claude-code:sonnet"]["fit_score"] == 0.8
    assert postings.get(a)["fit_score"] is None  # your saved posting is unchanged
    page = TestClient(create_app()).get("/compare").text
    assert "fit 0.4" in page and "fit 0.8" in page and "claude-code:sonnet" in page


def test_compare_keeps_going_when_one_backend_fails(tmp_path):
    from jarvie.career import compare

    prof = profile.load(write_profile(tmp_path))
    postings.save(
        company="CACI", title="SWE Intern", source="web", url="https://c/1", raw_text="Java"
    )
    agents = {
        "codex": CLIAgent(Scripted(SubscriptionLimit("codex limit"))),
        "claude-code:sonnet": CLIAgent(Scripted({"summary": "", "calls": []})),
    }
    t = compare.table(compare.run(prof, ["codex", "claude-code:sonnet"], compare.pick(5), agents))
    assert "codex limit" in t["errors"]["codex"] and "claude-code:sonnet" not in t["errors"]
