"""Phase 2: research agent, worker, cost caps, and the real SDK path (mocked HTTP)."""

import pytest
from agent_fakes import FakeAgent, message, mock_anthropic
from career_fakes import board_client, call, write_profile

from jarvie import briefs, config, jobs, worker
from jarvie.career import postings, profile
from jarvie.career import tools as career_tools
from jarvie.llm import ClaudeAgent, LLMError, cached_system

pytestmark = pytest.mark.usefixtures("temp_db")


@pytest.fixture
def prof(tmp_path, monkeypatch):
    path = write_profile(tmp_path)
    monkeypatch.setattr(config, "CAREER_PROFILE", path)
    return profile.load(path)


def ctx_with(agent, prof) -> worker.WorkerContext:
    return worker.WorkerContext(
        agent=agent, profile_loader=lambda: prof, extra={"http_client": board_client()}
    )


def score_everything(prompt, tools):
    """Script: score each posting_id in the prompt; for open web searches, save one lab."""
    import re

    save = tools["save_posting"]
    ids = [int(i) for i in re.findall(r"posting_id (\d+)", prompt)]
    for i in ids:
        save(
            company="x",
            title="x",
            posting_id=i,
            prestige=4,
            prestige_reason="known",
            competitiveness=4,
            competitiveness_reason="selective",
            ownership=3,
            ownership_reason="standard",
            fit_score=0.7,
            fit_reasons="good",
            process=["oa"],
        )
    if not ids and "Search the web" in prompt:
        save(
            company="Kim Lab, UC Berkeley",
            title="Undergraduate Researcher",
            kind="research",
            url="https://kimlab.berkeley.edu/join",
            fields=["biology"],
            prestige=2,
            fit_score=0.6,
        )
    return f"scored {len(ids)}"


# --- score_posting ---------------------------------------------------------------


def test_pasted_handshake_text_is_scored_and_saved(prof):
    def script(prompt, tools):
        assert "<posting>" in prompt
        tools["save_posting"](
            company="Acme",
            title="Bio Intern",
            kind="internship",
            fields=["biology"],
            paid=True,
            pay="$30/hr",
            fit_score=0.8,
            skills=["PCR"],
            process=["interview"],
        )

    agent = FakeAgent(script)
    job_id = call(career_tools.add_posting, url_or_text="Handshake: Acme Bio Intern, $30/hr")[
        "job_id"
    ]
    assert worker.run_pending(ctx_with(agent, prof)) == 1

    job = jobs.get(job_id)
    assert job["status"] == "done" and job["cost_usd"] == pytest.approx(0.10)
    [pid] = job["result"]["saved"]
    p = postings.get(pid)
    assert (p["source"], p["pay"], p["skills"]) == ("handshake", "$30/hr", ["PCR"])
    assert agent.calls[0]["web_fetch"] == 0  # pasted text needs no fetching
    # The resume and rubric go in the cached system prompt.
    assert "Alex K" in agent.calls[0]["system"][0]["text"]
    assert agent.calls[0]["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_url_posting_allows_fetching(prof):
    agent = FakeAgent(lambda prompt, tools: "login wall, nothing saved")
    job_id = jobs.enqueue("score_posting", {"url": "https://jobs.example.com/1"})
    worker.run_pending(ctx_with(agent, prof))
    assert agent.calls[0]["web_fetch"] == 2
    assert jobs.get(job_id)["result"]["saved"] == []
    assert "login wall" in jobs.get(job_id)["result"]["note"]


def test_bad_tool_arguments_are_returned_to_the_model(prof):
    replies = []

    def script(prompt, tools):
        replies.append(tools["save_posting"](company="A", title="B", prestige=9))
        replies.append(tools["save_posting"](company="A", title="B", fit_score=3))
        replies.append(tools["save_posting"](company="A", title="B", posting_id=999))

    jobs.enqueue("score_posting", {"text": "x"})
    worker.run_pending(ctx_with(FakeAgent(script), prof))
    assert all(r.startswith("error") for r in replies)
    assert postings.search() == []


# --- discover_postings -----------------------------------------------------------


def test_discovery_from_brief_ingests_then_scores_then_searches_web(prof):
    bid = briefs.start("discover_postings")
    briefs.answer(
        bid,
        {
            "kinds": "both",
            "fields": "machine learning, biology, biotech",
            "term": "Summer 2027",
            "max_results": 10,
        },
    )
    job_id = briefs.submit(bid)
    agent = FakeAgent(score_everything)
    worker.run_pending(ctx_with(agent, prof))

    job = jobs.get(job_id)
    assert job["status"] == "done", job["error"]
    result = job["result"]
    assert result["from_job_boards"] >= 1
    assert len(result["scored"]) == result["from_job_boards"] + 1  # + the web-found lab
    lab = [p for p in postings.search(kind="research") if "Kim Lab" in p["company"]]
    assert lab and lab[0]["url"] == "https://kimlab.berkeley.edu/join"
    scored = postings.get(result["scored"][0])
    assert scored["prestige"] == 4 and scored["fit_score"] == 0.7
    assert agent.calls[-1]["web_search"] == 8  # the open web search step


def test_discovery_stops_at_job_budget(prof, monkeypatch):
    monkeypatch.setenv("CAREER_JOB_BUDGET_USD", "0.05")
    job_id = jobs.enqueue(
        "discover_postings", {"kinds": ["internship", "research"], "max_results": 25}
    )
    agent = FakeAgent(score_everything, cost=0.10)
    worker.run_pending(ctx_with(agent, prof))
    result = jobs.get(job_id)["result"]
    assert result["stopped_early_for_budget"] is True
    assert len(agent.calls) == 1  # one chunk, then the budget was spent


# --- worker ----------------------------------------------------------------------


def test_worker_records_expected_failures_and_crashes(prof):
    def fail(prompt, tools):
        raise LLMError("Anthropic API key missing or invalid (set ANTHROPIC_API_KEY)")

    bad = jobs.enqueue("score_posting", {"text": "x"})
    empty = jobs.enqueue("score_posting", {})
    worker.run_pending(ctx_with(FakeAgent(fail), prof))
    assert "ANTHROPIC_API_KEY" in jobs.get(bad)["error"]
    assert "needs a url or text" in jobs.get(empty)["error"]

    def crash(prompt, tools):
        raise RuntimeError("bug")

    boom = jobs.enqueue("score_posting", {"text": "x"})
    after = jobs.enqueue("score_posting", {"text": "y"})
    worker.run_pending(ctx_with(FakeAgent(crash), prof))
    assert jobs.get(boom)["error"].startswith("internal error")
    assert jobs.get(after)["status"] == "failed"  # the loop kept going


def test_daily_budget_leaves_jobs_queued(prof, monkeypatch):
    monkeypatch.setenv("JARVIE_DAILY_BUDGET_USD", "0.15")
    for _ in range(3):
        jobs.enqueue("score_posting", {"text": "x"})
    ran = worker.run_pending(ctx_with(FakeAgent(lambda p, t: None, cost=0.10), prof))
    assert ran == 2
    assert [j["status"] for j in jobs.recent()] == ["queued", "done", "done"]


def test_missing_profile_fails_the_job_clearly(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CAREER_PROFILE", tmp_path / "missing.toml")
    job_id = jobs.enqueue("score_posting", {"text": "x"})
    worker.run_pending(worker.WorkerContext(agent=FakeAgent(lambda p, t: None)))
    error = jobs.get(job_id)["error"]
    assert "career_profile.example.toml" in error and not error.startswith("internal")


# --- real SDK path with a mocked HTTP layer (no API key needed) -------------------


def test_claude_agent_runs_tool_loop_against_mocked_api():
    seen: list = []
    client = mock_anthropic(
        [
            message(
                [{"type": "tool_use", "id": "t1", "name": "save_x", "input": {"name": "Acme"}}],
                stop="tool_use",
                cache_read_input_tokens=500,
                server_tool_use={"web_search_requests": 2},
            ),
            message([{"type": "text", "text": "Saved."}]),
        ],
        seen,
    )
    got = []

    def save_x(name: str) -> str:
        """Save x.

        Args:
            name: Name.
        """
        got.append(name)
        return "ok"

    result = ClaudeAgent(client=client).run(cached_system("stable"), "go", [save_x], web_search=3)
    assert got == ["Acme"] and result.text == "Saved."
    assert result.usage.web_searches == 2
    # 2200 in, 200 out, 500 cache reads, 2 searches at Opus 5.5 prices
    assert result.usage.cost_usd == pytest.approx((2000 * 4 + 200 * 20 + 500 * 0.2) / 1e6 + 0.02)
    body = seen[0]["body"]
    assert body["model"] == "claude-opus-5-5"
    assert body["thinking"] == {"type": "adaptive"}
    assert body["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in seen[0]["headers"]["anthropic-beta"]
    assert {t["name"] for t in body["tools"]} == {"save_x", "web_search"}


def test_refusal_becomes_an_llm_error():
    client = mock_anthropic([message([{"type": "text", "text": ""}], stop="refusal")])
    with pytest.raises(LLMError, match="declined"):
        ClaudeAgent(client=client).run(cached_system("s"), "go", [])
