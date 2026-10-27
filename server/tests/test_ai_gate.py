"""The AI gate on job searches: board-only searches use no model; AI only where asked."""

import pytest
from agent_fakes import FakeAgent
from career_fakes import call
from fastapi.testclient import TestClient
from test_career_phase2 import ctx_with, prof, score_everything  # noqa: F401 (fixture)

from jarvie import briefs, jobs, worker
from jarvie.career import postings
from jarvie.career import tools as career_tools
from jarvie.dashboard import create_app

pytestmark = pytest.mark.usefixtures("temp_db")


class NoModel:
    """Fails the test if a search that shouldn't use AI touches the model."""

    def run(self, *args, **kwargs):
        raise AssertionError("a board-only search called the model")


def search(prof, agent, **params):
    job_id = jobs.enqueue(
        "discover_postings", {"kinds": ["internship", "research"], "max_results": 10, **params}
    )
    worker.run_pending(ctx_with(agent, prof))
    return jobs.get(job_id)


def test_board_only_search_never_touches_the_model(prof):
    job = search(prof, NoModel(), ai_score=0, search_beyond_boards=False)
    assert job["status"] == "done", job["error"]
    r = job["result"]
    assert r["used_ai"] is False and r["scored"] == [] and job["cost_usd"] == 0
    assert r["from_job_boards"] >= 1 and r["unscored"] == r["from_job_boards"]


def test_board_only_search_needs_no_key(prof, monkeypatch):
    # The real context, with no ANTHROPIC_API_KEY: creating a model would fail the job.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    job_id = jobs.enqueue("discover_postings", {"ai_score": "0"})
    ctx = ctx_with(None, prof)
    worker.run_pending(ctx)
    assert jobs.get(job_id)["status"] == "done"
    assert ctx.agent is None  # never created


def test_free_jobs_run_after_the_ai_budget_is_spent(prof, monkeypatch):
    monkeypatch.setenv("JARVIE_DAILY_BUDGET_USD", "0")
    ai_job = jobs.enqueue("discover_postings", {"ai_score": -1})
    free_job = jobs.enqueue("discover_postings", {"ai_score": 0})
    worker.run_pending(ctx_with(NoModel(), prof))
    assert jobs.get(free_job)["status"] == "done"
    assert jobs.get(ai_job)["status"] == "queued"  # waits for tomorrow's budget


def test_ai_scores_only_the_top_n(prof):
    agent = FakeAgent(score_everything)
    job = search(prof, agent, ai_score=2)
    r = job["result"]
    assert len(r["scored"]) == 2 and r["unscored"] == r["from_job_boards"] - 2
    assert len(agent.calls) == 1 and agent.calls[0]["web_search"] == 4  # no open web step
    # The two scored are the best ranked, i.e. the first two the boards returned.
    assert postings.get(r["scored"][0])["fit_score"] == 0.7


def test_web_search_runs_only_when_asked(prof):
    agent = FakeAgent(score_everything)
    job = search(
        prof,
        agent,
        ai_score=0,
        search_beyond_boards=True,
        include_companies=["Eli Lilly"],
        exclude_companies=["Merck"],
    )
    r = job["result"]
    assert r["used_ai"] is True and len(agent.calls) == 1
    call_ = agent.calls[0]
    assert call_["web_search"] == 8
    assert (
        "Check these first: Eli Lilly" in call_["prompt"] and "Skip these: Merck" in call_["prompt"]
    )
    assert "internships and research positions" in call_["prompt"]


@pytest.mark.parametrize("said, stored", [("none", 0), ("no", 0), ("All", -1), ("5", 5), (12, 12)])
def test_ai_score_answers(said, stored):
    bid = briefs.start("discover_postings")
    assert briefs.answer(bid, {"ai_score": said})["answers"]["ai_score"] == stored


def test_gate_questions_are_required():
    bid = briefs.start("discover_postings")
    briefs.answer(bid, {"kinds": "both", "fields": "biotech", "term": "Summer 2027"})
    with pytest.raises(briefs.BriefError, match="search_beyond_boards, ai_score"):
        briefs.submit(bid)
    with pytest.raises(briefs.BriefError, match="number"):
        briefs.answer(bid, {"ai_score": "lots"})


def test_score_saved_postings_on_demand(prof):
    a = postings.save(company="CACI", title="SWE Intern", source="web", url="https://c/1")
    b = postings.save(company="Acme", title="ML Intern", source="web", url="https://a/1")
    started = call(career_tools.score_postings, posting_ids=[a])
    assert "error" in call(career_tools.score_postings, posting_ids=[999])
    agent = FakeAgent(score_everything)
    worker.run_pending(ctx_with(agent, prof))
    assert jobs.get(started["job_id"])["result"]["scored"] == [a]
    assert postings.get(a)["fit_score"] == 0.7 and postings.get(b)["fit_score"] is None
    assert f"posting_id {a}" in agent.calls[0]["prompt"]


def test_dashboard_score_button_and_gate_hint(prof):
    pid = postings.save(company="CACI", title="SWE Intern", source="web", url="https://c/1")
    web = TestClient(create_app())
    assert "Score with AI" in web.get(f"/postings/{pid}").text
    web.post(f"/postings/{pid}/score")
    assert jobs.recent()[0]["params"] == {"posting_ids": [pid]}
    web.post("/briefs", data={"kind": "discover_postings"})
    page = web.get("/briefs").text
    assert "That uses AI" in page and "none / all / a number" in page
