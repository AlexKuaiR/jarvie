"""Phase 1: job-board sources, hard filters, ingest, voice tools and dashboard."""

import asyncio
from datetime import date

import pytest
from career_fakes import board_client, call, write_profile
from fastapi.testclient import TestClient
from pipecat.processors.aggregators.llm_context import LLMContext

from jarvie import briefs, config, jobs, tools
from jarvie.career import filters, ingest, postings, profile, sources
from jarvie.career import tools as career_tools
from jarvie.dashboard import create_app
from jarvie.job_tools import JOB_TOOLS, job_status, open_briefs, start_brief, submit_brief
from jarvie.job_tools import answer_brief as answer_brief_tool

pytestmark = pytest.mark.usefixtures("temp_db")


@pytest.fixture
def profile_path(tmp_path, monkeypatch):
    path = write_profile(tmp_path)
    monkeypatch.setattr(config, "CAREER_PROFILE", path)
    return path


# --- sources --------------------------------------------------------------------


def fetch(source_cls, slug):
    return asyncio.run(source_cls(board_client()).fetch(slug))


def test_greenhouse_parses_and_unescapes_html():
    found = fetch(sources.GreenhouseSource, "acme")
    assert [p.title for p in found][0] == "Machine Learning Intern, Summer 2027"
    assert found[0].company == "Acme Bio"
    assert found[0].text == "Train protein models. Graduating 2028."
    assert found[0].ats == "greenhouse"


def test_lever_ashby_simplify_parse():
    [lever] = fetch(sources.LeverSource, "beta")
    assert lever.location == "Berkeley, CA" and "Python" in lever.text
    ashby = fetch(sources.AshbySource, "gamma")
    assert [p.title for p in ashby] == ["AI Research Intern"]  # unlisted one dropped
    [simplify] = fetch(sources.SimplifySource, "SimplifyJobs/Summer2027-Internships")
    assert simplify.terms == ["Summer 2027"]
    assert simplify.ats == "workday"  # from the apply URL


def test_unknown_board_is_a_named_error():
    with pytest.raises(sources.SourceError, match="check the board slug"):
        fetch(sources.GreenhouseSource, "nope")


def test_fetch_all_keeps_going_after_a_failure():
    raw, errors = asyncio.run(
        sources.fetch_all({"greenhouse": ["acme", "nope"], "lever": ["beta"]}, board_client())
    )
    assert len(raw) == 4
    assert len(errors) == 1 and "nope" in errors[0]


# --- filters --------------------------------------------------------------------


def crit(**kw) -> filters.Criteria:
    return filters.Criteria(graduation=date(2028, 5, 1), **kw)


def raw(title, text="", **kw) -> sources.RawPosting:
    return sources.RawPosting(
        source="greenhouse", company="Acme", title=title, url="u", text=text, **kw
    )


def test_field_tagging_uses_whole_words():
    assert filters.tag_fields("ML Intern", "") == ["machine learning"]
    assert "machine learning" not in filters.tag_fields("HTML Intern", "")  # 'ml' inside a word
    assert set(filters.tag_fields("Intern", "genomics and CRISPR in the wet lab")) == {
        "biology",
        "biotech",
    }


@pytest.mark.parametrize(
    "posting, criteria, keep, reason",
    [
        (raw("Senior Accountant"), crit(), False, "not an internship"),
        (raw("ML Intern", "Unpaid internship"), crit(), False, "unpaid"),
        (raw("ML Intern", "Unpaid internship"), crit(paid_only=False), True, ""),
        (raw("ML Intern", "Must be graduating in 2025"), crit(), False, "graduation year"),
        (raw("ML Intern", "Graduating December 2027 or 2028"), crit(), True, ""),
        (raw("Biology Intern"), crit(fields=["machine learning"]), False, "field"),
        (
            raw("Biology Intern"),
            crit(fields=["machine learning"], include_companies=["acme"]),
            True,
            "",
        ),
        (raw("ML Intern"), crit(exclude_companies=["ACME"]), False, "excluded"),
        (raw("ML Intern", location="Austin, TX"), crit(locations=["Boston"]), False, "location"),
        (raw("ML Intern", location="Remote - US"), crit(locations=["Boston"]), True, ""),
        (raw("Research Assistant, Lab"), crit(kinds=["internship"]), False, "research not wanted"),
        (raw("ML Intern", terms=["Summer 2026"]), crit(term="Summer 2027"), False, "term"),
    ],
)
def test_judge(posting, criteria, keep, reason):
    verdict = filters.judge(posting, criteria)
    assert verdict.keep is keep
    assert reason in verdict.reason


def test_unknown_facts_keep_the_posting():
    v = filters.judge(raw("ML Intern", "No details."), crit())
    assert v.keep and v.paid is None


# --- ingest ---------------------------------------------------------------------


def test_ingest_saves_only_what_fits(profile_path):
    prof = profile.load(profile_path)
    summary = asyncio.run(ingest.ingest(prof, ingest.criteria_from(prof), board_client()))
    saved = {postings.get(i)["title"] for i in summary["posting_ids"]}
    assert saved == {
        "Machine Learning Intern, Summer 2027",
        "AI Research Intern",
        "Computational Biology Intern",
    }
    assert summary["dropped"]["not an internship or research position"] == 1
    assert summary["dropped"]["unpaid"] == 1
    assert any("graduation year" in r for r in summary["dropped"])  # the lever one says 2025
    # Re-running updates rather than duplicating.
    again = asyncio.run(ingest.ingest(prof, ingest.criteria_from(prof), board_client()))
    assert again["posting_ids"] == summary["posting_ids"]


def test_cap_keeps_the_most_relevant(profile_path):
    prof = profile.load(profile_path)
    c = ingest.criteria_from(
        prof, {"term": "Summer 2027", "include_companies": ["Delta Therapeutics"]}
    )
    summary = asyncio.run(ingest.ingest(prof, c, board_client(), max_results=1))
    [only] = summary["posting_ids"]
    assert postings.get(only)["company"] == "Delta Therapeutics"


def test_brief_answers_override_profile(profile_path):
    prof = profile.load(profile_path)
    c = ingest.criteria_from(prof, {"fields": ["biotech"], "term": "Summer 2027", "locations": []})
    assert c.fields == ["biotech"] and c.term == "Summer 2027"
    summary = asyncio.run(ingest.ingest(prof, c, board_client(), max_results=1))
    assert summary["saved"] == 1 and summary["dropped"]["over max_results"] >= 1


# --- voice/text tools -----------------------------------------------------------


def test_all_tool_schemas_build_with_unique_names():
    from jarvie.outreach.tools import OUTREACH_TOOLS

    # The same list bot.py registers with the voice model.
    every = [*tools.ALL_TOOLS, *JOB_TOOLS, *career_tools.CAREER_TOOLS, *OUTREACH_TOOLS]
    names = [s.name for s in LLMContext(tools=every).tools.standard_tools]
    assert len(names) == len(set(names))
    assert {"start_brief", "answer_brief", "list_postings", "add_posting", "draft_emails"} <= set(names)
    assert len(names) == 30  # 10 existing + 5 job + 9 career + 6 outreach


def test_brief_by_voice_prefills_from_profile(profile_path):
    started = call(start_brief, kind="discover_postings")
    assert started["prefilled"]["paid_only"] is True
    assert "term" in [q["key"] for q in started["questions"]]
    bid = started["brief_id"]
    assert (
        call(answer_brief_tool, brief_id=bid, answers={"term": "Summer 2027"})["remaining"][0][
            "required"
        ]
        is False
    )
    assert call(open_briefs)["briefs"][0]["brief_id"] == bid
    job_id = call(submit_brief, brief_id=bid)["job_id"]
    assert call(job_status, job_id=job_id)["job"]["status"] == "queued"
    assert call(job_status)["jobs"][0]["job_id"] == job_id


def test_brief_tool_errors_are_returned_not_raised(profile_path):
    assert "error" in call(start_brief, kind="apply_everywhere")
    bid = call(start_brief, kind="find_contacts")["brief_id"]
    assert "choose" in call(answer_brief_tool, brief_id=bid, answers={"purpose": "spam"})["error"]
    assert "still need" in call(submit_brief, brief_id=bid)["error"]


def test_posting_tools(profile_path):
    pid = postings.save(
        company="Acme", title="ML Intern", source="web", fields=["machine learning"]
    )
    listed = call(career_tools.list_postings, field="machine learning")
    assert listed["postings"][0]["posting_id"] == pid
    assert "raw_text" not in call(career_tools.get_posting, posting_id=pid)["posting"]
    assert call(career_tools.set_application_status, posting_id=pid, status="oa")["updated"]
    assert "error" in call(career_tools.set_application_status, posting_id=pid, status="ghosted")
    assert call(career_tools.set_coffee_chat, posting_id=pid)["updated"] is True
    assert call(career_tools.pipeline_summary)["by_status"]["oa"] == 1


def test_add_posting_queues_url_or_text():
    url_job = call(career_tools.add_posting, url_or_text="https://app.joinhandshake.com/jobs/1")
    text_job = call(career_tools.add_posting, url_or_text="Acme ML intern, paid, summer 2027 ...")
    assert jobs.get(url_job["job_id"])["params"] == {"url": "https://app.joinhandshake.com/jobs/1"}
    assert "text" in jobs.get(text_job["job_id"])["params"]


# --- dashboard ------------------------------------------------------------------


@pytest.fixture
def web(profile_path):
    return TestClient(create_app())


def test_dashboard_pages_render(web):
    pid = postings.save(company="Acme", title="ML Intern", source="web", url="https://a/1")
    for path in ("/", "/pipeline", "/briefs", "/jobs", f"/postings/{pid}"):
        r = web.get(path)
        assert r.status_code == 200, path
    assert "ML Intern" in web.get("/").text


def test_dashboard_status_and_brief_forms(web):
    pid = postings.save(company="Acme", title="ML Intern", source="web", url="https://a/1")
    r = web.post(f"/postings/{pid}/status", data={"status": "applied", "note": "via site"})
    assert r.status_code == 200 and postings.get(pid)["status"] == "applied"
    web.post(f"/postings/{pid}/coffee", data={"had_chat": "1"})
    assert postings.get(pid)["has_coffee_chat"] is True

    web.post("/briefs", data={"kind": "discover_postings"})
    [b] = briefs.open_briefs()
    assert b["last_channel"] == "dashboard"
    r = web.post(f"/briefs/{b['id']}/submit")
    assert "still need" in r.text  # error shown, nothing queued
    web.post(f"/briefs/{b['id']}/answer", data={"term": "Summer 2027", "notes": ""})
    web.post(f"/briefs/{b['id']}/submit")
    assert briefs.get(b["id"])["status"] == "submitted"
    assert jobs.recent()[0]["kind"] == "discover_postings"
    assert "discover_postings" in web.get("/jobs").text


def test_dashboard_escapes_html(web):
    postings.save(company="<script>x</script>", title="Intern", source="web", url="https://a/2")
    assert "<script>x</script>" not in web.get("/").text
