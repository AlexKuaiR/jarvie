"""Starting a brief in the same call, never claiming a start that didn't happen, and
the resume answer for contact searches (one resume, the default, or one per person)."""

import pytest
from agent_fakes import FakeWriter
from career_fakes import call
from test_career_resumes import jobs_db_cleanup, write_multi

from jarvie import briefs, config, jobs
from jarvie.job_tools import answer_brief, start_brief
from jarvie.llm import Usage
from jarvie.outreach import contacts, writer
from jarvie.outreach.writer import DraftEmail, Personalization

pytestmark = pytest.mark.usefixtures("temp_db")

# The search from the bug report: answered by voice, then "I'll start searching now".
CONTACT_SEARCH = {
    "target": "Amplify and Sanofi",
    "roles": "recruiters, managers",
    "purpose": "coffee_chat",
    "max_contacts": 2,
    "draft_after": "yes",
}


@pytest.fixture
def prof(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "CAREER_PROFILE", write_multi(tmp_path))


# --- starting ---------------------------------------------------------------------


def test_answering_with_start_queues_the_job(prof):
    bid = call(start_brief, kind="find_contacts")["brief_id"]
    result = call(answer_brief, brief_id=bid, answers=CONTACT_SEARCH, start=True)
    assert result["started"] is True
    job = jobs.get(result["job_id"])
    assert job["kind"] == "find_contacts" and job["params"]["draft_after"] is True
    assert briefs.get(bid)["status"] == "submitted"


def test_start_brief_can_start_in_one_call(prof):
    result = call(start_brief, kind="find_contacts", answers=CONTACT_SEARCH, start=True)
    assert result["started"] is True and jobs.get(result["job_id"])


def test_complete_but_not_asked_to_start_says_so(prof):
    bid = call(start_brief, kind="find_contacts")["brief_id"]
    result = call(answer_brief, brief_id=bid, answers=CONTACT_SEARCH)
    assert result["started"] is False
    assert "Not started yet" in result["tell_user"]
    assert jobs.recent() == [] and briefs.get(bid)["status"] == "open"


def test_start_waits_for_required_answers(prof):
    bid = call(start_brief, kind="find_contacts")["brief_id"]
    result = call(answer_brief, brief_id=bid, answers={"target": "Sanofi"}, start=True)
    assert result["started"] is False and "roles, purpose" in result["tell_user"]
    assert jobs.recent() == []


def test_a_bad_answer_never_starts(prof):
    answers = {**CONTACT_SEARCH, "max_contacts": "lots"}
    result = call(start_brief, kind="find_contacts", answers=answers, start=True)
    assert "error" in result and result["started"] is False
    assert jobs.recent() == []


# --- the resume answer --------------------------------------------------------------


@pytest.mark.parametrize(
    "said, stored",
    [
        ("swe", "swe"),
        ("the MLE one", "mle"),
        ("my quant resume", "qr"),
        ("default", "default"),
        ("doesn't matter", "default"),
        ("match each person", "auto"),
        (
            "use software engineering resume for software roles; machine learning "
            "engineering resume for AI or machine learning roles",
            "auto",
        ),
    ],
)
def test_resume_answers(prof, said, stored):
    bid = briefs.start("find_contacts")
    assert briefs.answer(bid, {"resume": said})["answers"]["resume"] == stored


def test_an_unknown_resume_is_explained(prof):
    bid = briefs.start("find_contacts")
    with pytest.raises(briefs.BriefError, match="swe, mle, qr"):
        briefs.answer(bid, {"resume": "the blue one"})


def test_auto_resume_picks_per_person(prof):
    ml = contacts.add(
        name="Dana Lee",
        evidence_urls=["https://x/team"],
        source="user",
        company="Sanofi",
        role="Machine Learning Engineering Manager",
        email="dana@sanofi.com",
        features={"team": "AI for drug discovery"},
    )
    swe = contacts.add(
        name="Sam Ortiz",
        evidence_urls=["https://y/team"],
        source="user",
        company="Amplify",
        role="Software Engineering Manager",
        email="sam@amplify.com",
        features={"team": "platform"},
    )

    def make(prompt, schema):
        return DraftEmail(
            subject="Hi",
            body=f"Hi {prompt.split()[1]}, " + "word " * 100 + "Best, Alex Kuai",
            personalization_used=[Personalization(feature_key="team", how_used="x")],
        )

    fake = FakeWriter(make)
    job = {"id": 1, "params": {"contact_ids": [ml, swe], "resume": "auto"}}
    writer.draft_outreach(job, fake, Usage())
    dana_prompt, sam_prompt = fake.calls
    assert "PyTorch" in dana_prompt and "Go and Python" not in dana_prompt  # mle
    assert "Go and Python" in sam_prompt  # swe
    jobs_db_cleanup(ml)
    jobs_db_cleanup(swe)
