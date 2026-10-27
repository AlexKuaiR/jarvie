"""Phase 5: cover letters and essays, versions, flagged claims, tools and pages."""

import pytest
from agent_fakes import FakeAgent
from career_fakes import call, write_profile
from fastapi.testclient import TestClient

from jarvie import config, jobs, worker
from jarvie.career import documents, postings, profile
from jarvie.career import tools as career_tools
from jarvie.dashboard import create_app

pytestmark = pytest.mark.usefixtures("temp_db")

LETTER = " ".join(["word"] * 280)


@pytest.fixture
def prof(tmp_path, monkeypatch):
    path = write_profile(tmp_path)
    monkeypatch.setattr(config, "CAREER_PROFILE", path)
    return profile.load(path)


@pytest.fixture
def posting():
    return postings.save(
        company="Acme Bio",
        title="ML Intern",
        source="greenhouse",
        url="https://a/1",
        raw_text="Train protein models.",
    )


def letter_script(claims):
    def script(prompt, tools):
        return tools["save_document"](
            text=LETTER, claims_about_student=claims, research_notes="Acme builds protein models."
        )

    return script


def run(agent, prof, **params):
    job_id = jobs.enqueue("write_document", params)
    worker.run_pending(worker.WorkerContext(agent=agent, profile_loader=lambda: prof))
    return jobs.get(job_id)


def test_cover_letter_saved_with_unsupported_claims_flagged(prof, posting):
    claims = [
        "Built a protein structure model in PyTorch.",  # in the resume
        "Won first place at a national hackathon.",  # not in the resume
        "Improved accuracy by 37%.",  # a number the resume doesn't have
    ]
    agent = FakeAgent(letter_script(claims))
    job = run(agent, prof, posting_id=posting, kind="cover_letter")
    assert job["status"] == "done", job["error"]
    d = documents.get(job["result"]["document_id"])
    assert d["version"] == 1 and d["research_notes"].startswith("Acme")
    assert d["flagged"] == claims[1:]
    assert "Alex K" in agent.calls[0]["system"][0]["text"]  # resume in cached system prompt
    assert agent.calls[0]["web_fetch"] == 3


def test_versions_count_per_kind_and_prompt(prof, posting):
    agent = FakeAgent(letter_script([]))
    run(agent, prof, posting_id=posting, kind="cover_letter")
    second = run(agent, prof, posting_id=posting, kind="cover_letter")
    essay = run(agent, prof, posting_id=posting, kind="essay", question="Why Acme?")
    assert second["result"]["version"] == 2
    assert essay["result"]["version"] == 1


def test_essay_word_limit_is_enforced_through_the_tool(prof, posting):
    replies = []

    def script(prompt, tools):
        replies.append(tools["save_document"](text=LETTER, claims_about_student=[]))
        replies.append(tools["save_document"](text="short answer " * 50, claims_about_student=[]))

    job = run(
        FakeAgent(script), prof, posting_id=posting, kind="essay", question="Why?", word_limit=150
    )
    assert replies[0].startswith("error: 280 words")
    assert job["status"] == "done"
    assert len(documents.get(job["result"]["document_id"])["text"].split()) == 100


@pytest.mark.parametrize(
    "params, message",
    [
        ({"kind": "essay"}, "prompt"),
        ({"kind": "poem"}, "kind"),
        ({"posting_id": 999}, "no posting"),
    ],
)
def test_bad_requests_fail_clearly(prof, posting, params, message):
    job = run(FakeAgent(letter_script([])), prof, **{"posting_id": posting, **params})
    assert job["status"] == "failed" and message in job["error"]


def test_writer_that_saves_nothing_fails_the_job(prof, posting):
    job = run(FakeAgent(lambda p, t: "I couldn't"), prof, posting_id=posting, kind="cover_letter")
    assert "didn't save a document" in job["error"]


def test_unsupported_claims():
    resume = "Python, PyTorch. Protein structure model. Dean's list 2025."
    assert documents.unsupported(["Built a protein structure model"], resume) == []
    assert documents.unsupported(["Made the Dean's list in 2025"], resume) == []
    assert documents.unsupported(["Made the Dean's list in 2024"], resume) == [
        "Made the Dean's list in 2024"
    ]


def test_document_tools(posting):
    job = call(career_tools.write_essay, posting_id=posting, question="Why us?", word_limit=200)
    assert jobs.get(job["job_id"])["params"]["word_limit"] == 200
    call(career_tools.write_cover_letter, posting_id=posting)
    documents.add_version(posting, "cover_letter", "Dear team, ...", flagged=["x"])
    [d] = call(career_tools.list_documents)["documents"]
    assert (d["company"], d["flagged_claims"]) == ("Acme Bio", 1)


def test_document_pages(prof, posting):
    web = TestClient(create_app())
    doc = documents.add_version(posting, "cover_letter", "Dear team at Acme", flagged=["Won X"])
    assert "Won X" in web.get(f"/documents/{doc}").text
    assert "cover_letter" in web.get("/documents").text
    assert "cover_letter v1" in web.get(f"/postings/{posting}").text
    assert "prompt" in web.post(f"/postings/{posting}/documents", data={"kind": "essay"}).text
    web.post(
        f"/postings/{posting}/documents",
        data={"kind": "essay", "question": "Why?", "word_limit": "250"},
    )
    assert jobs.recent()[0]["params"]["word_limit"] == 250
