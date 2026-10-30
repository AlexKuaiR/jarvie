"""Working on emails together: a summary is written first and the email from it, Jarvie
rewrites drafts on request (keeping every version), and facts you add about a person
are what emails may use."""

import pytest
from agent_fakes import SUMMARY, FakeWriter
from career_fakes import call
from fastapi.testclient import TestClient

from jarvie import jobs, worker
from jarvie.dashboard import create_app
from jarvie.outreach import contacts, messages
from jarvie.outreach import tools as outreach_tools
from jarvie.outreach.writer import DraftEmail, Personalization, Sender

pytestmark = pytest.mark.usefixtures("temp_db")

ME = Sender("Alex Kuai", "UC Berkeley EECS '28. Built a protein structure model in PyTorch.")


def saranya() -> int:
    return contacts.add(
        name="Saranya Sridhar",
        evidence_urls=["https://terrapinn.com/speaker"],
        company="Sanofi",
        role="Head of Global Clinical Development for Vaccines",
        alumni_of=["berkeley"],
        email="saranya.sridhar@sanofi.com",
        features={"education": "MPH, UC Berkeley", "focus": "new vaccine platforms"},
    )


def email(body_extra: str = "", used=("education",), subject="Berkeley student, quick question"):
    body = (
        f"Hi Saranya, I'm a Berkeley EECS student who builds protein models in PyTorch. "
        f"{body_extra}" + "I'd love to hear how your team works. " * 12 + "Best, Alex"
    )
    return DraftEmail(
        subject=subject,
        body=body,
        personalization_used=[Personalization(feature_key=k, how_used="x") for k in used],
    )


def run(fake: FakeWriter) -> None:
    worker.run_pending(worker.WorkerContext(writer=fake, extra={"sender": ME}))


def draft_one(cid: int) -> int:
    fake = FakeWriter(lambda p, s: email())
    job = jobs.enqueue("draft_outreach", {"contact_ids": [cid], "resume": "mle"})
    run(fake)
    assert fake.summary_calls and "Sanofi" in fake.summary_calls[0]  # planned first...
    assert "Gist: " + SUMMARY["gist"] in fake.calls[0]  # ...then written from the plan
    return jobs.get(job)["result"]["drafted"][0]


def test_a_draft_keeps_its_summary_and_first_version():
    mid = draft_one(saranya())
    m = messages.get(mid)
    assert m["summary"]["gist"] == SUMMARY["gist"] and m["resume"] == "mle"
    assert [v["by"] for v in messages.revisions(mid)] == ["jarvie"]


def test_rewrite_follows_the_request_and_keeps_your_edits():
    mid = draft_one(saranya())
    messages.edit_draft(mid, body=messages.get(mid)["body"] + " P.S. I'm free Fridays.")

    def summarize(prompt):
        assert "I'm free Fridays" in prompt and "lead with her vaccine work" in prompt
        return {**SUMMARY, "angle": "Her work on new vaccine platforms"}

    fake = FakeWriter(
        lambda p, s: email("Your vaccine platform work caught my eye. "), summarize=summarize
    )
    job = jobs.enqueue(
        "revise_outreach", {"outreach_id": mid, "instruction": "lead with her vaccine work"}
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("jarvie.outreach.writer.load_sender", lambda *a, **k: ME)
        run(fake)
    assert jobs.get(job)["result"]["revised"] is True
    m = messages.get(mid)
    assert "vaccine platform" in m["body"] and m["summary"]["angle"].startswith("Her work")
    assert "I'm free Fridays" in fake.calls[0]  # the rewrite saw your edit
    versions = messages.revisions(mid)
    assert [v["by"] for v in versions] == ["jarvie", "you", "jarvie"]
    assert versions[-1]["instruction"] == "lead with her vaccine work"


def test_a_rewrite_using_unstored_facts_is_refused_and_changes_nothing(monkeypatch):
    mid = draft_one(saranya())
    before = messages.get(mid)["body"]
    monkeypatch.setattr("jarvie.outreach.writer.load_sender", lambda *a, **k: ME)
    fake = FakeWriter(lambda p, s: email("Loved your podcast! ", used=("hobbies",)))
    job = jobs.enqueue("revise_outreach", {"outreach_id": mid, "instruction": "be warmer"})
    run(fake)
    result = jobs.get(job)["result"]
    assert result["revised"] is False and any("isn't stored" in p for p in result["problems"])
    assert messages.get(mid)["body"] == before
    assert len(messages.revisions(mid)) == 1


def test_restore_an_earlier_version():
    mid = draft_one(saranya())
    first = messages.revisions(mid)[0]
    messages.edit_draft(mid, subject="Changed my mind")
    messages.restore(mid, first["id"])
    assert messages.get(mid)["subject"] == first["subject"]
    assert len(messages.revisions(mid)) == 3  # the restore is a version too, so it's undoable


def test_facts_you_add_reach_the_writer_and_the_check():
    cid = saranya()
    call(outreach_tools.add_contact_fact, contact_id=cid, fact="We met at the career fair")
    assert contacts.get(cid)["features"]["from_you"] == "We met at the career fair"
    fake = FakeWriter(lambda p, s: email("It was great meeting you. ", used=("from_you",)))
    jobs.enqueue("draft_outreach", {"contact_ids": [cid]})
    run(fake)
    assert "We met at the career fair" in fake.summary_calls[0]
    assert messages.queue("drafted")  # using the new fact passed the grounding check


def test_your_contact_notes_are_facts_too():
    cid = contacts.add("Sam Lee", source="user", email="sam@x.com", notes="My CS 61B TA")
    fake = FakeWriter(
        lambda p, s: DraftEmail(
            subject="Hi Sam",
            body="Hi Sam, " + "word " * 80 + "Best, Alex",
            personalization_used=[Personalization(feature_key="your_notes", how_used="x")],
        )
    )
    jobs.enqueue("draft_outreach", {"contact_ids": [cid]})
    run(fake)
    assert "My CS 61B TA" in fake.calls[0] and messages.queue("drafted")


def test_voice_summary_and_revise_tools():
    mid = draft_one(saranya())
    s = call(outreach_tools.email_summary, outreach_id=mid)
    assert s["summary"]["gist"] == SUMMARY["gist"] and s["known_about_them"]["focus"]
    queued = call(outreach_tools.outreach_queue)["drafts"][0]
    assert queued["gist"] == SUMMARY["gist"]
    r = call(outreach_tools.revise_email, outreach_id=mid, instruction="shorter")
    assert jobs.get(r["job_id"])["kind"] == "revise_outreach"
    messages.set_status(mid, "approved")
    assert "error" in call(outreach_tools.revise_email, outreach_id=mid, instruction="x")


def test_dashboard_shows_summary_facts_and_rewrite():
    cid = saranya()
    mid = draft_one(cid)
    client = TestClient(create_app())
    page = client.get("/outreach").text
    assert SUMMARY["gist"] in page and "new vaccine platforms" in page and "Rewrite" in page
    client.post(f"/outreach/{mid}/revise", data={"instruction": "shorter"})
    assert "Rewriting" in client.get("/outreach").text
    client.post(f"/contacts/{cid}/fact", data={"fact": "Runs the respiratory team"})
    assert "Runs the respiratory team" in client.get("/outreach").text
