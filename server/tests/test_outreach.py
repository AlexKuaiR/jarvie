import pytest

from jarvie.outreach import contacts, messages

pytestmark = pytest.mark.usefixtures("temp_db")


def add_contact(**overrides) -> int:
    fields = {
        "name": "Dana Recruiter",
        "evidence_urls": ["https://acme.bio/team"],
        "company": "Acme Bio",
        "role": "University Recruiter",
        "is_recruiter_or_admissions": True,
        "alumni_of": ["berkeley"],
        "email": "Dana@Acme.bio",
        "features": {"school": "Berkeley", "major": "MCB"},
    }
    fields.update(overrides)
    return contacts.add(**fields)


# --- contacts -------------------------------------------------------------------


def test_contact_round_trip_and_email_dedupe():
    cid = add_contact()
    c = contacts.get(cid)
    assert c["email"] == "dana@acme.bio"
    assert c["alumni_of"] == ["berkeley"]
    assert c["features"]["major"] == "MCB"
    assert c["is_recruiter_or_admissions"] is True
    assert c["source"] == "agent"
    assert add_contact(email="dana@acme.bio ") == cid
    assert [x["id"] for x in contacts.search(school="berkeley")] == [cid]
    assert contacts.search(school="nnhs") == []
    assert [x["id"] for x in contacts.search(company="acme bio")] == [cid]


def test_standalone_contact_needs_no_company_or_posting():
    # Added by you, e.g. a professor you met: no evidence URL, no company, no job search.
    cid = contacts.add("Prof. Lee", source="user", email="lee@berkeley.edu")
    c = contacts.get(cid)
    assert c["company"] is None
    assert c["evidence_urls"] == []
    assert c["email_source"] == "user"
    oid = messages.add_draft(cid, "Research question", "Hi Professor Lee, ...", "research_inquiry")
    assert messages.get(oid)["posting_id"] is None


@pytest.mark.parametrize(
    "bad",
    [{"evidence_urls": []}, {"name": " "}, {"alumni_of": ["stanford"]}, {"source": "guess"}],
)
def test_agent_contacts_require_evidence_and_known_values(bad):
    with pytest.raises(ValueError):
        add_contact(**bad)


def test_set_email_later_and_has_email_filter():
    cid = add_contact(email=None)
    assert [c["id"] for c in contacts.search(has_email=False)] == [cid]
    assert contacts.set_email(cid, "Dana@acme.bio", "hunter", "valid") is True
    c = contacts.get(cid)
    assert (c["email"], c["email_source"], c["email_verified"]) == (
        "dana@acme.bio",
        "hunter",
        "valid",
    )
    assert [c["id"] for c in contacts.search(has_email=True)] == [cid]

    other = add_contact(name="Someone Else", email=None)
    with pytest.raises(ValueError, match="already belongs"):
        contacts.set_email(other, "dana@acme.bio", "hunter", "valid")


# --- messages -------------------------------------------------------------------


def test_outreach_lifecycle():
    cid = add_contact()
    oid = messages.add_draft(cid, "Berkeley MCB student", "Hi Dana, ...", posting_id=7)
    messages.edit_draft(oid, body="Hi Dana, shorter.")
    messages.set_status(oid, "approved")
    with pytest.raises(messages.OutreachRefused):
        messages.edit_draft(oid, body="too late")
    messages.set_status(oid, "scheduled", scheduled_at="2026-11-02T10:00:00-08:00")
    messages.set_status(oid, "sent", sent_at="2026-11-02T10:00:05-08:00")
    messages.set_status(oid, "replied")
    item = messages.get(oid)
    assert item["status"] == "replied"
    assert item["purpose"] == "coffee_chat"
    assert item["posting_id"] == 7
    assert item["body"] == "Hi Dana, shorter."


def test_cannot_skip_approval_or_use_unknown_purpose():
    cid = add_contact()
    with pytest.raises(ValueError):
        messages.add_draft(cid, "s", "b", purpose="spam")
    oid = messages.add_draft(cid, "s", "b")
    with pytest.raises(messages.OutreachRefused, match="drafted to sent"):
        messages.set_status(oid, "sent")


def test_one_live_thread_per_contact():
    cid = add_contact()
    first = messages.add_draft(cid, "s", "b")
    with pytest.raises(messages.OutreachRefused, match="already has an email"):
        messages.add_draft(cid, "s2", "b2")
    messages.set_status(first, "stopped")
    messages.add_draft(cid, "s2", "b2")  # allowed once the first is stopped


def test_do_not_contact_blocks_and_stops_queued_mail():
    cid = add_contact()
    oid = messages.add_draft(cid, "s", "b")
    messages.set_status(oid, "approved")
    assert contacts.mark_do_not_contact(cid) is True
    assert messages.get(oid)["status"] == "stopped"
    with pytest.raises(messages.OutreachRefused, match="do-not-contact"):
        messages.add_draft(cid, "s", "b")


def test_draft_needs_email_address():
    cid = add_contact(email=None)
    with pytest.raises(messages.OutreachRefused, match="no email"):
        messages.add_draft(cid, "s", "b")


def test_queue_joins_contact_details():
    cid = add_contact()
    messages.add_draft(cid, "s", "b")
    [item] = messages.queue("drafted")
    assert item["name"] == "Dana Recruiter"
    assert item["email"] == "dana@acme.bio"
    assert messages.queue("sent") == []
