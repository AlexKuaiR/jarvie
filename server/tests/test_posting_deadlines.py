"""Posting deadlines: found in the text, shown when close, and past-due postings you
never applied to are hidden as expired (kept, not deleted)."""

from datetime import date, timedelta

import pytest
from career_fakes import call, write_profile
from fastapi.testclient import TestClient

from jarvie import config
from jarvie.career import postings
from jarvie.career import tools as career_tools
from jarvie.career.filters import find_deadline
from jarvie.dashboard import create_app

pytestmark = pytest.mark.usefixtures("temp_db")

TODAY = date(2026, 10, 30)


@pytest.mark.parametrize(
    "text, deadline",
    [
        ("Apply by December 11, 2026.", "2026-12-11"),
        ("Deadline: 12/11/2026", "2026-12-11"),
        ("Application deadline 2027-01-24", "2027-01-24"),
        ("Applications close Jan 15", "2027-01-15"),  # no year: the nearest Jan 15
        ("The deadline to apply is Oct 1st", "2026-10-01"),  # recent, so this year
        ("Start date: June 1, 2027. Posted Oct 2, 2026.", None),  # not deadlines
        ("Applications are due Feb 30", None),  # not a real date
        ("", None),
    ],
)
def test_find_deadline(text, deadline):
    assert find_deadline(text, TODAY) == deadline


def save(title, deadline=None, status=None, **kw):
    pid = postings.save(company="Acme", title=title, source="web", deadline=deadline, **kw)
    if status:
        postings.set_status(pid, status)
    return pid


def test_past_due_and_never_applied_expires():
    past = (date.today() - timedelta(days=3)).isoformat()
    soon = (date.today() + timedelta(days=5)).isoformat()
    untouched = save("Untouched", past)
    interested = save("Interested", past, "interested")
    applied = save("Applied", past, "applied")
    open_ = save("Still open", soon)
    unknown = save("No deadline")

    shown = {p["id"] for p in postings.search(limit=50)}
    assert shown == {applied, open_, unknown}  # expired ones are hidden from lists...
    assert {p["id"] for p in postings.search(status="expired")} == {untouched, interested}
    assert postings.get(untouched)["status"] == "expired"  # ...but kept, not deleted
    assert "never applied" in postings.history(untouched)[-1]["note"]


def test_naming_an_expired_posting_still_finds_it():
    pid = save("Quant Research Intern", (date.today() - timedelta(days=1)).isoformat())
    found = postings.search(query="acme quant")
    assert [p["id"] for p in found] == [pid] and found[0]["status"] == "expired"
    postings.set_status(pid, "applied")  # "I did apply to that one" brings it back
    assert pid in {p["id"] for p in postings.search()}


def test_voice_list_has_days_left():
    save("Soon", (date.today() + timedelta(days=4)).isoformat())
    result = call(career_tools.list_postings)
    assert result["postings"][0]["days_left"] == 4


def test_fill_deadlines_from_saved_text():
    pid = save("SWE Intern", raw_text="Great team. Apply by December 11, 2026.")
    other = save("Data Intern", raw_text="No date here.")
    assert postings.fill_deadlines(TODAY) == [pid]
    assert postings.get(pid)["deadline"] == "2026-12-11"
    assert postings.get(other)["deadline"] is None


def test_a_refetch_without_a_date_keeps_the_deadline():
    pid = postings.save(company="Acme", title="Intern", source="greenhouse", url="https://a/1")
    postings.update(pid, deadline="2026-12-11")
    postings.save(company="Acme", title="Intern", source="greenhouse", url="https://a/1")
    assert postings.get(pid)["deadline"] == "2026-12-11"


def test_dashboard_hides_expired_and_flags_soon(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CAREER_PROFILE", write_profile(tmp_path))
    save("Gone Intern", (date.today() - timedelta(days=2)).isoformat())
    save("Soon Intern", (date.today() + timedelta(days=3)).isoformat())
    client = TestClient(create_app())
    page = client.get("/").text
    assert "Soon Intern" in page and "in 3 days" in page
    assert "Gone Intern" not in page and "1 expired" in page
    assert "Gone Intern" in client.get("/?status=expired").text
    assert "Gone Intern" not in client.get("/pipeline").text
