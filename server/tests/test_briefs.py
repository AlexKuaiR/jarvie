import pytest

from jarvie import briefs, jobs

pytestmark = pytest.mark.usefixtures("temp_db")


def test_voice_then_text_then_submit():
    bid = briefs.start("discover_postings")
    assert [q["key"] for q in briefs.remaining(bid)][:3] == ["kinds", "fields", "term"]

    # Started by voice...
    briefs.answer(bid, {"kinds": "both", "fields": "machine learning and biotech"}, "voice")
    # ...finished by typing in the localhost text box.
    brief = briefs.answer(
        bid,
        {"term": "Summer 2027", "locations": "none", "search_beyond_boards": "no", "ai_score": 3},
        "text",
    )
    assert brief["answers"]["kinds"] == ["internship", "research"]
    assert brief["answers"]["fields"] == ["machine learning", "biotech"]
    assert brief["answers"]["locations"] == []
    assert brief["last_channel"] == "text"

    job_id = briefs.submit(bid, "text")
    job = jobs.get(job_id)
    assert job["kind"] == "discover_postings"
    assert job["params"]["term"] == "Summer 2027"
    assert job["params"]["paid_only"] is True  # skipped optional -> default
    assert job["params"]["max_results"] == 25
    assert job["params"]["brief_id"] == bid
    assert briefs.get(bid)["status"] == "submitted"
    assert briefs.get(bid)["job_id"] == job_id


def test_submit_refused_until_required_answered():
    bid = briefs.start("discover_postings", prefill={"kinds": ["internship"]})
    with pytest.raises(briefs.BriefError, match="still need: fields, term"):
        briefs.submit(bid)
    assert jobs.recent() == []


def test_required_questions_are_asked_first():
    bid = briefs.start("find_contacts")
    keys = [q["key"] for q in briefs.remaining(bid)]
    assert keys[:3] == ["target", "roles", "purpose"]
    assert all(not q["required"] for q in briefs.remaining(bid)[3:])


def test_prefill_counts_as_answered_but_can_change():
    bid = briefs.start("discover_postings", prefill={"fields": ["biology"], "unrelated": 1})
    assert "fields" not in [q["key"] for q in briefs.remaining(bid)]
    briefs.answer(bid, {"fields": "biotech"})
    assert briefs.get(bid)["answers"]["fields"] == ["biotech"]
    assert "unrelated" not in briefs.get(bid)["answers"]


def test_contact_brief_without_alumni_filter():
    # Standalone outreach: professors, no school filter, no job search involved.
    bid = briefs.start("find_contacts")
    briefs.answer(
        bid,
        {
            "target": "UC Berkeley EECS",
            "roles": "professor",
            "purpose": "research inquiry",
            "schools": "no preference",
            "draft_after": "yes",
        },
    )
    job = jobs.get(briefs.submit(bid))
    assert job["kind"] == "find_contacts"
    assert job["params"]["schools"] == []
    assert job["params"]["purpose"] == "research_inquiry"
    assert job["params"]["draft_after"] is True
    assert job["params"]["max_contacts"] == 10


@pytest.mark.parametrize(
    "answers, message",
    [
        ({"kinds": "jobs"}, "choose from"),
        ({"paid_only": "maybe"}, "yes or no"),
        ({"min_prestige": "high"}, "number"),
        ({"min_prestige": 9}, "between 1 and 5"),
        ({"salary": 100}, "no such questions"),
    ],
)
def test_bad_answers_are_explained_and_change_nothing(answers, message):
    bid = briefs.start("discover_postings", prefill={"term": "Summer 2027"})
    with pytest.raises(briefs.BriefError, match=message):
        briefs.answer(bid, {"fields": "biology", **answers})
    assert briefs.get(bid)["answers"] == {"term": "Summer 2027"}


def test_single_choice_takes_one():
    bid = briefs.start("find_contacts")
    with pytest.raises(briefs.BriefError, match="just one"):
        briefs.answer(bid, {"purpose": "coffee chat, referral"})
    briefs.answer(bid, {"purpose": "Coffee chat"})
    assert briefs.get(bid)["answers"]["purpose"] == "coffee_chat"


def test_cancel_and_closed_briefs():
    bid = briefs.start("find_contacts")
    other = briefs.start("discover_postings")
    assert [b["id"] for b in briefs.open_briefs()] == [other, bid]
    briefs.cancel(bid)
    assert [b["id"] for b in briefs.open_briefs()] == [other]
    with pytest.raises(briefs.BriefError, match="already cancelled"):
        briefs.answer(bid, {"target": "x"})
    with pytest.raises(briefs.BriefError, match="no questions"):
        briefs.start("find_email")  # single lookups take parameters directly
