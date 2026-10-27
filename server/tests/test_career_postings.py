import pytest

from jarvie.career import postings

pytestmark = pytest.mark.usefixtures("temp_db")


def add_posting(**overrides) -> int:
    fields = {
        "company": "Acme Bio",
        "title": "ML Intern",
        "source": "greenhouse",
        "url": "https://boards.greenhouse.io/acme/jobs/1",
        "fields": ["machine learning", "biotech"],
        "paid": True,
        "fit_score": 0.8,
    }
    fields.update(overrides)
    return postings.save(**fields)


def test_save_and_get_round_trips_lists_and_flags():
    pid = add_posting(skills=["python", "pytorch"], process=["oa", "essays"], prestige=4)
    p = postings.get(pid)
    assert p["fields"] == ["machine learning", "biotech"]
    assert p["skills"] == ["python", "pytorch"]
    assert p["process"] == ["oa", "essays"]
    assert p["paid"] is True
    assert p["prestige"] == 4
    assert p["status"] is None
    assert p["has_coffee_chat"] is False


def test_same_url_updates_without_erasing_other_fields():
    pid = add_posting(location="Berkeley, CA")
    again = postings.save(
        company="Acme Bio",
        title="ML Intern",
        source="greenhouse",
        url="https://boards.greenhouse.io/acme/jobs/1",
        fit_score=0.9,
    )
    assert again == pid
    p = postings.get(pid)
    assert p["fit_score"] == 0.9
    assert p["location"] == "Berkeley, CA"


def test_postings_without_url_are_separate():
    a = postings.save(company="Lab A", title="RA", source="handshake", kind="research")
    b = postings.save(company="Lab A", title="RA", source="handshake", kind="research")
    assert a != b


@pytest.mark.parametrize(
    "bad",
    [{"kind": "job"}, {"prestige": 6}, {"ownership": 0}, {"salary": 10}, {"company": ""}],
)
def test_save_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        add_posting(**bad)


def test_search_filters_and_sorts():
    ml = add_posting(fit_score=0.5)
    bio = add_posting(
        url="https://x/2", fields=["biology"], kind="research", fit_score=0.9, deadline="2026-12-01"
    )
    swe = add_posting(url="https://x/3", fields=["software engineering"], fit_score=None)

    assert [p["id"] for p in postings.search()] == [bio, ml, swe]  # unscored last
    assert [p["id"] for p in postings.search(field="biology")] == [bio]
    assert [p["id"] for p in postings.search(kind="research")] == [bio]
    assert [p["id"] for p in postings.search(min_fit=0.6)] == [bio]
    assert [p["id"] for p in postings.search(sort="deadline")][0] == bio

    postings.set_status(ml, "applied")
    assert [p["id"] for p in postings.search(status="applied")] == [ml]
    assert {p["id"] for p in postings.search(status="none")} == {bio, swe}

    with pytest.raises(ValueError):
        postings.search(sort="random")


def test_status_history_and_summary():
    pid = add_posting()
    postings.set_status(pid, "applied")
    postings.set_status(pid, "oa", note="HackerRank due Friday")
    postings.set_status(pid, "interviewing")
    assert postings.get(pid)["status"] == "interviewing"
    assert [e["status"] for e in postings.history(pid)] == ["applied", "oa", "interviewing"]
    assert postings.history(pid)[1]["note"] == "HackerRank due Friday"

    assert postings.set_coffee_chat(pid) is True
    summary = postings.pipeline_summary()
    assert summary["by_status"]["interviewing"] == 1
    assert summary["by_status"]["rejected"] == 0
    assert summary["coffee_chats"] == 1


def test_set_status_validates():
    pid = add_posting()
    with pytest.raises(ValueError):
        postings.set_status(pid, "ghosted")
    with pytest.raises(ValueError):
        postings.set_status(999, "applied")
    assert postings.set_coffee_chat(pid) is False  # no application yet
