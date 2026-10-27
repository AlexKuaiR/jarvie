import pytest

from jarvie import papers

pytestmark = pytest.mark.usefixtures("temp_db")


def ids(found: list[papers.Paper]) -> list[str]:
    return [p.paper_id for p in found]


def add_spindles() -> str:
    return papers.add(
        title="Sleep spindles and memory consolidation",
        authors=["M. Walker", "R. Stickgold"],
        year=2009,
        abstract="Spindles during NREM stage 2 predict overnight memory gains.",
        doi="10.1000/spindles",
    )


def test_add_then_get_round_trips_fields():
    paper_id = add_spindles()
    assert paper_id == "manual:1"
    p = papers.get(paper_id)
    assert p.title == "Sleep spindles and memory consolidation"
    assert p.authors == ["M. Walker", "R. Stickgold"]
    assert p.year == 2009
    assert p.source == "manual"
    assert p.notes is None
    assert p.added_at is not None


def test_get_missing_is_none():
    assert papers.get("manual:99") is None


def test_add_requires_title():
    with pytest.raises(ValueError):
        papers.add(title="  ")


def test_same_doi_is_not_added_twice():
    first = add_spindles()
    again = papers.add(title="Different title", doi="https://doi.org/10.1000/SPINDLES")
    assert again == first
    assert len(papers.list_recent()) == 1


def test_by_doi_normalizes():
    paper_id = add_spindles()
    assert papers.by_doi("doi:10.1000/Spindles").paper_id == paper_id
    assert papers.by_doi("") is None


def test_explicit_paper_id_for_other_sources():
    paper_id = papers.add(title="An S2 paper", source="s2", paper_id="abc123")
    assert paper_id == "abc123"
    assert papers.get("abc123").source == "s2"


def test_search_matches_title_abstract_and_authors():
    spindles = add_spindles()
    other = papers.add(title="Circadian phase and caffeine", authors=["T. Roenneberg"])
    assert ids(papers.search("spindles")) == [spindles]
    assert ids(papers.search("NREM")) == [spindles]
    assert ids(papers.search("Roenneberg")) == [other]


def test_search_tolerates_spoken_queries_and_syntax():
    spindles = add_spindles()
    # Extra words and FTS5 syntax characters shouldn't break or empty the search.
    assert ids(papers.search("papers on sleep-spindles: memory?")) == [spindles]
    assert papers.search("   ") == []
    assert papers.search("quantum") == []


def test_search_respects_limit():
    for n in range(4):
        papers.add(title=f"Sleep study {n}")
    assert len(papers.search("sleep", limit=2)) == 2


def test_update_notes_and_search_index_follow_changes():
    paper_id = add_spindles()
    assert papers.update_notes(paper_id, "small sample") is True
    assert papers.get(paper_id).notes == "small sample"
    assert papers.update_notes("manual:99", "x") is False


def test_delete_removes_from_store_and_search():
    paper_id = add_spindles()
    assert papers.delete(paper_id) is True
    assert papers.get(paper_id) is None
    assert papers.search("spindles") == []
    assert papers.delete(paper_id) is False


def test_list_recent_newest_first():
    first = papers.add(title="First")
    second = papers.add(title="Second")
    assert ids(papers.list_recent()) == [second, first]
    assert ids(papers.list_recent(limit=1)) == [second]
