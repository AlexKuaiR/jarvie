import asyncio

import httpx
import pytest

from jarvie import literature, papers, tools

pytestmark = pytest.mark.usefixtures("temp_db")


class FakeParams:
    """Stands in for Pipecat's FunctionCallParams (same as in test_tools.py)."""

    def __init__(self):
        self.result = None

    async def result_callback(self, result):
        self.result = result


def call(tool, **kwargs) -> dict:
    params = FakeParams()
    asyncio.run(tool(params, **kwargs))
    return params.result


SPINDLES = {
    "paperId": "abc123",
    "title": "Sleep spindles and memory consolidation",
    "authors": [{"authorId": "1", "name": "M. Walker"}, {"authorId": "2", "name": "R. Stickgold"}],
    "year": 2009,
    "venue": "Neuron",
    "abstract": "Spindles during NREM stage 2 predict overnight memory gains.",
    "externalIds": {"DOI": "10.1000/Spindles"},
    "url": "https://www.semanticscholar.org/paper/abc123",
}


class FakeS2:
    """Answers S2 requests from canned data and records what was asked."""

    def __init__(self, status: int = 200):
        self.status = status
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.status != 200:
            return httpx.Response(self.status, text="nope")
        path = request.url.path
        if path.endswith("/paper/search"):
            return httpx.Response(200, json={"total": 1, "data": [SPINDLES, {"paperId": "x"}]})
        if path.endswith("/paper/abc123") or path.endswith("/paper/DOI:10.1000/spindles"):
            return httpx.Response(200, json=SPINDLES)
        return httpx.Response(404, json={"error": "not found"})

    def source(self, api_key: str | None = None) -> literature.SemanticScholarSource:
        client = httpx.AsyncClient(transport=httpx.MockTransport(self.handler))
        source = literature.SemanticScholarSource(api_key=api_key, client=client)
        source.retry_delay = 0
        return source


@pytest.fixture
def s2(monkeypatch) -> FakeS2:
    """Point the tools at a fake Semantic Scholar."""
    fake = FakeS2()
    monkeypatch.setattr(literature, "get_source", fake.source)
    return fake


# --- SemanticScholarSource ------------------------------------------------------


def test_search_parses_and_caches_results():
    fake = FakeS2()
    found = asyncio.run(fake.source().search("spindles", limit=3, year_from=2005))

    # The malformed second result (no title) is dropped.
    assert [p.paper_id for p in found] == ["abc123"]
    p = found[0]
    assert p.authors == ["M. Walker", "R. Stickgold"]
    assert p.doi == "10.1000/spindles"
    assert p.source == "s2"
    assert papers.get("abc123") is not None

    params = fake.requests[0].url.params
    assert params["query"] == "spindles"
    assert params["limit"] == "3"
    assert params["year"] == "2005-"
    assert "x-api-key" not in fake.requests[0].headers


def test_api_key_is_sent_when_set():
    fake = FakeS2()
    asyncio.run(fake.source(api_key="secret").search("spindles"))
    assert fake.requests[0].headers["x-api-key"] == "secret"


def test_repeat_search_keeps_notes():
    fake = FakeS2()
    asyncio.run(fake.source().search("spindles"))
    papers.update_notes("abc123", "small sample")
    found = asyncio.run(fake.source().search("spindles"))
    assert found[0].notes == "small sample"


def test_get_uses_cache_before_network():
    fake = FakeS2()
    p = asyncio.run(fake.source().get("abc123"))
    assert p.title == SPINDLES["title"]
    asyncio.run(fake.source().get("abc123"))
    assert len(fake.requests) == 1


def test_get_by_doi_normalizes_and_fetches():
    fake = FakeS2()
    p = asyncio.run(fake.source().get_by_doi("https://doi.org/10.1000/SPINDLES"))
    assert p.paper_id == "abc123"
    assert fake.requests[0].url.path.endswith("/paper/DOI:10.1000/spindles")
    # Second lookup is answered from the local cache.
    asyncio.run(fake.source().get_by_doi("10.1000/spindles"))
    assert len(fake.requests) == 1


def test_unknown_paper_is_none():
    fake = FakeS2()
    assert asyncio.run(fake.source().get("missing")) is None
    assert asyncio.run(fake.source().get_by_doi("10.9/missing")) is None
    assert asyncio.run(fake.source().get_by_doi("  ")) is None


@pytest.mark.parametrize("status, message", [(429, "rate limiting"), (500, r"error \(500\)")])
def test_http_errors_raise_source_error(status, message):
    with pytest.raises(literature.SourceError, match=message):
        asyncio.run(FakeS2(status=status).source().search("spindles"))


def test_rate_limit_retries_once():
    fake = FakeS2(status=429)
    with pytest.raises(literature.SourceError):
        asyncio.run(fake.source().search("spindles"))
    assert len(fake.requests) == 2


def test_rate_limit_then_success():
    fake = FakeS2()
    statuses = iter([429, 200])

    def flaky(request):
        fake.status = next(statuses)
        return fake.handler(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(flaky))
    source = literature.SemanticScholarSource(client=client)
    source.retry_delay = 0
    assert [p.paper_id for p in asyncio.run(source.search("spindles"))] == ["abc123"]


def test_network_failure_raises_source_error():
    def broken(request):
        raise httpx.ConnectError("no network")

    client = httpx.AsyncClient(transport=httpx.MockTransport(broken))
    source = literature.SemanticScholarSource(client=client)
    with pytest.raises(literature.SourceError, match="unreachable"):
        asyncio.run(source.search("spindles"))


def test_get_source_defaults_to_semantic_scholar(monkeypatch):
    monkeypatch.setenv("S2_API_KEY", "k")
    source = literature.get_source()
    assert isinstance(source, literature.SemanticScholarSource)
    assert source.api_key == "k"


# --- ManualSource -----------------------------------------------------------------


def test_manual_source_reads_local_store():
    old = papers.add(title="Sleep spindles", year=1990, abstract="NREM stage 2.")
    new = papers.add(title="Sleep spindles revisited", year=2020, doi="10.1/new")
    source = literature.ManualSource()
    assert {p.paper_id for p in asyncio.run(source.search("spindles"))} == {old, new}
    assert [p.paper_id for p in asyncio.run(source.search("spindles", year_from=2000))] == [new]
    assert asyncio.run(source.get(old)).abstract == "NREM stage 2."
    assert asyncio.run(source.get_by_doi("10.1/NEW")).paper_id == new


# --- tools ------------------------------------------------------------------------


def test_find_papers_tool_returns_summaries(s2):
    result = call(tools.find_papers, query="spindles")
    assert result == {
        "found": True,
        "papers": [
            {
                "paper_id": "abc123",
                "title": SPINDLES["title"],
                "authors": ["M. Walker", "R. Stickgold"],
                "year": 2009,
            }
        ],
    }


def test_find_papers_tool_reports_errors(monkeypatch):
    monkeypatch.setattr(literature, "get_source", FakeS2(status=429).source)
    result = call(tools.find_papers, query="spindles")
    assert result["found"] is False
    assert "rate limiting" in result["error"]


def test_find_paper_by_doi_tool(s2):
    result = call(tools.find_paper_by_doi, doi="doi:10.1000/spindles")
    assert result["found"] is True
    assert result["paper"]["abstract"] == SPINDLES["abstract"]
    assert call(tools.find_paper_by_doi, doi="10.9/missing") == {"found": False}


def test_search_papers_tool_only_searches_local_library(s2):
    assert call(tools.search_papers, query="spindles") == {"found": False, "papers": []}
    call(tools.find_papers, query="spindles")
    assert call(tools.search_papers, query="Walker")["papers"][0]["paper_id"] == "abc123"
    # find_papers made the only network call.
    assert len(s2.requests) == 1


def test_get_paper_tool_includes_abstract_and_notes(s2):
    call(tools.find_papers, query="spindles")
    papers.update_notes("abc123", "small sample")
    result = call(tools.get_paper, paper_id="abc123")
    assert result["found"] is True
    assert result["paper"]["abstract"] == SPINDLES["abstract"]
    assert result["paper"]["notes"] == "small sample"
    assert call(tools.get_paper, paper_id="missing") == {"found": False}


def test_add_paper_note_tool():
    paper_id = papers.add(title="Sleep spindles")
    assert call(tools.add_paper_note, paper_id=paper_id, text="n=20") == {"saved": True}
    assert papers.get(paper_id).notes == "n=20"
    assert call(tools.add_paper_note, paper_id="manual:99", text="x") == {"saved": False}
