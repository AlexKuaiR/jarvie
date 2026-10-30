import asyncio
import os
from typing import Protocol

import httpx
from loguru import logger

from . import papers
from .papers import Paper


class SourceError(Exception):
    """A paper source couldn't answer (network down, rate limited, bad response)."""


# A Protocol is a "shape" any class can match without inheriting from it. The voice tools
# only ever talk to a PaperSource, so tests (or an offline mode) can swap in ManualSource
# or a fake without the tools changing. The methods are async so a slow network call
# doesn't freeze the voice pipeline while it waits.
class PaperSource(Protocol):
    async def search(
        self, query: str, limit: int = 5, year_from: int | None = None
    ) -> list[Paper]: ...

    async def get(self, paper_id: str) -> Paper | None: ...

    async def get_by_doi(self, doi: str) -> Paper | None: ...


class ManualSource:
    """Only papers already in the local SQLite database; never touches the network."""

    async def search(self, query: str, limit: int = 5, year_from: int | None = None) -> list[Paper]:
        found = papers.search(query, limit=limit if year_from is None else 100)
        if year_from is not None:
            found = [p for p in found if p.year is not None and p.year >= year_from]
        return found[:limit]

    async def get(self, paper_id: str) -> Paper | None:
        return papers.get(paper_id)

    async def get_by_doi(self, doi: str) -> Paper | None:
        return papers.by_doi(doi)


S2_BASE = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = "paperId,title,authors,year,venue,abstract,externalIds,url"


class SemanticScholarSource:
    """Searches Semantic Scholar and caches every paper it returns in the local database.

    Caching means follow-up questions ("what was their method?") and notes work
    without another API call. An API key is optional; without one, S2 shares a
    small rate limit across all keyless users, so expect occasional refusals.
    """

    def __init__(self, api_key: str | None = None, client: httpx.AsyncClient | None = None):
        # client is injectable so tests can hand in one that answers from canned data
        # instead of the network. Normally it's None and each call opens its own.
        self.api_key = api_key
        self._client = client
        self.retry_delay = 1.5  # seconds to wait before retrying a rate-limited request

    async def _request(self, path: str, params: dict) -> httpx.Response:
        headers = {"x-api-key": self.api_key} if self.api_key else {}
        try:
            if self._client is not None:
                return await self._client.get(S2_BASE + path, params=params, headers=headers)
            async with httpx.AsyncClient(timeout=10.0) as client:
                return await client.get(S2_BASE + path, params=params, headers=headers)
        except httpx.HTTPError as e:
            logger.error("semantic scholar request failed: {}", e)
            raise SourceError("Semantic Scholar is unreachable.") from e

    async def _get_json(self, path: str, params: dict) -> dict | None:
        """GET an S2 endpoint. Returns None on 404; raises SourceError on other failures."""
        resp = await self._request(path, params)
        if resp.status_code == 429:
            # Keyless users share one rate limit, so a single short wait often gets through.
            logger.warning("semantic scholar rate limited; retrying once")
            await asyncio.sleep(self.retry_delay)
            resp = await self._request(path, params)

        if resp.status_code == 404:
            return None
        if resp.status_code == 429:
            raise SourceError("Semantic Scholar is rate limiting requests; try again shortly.")
        if resp.status_code != 200:
            logger.error("semantic scholar {} returned {}: {}", path, resp.status_code, resp.text)
            raise SourceError(f"Semantic Scholar returned an error ({resp.status_code}).")
        return resp.json()

    def _cache(self, data: dict) -> Paper | None:
        """Store one S2 paper object locally and return the stored version (with notes)."""
        if not data.get("paperId") or not data.get("title"):
            return None
        paper_id = papers.add(
            title=data["title"],
            authors=[a["name"] for a in data.get("authors") or [] if a.get("name")],
            year=data.get("year"),
            venue=data.get("venue") or None,
            abstract=data.get("abstract"),
            doi=(data.get("externalIds") or {}).get("DOI"),
            url=data.get("url"),
            source="s2",
            paper_id=data["paperId"],
        )
        return papers.get(paper_id)

    async def search(self, query: str, limit: int = 5, year_from: int | None = None) -> list[Paper]:
        params = {"query": query, "limit": limit, "fields": S2_FIELDS}
        if year_from is not None:
            params["year"] = f"{year_from}-"  # S2 syntax for "this year or later"
        data = await self._get_json("/paper/search", params) or {}
        cached = (self._cache(item) for item in data.get("data") or [])
        return [p for p in cached if p is not None]

    async def get(self, paper_id: str) -> Paper | None:
        # Papers the user found earlier are already cached, notes and all.
        if (local := papers.get(paper_id)) is not None:
            return local
        data = await self._get_json(f"/paper/{paper_id}", {"fields": S2_FIELDS})
        return self._cache(data) if data else None

    async def get_by_doi(self, doi: str) -> Paper | None:
        doi = papers.normalize_doi(doi)
        if not doi:
            return None
        if (local := papers.by_doi(doi)) is not None:
            return local
        data = await self._get_json(f"/paper/DOI:{doi}", {"fields": S2_FIELDS})
        return self._cache(data) if data else None


def get_source() -> PaperSource:
    """The single place that decides where papers come from."""
    # Set S2_API_KEY in server/.env (bot.py loads it) for a higher rate limit.
    return SemanticScholarSource(api_key=os.getenv("S2_API_KEY"))
