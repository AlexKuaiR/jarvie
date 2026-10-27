"""Public job-board feeds: Greenhouse, Lever, Ashby, and SimplifyJobs' internship lists.

All of these are public JSON endpoints that companies (or SimplifyJobs) publish for
exactly this purpose, so no scraping or login is involved. Handshake is deliberately
absent: its terms forbid automated access, so Handshake postings come in pasted.
"""

import html
import re
from dataclasses import dataclass, field
from typing import Protocol

import httpx
from bs4 import BeautifulSoup
from loguru import logger


class SourceError(Exception):
    """A feed couldn't be fetched or parsed; the message names which."""


@dataclass
class RawPosting:
    """A posting as a feed reports it, before filtering and scoring."""

    source: str
    company: str
    title: str
    url: str
    location: str = ""
    text: str = ""  # description as plain text, for filters and the research agent
    terms: list[str] = field(default_factory=list)  # e.g. ["Summer 2027"] (Simplify only)
    ats: str | None = None


class PostingSource(Protocol):
    name: str

    async def fetch(self, slug: str) -> list[RawPosting]: ...


def html_to_text(markup: str) -> str:
    # Greenhouse double-escapes its HTML; unescape once before parsing.
    soup = BeautifulSoup(html.unescape(markup or ""), "html.parser")
    return re.sub(r"\n{3,}", "\n\n", soup.get_text("\n")).strip()


class _HttpSource:
    name = ""

    def __init__(self, client: httpx.AsyncClient | None = None):
        # Injectable so tests can answer from canned JSON instead of the network.
        self._client = client

    async def _get_json(self, url: str, params: dict | None = None):
        try:
            if self._client is not None:
                resp = await self._client.get(url, params=params)
            else:
                async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
                    resp = await client.get(url, params=params)
        except httpx.HTTPError as e:
            raise SourceError(f"{self.name}: {url} unreachable ({e})") from e
        if resp.status_code == 404:
            raise SourceError(f"{self.name}: {url} not found; check the board slug")
        if resp.status_code != 200:
            raise SourceError(f"{self.name}: {url} returned {resp.status_code}")
        try:
            return resp.json()
        except ValueError as e:
            raise SourceError(f"{self.name}: {url} didn't return JSON") from e


class GreenhouseSource(_HttpSource):
    name = "greenhouse"

    async def fetch(self, slug: str) -> list[RawPosting]:
        data = await self._get_json(
            f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs", {"content": "true"}
        )
        company = data.get("meta", {}).get("name") or slug
        return [
            RawPosting(
                source=self.name,
                ats="greenhouse",
                company=job.get("company_name") or company,
                title=job.get("title", ""),
                url=job.get("absolute_url", ""),
                location=(job.get("location") or {}).get("name", ""),
                text=html_to_text(job.get("content", "")),
            )
            for job in data.get("jobs", [])
            if job.get("title") and job.get("absolute_url")
        ]


class LeverSource(_HttpSource):
    name = "lever"

    async def fetch(self, slug: str) -> list[RawPosting]:
        data = await self._get_json(f"https://api.lever.co/v0/postings/{slug}", {"mode": "json"})
        out = []
        for job in data if isinstance(data, list) else []:
            if not job.get("text") or not job.get("hostedUrl"):
                continue
            lists = "\n".join(
                f"{item.get('text', '')}\n{html_to_text(item.get('content', ''))}"
                for item in job.get("lists", [])
            )
            out.append(
                RawPosting(
                    source=self.name,
                    ats="lever",
                    company=slug,
                    title=job["text"],
                    url=job["hostedUrl"],
                    location=(job.get("categories") or {}).get("location", "") or "",
                    text=f"{job.get('descriptionPlain', '')}\n{lists}".strip(),
                )
            )
        return out


class AshbySource(_HttpSource):
    name = "ashby"

    async def fetch(self, slug: str) -> list[RawPosting]:
        data = await self._get_json(f"https://api.ashbyhq.com/posting-api/job-board/{slug}")
        return [
            RawPosting(
                source=self.name,
                ats="ashby",
                company=slug,
                title=job.get("title", ""),
                url=job.get("jobUrl") or job.get("applyUrl", ""),
                location=job.get("location", "") or "",
                text=job.get("descriptionPlain") or html_to_text(job.get("descriptionHtml", "")),
            )
            for job in data.get("jobs", [])
            if job.get("title")
            and (job.get("jobUrl") or job.get("applyUrl"))
            and job.get("isListed", True)
        ]


# The repos SimplifyJobs publishes each season; the slug passed to fetch() is the repo,
# e.g. "SimplifyJobs/Summer2027-Internships". Each repo keeps a machine-readable
# listings.json next to its README.
SIMPLIFY_LISTINGS = "https://raw.githubusercontent.com/{repo}/dev/.github/scripts/listings.json"


def ats_from_url(url: str) -> str | None:
    for marker, ats in (
        ("greenhouse.io", "greenhouse"),
        ("lever.co", "lever"),
        ("ashbyhq.com", "ashby"),
        ("myworkdayjobs.com", "workday"),
        ("joinhandshake.com", "handshake"),
    ):
        if marker in url:
            return ats
    return None


class SimplifySource(_HttpSource):
    name = "simplify"

    async def fetch(self, slug: str) -> list[RawPosting]:
        data = await self._get_json(SIMPLIFY_LISTINGS.format(repo=slug))
        out = []
        for item in data if isinstance(data, list) else []:
            if not item.get("active", True) or not item.get("is_visible", True):
                continue
            url = item.get("url", "")
            if not url or not item.get("title") or not item.get("company_name"):
                continue
            out.append(
                RawPosting(
                    source=self.name,
                    ats=ats_from_url(url),
                    company=item["company_name"],
                    title=item["title"],
                    url=url,
                    location="; ".join(item.get("locations") or []),
                    text=" ".join(filter(None, [item.get("category"), item.get("sponsorship")])),
                    terms=list(item.get("terms") or []),
                )
            )
        return out


SOURCES: dict[str, type[_HttpSource]] = {
    "greenhouse": GreenhouseSource,
    "lever": LeverSource,
    "ashby": AshbySource,
    "simplify": SimplifySource,
}


async def fetch_all(
    watchlist: dict[str, list[str]], client: httpx.AsyncClient | None = None
) -> tuple[list[RawPosting], list[str]]:
    """Fetch every slug in the watchlist. A failing feed is logged and reported, not fatal."""
    postings, errors = [], []
    for source_name, slugs in watchlist.items():
        source = SOURCES[source_name](client)
        for slug in slugs:
            try:
                found = await source.fetch(slug)
            except SourceError as e:
                logger.warning("career source failed: {}", e)
                errors.append(str(e))
                continue
            logger.info("career source {} {}: {} postings", source_name, slug, len(found))
            postings.extend(found)
    return postings, errors
