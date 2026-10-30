"""Finding and verifying work email addresses, behind a small interface.

HunterFinder uses Hunter.io's Email Finder and Email Verifier APIs (HUNTER_API_KEY).
FakeFinder-style stand-ins in tests implement the same two methods.
"""

import os
import re
from dataclasses import dataclass
from typing import Protocol

import httpx
from loguru import logger


class EmailFinderError(ValueError):
    """The finder couldn't run (no key, rate limit, bad response); the message says why."""


@dataclass
class FoundEmail:
    email: str
    confidence: int  # 0-100, the finder's own score
    source: str  # e.g. "hunter"


class EmailFinder(Protocol):
    def find(
        self, first: str, last: str, domain: str | None = None, company: str | None = None
    ) -> FoundEmail | None: ...

    def verify(self, email: str) -> str:
        """'valid', 'invalid', 'accept_all', 'webmail', 'disposable' or 'unknown'."""
        ...


TITLES = {"dr", "prof", "professor", "mr", "ms", "mrs", "mx", "phd", "md", "jr", "sr", "ii", "iii"}


def split_name(name: str) -> tuple[str, str]:
    """("Dana", "Lee") from "Dr. Dana M. Lee, PhD"; middle names and titles dropped."""
    words = [w for w in re.split(r"[\s,]+", name.strip()) if w]
    words = [w for w in words if w.lower().strip(".") not in TITLES]
    if len(words) < 2:
        raise ValueError(f"need a first and last name, got {name!r}")
    return words[0].strip("."), words[-1].strip(".")


def domain_of(value: str) -> str | None:
    """ "acme.bio" from "https://www.acme.bio/team" or "acme.bio"; None for a plain name."""
    value = value.strip().lower()
    m = re.match(r"^(?:https?://)?(?:www\.)?([a-z0-9-]+(?:\.[a-z0-9-]+)+)", value)
    return m.group(1) if m and " " not in value else None


class HunterFinder:
    BASE = "https://api.hunter.io/v2"

    def __init__(self, api_key: str, client: httpx.Client | None = None):
        self.api_key = api_key
        self._client = client  # injectable so tests can answer from canned JSON

    def _get(self, path: str, params: dict) -> dict | None:
        params = {**params, "api_key": self.api_key}
        try:
            if self._client is not None:
                resp = self._client.get(self.BASE + path, params=params)
            else:
                with httpx.Client(timeout=20.0) as client:
                    resp = client.get(self.BASE + path, params=params)
        except httpx.HTTPError as e:
            raise EmailFinderError(f"Hunter unreachable ({e})") from e
        if resp.status_code == 401:
            raise EmailFinderError("Hunter API key missing or invalid (set HUNTER_API_KEY)")
        if resp.status_code == 429:
            raise EmailFinderError("Hunter rate limit or monthly quota reached")
        if resp.status_code in (400, 404, 451):
            return None  # no result for this person (or Hunter won't process them)
        if resp.status_code != 200:
            raise EmailFinderError(f"Hunter returned {resp.status_code}")
        return resp.json().get("data")

    def find(self, first, last, domain=None, company=None):
        if not domain and not company:
            raise EmailFinderError("need a domain or company to look up an email")
        params = {"first_name": first, "last_name": last}
        params.update({"domain": domain} if domain else {"company": company})
        data = self._get("/email-finder", params)
        if not data or not data.get("email"):
            return None
        return FoundEmail(data["email"].lower(), int(data.get("score") or 0), "hunter")

    def verify(self, email):
        data = self._get("/email-verifier", {"email": email})
        return (data or {}).get("status") or "unknown"


def default_finder() -> EmailFinder:
    key = os.getenv("HUNTER_API_KEY")
    if not key:
        raise EmailFinderError("set HUNTER_API_KEY in server/.env to look up emails")
    return HunterFinder(key)


def lookup(finder: EmailFinder, name: str, domain: str | None, company: str | None):
    """Find and verify one person's address. Returns (FoundEmail, status) or (None, reason)."""
    first, last = split_name(name)
    found = finder.find(first, last, domain=domain, company=company)
    if found is None:
        return None, "not found"
    status = finder.verify(found.email)
    if status in ("invalid", "disposable"):
        logger.info("email for {} failed verification: {}", name, status)
        return None, f"found {found.email} but it verified as {status}"
    return found, status
