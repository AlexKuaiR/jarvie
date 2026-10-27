"""Your requirements and resume, loaded from ~/jarvie-data/career_profile.toml.

See career_profile.example.toml next to this file for every setting.
"""

import re
import tomllib
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from .. import config
from .postings import KINDS

WATCHLIST_SOURCES = {"greenhouse", "lever", "ashby", "simplify"}


class ProfileError(ValueError):
    """The profile file is missing or has a bad value; the message says which."""


@dataclass
class CareerProfile:
    resume_path: Path
    graduation: date  # first day of the graduation month
    fields: list[str]
    graduation_window_months: int = 6
    kinds: list[str] = field(default_factory=lambda: list(KINDS))
    paid_only: bool = True
    min_hourly_pay: float = 0
    locations: list[str] = field(default_factory=list)
    min_prestige: int = 1
    dealbreakers: list[str] = field(default_factory=list)
    watchlist: dict[str, list[str]] = field(default_factory=dict)
    # Your details for signing emails and filling application forms ([applicant]).
    applicant: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.applicant.get("name", "")

    def resume_text(self) -> str:
        if not self.resume_path.exists():
            raise ProfileError(f"resume not found at {self.resume_path}")
        if self.resume_path.suffix.lower() not in (".md", ".txt"):
            raise ProfileError("resume must be a .md or .txt file for now")
        return self.resume_path.read_text()


def _parse_month(value: str) -> date:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}", value):
        raise ProfileError(f'graduation must look like "2028-05", got {value!r}')
    year, month = map(int, value.split("-"))
    if not 1 <= month <= 12:
        raise ProfileError(f"graduation month must be 01-12, got {value!r}")
    return date(year, month, 1)


def load(path: Path | None = None) -> CareerProfile:
    path = path or config.CAREER_PROFILE
    if not path.exists():
        raise ProfileError(
            f"no career profile at {path}; copy career_profile.example.toml there and fill it in"
        )
    try:
        data = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as e:
        raise ProfileError(f"{path} isn't valid TOML: {e}") from e

    for key in ("resume", "graduation", "fields"):
        if key not in data:
            raise ProfileError(f"{path} is missing '{key}'")
    if not data["fields"]:
        raise ProfileError("'fields' needs at least one field")
    kinds = data.get("kinds", list(KINDS))
    if set(kinds) - set(KINDS):
        raise ProfileError(f"'kinds' must be from {KINDS}")
    min_prestige = data.get("min_prestige", 1)
    if not 1 <= min_prestige <= 5:
        raise ProfileError("'min_prestige' must be 1-5")
    watchlist = data.get("watchlist", {})
    if set(watchlist) - WATCHLIST_SOURCES:
        raise ProfileError(f"'watchlist' may only list {', '.join(sorted(WATCHLIST_SOURCES))}")

    return CareerProfile(
        resume_path=Path(data["resume"]).expanduser(),
        graduation=_parse_month(data["graduation"]),
        fields=list(data["fields"]),
        graduation_window_months=data.get("graduation_window_months", 6),
        kinds=list(kinds),
        paid_only=data.get("paid_only", True),
        min_hourly_pay=data.get("min_hourly_pay", 0),
        locations=list(data.get("locations", [])),
        min_prestige=min_prestige,
        dealbreakers=list(data.get("dealbreakers", [])),
        watchlist={k: list(v) for k, v in watchlist.items()},
        applicant=_applicant(data.get("applicant", {})),
    )


APPLICANT_KEYS = {
    "name",
    "email",
    "phone",
    "school",
    "major",
    "links",
    "resume_pdf",
    "location",
    "work_authorization",
    "pronouns",
}


def _applicant(data: dict) -> dict:
    unknown = set(data) - APPLICANT_KEYS
    if unknown:
        raise ProfileError(f"unknown [applicant] keys: {sorted(unknown)}")
    out = dict(data)
    if "resume_pdf" in out:
        out["resume_pdf"] = str(Path(out["resume_pdf"]).expanduser())
    out["links"] = list(out.get("links", []))
    return out
