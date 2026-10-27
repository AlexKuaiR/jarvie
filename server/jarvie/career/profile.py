"""Your requirements and resumes, loaded from ~/jarvie-data/career_profile.toml.

See career_profile.example.toml next to this file for every setting. You can have one
resume (`resume = ...`) or several named ones ([resumes.swe], [resumes.qr], ...); each
posting uses the one whose fields match it best, unless you pick one for it.
"""

import re
import tomllib
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from .. import config
from .filters import tag_fields
from .postings import KINDS

WATCHLIST_SOURCES = {"greenhouse", "lever", "ashby", "simplify"}


class ProfileError(ValueError):
    """The profile file is missing or has a bad value; the message says which."""


@dataclass
class Resume:
    name: str  # e.g. "swe"; "default" for a single `resume = ...`
    text_path: Path  # .md or .txt, what the models read
    pdf_path: Path | None = None  # what the apply assistant uploads
    fields: list[str] = field(default_factory=list)  # postings it's meant for

    def text(self) -> str:
        if not self.text_path.exists():
            raise ProfileError(f"resume '{self.name}' not found at {self.text_path}")
        if self.text_path.suffix.lower() not in (".md", ".txt"):
            raise ProfileError(f"resume '{self.name}' must be a .md or .txt file for now")
        return self.text_path.read_text()


@dataclass
class CareerProfile:
    resumes: dict[str, Resume]
    default_resume: str
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
    # Which model runs each kind of job ([models]); see jarvie/cli_models.py.
    models: dict[str, str] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.applicant.get("name", "")

    def resume(self, name: str | None = None) -> Resume:
        """A resume by name (any case; "default" means the default one), or the default."""
        if not name or (name.lower() == "default" and "default" not in self.resumes):
            name = self.default_resume
        name = next((n for n in self.resumes if n.lower() == name.lower()), name)
        if name not in self.resumes:
            raise ProfileError(f"no resume named '{name}'; you have {', '.join(self.resumes)}")
        return self.resumes[name]

    def resume_text(self, name: str | None = None) -> str:
        return self.resume(name).text()

    def resume_for(self, posting: dict) -> Resume:
        """The resume to use for a posting: the one picked for it, else the best match.

        Best match = the most fields in common with the posting's title, then with all of
        its fields; ties go to the default resume, then to the order in the profile.
        """
        if posting.get("resume") in self.resumes:
            return self.resumes[posting["resume"]]
        title_fields = set(tag_fields(posting.get("title") or "", ""))
        all_fields = set(posting.get("fields") or [])
        order = list(self.resumes)

        def rank(r: Resume):
            wanted = set(r.fields)
            return (
                len(wanted & title_fields),
                len(wanted & all_fields),
                r.name == self.default_resume,
                -order.index(r.name),
            )

        return max(self.resumes.values(), key=rank)

    def labeled_resumes(self) -> str:
        """Every resume with its name and fields, for prompts that may need any of them."""
        if len(self.resumes) == 1:
            return self.resume_text()
        return "\n\n".join(
            f"### Resume '{r.name}' (for {', '.join(r.fields) or 'anything'})\n{r.text()}"
            for r in self.resumes.values()
        )


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

    for key in ("graduation", "fields"):
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

    applicant = _applicant(data.get("applicant", {}))
    resumes, default_resume = _resumes(data, applicant)

    return CareerProfile(
        resumes=resumes,
        default_resume=default_resume,
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
        applicant=applicant,
        models=_models(data.get("models", {})),
    )


def _models(data: dict) -> dict[str, str]:
    from ..cli_models import DEFAULT_SPECS, parse_spec

    unknown = set(data) - set(DEFAULT_SPECS)
    if unknown:
        raise ProfileError(f"unknown [models] keys {sorted(unknown)}; use {sorted(DEFAULT_SPECS)}")
    out = {**DEFAULT_SPECS, **data}
    for role, spec in out.items():
        try:
            parse_spec(spec)
        except ValueError as e:
            raise ProfileError(f"[models] {role}: {e}") from e
    return out


def _resumes(data: dict, applicant: dict) -> tuple[dict[str, Resume], str]:
    """Either one `resume = ...` (named "default") or named [resumes.<name>] tables."""
    pdf = Path(applicant["resume_pdf"]) if applicant.get("resume_pdf") else None
    if "resumes" in data and "resume" in data:
        raise ProfileError("use either 'resume' or [resumes.<name>] tables, not both")
    if "resumes" not in data:
        if "resume" not in data:
            raise ProfileError("missing 'resume' (or [resumes.<name>] tables)")
        one = Resume("default", Path(data["resume"]).expanduser(), pdf, list(data["fields"]))
        return {"default": one}, "default"

    resumes = {}
    for name, r in data["resumes"].items():
        if not isinstance(r, dict) or "text" not in r:
            raise ProfileError(f'[resumes.{name}] needs text = "path to a .md or .txt"')
        unknown = set(r) - {"text", "pdf", "fields"}
        if unknown:
            raise ProfileError(f"unknown [resumes.{name}] keys: {sorted(unknown)}")
        resumes[name] = Resume(
            name,
            Path(r["text"]).expanduser(),
            Path(r["pdf"]).expanduser() if r.get("pdf") else pdf,
            list(r.get("fields", [])),
        )
    if not resumes:
        raise ProfileError("[resumes] has no resumes in it")
    default = data.get("default_resume", next(iter(resumes)))
    if default not in resumes:
        raise ProfileError(f"default_resume '{default}' isn't one of {', '.join(resumes)}")
    return resumes, default


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
