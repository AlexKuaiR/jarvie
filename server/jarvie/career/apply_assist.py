"""Apply assistant: fills an application form from your profile, then stops for you.

    .venv/bin/python -m jarvie.career.apply_assist <posting_id>

Opens a visible browser on the posting's application form (Greenhouse, Lever or Ashby),
fills what it can from [applicant] in your profile, attaches your resume and the latest
cover letter Jarvie wrote, and takes a screenshot. It never clicks Submit: you review
every field, answer what it left blank, and submit yourself.

Left blank on purpose: demographic/EEO questions (gender, race, veteran, disability),
anything it can't map with confidence, and custom questions (use write_essay for those).
Workday and Handshake are refused: Workday needs a per-company account and blocks
automation; Handshake's terms forbid it.

Needs the optional extra:  uv sync --extra apply && .venv/bin/playwright install chromium
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from loguru import logger

from .. import config
from . import documents, postings
from .profile import CareerProfile

SUPPORTED = ("greenhouse", "lever", "ashby")
REFUSED = {
    "workday": "Workday needs an account per company and blocks automation; apply on its site",
    "handshake": "Handshake's terms forbid automation; apply on Handshake",
}


@dataclass
class FormField:
    selector: str
    label: str
    type: str  # text, email, tel, url, textarea, select, file, checkbox, radio
    required: bool = False
    options: list[str] = field(default_factory=list)


@dataclass
class Plan:
    fills: list[tuple[FormField, str, str]] = field(default_factory=list)  # field, value, from
    uploads: list[tuple[FormField, Path, str]] = field(default_factory=list)
    left_blank: list[tuple[FormField, str]] = field(default_factory=list)  # field, why


class Browser(Protocol):
    def open(self, url: str) -> None: ...
    def fields(self) -> list[FormField]: ...
    def fill(self, f: FormField, value: str) -> None: ...
    def upload(self, f: FormField, path: Path) -> None: ...
    def screenshot(self, path: Path) -> None: ...


SENSITIVE = re.compile(
    r"gender|sex\b|race|ethnic|hispanic|latin|veteran|disabilit|sexual orientation|"
    r"transgender|pronoun|age\b|date of birth|birth|religio|criminal|conviction|ssn|"
    r"social security",
    re.I,
)


def _link(applicant: dict, *needles: str) -> str | None:
    return next(
        (u for u in applicant.get("links", []) if any(n in u.lower() for n in needles)), None
    )


def value_for(label: str, applicant: dict) -> tuple[str | None, str]:
    """(value, where it came from) for a text-like field, or (None, why not)."""
    lab = label.lower()
    name = applicant.get("name", "").split()
    rules = [
        (r"first name|given name|preferred name", name[0] if name else None, "name"),
        (r"last name|family name|surname", name[-1] if len(name) > 1 else None, "name"),
        (r"^\W*(full )?name\b", applicant.get("name"), "name"),
        (r"e-?mail", applicant.get("email"), "email"),
        (r"phone|mobile", applicant.get("phone"), "phone"),
        (r"github", _link(applicant, "github.com"), "links"),
        (r"linkedin", _link(applicant, "linkedin.com"), "links"),
        (r"website|portfolio|personal (site|url)|other link", _link(applicant, "http"), "links"),
        (r"school|university|college|institution", applicant.get("school"), "school"),
        (r"major|discipline|field of study|degree", applicant.get("major"), "major"),
        (
            r"location|city|where are you based|current address",
            applicant.get("location"),
            "location",
        ),
        (
            r"authori[sz]ed to work|work authori[sz]ation|sponsorship|visa",
            applicant.get("work_authorization"),
            "work_authorization",
        ),
    ]
    for pattern, value, source in rules:
        if re.search(pattern, lab):
            return (value, source) if value else (None, f"no {source} in [applicant]")
    return None, "not something it fills automatically"


def plan_fill(fields: list[FormField], applicant: dict, cover_letter: Path | None) -> Plan:
    plan = Plan()
    resume = Path(applicant["resume_pdf"]) if applicant.get("resume_pdf") else None
    for f in fields:
        lab = f.label.lower()
        if SENSITIVE.search(lab):
            plan.left_blank.append((f, "demographic/sensitive: yours to answer"))
        elif f.type == "file":
            if re.search(r"cover", lab):
                if cover_letter:
                    plan.uploads.append((f, cover_letter, "latest cover letter"))
                else:
                    plan.left_blank.append((f, "no cover letter written yet"))
            elif re.search(r"resume|cv\b|curriculum", lab):
                if resume and resume.exists():
                    plan.uploads.append((f, resume, "resume_pdf"))
                else:
                    plan.left_blank.append(
                        (f, "set resume_pdf in [applicant] to a file that exists")
                    )
            else:
                plan.left_blank.append((f, "unknown file upload"))
        elif f.type in ("checkbox", "radio"):
            plan.left_blank.append((f, "choices are yours to make"))
        elif f.type == "textarea" and re.search(r"cover letter", lab) and cover_letter:
            plan.fills.append((f, cover_letter.read_text(), "latest cover letter"))
        else:
            value, source = value_for(f.label, applicant)
            if value is None:
                plan.left_blank.append((f, source))
            elif f.type == "select":
                match = next((o for o in f.options if o.lower() == value.lower()), None)
                if match:
                    plan.fills.append((f, match, source))
                else:
                    plan.left_blank.append((f, f"'{value}' isn't one of the options"))
            else:
                plan.fills.append((f, value, source))
    return plan


def application_url(posting: dict) -> str:
    """The form's URL. Greenhouse and Lever forms live at the posting or /apply."""
    url = posting["url"] or ""
    if posting.get("ats") == "lever" and not url.rstrip("/").endswith("/apply"):
        return url.rstrip("/") + "/apply"
    if posting.get("ats") == "ashby" and "/application" not in url:
        return url.rstrip("/") + "/application"
    return url


def cover_letter_file(posting_id: int) -> Path | None:
    letters = [d for d in documents.for_posting(posting_id) if d["kind"] == "cover_letter"]
    if not letters:
        return None
    out = config.DATA_DIR / "applications" / str(posting_id)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "cover_letter.txt"
    path.write_text(letters[0]["text"])  # newest version
    return path


def prepare(posting_id: int, browser: Browser, profile: CareerProfile) -> tuple[Plan, Path]:
    """Fill the form for one posting and screenshot it. Never submits."""
    posting = postings.get(posting_id)
    if posting is None:
        raise ValueError(f"no posting {posting_id}")
    ats = posting.get("ats")
    if ats in REFUSED:
        raise ValueError(REFUSED[ats])
    if ats not in SUPPORTED:
        raise ValueError(
            f"only {', '.join(SUPPORTED)} forms are supported (this one: {ats or 'unknown'})"
        )
    if not profile.applicant.get("name") or not profile.applicant.get("email"):
        raise ValueError("fill in [applicant] name and email in career_profile.toml first")

    browser.open(application_url(posting))
    plan = plan_fill(browser.fields(), profile.applicant, cover_letter_file(posting_id))
    for f, value, _ in plan.fills:
        browser.fill(f, value)
    for f, path, _ in plan.uploads:
        browser.upload(f, path)
    shot = config.DATA_DIR / "applications" / str(posting_id) / "filled.png"
    shot.parent.mkdir(parents=True, exist_ok=True)
    browser.screenshot(shot)
    if posting["status"] is None:
        postings.set_status(posting_id, "interested", "application form pre-filled; not submitted")
    logger.info(
        "apply assist {}: {} filled, {} uploads, {} left blank",
        posting_id,
        len(plan.fills),
        len(plan.uploads),
        len(plan.left_blank),
    )
    return plan, shot


# JavaScript that lists the form's visible inputs with their labels.
FIELDS_JS = """
() => {
  const out = [];
  const els = document.querySelectorAll('input, textarea, select');
  els.forEach((el, i) => {
    const type = el.tagName === 'INPUT' ? (el.type || 'text') : el.tagName.toLowerCase();
    if (['hidden', 'submit', 'button', 'image', 'reset', 'search'].includes(type)) return;
    if (type !== 'file' && el.offsetParent === null) return;
    if (!el.dataset.jarvie) el.dataset.jarvie = 'f' + i;
    let label = '';
    if (el.id) { const l = document.querySelector(`label[for="${CSS.escape(el.id)}"]`); if (l) label = l.innerText; }
    if (!label) { const l = el.closest('label'); if (l) label = l.innerText; }
    if (!label) label = el.getAttribute('aria-label') || el.placeholder || el.name || '';
    out.push({selector: `[data-jarvie="${el.dataset.jarvie}"]`, label: label.trim().slice(0, 200),
              type, required: el.required || el.getAttribute('aria-required') === 'true',
              options: el.tagName === 'SELECT' ? Array.from(el.options).map(o => o.text.trim()) : []});
  });
  return out;
}
"""


class PlaywrightBrowser:
    """A real, visible Chromium window. Close it yourself after reviewing and submitting."""

    def __init__(self, headless: bool = False):
        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=headless)
        self.page = self._browser.new_page()

    def open(self, url):
        self.page.goto(url, wait_until="domcontentloaded")
        self.page.wait_for_timeout(1500)  # let the form's scripts render it

    def fields(self):
        return [FormField(**f) for f in self.page.evaluate(FIELDS_JS)]

    def fill(self, f, value):
        if f.type == "select":
            self.page.select_option(f.selector, label=value)
        else:
            self.page.fill(f.selector, value)

    def upload(self, f, path):
        self.page.set_input_files(f.selector, str(path))

    def screenshot(self, path):
        self.page.screenshot(path=str(path), full_page=True)

    def close(self):
        self._browser.close()
        self._pw.stop()


def main() -> None:
    import argparse

    from .profile import load

    parser = argparse.ArgumentParser(description="Pre-fill an application form (never submits).")
    parser.add_argument("posting_id", type=int)
    args = parser.parse_args()
    browser = PlaywrightBrowser(headless=False)
    try:
        plan, shot = prepare(args.posting_id, browser, load())
        print(
            f"Filled {len(plan.fills)} fields and {len(plan.uploads)} uploads. Screenshot: {shot}"
        )
        for f, why in plan.left_blank:
            print(f"  left blank: {f.label or f.selector} ({why})")
        input("Review the form in the browser and submit it yourself. Press Enter here to close. ")
    finally:
        browser.close()


if __name__ == "__main__":
    main()
