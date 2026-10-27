"""Finding people to contact, and their email addresses. Works without the career pipeline.

The agent searches public pages (team and lab pages, alumni spotlights, news, talks,
university directories). It never uses LinkedIn, and every person it saves needs a
source URL; an alumni claim needs a URL that shows it.
"""

from dataclasses import dataclass, field

from loguru import logger

from .. import jobs
from ..llm import AgentModel, Usage, cached_system
from . import contacts
from .email_finder import EmailFinder, EmailFinderError, domain_of, lookup

SCHOOL_NAMES = {"berkeley": "UC Berkeley", "nnhs": "Newton North High School (Newton, MA)"}
BLOCKED_SOURCES = ("linkedin.com",)

SYSTEM = """\
You find specific people a student could email, using only public web pages: company
team pages, lab and faculty pages, alumni spotlights, school and news articles,
conference talks, and university directories. Never use or cite LinkedIn.

Save each person with save_contact. Rules:
- Only real, currently-employed people you found on a page; never guess names.
- evidence_urls must include the page(s) where you found them.
- Only mark alumni_of when a page you cite says they attended that school, and pass
  that page as alumni_evidence_url.
- features are short facts the email writer may use, each true per your sources:
  e.g. {"school": "UC Berkeley '18", "team": "protein design", "recent": "spoke at
  NeurIPS 2025 on ..."}. Nothing personal beyond their professional life.
- domain is the organization's email domain if you know it (e.g. "acme.bio").
Stop when you reach the limit you're given or run out of well-supported people.
"""


@dataclass
class ContactLog:
    saved: list[int] = field(default_factory=list)
    emails_found: int = 0
    notes: list[str] = field(default_factory=list)
    finder_error: str | None = None


def make_save_contact(log: ContactLog, job_id: int, finder: EmailFinder | None, limit: int):
    def save_contact(
        name: str,
        evidence_urls: list[str],
        role: str | None = None,
        company: str | None = None,
        is_recruiter_or_admissions: bool = False,
        alumni_of: list[str] | None = None,
        alumni_evidence_url: str | None = None,
        features: dict[str, str] | None = None,
        domain: str | None = None,
    ) -> str:
        """Save one person and look up their work email.

        Args:
            name: Full name as written on the page.
            evidence_urls: Pages where you found them.
            role: Their title, e.g. "University Recruiter" or "Associate Professor".
            company: Company, lab, school or organization.
            is_recruiter_or_admissions: True for recruiting, university relations or
                admissions roles.
            alumni_of: Any of "berkeley", "nnhs" that a cited page supports.
            alumni_evidence_url: The page showing the alumni connection.
            features: Short true facts for personalizing an email.
            domain: The organization's email domain, if known.
        """
        if len(log.saved) >= limit:
            return f"limit of {limit} reached; stop and summarize"
        urls = [u for u in evidence_urls if u and not any(b in u.lower() for b in BLOCKED_SOURCES)]
        if not urls:
            return "error: need at least one non-LinkedIn evidence URL"
        alumni = list(alumni_of or [])
        if alumni and not alumni_evidence_url:
            return "error: alumni_of needs alumni_evidence_url, or leave alumni_of empty"
        if alumni_evidence_url and alumni_evidence_url not in urls:
            urls.append(alumni_evidence_url)
        try:
            cid = contacts.add(
                name,
                urls,
                source="agent",
                company=company,
                role=role,
                is_recruiter_or_admissions=is_recruiter_or_admissions,
                alumni_of=alumni,
                features=features or {},
                found_by_job=job_id,
            )
        except ValueError as e:
            return f"error: {e}"
        log.saved.append(cid)
        if finder is None or log.finder_error:
            return f"saved contact {cid} (no email lookup: {log.finder_error or 'disabled'})"
        where = domain_of(domain or "")
        if not where and not company:
            return f"saved contact {cid}; no company or domain to look up an email"
        try:
            found, status = lookup(finder, name, where, company)
        except EmailFinderError as e:
            # No key, quota, or Hunter down: stop trying for the rest of this job.
            log.finder_error = str(e)
            return f"saved contact {cid}; email lookup unavailable: {e}"
        except ValueError as e:  # this person only, e.g. a single-word name
            return f"saved contact {cid}; email lookup skipped: {e}"
        if found is None:
            return f"saved contact {cid}; email {status}"
        try:
            contacts.set_email(cid, found.email, found.source, status)
        except ValueError:
            return f"saved contact {cid}; their email belongs to a contact already saved"
        log.emails_found += 1
        return f"saved contact {cid} with email ({status})"

    return save_contact


def find_contacts(job: dict, agent: AgentModel, finder: EmailFinder | None, spend: Usage) -> dict:
    p = job["params"]
    schools = [SCHOOL_NAMES.get(s, s) for s in p.get("schools") or []]
    limit = int(p.get("max_contacts", 10))
    prompt = (
        f"Find up to {limit} people at: {p.get('target') or 'any organization'}.\n"
        f"Roles wanted: {', '.join(p.get('roles') or ['any'])}.\n"
        + (f"Only alumni of: {' or '.join(schools)}.\n" if schools else "No alumni requirement.\n")
        + f"Purpose of reaching out: {str(p.get('purpose', 'coffee_chat')).replace('_', ' ')}.\n"
        + f"Notes from the student: {p.get('notes') or 'none'}."
    )
    log = ContactLog()
    result = agent.run(
        cached_system(SYSTEM),
        prompt,
        [make_save_contact(log, job["id"], finder, limit)],
        web_search=min(4 + 2 * limit, 30),
        web_fetch=min(2 + limit, 20),
    )
    spend.add(result.usage)
    out = {
        "contacts": log.saved,
        "emails_found": log.emails_found,
        "email_lookup_error": log.finder_error,
        "note": result.text[:500],
    }
    with_email = [c for c in log.saved if (x := contacts.get(c)) and x["email"]]
    if p.get("draft_after") and with_email:
        out["draft_job"] = jobs.enqueue(
            "draft_outreach",
            {
                "contact_ids": with_email,
                "purpose": p.get("purpose", "coffee_chat"),
                "notes": p.get("notes", ""),
            },
        )
    logger.info(
        "find_contacts job {}: {} people, {} emails", job["id"], len(log.saved), log.emails_found
    )
    return out


def find_email(job: dict, finder: EmailFinder) -> dict:
    """One person's address: an existing contact (contact_id) or a name plus company/domain."""
    p = job["params"]
    if p.get("contact_id"):
        c = contacts.get(int(p["contact_id"]))
        if c is None:
            raise ValueError(f"no contact {p['contact_id']}")
        name, company = c["name"], c["company"]
        domain = domain_of(p.get("domain") or "") or domain_of(company or "")
    else:
        name = p.get("name") or ""
        where = p.get("domain") or p.get("company") or ""
        domain = domain_of(where)
        company = None if domain else (where or None)
        c = None
    found, status = lookup(finder, name, domain, company)
    if found is None:
        return {"found": False, "reason": status, "contact_id": c and c["id"]}
    if c is None:
        cid = contacts.add(
            name,
            source="user",
            company=company or domain,
            role=p.get("role"),
            email=found.email,
            email_source=found.source,
            email_verified=status,
        )
    else:
        cid = c["id"]
        contacts.set_email(cid, found.email, found.source, status)
    return {"found": True, "contact_id": cid, "email": found.email, "verified": status}
