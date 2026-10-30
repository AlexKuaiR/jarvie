import sqlite3
from datetime import datetime

from ..config import TZ
from ..storage import dumps, loads
from .schema import connect

SCHOOLS = ("berkeley", "nnhs")
SOURCES = ("agent", "user")


def _now() -> str:
    return datetime.now(TZ).isoformat()


def _decode(r) -> dict:
    out = dict(r)
    out["alumni_of"] = loads(out["alumni_of"], [])
    out["evidence_urls"] = loads(out["evidence_urls"], [])
    out["features"] = loads(out["features"], {})
    out["is_recruiter_or_admissions"] = bool(out["is_recruiter_or_admissions"])
    out["do_not_contact"] = bool(out["do_not_contact"])
    return out


def add(
    name: str,
    evidence_urls: list[str] | None = None,
    source: str = "agent",
    company: str | None = None,
    role: str | None = None,
    is_recruiter_or_admissions: bool = False,
    alumni_of: list[str] | None = None,
    email: str | None = None,
    email_source: str | None = None,
    email_verified: str | None = None,
    features: dict | None = None,
    notes: str | None = None,
    found_by_job: int | None = None,
) -> int:
    """Save a person, or return the existing id if they're already stored (same email, or
    same name and company when there's no email).

    People the research agent finds (source "agent") need at least one evidence URL:
    where they were found, and the page backing any alumni claim, so nobody is
    emailed on a guess. People you add yourself (source "user") don't.
    """
    if not name or not name.strip():
        raise ValueError("a contact needs a name")
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}")
    if source == "agent" and not evidence_urls:
        raise ValueError("a contact found by the agent needs at least one evidence URL")
    unknown = set(alumni_of or []) - set(SCHOOLS)
    if unknown:
        raise ValueError(f"alumni_of must be from {SCHOOLS}")
    email = email.strip().lower() if email else None
    with connect() as conn:
        if email:
            existing = conn.execute("SELECT id FROM contacts WHERE email = ?", (email,)).fetchone()
        else:
            # Without an email, the same name at the same place is the same person.
            existing = conn.execute(
                """SELECT id FROM contacts WHERE name = ? COLLATE NOCASE
                   AND COALESCE(company, '') = COALESCE(?, '') COLLATE NOCASE""",
                (name.strip(), company),
            ).fetchone()
        if existing:
            return existing["id"]
        cur = conn.execute(
            """INSERT INTO contacts
               (name, company, role, is_recruiter_or_admissions, alumni_of, source,
                evidence_urls, email, email_source, email_verified, features, notes,
                found_at, found_by_job)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                name.strip(),
                company,
                role,
                int(is_recruiter_or_admissions),
                dumps(alumni_of or []),
                source,
                dumps(evidence_urls or []),
                email,
                email_source or ("user" if source == "user" and email else None),
                email_verified,
                dumps(features or {}),
                notes,
                _now(),
                found_by_job,
            ),
        )
        return cur.lastrowid


def get(contact_id: int) -> dict | None:
    with connect() as conn:
        r = conn.execute("SELECT * FROM contacts WHERE id = ?", (contact_id,)).fetchone()
    return _decode(r) if r else None


def set_email(contact_id: int, email: str, email_source: str, email_verified: str | None) -> bool:
    """Record an address found later, e.g. by the email finder for an existing contact."""
    try:
        with connect() as conn:
            cur = conn.execute(
                "UPDATE contacts SET email = ?, email_source = ?, email_verified = ? WHERE id = ?",
                (email.strip().lower(), email_source, email_verified, contact_id),
            )
            return cur.rowcount > 0
    except sqlite3.IntegrityError as e:
        raise ValueError(f"{email} already belongs to another contact") from e


def add_fact(contact_id: int, fact: str, key: str | None = None) -> str:
    """Remember something about a person for future emails ("we met at the career fair").

    Facts are what the writer may personalize from; anything else it mentions is
    rejected by the grounding check. Returns the key it was stored under.
    """
    contact = get(contact_id)
    if contact is None:
        raise ValueError(f"no contact {contact_id}")
    if not fact.strip():
        raise ValueError("the fact is empty")
    features = contact["features"]
    if not key:
        n = 1 + sum(k.startswith("from_you") for k in features)
        key = "from_you" if n == 1 else f"from_you_{n}"
    features[key] = fact.strip()
    with connect() as conn:
        conn.execute("UPDATE contacts SET features = ? WHERE id = ?", (dumps(features), contact_id))
    return key


def search(
    company: str | None = None,
    school: str | None = None,
    has_email: bool | None = None,
    limit: int = 50,
) -> list[dict]:
    sql = "SELECT * FROM contacts WHERE 1 = 1"
    args: list = []
    if company:
        sql += " AND company = ? COLLATE NOCASE"
        args.append(company)
    if school:
        # alumni_of is a JSON list; json_each lets SQLite look inside it.
        sql += " AND EXISTS (SELECT 1 FROM json_each(contacts.alumni_of) WHERE value = ?)"
        args.append(school)
    if has_email is True:
        sql += " AND email IS NOT NULL"
    elif has_email is False:
        sql += " AND email IS NULL"
    sql += " ORDER BY id DESC LIMIT ?"
    args.append(limit)
    with connect() as conn:
        rows = conn.execute(sql, args).fetchall()
    return [_decode(r) for r in rows]


def mark_do_not_contact(contact_id: int) -> bool:
    """Never email this person again, and stop anything already queued for them."""
    with connect() as conn:
        ok = conn.execute(
            "UPDATE contacts SET do_not_contact = 1 WHERE id = ?", (contact_id,)
        ).rowcount
        conn.execute(
            """UPDATE outreach SET status = 'stopped', updated_at = ?
               WHERE contact_id = ? AND status IN ('drafted', 'approved', 'scheduled')""",
            (_now(), contact_id),
        )
        return ok > 0
