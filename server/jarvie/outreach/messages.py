"""Email drafts and their path to being sent: drafted -> approved -> scheduled -> sent."""

import sqlite3
from datetime import datetime

from ..config import TZ
from ..storage import dumps, loads
from . import contacts
from .schema import connect

PURPOSES = ("coffee_chat", "research_inquiry", "referral", "advice", "other")

# Allowed status changes. Anything else is refused, so e.g. a draft can't be marked
# sent without being approved first.
TRANSITIONS = {
    "drafted": {"approved", "stopped"},
    "approved": {"scheduled", "drafted", "stopped"},
    "scheduled": {"sent", "approved", "stopped"},
    "sent": {"replied", "bounced", "stopped"},
    "replied": set(),
    "bounced": set(),
    "stopped": set(),
}


class OutreachRefused(Exception):
    """A draft or status change that would break an outreach rule."""


def _now() -> str:
    return datetime.now(TZ).isoformat()


def add_draft(
    contact_id: int,
    subject: str,
    body: str,
    purpose: str = "coffee_chat",
    posting_id: int | None = None,
    summary: dict | None = None,
    resume: str | None = None,
) -> int:
    """Store a draft email to one person. posting_id optionally ties it to a career posting.

    summary is the plan the writer wrote it from; resume is the resume choice used, so a
    rewrite later draws on the same one.
    """
    if purpose not in PURPOSES:
        raise ValueError(f"purpose must be one of {PURPOSES}")
    contact = contacts.get(contact_id)
    if contact is None:
        raise OutreachRefused(f"no contact {contact_id}")
    if contact["do_not_contact"]:
        raise OutreachRefused(f"{contact['name']} is marked do-not-contact")
    if not contact["email"]:
        raise OutreachRefused(f"no email address for {contact['name']}")
    if not subject.strip() or not body.strip():
        raise OutreachRefused("a draft needs a subject and a body")
    now = _now()
    try:
        with connect() as conn:
            outreach_id = conn.execute(
                """INSERT INTO outreach
                   (contact_id, purpose, posting_id, subject, body, status, summary, resume,
                    created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, 'drafted', ?, ?, ?, ?)""",
                (
                    contact_id,
                    purpose,
                    posting_id,
                    subject.strip(),
                    body.strip(),
                    dumps(summary),
                    resume,
                    now,
                    now,
                ),
            ).lastrowid
    except sqlite3.IntegrityError as e:
        # The partial unique index: this person already has a live thread.
        raise OutreachRefused(f"{contact['name']} already has an email in progress") from e
    _record(outreach_id, "jarvie")
    return outreach_id


def _decode(r) -> dict:
    out = dict(r)
    out["summary"] = loads(out.get("summary"))
    return out


def _record(outreach_id: int, by: str, instruction: str | None = None) -> None:
    """Save the draft as it is now as a version, so any version can be restored."""
    m = get(outreach_id)
    with connect() as conn:
        conn.execute(
            """INSERT INTO outreach_revisions
               (outreach_id, subject, body, summary, by, instruction, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (outreach_id, m["subject"], m["body"], dumps(m["summary"]), by, instruction, _now()),
        )


def _editable(outreach_id: int) -> dict:
    current = get(outreach_id)
    if current is None or current["status"] != "drafted":
        raise OutreachRefused("only drafts can be edited")
    return current


def get(outreach_id: int) -> dict | None:
    with connect() as conn:
        r = conn.execute("SELECT * FROM outreach WHERE id = ?", (outreach_id,)).fetchone()
    return _decode(r) if r else None


def set_status(outreach_id: int, status: str, **fields) -> None:
    """Move an email along its path. fields may set scheduled_at, sent_at,
    gmail_message_id or gmail_thread_id."""
    allowed_fields = {"scheduled_at", "sent_at", "gmail_message_id", "gmail_thread_id"}
    if set(fields) - allowed_fields:
        raise ValueError(f"unknown outreach fields: {sorted(set(fields) - allowed_fields)}")
    current = get(outreach_id)
    if current is None:
        raise OutreachRefused(f"no outreach {outreach_id}")
    if status not in TRANSITIONS[current["status"]]:
        raise OutreachRefused(f"can't go from {current['status']} to {status}")
    if status in ("approved", "scheduled", "sent"):
        contact = contacts.get(current["contact_id"])
        if contact["do_not_contact"]:
            raise OutreachRefused(f"{contact['name']} is marked do-not-contact")
    values = {"status": status, "updated_at": _now(), **fields}
    sets = ", ".join(f"{k} = ?" for k in values)
    with connect() as conn:
        conn.execute(f"UPDATE outreach SET {sets} WHERE id = ?", (*values.values(), outreach_id))


def edit_draft(outreach_id: int, subject: str | None = None, body: str | None = None) -> None:
    """Change a draft's text by hand. Only drafts can be edited; approve again after.

    The summary is kept: a rewrite with Jarvie starts from your edited text anyway.
    """
    current = _editable(outreach_id)
    subject, body = subject or current["subject"], body or current["body"]
    if (subject, body) == (current["subject"], current["body"]):
        return  # nothing changed, so no new version
    with connect() as conn:
        conn.execute(
            "UPDATE outreach SET subject = ?, body = ?, updated_at = ? WHERE id = ?",
            (subject, body, _now(), outreach_id),
        )
    _record(outreach_id, "you")


def revise(outreach_id: int, subject: str, body: str, summary: dict, instruction: str) -> None:
    """Replace a draft with Jarvie's rewrite, keeping the old one as a version."""
    _editable(outreach_id)
    with connect() as conn:
        conn.execute(
            "UPDATE outreach SET subject = ?, body = ?, summary = ?, updated_at = ? WHERE id = ?",
            (subject.strip(), body.strip(), dumps(summary), _now(), outreach_id),
        )
    _record(outreach_id, "jarvie", instruction)


def revisions(outreach_id: int) -> list[dict]:
    """Every version of a draft, oldest first."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM outreach_revisions WHERE outreach_id = ? ORDER BY id", (outreach_id,)
        ).fetchall()
    return [_decode(r) for r in rows]


def restore(outreach_id: int, revision_id: int) -> None:
    """Go back to an earlier version (itself saved as a new version, so it can be undone)."""
    _editable(outreach_id)
    old = next((r for r in revisions(outreach_id) if r["id"] == revision_id), None)
    if old is None:
        raise OutreachRefused(f"no version {revision_id} of this draft")
    with connect() as conn:
        conn.execute(
            "UPDATE outreach SET subject = ?, body = ?, summary = ?, updated_at = ? WHERE id = ?",
            (old["subject"], old["body"], dumps(old["summary"]), _now(), outreach_id),
        )
    _record(outreach_id, "you", f"went back to version {revision_id}")


def queue(status: str | None = None) -> list[dict]:
    """Emails with the person's name, email and company, oldest first."""
    sql = """SELECT outreach.*, contacts.name, contacts.email, contacts.company
             FROM outreach JOIN contacts ON contacts.id = outreach.contact_id"""
    args: tuple = ()
    if status:
        if status not in TRANSITIONS:
            raise ValueError(f"status must be one of {sorted(TRANSITIONS)}")
        sql += " WHERE outreach.status = ?"
        args = (status,)
    sql += " ORDER BY outreach.id"
    with connect() as conn:
        return [_decode(r) for r in conn.execute(sql, args).fetchall()]
