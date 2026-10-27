"""Voice/text tools for people and emails, independent of the career pipeline.

Broad people searches go through a brief (start_brief with kind "find_contacts" in
job_tools.py); these tools cover single lookups, adding people, and reading results.
"""

from pipecat.services.llm_service import FunctionCallParams

from .. import jobs
from . import contacts, messages


def _short(c: dict) -> dict:
    return {
        "contact_id": c["id"],
        "name": c["name"],
        "role": c["role"],
        "company": c["company"],
        "alumni_of": c["alumni_of"],
        "has_email": bool(c["email"]),
        "email_verified": c["email_verified"],
        "do_not_contact": c["do_not_contact"],
    }


async def find_email(
    params: FunctionCallParams,
    name: str | None = None,
    company_or_domain: str | None = None,
    contact_id: int | None = None,
):
    """Look up one person's work email address in the background.

    Call this for "find the email for Dana Lee at Genentech" (name + company_or_domain)
    or to fill in a saved contact's missing email (contact_id). For finding new people,
    start a brief with kind "find_contacts" instead. Results arrive via job_status.

    Args:
        name: The person's full name.
        company_or_domain: e.g. "Genentech" or "gene.com".
        contact_id: A saved contact, instead of name and company.
    """
    if contact_id is not None:
        job_id = jobs.enqueue("find_email", {"contact_id": contact_id})
    elif name and company_or_domain:
        job_id = jobs.enqueue("find_email", {"name": name, "domain": company_or_domain})
    else:
        await params.result_callback({"error": "need a name and company, or a contact_id"})
        return
    await params.result_callback({"started": True, "job_id": job_id, **jobs.worker_note()})


async def add_contact(
    params: FunctionCallParams,
    name: str,
    email: str | None = None,
    company: str | None = None,
    role: str | None = None,
    notes: str | None = None,
):
    """Save someone the user names themselves, e.g. a professor they met.

    Call this for "add Professor Lee, lee@berkeley.edu, to my contacts". Use only what
    the user said.
    """
    try:
        cid = contacts.add(
            name, source="user", email=email, company=company, role=role, notes=notes
        )
    except ValueError as e:
        await params.result_callback({"error": str(e)})
        return
    await params.result_callback({"added": True, "contact_id": cid})


async def list_contacts(
    params: FunctionCallParams,
    company: str | None = None,
    school: str | None = None,
    has_email: bool | None = None,
    limit: int = 10,
):
    """List saved people, newest first.

    Call this for "who did you find at Genentech?" or "which Berkeley alumni do I have?".
    Say names and roles; never read email addresses or ids aloud unless asked.

    Args:
        school: "berkeley" or "nnhs".
        has_email: true for only people with an address, false for those without.
    """
    found = contacts.search(company, school, has_email, limit)
    await params.result_callback({"found": bool(found), "contacts": [_short(c) for c in found]})


async def do_not_contact(params: FunctionCallParams, contact_id: int):
    """Never email this person, and cancel anything queued to them.

    Call this when the user says "don't email her", "take him off the list".
    """
    await params.result_callback({"updated": contacts.mark_do_not_contact(contact_id)})


async def outreach_queue(params: FunctionCallParams):
    """How many emails are waiting at each step, and the drafts awaiting approval.

    Call this for "what emails are waiting for me?". Approving and sending happen on the
    dashboard (localhost:7861/outreach), not by voice; tell the user to review there.
    """
    counts: dict[str, int] = {}
    for item in messages.queue():
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    drafts = [
        {"outreach_id": m["id"], "to": m["name"], "company": m["company"], "subject": m["subject"]}
        for m in messages.queue("drafted")
    ]
    await params.result_callback({"counts": counts, "drafts": drafts[:10]})


async def draft_emails(
    params: FunctionCallParams,
    contact_ids: list[int],
    purpose: str = "coffee_chat",
    notes: str | None = None,
    posting_id: int | None = None,
    resume: str | None = None,
):
    """Have the writer draft personal emails to saved contacts, in the background.

    Call this for "draft coffee chat emails to the Genentech recruiters" (use
    list_contacts to get their ids). Drafts wait on the dashboard for the user to edit
    and approve; nothing is sent without approval.

    Args:
        contact_ids: Who to write to.
        purpose: coffee_chat, research_inquiry, referral, advice or other.
        notes: Anything the user wants mentioned, in their words.
        posting_id: A related saved posting, if any.
        resume: Which of the user's resumes to draw on (e.g. "swe", "mle", "qr"), if
            they say; otherwise the posting's resume or their default.
    """
    if purpose not in messages.PURPOSES:
        await params.result_callback({"error": f"purpose must be one of {messages.PURPOSES}"})
        return
    job_id = jobs.enqueue(
        "draft_outreach",
        {
            "contact_ids": contact_ids,
            "purpose": purpose,
            "notes": notes or "",
            "posting_id": posting_id,
            "resume": resume or "",
        },
    )
    await params.result_callback({"started": True, "job_id": job_id, **jobs.worker_note()})


OUTREACH_TOOLS = [
    find_email,
    add_contact,
    list_contacts,
    do_not_contact,
    draft_emails,
    outreach_queue,
]
