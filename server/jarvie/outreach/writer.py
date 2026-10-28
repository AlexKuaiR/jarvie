"""Drafting personal outreach emails from what the research agent stored about a person.

The writer (Claude Sonnet 5.5) only sees the person's stored features, your resume, and
the purpose. Every draft then passes a code-level grounding check before it's saved:
the writer must say which stored facts it used, and drafts that mention specific
things (a talk, a paper) not present in those facts are rejected.
"""

import re
from dataclasses import dataclass

from pydantic import BaseModel, Field

from ..briefs import AUTO_RESUME
from ..llm import ClaudeWriter, LLMError, Usage, WriterModel, cached_system, write_many
from . import contacts, messages


class Personalization(BaseModel):
    feature_key: str = Field(description="Which stored fact was used, e.g. 'school', or 'resume'")
    how_used: str = Field(description="The phrase in the email that uses it")


class DraftEmail(BaseModel):
    subject: str
    body: str
    personalization_used: list[Personalization]


PURPOSE_ASKS = {
    "coffee_chat": "a 15-minute call or coffee chat about their path and their team",
    "research_inquiry": "whether they're taking undergraduate researchers, and a short meeting",
    "referral": "advice on applying, and whether they'd be open to a referral if it fits",
    "advice": "15 minutes of advice on one specific question",
    "other": "a short conversation",
}

SYSTEM = """\
You write short, personal cold emails from a student to one person. Rules:
- 90 to 140 words in the body. Plain text, no markdown, no emoji.
- Open by addressing them by first name (or "Professor <Last>" for faculty).
- Use exactly one or two specific, true connections from the person's stored facts
  (shared school, their team, their work). Use only facts given; never invent
  shared interests, events, talks, papers or compliments.
- One sentence on who the student is, drawn from the resume.
- One clear, modest ask, as given. Offer flexibility on timing.
- Sign off with the student's name only. No placeholders like [Name].
- Subject under 70 characters, specific, not clickbait.
List every fact you used in personalization_used (feature_key = the fact's key, or
"resume" for the student's own background).
"""


@dataclass
class Sender:
    name: str
    background: str  # the resume text


def load_sender(
    resume: str | None = None, posting: dict | None = None, contact: dict | None = None
) -> Sender:
    """Your name and resume, from the career profile (shared with the job search).

    resume names one of your resumes; otherwise a related posting's resume is used, or
    your default one. resume "auto" picks for the contact: their role and team are
    matched like a posting's title and fields (an ML manager gets your ML resume).
    """
    from ..career import profile
    from ..career.filters import tag_fields

    p = profile.load()
    if not p.name:
        raise ValueError("add your name under [applicant] in career_profile.toml to sign emails")
    if resume == AUTO_RESUME:
        resume = None
        if contact:
            role = contact.get("role") or ""
            team = " ".join(str(v) for v in (contact.get("features") or {}).values())
            posting = {"title": role, "fields": tag_fields(role, team)}
    chosen = p.resume(resume) if resume else p.resume_for(posting) if posting else p.resume()
    return Sender(p.name, chosen.text())


# Specific things a cold email might claim to know; each must appear in the stored facts.
SPECIFIC_CLAIMS = (
    "talk",
    "paper",
    "podcast",
    "article",
    "post",
    "keynote",
    "interview",
    "award",
    "book",
)
PLACEHOLDER = re.compile(r"\[[^\]]+\]|\{[^}]+\}|<[^>]+>")


def check_draft(draft: DraftEmail, contact: dict, sender: Sender) -> list[str]:
    """Problems with a draft, or [] if it may be queued for your review."""
    problems = []
    words = len(draft.body.split())
    if not 60 <= words <= 170:
        problems.append(f"body is {words} words; keep it 90-140")
    if not draft.subject.strip() or len(draft.subject) > 90:
        problems.append("subject missing or too long")
    if PLACEHOLDER.search(draft.subject + draft.body):
        problems.append("contains a placeholder")
    if sender.name.split()[0].lower() not in draft.body.lower():
        problems.append("not signed with the student's name")
    first = contact["name"].replace("Dr.", "").replace("Prof.", "").split()[0].lower()
    last = contact["name"].split()[-1].lower()
    if first not in draft.body.lower() and last not in draft.body.lower():
        problems.append("doesn't address the person by name")
    features = contact.get("features") or {}
    for item in draft.personalization_used:
        if item.feature_key != "resume" and item.feature_key not in features:
            problems.append(f"used a fact that isn't stored: {item.feature_key}")
    known = " ".join(str(v) for v in features.values()).lower()
    for claim in SPECIFIC_CLAIMS:
        if re.search(rf"\byour {claim}s?\b", draft.body, re.I) and claim not in known:
            problems.append(f"mentions their {claim}, which isn't in the stored facts")
    return problems


def prompt_for(
    contact: dict, purpose: str, notes: str, sender: Sender, posting: dict | None
) -> str:
    facts = "\n".join(f"- {k}: {v}" for k, v in (contact.get("features") or {}).items()) or "- none"
    lines = [
        f"Recipient: {contact['name']}, {contact.get('role') or 'role unknown'}"
        f"{' at ' + contact['company'] if contact.get('company') else ''}.",
        f"Stored facts about them:\n{facts}",
        f"Alumni of: {', '.join(contact.get('alumni_of') or []) or 'no shared school known'}",
        f"Purpose: {purpose.replace('_', ' ')}. Ask for: {PURPOSE_ASKS.get(purpose, PURPOSE_ASKS['other'])}.",
    ]
    if posting:
        lines.append(
            f"Related opening (mention briefly, don't ask them to get you the job): "
            f"{posting['title']} at {posting['company']}."
        )
    if notes:
        lines.append(f"Student's notes: {notes}")
    lines.append(f"Student: {sender.name}. Resume:\n{sender.background}")
    return "\n".join(lines)


BATCH_THRESHOLD = 10  # at least this many drafts go through the half-price Batches API


def draft_outreach(
    job: dict, writer: WriterModel, spend: Usage, sender: Sender | None = None
) -> dict:
    """Draft one email per contact, check each, and queue passing drafts for review."""
    p = job["params"]
    purpose = p.get("purpose", "coffee_chat")
    if purpose not in messages.PURPOSES:
        raise ValueError(f"purpose must be one of {messages.PURPOSES}")
    posting = None
    if p.get("posting_id"):
        from ..career import postings  # optional link to the job search

        posting = postings.get(int(p["posting_id"]))
    resume = p.get("resume") or None
    per_person = sender is None and resume == AUTO_RESUME  # each contact gets their own resume
    if sender is None and not per_person:
        sender = load_sender(resume, posting)

    ids = [int(i) for i in p.get("contact_ids") or []]
    if not ids:
        raise ValueError("draft_outreach needs contact_ids")
    skipped: dict[int, str] = {}
    targets = []
    for cid in ids:
        c = contacts.get(cid)
        if c is None:
            skipped[cid] = "no such contact"
        elif c["do_not_contact"]:
            skipped[cid] = "do not contact"
        elif not c["email"]:
            skipped[cid] = "no email address"
        elif any(
            m["contact_id"] == cid and m["status"] in ("drafted", "approved", "scheduled", "sent")
            for m in messages.queue()
        ):
            skipped[cid] = "already has an email in progress"
        else:
            targets.append(c)

    senders = [load_sender(resume, posting, c) if per_person else sender for c in targets]
    system = cached_system(SYSTEM)
    prompts = [
        prompt_for(c, purpose, p.get("notes", ""), me, posting) for c, me in zip(targets, senders)
    ]
    if len(prompts) >= BATCH_THRESHOLD and isinstance(writer, ClaudeWriter):
        first_pass, usage = write_many(writer, system, prompts, DraftEmail)
        spend.add(usage)
    else:
        first_pass = []
        for prompt in prompts:
            try:
                draft, usage = writer.write(system, prompt, DraftEmail)
                spend.add(usage)
                first_pass.append((draft, None))
            except LLMError as e:
                first_pass.append((None, str(e)))

    drafted, rejected = [], {}
    for c, me, prompt, (draft, error) in zip(targets, senders, prompts, first_pass):
        problems = [error] if error else check_draft(draft, c, me)
        if problems and not error:
            # One retry, telling the writer exactly what to fix.
            retry = (
                f"{prompt}\n\nYour previous draft had these problems; fix them:\n- "
                + "\n- ".join(problems)
            )
            draft, usage = writer.write(system, retry, DraftEmail)
            spend.add(usage)
            problems = check_draft(draft, c, me)
        if problems:
            rejected[c["id"]] = problems
            continue
        drafted.append(
            messages.add_draft(
                c["id"], draft.subject, draft.body, purpose, posting and posting["id"]
            )
        )
    return {"drafted": drafted, "rejected": rejected, "skipped": skipped}
