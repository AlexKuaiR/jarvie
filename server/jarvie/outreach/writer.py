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


class EmailSummary(BaseModel):
    """The plan for one email, written before the email itself and kept with the draft."""

    gist: str = Field(description="One or two sentences: what the email says and asks for")
    angle: str = Field(description="Why this person: the connection the email leads with")
    points: list[str] = Field(description="The 2-3 things the email will say, in order")
    facts_used: list[str] = Field(
        description="Keys of the stored facts it relies on, or 'resume' for the student"
    )
    tone: str = Field(description="e.g. warm and brief, formal for faculty")


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

SUMMARY_SYSTEM = """\
You plan short, personal cold emails from a student to one person, before they are
written. Read what's known about the person and the student, then decide: the gist (what
the email says and asks), the angle (the one true connection to lead with), two or three
points in order, which stored facts it relies on (by key), and the tone. Use only the
facts given; never plan around anything not stated. Keep it brief: the summary is read
aloud to the student and is the basis for the email.
"""

DRAFT_FROM_SUMMARY = """\
Write the email from this summary. Follow its angle, points and tone; use only facts
it lists, plus the stored facts above.
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
    features = facts_of(contact)
    for item in draft.personalization_used:
        if item.feature_key != "resume" and item.feature_key not in features:
            problems.append(f"used a fact that isn't stored: {item.feature_key}")
    known = " ".join(str(v) for v in features.values()).lower()
    for claim in SPECIFIC_CLAIMS:
        if re.search(rf"\byour {claim}s?\b", draft.body, re.I) and claim not in known:
            problems.append(f"mentions their {claim}, which isn't in the stored facts")
    return problems


def facts_of(contact: dict) -> dict:
    """What the writer may personalize from: stored facts, plus your notes on the person."""
    facts = dict(contact.get("features") or {})
    if contact.get("notes"):
        facts["your_notes"] = contact["notes"]
    return facts


def format_summary(summary: dict) -> str:
    points = "\n".join(f"  {i}. {pt}" for i, pt in enumerate(summary.get("points") or [], 1))
    return (
        f"Gist: {summary.get('gist', '')}\nAngle: {summary.get('angle', '')}\n"
        f"Points:\n{points}\nFacts used: {', '.join(summary.get('facts_used') or [])}\n"
        f"Tone: {summary.get('tone', '')}"
    )


def prompt_for(
    contact: dict, purpose: str, notes: str, sender: Sender, posting: dict | None
) -> str:
    """Everything known about the person and the student: the basis for every step."""
    facts = "\n".join(f"- {k}: {v}" for k, v in facts_of(contact).items()) or "- none"
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
    contexts = [
        prompt_for(c, purpose, p.get("notes", ""), me, posting) for c, me in zip(targets, senders)
    ]
    # Two steps: plan each email (the summary you'll read and revise), then write it.
    summaries = _write_all(writer, SUMMARY_SYSTEM, contexts, EmailSummary, spend)
    drafted, rejected = [], {}
    ready = []
    for c, me, context, (summary, error) in zip(targets, senders, contexts, summaries):
        if error:
            rejected[c["id"]] = [error]
        else:
            ready.append(
                (
                    c,
                    me,
                    f"{context}\n\n{DRAFT_FROM_SUMMARY}\n{format_summary(summary.model_dump())}",
                    summary,
                )
            )
    first_pass = _write_all(writer, SYSTEM, [r[2] for r in ready], DraftEmail, spend)
    for (c, me, prompt, summary), (draft, error) in zip(ready, first_pass):
        draft, problems = _checked(writer, prompt, draft, error, c, me, spend)
        if problems:
            rejected[c["id"]] = problems
            continue
        drafted.append(
            messages.add_draft(
                c["id"],
                draft.subject,
                draft.body,
                purpose,
                posting and posting["id"],
                summary=summary.model_dump(),
                resume=resume,
            )
        )
    return {"drafted": drafted, "rejected": rejected, "skipped": skipped}


def _write_all(
    writer: WriterModel, system: str, prompts: list[str], schema, spend: Usage
) -> list[tuple]:
    """One (result, error) per prompt; many at once go through the half-price batch API."""
    if len(prompts) >= BATCH_THRESHOLD and isinstance(writer, ClaudeWriter):
        out, usage = write_many(writer, cached_system(system), prompts, schema)
        spend.add(usage)
        return out
    out = []
    for prompt in prompts:
        try:
            result, usage = writer.write(cached_system(system), prompt, schema)
            spend.add(usage)
            out.append((result, None))
        except LLMError as e:
            out.append((None, str(e)))
    return out


def _checked(writer, prompt, draft, error, contact, sender, spend) -> tuple:
    """The grounding check, with one retry that says exactly what to fix."""
    if error:
        return None, [error]
    problems = check_draft(draft, contact, sender)
    if problems:
        retry = f"{prompt}\n\nYour previous draft had these problems; fix them:\n- " + "\n- ".join(
            problems
        )
        draft, usage = writer.write(cached_system(SYSTEM), retry, DraftEmail)
        spend.add(usage)
        problems = check_draft(draft, contact, sender)
    return draft, problems


REVISE_SUMMARY = """\
The student wants this email changed. Here is the current summary and the current text
(which may include the student's own edits; keep those unless asked otherwise):

{summary}

Subject: {subject}
{body}

The student asks: {instruction}

Write the updated summary: change what they asked for and keep the rest.
"""


def revise_outreach(job: dict, writer: WriterModel, spend: Usage) -> dict:
    """Rewrite one draft with you: update its summary from your request, then the email.

    The grounding check still applies, so a request to mention something not stored
    about the person is refused (save it with add_contact_fact first). A rejected
    rewrite leaves the draft as it was.
    """
    p = job["params"]
    instruction = (p.get("instruction") or "").strip()
    if not instruction:
        raise ValueError("say what to change")
    m = messages.get(int(p["outreach_id"]))
    if m is None or m["status"] != "drafted":
        raise ValueError("only drafts can be rewritten")
    contact = contacts.get(m["contact_id"])
    posting = None
    if m["posting_id"]:
        from ..career import postings

        posting = postings.get(m["posting_id"])
    me = load_sender(m["resume"] or None, posting, contact)
    context = prompt_for(contact, m["purpose"], "", me, posting)
    summary_text = format_summary(m["summary"]) if m["summary"] else "(none yet)"
    ask = REVISE_SUMMARY.format(
        summary=summary_text, subject=m["subject"], body=m["body"], instruction=instruction
    )
    [(summary, error)] = _write_all(
        writer, SUMMARY_SYSTEM, [f"{context}\n\n{ask}"], EmailSummary, spend
    )
    if error:
        raise LLMError(error)
    prompt = (
        f"{context}\n\nThe student's current version, to rework (keep their edits unless "
        f"the summary changes them):\nSubject: {m['subject']}\n{m['body']}\n\n"
        f"{DRAFT_FROM_SUMMARY}\n{format_summary(summary.model_dump())}"
    )
    [(draft, error)] = _write_all(writer, SYSTEM, [prompt], DraftEmail, spend)
    draft, problems = _checked(writer, prompt, draft, error, contact, me, spend)
    if problems:
        return {"revised": False, "outreach_id": m["id"], "problems": problems}
    messages.revise(m["id"], draft.subject, draft.body, summary.model_dump(), instruction)
    return {"revised": True, "outreach_id": m["id"], "summary": summary.model_dump()}
