"""Voice/text tools for postings and application tracking."""

from pipecat.services.llm_service import FunctionCallParams

from .. import jobs
from . import postings


def _short(p: dict) -> dict:
    """What the voice model needs to talk about a posting, without the full description."""
    return {
        "posting_id": p["id"],
        "company": p["company"],
        "title": p["title"],
        "kind": p["kind"],
        "location": p["location"],
        "deadline": p["deadline"],
        "fields": p["fields"],
        "paid": p["paid"],
        "fit_score": p["fit_score"],
        "status": p["status"],
    }


async def list_postings(
    params: FunctionCallParams,
    kind: str | None = None,
    field: str | None = None,
    status: str | None = None,
    min_fit: float | None = None,
    sort: str = "fit",
    limit: int = 5,
    query: str | None = None,
):
    """List saved internships and research positions.

    Call this for "what internships did you find", "show me ML ones", "what have I
    applied to". Read back company and title, a few at a time; ids are for later calls,
    never say them aloud.

    When the user names a company or role ("the CACI one", "the Jane Street quant
    internship"), pass it as query to find that posting. If several match, read their
    titles back and ask which one. If none match, the name may have been misheard, so
    try other spellings of it (e.g. "Casey" may be "CACI") before saying it isn't saved.

    Args:
        query: Part of the company name or title, e.g. "caci" or "quant".
        kind: "internship" or "research".
        field: e.g. "machine learning", "software engineering", "biology", "biotech".
        status: "none" for ones not acted on yet, or interested, applied, oa,
            interviewing, offer, rejected, withdrawn.
        min_fit: Only postings scored at least this (0 to 1).
        sort: "fit", "deadline" or "newest".
        limit: How many to return.
    """
    try:
        found = postings.search(kind, field, status, min_fit, sort, limit, query)
    except ValueError as e:
        await params.result_callback({"error": str(e)})
        return
    await params.result_callback({"found": bool(found), "postings": [_short(p) for p in found]})


async def get_posting(params: FunctionCallParams, posting_id: int):
    """Everything known about one posting: scores with reasons, skills, application steps.

    Call this before answering questions about a specific posting ("how competitive is
    it?", "does it need essays?", "who owns my work there?").
    """
    p = postings.get(posting_id)
    if p is None:
        await params.result_callback({"found": False})
        return
    p = {k: v for k, v in p.items() if k != "raw_text"}
    p["history"] = postings.history(posting_id)
    p["resume_used"] = _resume_used(p)
    await params.result_callback({"found": True, "posting": p})


def _resume_used(p: dict) -> str | None:
    from .profile import ProfileError, load

    try:
        return load().resume_for(p).name
    except ProfileError:
        return None


async def set_posting_resume(params: FunctionCallParams, posting_id: int, resume: str):
    """Choose which of the user's resumes a posting uses (for cover letters, essays and
    the apply assistant).

    Call this for "use my QR resume for the Citadel one". Say "auto" to go back to
    picking by the posting's fields.

    Args:
        resume: The resume's name from the profile, e.g. "swe", "mle", "qr", or "auto".
    """
    from .profile import ProfileError, load

    if postings.get(posting_id) is None:
        await params.result_callback({"error": f"no posting {posting_id}"})
        return
    try:
        name = None if resume.strip().lower() == "auto" else load().resume(resume).name
    except ProfileError as e:
        await params.result_callback({"error": str(e)})
        return
    postings.set_resume(posting_id, name)
    await params.result_callback({"updated": True, "resume": name or "auto"})


async def set_application_status(
    params: FunctionCallParams, posting_id: int, status: str, note: str | None = None
):
    """Record where an application stands.

    Call this when the user says "I applied to the Acme one", "got an OA from Acme",
    "Acme rejected me", "I have an interview Friday" (note the date in note). Find the
    posting_id first with list_postings(query="Acme"); it works for any saved posting,
    whether or not it has a status yet.

    Args:
        status: interested, applied, oa, interviewing, offer, rejected or withdrawn.
        note: Anything worth remembering, in the user's words.
    """
    try:
        postings.set_status(posting_id, status, note)
    except ValueError as e:
        await params.result_callback({"error": str(e)})
        return
    await params.result_callback({"updated": True})


async def set_coffee_chat(params: FunctionCallParams, posting_id: int, had_chat: bool = True):
    """Mark that the user had (or didn't have) a coffee chat for this posting's company.

    Call this when the user says "I had a coffee chat with someone at Acme".
    """
    ok = postings.set_coffee_chat(posting_id, had_chat)
    await params.result_callback(
        {"updated": ok, **({} if ok else {"hint": "set an application status first"})}
    )


async def pipeline_summary(params: FunctionCallParams):
    """Counts of applications at each stage, plus coffee chats.

    Call this for "how's my internship search going?" or "how many OAs do I have?".
    """
    await params.result_callback(postings.pipeline_summary())


async def add_posting(params: FunctionCallParams, url_or_text: str):
    """Save and score one posting the user gives: a link, or pasted posting text.

    Call this when the user pastes a Handshake link or posting text, or gives a URL.
    It runs in the background (the research agent reads and scores it); tell the user,
    and use job_status later. Handshake links usually need login, so suggest pasting
    the posting text if the link alone fails.
    """
    text = url_or_text.strip()
    if not text:
        await params.result_callback({"error": "nothing to add"})
        return
    is_url = text.startswith(("http://", "https://")) and " " not in text
    job_id = jobs.enqueue("score_posting", {"url": text} if is_url else {"text": text})
    await params.result_callback({"started": True, "job_id": job_id})


async def write_cover_letter(params: FunctionCallParams, posting_id: int, notes: str | None = None):
    """Have Claude write a tailored cover letter for a saved posting, in the background.

    Call this for "write me a cover letter for the Acme internship". It appears on the
    dashboard's Documents tab; claims not backed by the resume are flagged there.

    Args:
        notes: Anything the user wants emphasized, in their words.
    """
    job_id = jobs.enqueue(
        "write_document", {"posting_id": posting_id, "kind": "cover_letter", "notes": notes or ""}
    )
    await params.result_callback({"started": True, "job_id": job_id})


async def write_essay(
    params: FunctionCallParams,
    posting_id: int,
    question: str,
    word_limit: int | None = None,
    notes: str | None = None,
):
    """Have Claude draft an application essay answering a specific prompt.

    Call this for "help with Acme's 'why us' essay, 250 words". Long prompts are
    usually typed; pass the prompt exactly as given.

    Args:
        question: The essay prompt, verbatim.
        word_limit: The application's limit, if any.
        notes: Points the user wants made.
    """
    job_id = jobs.enqueue(
        "write_document",
        {
            "posting_id": posting_id,
            "kind": "essay",
            "question": question,
            "word_limit": word_limit,
            "notes": notes or "",
        },
    )
    await params.result_callback({"started": True, "job_id": job_id})


async def list_documents(params: FunctionCallParams, posting_id: int | None = None):
    """Cover letters and essays written so far, newest first, without their full text.

    Call this for "is my Acme cover letter ready?". Point the user to the dashboard to
    read or copy them rather than reading a whole letter aloud.
    """
    from . import documents

    items = documents.for_posting(posting_id)
    await params.result_callback(
        {
            "documents": [
                {
                    "document_id": d["id"],
                    "kind": d["kind"],
                    "company": d["company"],
                    "title": d["title"],
                    "version": d["version"],
                    "words": len(d["text"].split()),
                    "flagged_claims": len(d["flagged"]),
                }
                for d in items[:10]
            ]
        }
    )


CAREER_TOOLS = [
    write_cover_letter,
    write_essay,
    list_documents,
    list_postings,
    get_posting,
    set_posting_resume,
    set_application_status,
    set_coffee_chat,
    pipeline_summary,
    add_posting,
]
