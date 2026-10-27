"""Voice/text tools for search briefs and background jobs, shared by career and outreach.

Typed messages from the localhost text box reach the same conversation as speech, so
these tools serve both: a brief started by voice can be finished by typing, and the
dashboard edits the same brief in the database.
"""

from pipecat.services.llm_service import FunctionCallParams

from . import briefs, jobs


def _brief_prefill(kind: str) -> dict:
    """Answers already known from your career profile, so Jarvie doesn't re-ask them."""
    from .career import profile  # imported here so outreach works without a profile

    try:
        p = profile.load()
    except profile.ProfileError:
        return {}
    if kind == "find_contacts":
        # Only worth asking which resume to use when there's more than one.
        return {"resume": ""} if len(p.resumes) == 1 else {}
    if kind != "discover_postings":
        return {}
    return {
        "kinds": p.kinds,
        "fields": p.fields,
        "locations": p.locations,
        "paid_only": p.paid_only,
        "min_prestige": p.min_prestige,
    }


async def start_brief(params: FunctionCallParams, kind: str):
    """Start the questions for a big search, before anything runs.

    Call this when the user wants a broad search: "find me internships" uses kind
    "discover_postings"; "find Berkeley alumni recruiters at Genentech" or "find
    professors doing protein design" uses kind "find_contacts". Then ask the returned
    questions one or two at a time, in plain words, and record answers with
    answer_brief. If open_briefs shows one of the same kind already open, continue that
    one instead of starting another.

    Answers already known from the user's profile are pre-filled; read them back
    briefly so the user can change them.

    Args:
        kind: "discover_postings" or "find_contacts".
    """
    try:
        brief_id = briefs.start(kind, prefill=_brief_prefill(kind))
    except briefs.BriefError as e:
        await params.result_callback({"error": str(e)})
        return
    await params.result_callback(
        {
            "brief_id": brief_id,
            "prefilled": briefs.get(brief_id)["answers"],
            "questions": briefs.remaining(brief_id),
        }
    )


async def answer_brief(
    params: FunctionCallParams, brief_id: int, answers: dict, channel: str = "voice"
):
    """Record the user's answers to brief questions, in their words.

    Pass whatever the user said for each question key, e.g. {"term": "Summer 2027",
    "fields": "machine learning and biotech"}; lists and yes/no are understood. If the
    result has an error, explain it and ask again. When "remaining" has no required
    questions left, offer to start the search with submit_brief.

    Args:
        brief_id: From start_brief or open_briefs.
        answers: Question key -> the user's answer.
        channel: "text" if the user typed (long pasted requests usually are), else "voice".
    """
    try:
        brief = briefs.answer(brief_id, answers, channel if channel in briefs.CHANNELS else "voice")
    except briefs.BriefError as e:
        await params.result_callback({"error": str(e)})
        return
    await params.result_callback(
        {"answers": brief["answers"], "remaining": briefs.remaining(brief_id)}
    )


async def submit_brief(params: FunctionCallParams, brief_id: int):
    """Start the search described by a brief. It runs in the background.

    Only call this after the user agrees to start. Tell them it's running and that
    they can ask for its status; don't wait for it.
    """
    try:
        job_id = briefs.submit(brief_id)
    except briefs.BriefError as e:
        await params.result_callback({"error": str(e)})
        return
    await params.result_callback({"started": True, "job_id": job_id})


async def open_briefs(params: FunctionCallParams):
    """List searches whose questions aren't finished yet, so they can be continued.

    Call this when the user says "let's continue that search" or before starting a new
    brief of the same kind.
    """
    items = briefs.open_briefs()
    await params.result_callback(
        {
            "briefs": [
                {"brief_id": b["id"], "kind": b["kind"], "answers": b["answers"]} for b in items
            ]
        }
    )


async def job_status(params: FunctionCallParams, job_id: int | None = None):
    """How a background search or lookup is going, or the most recent ones.

    Call this when the user asks "is my search done?" or "what did it find?". Leave
    job_id out for the five most recent jobs. Summarize results briefly; for postings,
    use list_postings to read them.

    Args:
        job_id: A specific job, from submit_brief or another tool that started one.
    """
    if job_id is not None:
        job = jobs.get(job_id)
        await params.result_callback({"found": job is not None, "job": job and _brief(job)})
        return
    await params.result_callback({"jobs": [_brief(j) for j in jobs.recent(5)]})


def _brief(job: dict) -> dict:
    return {
        "job_id": job["id"],
        "kind": job["kind"],
        "status": job["status"],
        "result": job["result"],
        "error": job["error"],
        "cost_usd": round(job["cost_usd"], 2),
    }


JOB_TOOLS = [start_brief, answer_brief, submit_brief, open_briefs, job_status]
