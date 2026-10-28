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


async def start_brief(
    params: FunctionCallParams, kind: str, answers: dict | None = None, start: bool = False
):
    """Start the questions for a big search, before anything runs.

    Call this when the user wants a broad search: "find me internships" uses kind
    "discover_postings"; "find Berkeley alumni recruiters at Genentech" or "find
    professors doing protein design" uses kind "find_contacts". Then ask the returned
    questions one or two at a time, in plain words, and record answers with
    answer_brief. If open_briefs shows one of the same kind already open, continue that
    one instead of starting another.

    For "discover_postings", two questions decide whether AI is used at all; always ask
    them, plainly: search_beyond_boards (AI web search beyond the job boards) and
    ai_score (how many results AI scores: none, a number, or all). Answering no and
    none makes a free search that uses only the job boards.

    Answers from the user's profile are pre-filled, but they are only defaults: pass
    everything the user already said in their request as answers, which replaces them
    (e.g. "find me five paid biotech internships" -> {"fields": "biotech",
    "max_results": 5, "paid_only": "yes"}). Read the remaining pre-filled answers back
    briefly so the user can change them.

    Args:
        kind: "discover_postings" or "find_contacts".
        answers: Question key -> what the user already said, in their words.
        start: True to start the search right away if nothing required is left (see
            answer_brief).
    """
    try:
        brief_id = briefs.start(kind, prefill=_brief_prefill(kind))
    except briefs.BriefError as e:
        await params.result_callback({"error": str(e)})
        return
    result: dict = {"brief_id": brief_id}
    if answers:
        try:
            briefs.answer(brief_id, answers)
        except briefs.BriefError as e:
            result["error"] = f"{e}; the brief is open, fix that answer with answer_brief"
    result.update(_maybe_start(brief_id, start and "error" not in result))
    if not result.get("started"):
        result["questions"] = briefs.remaining(brief_id)
    await params.result_callback(result)


async def answer_brief(
    params: FunctionCallParams,
    brief_id: int,
    answers: dict,
    channel: str = "voice",
    start: bool = False,
):
    """Record the user's answers to brief questions, in their words.

    Pass whatever the user said for each question key, e.g. {"term": "Summer 2027",
    "fields": "machine learning and biotech"}; lists and yes/no are understood. An
    answer replaces a pre-filled one. Record each answer as soon as the user gives it,
    before asking the next question. Company lists (include_companies,
    exclude_companies) must be a JSON list of names, e.g. ["Merck", "Johnson & Johnson"],
    never one sentence; put descriptions such as "big companies" or "promising startups"
    in notes. If the result has an error,
    explain it and ask again. When "remaining" has no required questions left, read
    the answers back in one sentence and offer to start.

    Pass start=true when the user has asked for the search to run: they said "start",
    "go", "yes", or their request already asked for results ("find them and draft
    emails"). It starts in this same call once nothing required is left. Only say a
    search is running when a result has "started": true; otherwise it hasn't started.

    Args:
        brief_id: From start_brief or open_briefs.
        answers: Question key -> the user's answer.
        channel: "text" if the user typed (long pasted requests usually are), else "voice".
        start: True to start the search as soon as the required answers are in.
    """
    try:
        briefs.answer(brief_id, answers, channel if channel in briefs.CHANNELS else "voice")
    except briefs.BriefError as e:
        await params.result_callback({"error": str(e)})
        return
    result = _maybe_start(brief_id, start)
    if not result.get("started"):
        result["remaining"] = briefs.remaining(brief_id)
    await params.result_callback(result)


def _maybe_start(brief_id: int, start: bool) -> dict:
    """The brief's answers, and whether it started: it starts only if asked and ready."""
    brief = briefs.get(brief_id)
    missing = [q["key"] for q in briefs.remaining(brief_id) if q["required"]]
    if start and not missing:
        job_id = briefs.submit(brief_id)
        return {
            "answers": brief["answers"],
            "started": True,
            "job_id": job_id,
            **jobs.worker_note(),
        }
    note = (
        f"Not started: still need {', '.join(missing)}."
        if missing
        else "Not started yet. Ask whether to start, then call submit_brief (or answer_brief "
        "with start=true). Don't say it's running until a result says started."
    )
    return {"answers": brief["answers"], "started": False, "tell_user": note}


async def submit_brief(params: FunctionCallParams, brief_id: int):
    """Start the search described by a brief. It runs in the background.

    Call this once the user wants it to run: they agreed, or their request already
    asked for the results. Tell them it's queued (with the job id) and that they can
    ask for its status; don't wait for it. Never say a search is running unless a
    result says "started": true.
    """
    try:
        job_id = briefs.submit(brief_id)
    except briefs.BriefError as e:
        await params.result_callback({"error": str(e)})
        return
    await params.result_callback({"started": True, "job_id": job_id, **jobs.worker_note()})


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
    job_id out for the five most recent jobs. If the result says the worker isn't
    running, say so: queued jobs won't start until it is. Summarize results briefly; for postings,
    use list_postings to read them.

    Args:
        job_id: A specific job, from submit_brief or another tool that started one.
    """
    if job_id is not None:
        job = jobs.get(job_id)
        note = jobs.worker_note() if job and job["status"] == "queued" else {}
        await params.result_callback({"found": job is not None, "job": job and _brief(job), **note})
        return
    recent = jobs.recent(5)
    note = jobs.worker_note() if any(j["status"] == "queued" for j in recent) else {}
    await params.result_callback({"jobs": [_brief(j) for j in recent], **note})


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
