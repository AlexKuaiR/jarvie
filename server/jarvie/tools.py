import asyncio
import json
from datetime import datetime, time, timedelta

from loguru import logger
from pipecat.services.llm_service import FunctionCallParams

from . import calendar, deadlines, note
from .config import TZ


async def warmup():
    # prebuild slow clients while bot sets up
    logger.info("warmup: building calendar service")
    await asyncio.to_thread(calendar._service)
    logger.info("warmup: calendar ready")


def briefing() -> str | None:
    items = deadlines.upcoming(days=7)
    if not items:
        return None
    return (
        f"Upcoming this week: {json.dumps(items)}. "
        "Mention at most the two most urgent in your greeting, briefly. "
        "If the user acknowledges one, call mute_deadline."
    )


async def get_events(params: FunctionCallParams, day_offset: int = 0):
    """Get the user's calendar events for a single day.

    Call this for any question about classes, schedule, or plans on a
    specific day. Always call it rather than guessing.

    Args:
        day_offset: 0 for today, 1 for tomorrow, -1 for yesterday, 2 for
            the day after tomorrow, and so on.
    """
    target = datetime.now(TZ).date() + timedelta(days=day_offset)
    start = datetime.combine(target, time.min, tzinfo=TZ)
    try:
        events = calendar.between(start, start + timedelta(days=1))
    except Exception as e:
        logger.error("calendar failed: {}", e)
        await params.result_callback({"found": False, "error": "Calendar unavailable."})
        return
    await params.result_callback(
        {
            "date": target.isoformat(),
            "found": bool(events),
            "events": events,
        }
    )


async def get_deadlines(params: FunctionCallParams, days: int = 7):
    """Get the user's upcoming exams, assignments, and application deadlines.

    Call this for any question about tests, due dates, or what's coming up academically.
    Always call it rather than guessing. Each item's id is what mute_deadline,
    update_deadline and complete_deadline need.

    Args:
        days: How many days ahead to look.
    """
    out = deadlines.upcoming(days=days)
    await params.result_callback({"found": bool(out), "items": out})


async def add_deadlines(
    params: FunctionCallParams,
    name: str,
    date: str,
    course: str | None = None,
    kind: str = "exam",
    weight: float | None = None,
    prep_hours_needed: float | None = None,
    topics: str | None = None,
):
    """Record a new exam, assignment, or application deadline.

    Call this when a user mentions something due in the future that isn't already tracked,
    for example "I have a chem midterm on the 24th", "I have a chem homework due on the 20th".

    Resolve relative dates yourself using today's date from the system message. Never guess
    a date; if the user is too vague, ask. Only fill optional fields the user mentioned.

    Args:
        name: Short label, e.g. "Midterm 1".
        date: ISO 8601, "YYYY-MM-DD".
        course: Course name or code, e.g. "CHEM 101".
        kind: "exam", "assignment", or "application".
        weight: Fraction of the course grade, e.g. 0.25 for 25%.
        prep_hours_needed: The user's estimate of study hours needed.
        topics: What it covers, in the user's own words.
    """

    new_id = deadlines.add(
        name=name,
        date=date,
        course=course,
        kind=kind,
        weight=weight,
        topics=topics,
        prep_hours_needed=prep_hours_needed,
    )
    await params.result_callback({"added": True, "id": new_id, "name": name})


async def mute_deadline(params: FunctionCallParams, item_id: int, days: int = 3):
    """Stop reminding the user about a specific deadline for a few days.

    Call this when the user acknowledges an item — "got it", "I know about
    that one", "stop reminding me about the midterm".

    Use the id from get_deadlines. If you don't have one, call get_deadlines
    first to find it.

    Args:
        days: How many days to stay quiet. Use 0 to un-mute.
    """
    ok = deadlines.mute(item_id, days)
    await params.result_callback({"muted": ok})


async def complete_deadline(params: FunctionCallParams, item_id: int):
    """Mark a deadline as finished so it stops appearing in reminders.

    Call this when the user says they've done something — "I finished the
    essay", "took the midterm", "submitted the application".

    This is different from mute_deadline: complete means the work is done,
    mute means the user doesn't want to hear about it right now.

    Use the id from get_deadlines. If you don't have one, call get_deadlines
    first to find it.
    """
    ok = deadlines.complete(item_id)
    await params.result_callback({"completed": ok})


async def update_deadline(
    params: FunctionCallParams,
    item_id: int,
    name: str | None = None,
    date: str | None = None,
    course: str | None = None,
    kind: str | None = None,
    weight: float | None = None,
    prep_hours_needed: float | None = None,
    topics: str | None = None,
):
    """Change details of a deadline that already exists.

    Call this when something about a tracked item changes — the date moved,
    the topics got clearer, the weight was announced. "The midterm got pushed
    to Friday", "it also covers chapter 9".

    Only pass the fields that are changing; leave the rest out. This replaces
    a field's value rather than adding to it, so when updating topics, include
    the full new list, not just the addition.

    Use the id from get_deadlines. If you don't have one, call get_deadlines first.

    Args:
        date: ISO 8601, "YYYY-MM-DD".
        kind: "exam", "assignment", or "application".
        weight: Fraction of the course grade, e.g. 0.25 for 25%.
    """
    fields = {
        k: v
        for k, v in {
            "name": name,
            "date": date,
            "course": course,
            "kind": kind,
            "weight": weight,
            "prep_hours_needed": prep_hours_needed,
            "topics": topics,
        }.items()
        if v is not None
    }

    ok = deadlines.update(item_id, **fields)
    await params.result_callback({"updated": ok})


async def get_note(params: FunctionCallParams, status: str = "open", limit: int = 20):
    """Look up the user's dev notes: ideas, bugs and todos about Jarvie itself.

    Call this when the user asks what's on their list for the project —
    "what notes do I have", "what bugs did I log", "what have I finished".
    These are notes about building Jarvie, not course deadlines; use
    get_deadlines for those.

    Newest notes come first. Each item has an id, text, category,
    created_at and done_at (null while the note is still open).

    Args:
        status: "open" for unfinished notes (default), "done" for finished
            ones, or "all" for both.
        limit: Maximum number of notes to return.
    """
    try:
        out = note.open_notes(status, limit)
    except ValueError:
        # Hand the error back to the model so it can retry, instead of crashing the call.
        await params.result_callback(
            {"error": f'unknown status "{status}"; use "open", "done" or "all"'}
        )
        return
    await params.result_callback({"found": bool(out), "items": out})

async def add_note(params: FunctionCallParams, 
                   text: str, category: str = "feature"):
    """Save a dev note about building Jarvie: an idea, a bug, or a todo.

    Call this when the user wants to remember something about the project,
    e.g. "note that the wake word missed me twice", "idea: read my Zotero
    papers aloud". For exams, assignments or applications, use add_deadlines.

    Use the user's own words, trimmed to a sentence or two, without adding
    details. If they don't say what the note is, ask. Call once per note, and
    don't re-save a note you just added.

    Args:
        text: The note, e.g. "Wake word misses when music is playing".
        category: "bug" (something broken), "todo" (a concrete task), or
            "feature" (an idea or improvement; use this if unclear).
    """
    new_id = note.add(
        text=text,
        category=category,
    )
    await params.result_callback({"added": True, "id": new_id, "text": text})
    
async def update_note(params: FunctionCallParams, item_id: int, text: str | None = None,
                      category: str | None = None):
    """Change the text or category of an existing dev note.

    Call this when the user corrects or reclassifies a saved note, e.g.
    "actually that wake word thing is a bug", "make the Zotero idea a todo".
    For deadlines, use update_deadline.

    Pass only the fields that are changing, at least one. Text replaces the
    whole note, so write the full new version in the user's own words.

    Use the id from get_note (finished notes only appear with status "all").
    If it's unclear which note they mean, ask. If the result says it wasn't
    updated, tell the user you couldn't find that note.

    Args:
        text: The full new note text.
        category: "bug", "todo", or "feature".
    """
    fields = {
        k: v
        for k, v in {
            "text": text,
            "category": category
        }.items()
        if v is not None
    }
    ok = note.edit_notes(item_id, **fields)
    await params.result_callback({"updated": ok})

    
async def complete_note(params: FunctionCallParams, item_id: int):
    """Mark a dev note as done so it drops off the open list.

    Call this when the user says a bug is fixed, a todo is finished, or an
    idea is built, e.g. "I fixed the wake word bug", "mark the mute tests
    done". For deadlines, use complete_deadline.

    Use the id from get_note. If it's unclear which note they mean, ask. If
    the result says it wasn't completed, tell the user you couldn't find
    that note.
    """
    ok = note.complete_note(item_id)
    await params.result_callback({"completed": ok})

ALL_TOOLS = [
    get_events,
    get_deadlines,
    add_deadlines,
    complete_deadline,
    mute_deadline,
    update_deadline,
    get_note,
    add_note,
    update_note,
    complete_note,
]
