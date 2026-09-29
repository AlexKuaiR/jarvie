from datetime import datetime, timedelta, time
from pipecat.services.llm_service import FunctionCallParams
from . import calendar
from .config import TZ
import asyncio
from loguru import logger
from . import deadlines
import json


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
        f"Upcoming this week: {json.dumps(items)}."
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
    """Get the user's upcoming exams, assignments, and application deadlines

    Call this for any question about tests, due dates, or what's coming up academically.
    Always call it rather than guessing.

    Each item contains an id. You need the id to mute, update, or complete an item later,
    so keep track of it, but never say it out loud.

    Args:
        days: how many days forward ahead to look. Defaults to 7.
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
    a date, if the user is too vague, ask.

    Args:
        name: Short label, e.g. "Midterm 1".
        date: ISO 8601, "YYYY-MM-DD".
        kind: "exam", "assignment", or "application".
        weight: Fraction of the course grade, e.g. 0.25 for 25%.
        topics: What it covers, in the user's own words.
    """

    new_id = deadlines.add(
        name=name, date=date, course=course, kind=kind, weight=weight, topics=topics
    )
    await params.result_callback({"added": True, "id": new_id, "name": name})


async def mute_deadline(params: FunctionCallParams, item_id: int, days: int = 3):
    """Stop reminding the user about a specific deadline for a few days.

    Call this when the user acknowledges an item — "got it", "I know about
    that one", "stop reminding me about the midterm".
    To un-mute something, call this with days=0.

    Use the id from get_deadlines. If you don't have one, call get_deadlines
    first to find it.
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
    name: str,
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


ALL_TOOLS = [
    get_events,
    get_deadlines,
    add_deadlines,
    complete_deadline,
    mute_deadline,
    update_deadline,
]
