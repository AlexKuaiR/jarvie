"""Outreach job handlers and ticks, registered with the worker."""

import time

from loguru import logger

from ..llm import Usage
from ..worker import BUSY_CHECKS, TICKS, WorkerContext, handler
from . import messages, research, sender, writer
from .email_finder import EmailFinderError


@handler("find_contacts")
def find_contacts(job: dict, ctx: WorkerContext, spend: Usage) -> dict:
    try:
        finder = ctx.get_email_finder()
    except EmailFinderError as e:
        # Still worth finding the people; emails can be looked up later.
        logger.warning("find_contacts without email lookup: {}", e)
        finder = None
    result = research.find_contacts(job, ctx.get_agent(), finder, spend)
    if finder is None:
        result["email_lookup_error"] = "set HUNTER_API_KEY in server/.env to look up emails"
    return result


@handler("find_email", uses_ai=lambda params: False)  # Hunter, not a model
def find_email(job: dict, ctx: WorkerContext, spend: Usage) -> dict:
    return research.find_email(job, ctx.get_email_finder())


@handler("draft_outreach")
def draft_outreach(job: dict, ctx: WorkerContext, spend: Usage) -> dict:
    return writer.draft_outreach(job, ctx.get_writer(), spend, ctx.extra.get("sender"))


@handler("revise_outreach")
def revise_outreach(job: dict, ctx: WorkerContext, spend: Usage) -> dict:
    return writer.revise_outreach(job, ctx.get_writer(), spend)


REPLY_CHECK_SECONDS = 600


def outreach_tick(ctx: WorkerContext) -> None:
    """Every worker loop: schedule approved email, send what's due, check for replies."""
    mailer = ctx.get_mailer()
    sender.schedule_approved()
    sender.send_due(mailer)
    last = ctx.extra.get("last_reply_check", 0.0)
    if time.monotonic() - last >= REPLY_CHECK_SECONDS:
        ctx.extra["last_reply_check"] = time.monotonic()
        sender.check_replies(mailer)


if outreach_tick not in TICKS:
    TICKS.append(outreach_tick)


def emails_waiting() -> bool:
    """Approved or scheduled emails still to send keep an auto-started worker running."""
    return bool(messages.queue("approved") or messages.queue("scheduled"))


if emails_waiting not in BUSY_CHECKS:
    BUSY_CHECKS.append(emails_waiting)
