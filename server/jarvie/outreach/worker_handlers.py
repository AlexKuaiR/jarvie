"""Outreach job handlers and ticks, registered with the worker."""

from loguru import logger

from ..llm import Usage
from ..worker import WorkerContext, handler
from . import research
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


@handler("find_email")
def find_email(job: dict, ctx: WorkerContext, spend: Usage) -> dict:
    return research.find_email(job, ctx.get_email_finder())
