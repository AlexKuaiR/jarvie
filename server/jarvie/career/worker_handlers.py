"""Career job handlers, registered with the worker."""

from ..llm import Usage
from ..worker import WorkerContext, handler
from . import research_agent


@handler("score_posting")
def score_posting(job: dict, ctx: WorkerContext, spend: Usage) -> dict:
    return research_agent.score_posting(job, ctx.get_agent(), ctx.get_profile(), spend)


@handler("discover_postings")
def discover_postings(job: dict, ctx: WorkerContext, spend: Usage) -> dict:
    return research_agent.discover_postings(
        job, ctx.get_agent(), ctx.get_profile(), spend, ctx.extra.get("http_client")
    )
