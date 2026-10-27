"""Career job handlers, registered with the worker."""

from ..llm import Usage
from ..worker import WorkerContext, handler
from . import research_agent


@handler("score_posting")
def score_posting(job: dict, ctx: WorkerContext, spend: Usage) -> dict:
    return research_agent.score_posting(job, ctx.get_agent, ctx.get_profile(), spend)


def _search_uses_ai(params: dict) -> bool:
    return int(params.get("ai_score", 0)) != 0 or bool(params.get("search_beyond_boards"))


@handler("discover_postings", uses_ai=_search_uses_ai)
def discover_postings(job: dict, ctx: WorkerContext, spend: Usage) -> dict:
    # ctx.get_agent is passed uncalled: a board-only search never creates a model.
    return research_agent.discover_postings(
        job, ctx.get_agent, ctx.get_profile(), spend, ctx.extra.get("http_client")
    )


@handler("write_document")
def write_document(job: dict, ctx: WorkerContext, spend: Usage) -> dict:
    from . import documents

    return documents.write_document(job, ctx.get_agent("documents"), ctx.get_profile(), spend)
