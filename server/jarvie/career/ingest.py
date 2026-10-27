"""Fetch the job boards on your watchlist, filter, and save what fits. No model, no API key.

Run from server/:  .venv/bin/python -m jarvie.career.ingest
"""

import asyncio
from collections import Counter

import httpx
from loguru import logger

from . import postings
from .filters import Criteria, judge, relevance
from .profile import CareerProfile, load
from .sources import fetch_all


def criteria_from(profile: CareerProfile, brief: dict | None = None) -> Criteria:
    """The profile's settings, with any answers from a search brief taking precedence."""
    brief = brief or {}
    return Criteria(
        graduation=profile.graduation,
        window_months=profile.graduation_window_months,
        fields=brief.get("fields") or profile.fields,
        kinds=brief.get("kinds") or profile.kinds,
        locations=brief["locations"] if "locations" in brief else profile.locations,
        paid_only=brief.get("paid_only", profile.paid_only),
        term=brief.get("term", ""),
        include_companies=brief.get("include_companies", []),
        exclude_companies=brief.get("exclude_companies", []),
    )


async def ingest(
    profile: CareerProfile,
    criteria: Criteria,
    client: httpx.AsyncClient | None = None,
    max_results: int | None = None,
    job_id: int | None = None,
) -> dict:
    """Fetch, filter and save. Returns the saved posting ids and why the rest were dropped."""
    raw, errors = await fetch_all(profile.watchlist, client)
    dropped: Counter[str] = Counter()
    kept = []
    for p in raw:
        verdict = judge(p, criteria)
        if verdict.keep:
            kept.append((p, verdict))
        else:
            dropped[verdict.reason] += 1
    # Strongest first, so a cap keeps the best matches rather than whatever came first.
    kept.sort(key=lambda pv: relevance(pv[0], pv[1], criteria), reverse=True)
    if max_results is not None and len(kept) > max_results:
        dropped["over max_results"] += len(kept) - max_results
        kept = kept[:max_results]
    saved: list[int] = []
    for p, verdict in kept:
        saved.append(
            postings.save(
                company=p.company,
                title=p.title,
                source=p.source,
                url=p.url,
                ats=p.ats,
                kind=verdict.kind,
                location=p.location or None,
                fields=verdict.fields,
                paid=verdict.paid,
                raw_text=p.text[:20000] or None,
                found_by_job=job_id,
            )
        )
    summary = {
        "fetched": len(raw),
        "saved": len(saved),
        "posting_ids": saved,
        "dropped": dict(dropped),
        "errors": errors,
    }
    logger.info("career ingest: {}", {k: v for k, v in summary.items() if k != "posting_ids"})
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Fetch your watchlist's job boards and save fits.")
    parser.add_argument("--max", type=int, default=100, help="most postings to save (best first)")
    parser.add_argument("--term", default="", help='only this term, e.g. "Summer 2027"')
    args = parser.parse_args()
    profile = load()
    criteria = criteria_from(profile, {"term": args.term} if args.term else None)
    summary = asyncio.run(ingest(profile, criteria, max_results=args.max))
    print(f"fetched {summary['fetched']}, saved {summary['saved']}")
    for reason, n in sorted(summary["dropped"].items(), key=lambda kv: -kv[1]):
        print(f"  dropped {n}: {reason}")
    for err in summary["errors"]:
        print(f"  error: {err}")


if __name__ == "__main__":
    main()
