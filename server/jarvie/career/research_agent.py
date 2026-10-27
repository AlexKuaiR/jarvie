"""The research agent: reads postings, scores them against your profile, finds more.

It runs in the worker, not the voice bot. Everything it learns reaches the database
only through save_posting(), whose arguments are validated like any other write, so
the voice model only ever sees structured, checked results.
"""

import asyncio
import os
from collections.abc import Callable
from dataclasses import dataclass, field

from loguru import logger

from ..llm import AgentModel, Usage, cached_system
from . import postings
from .ingest import criteria_from, ingest
from .profile import CareerProfile

# Written-out definitions so scores mean the same thing on every run.
RUBRIC = """\
Score every posting on these 1-5 scales, each with a one-sentence reason.

prestige (how much the name helps a resume in this field):
  5 = top-tier lab or company known across the field (e.g. DeepMind, Broad Institute)
  4 = widely respected, competitive brand or well-known lab at a top university
  3 = solid, recognizable company or established lab
  2 = small or little-known company/lab with some track record
  1 = unknown, or no information
competitiveness (how hard it is to get):
  5 = extremely selective (well under 2% / very large applicant pools, multi-round)
  4 = highly selective, usually technical interviews or an OA plus interviews
  3 = moderately selective
  2 = some screening
  1 = open or nearly so
ownership (who owns what you make, and can you publish or show it):
  5 = you can publish/open-source; the agreement is limited to work done on the job
  3 = standard employer IP agreement; publication needs approval; or unknown
  1 = claims rights to outside or personal work, or forbids showing any of it
fit_score: 0 to 1, how well it matches the candidate's profile and requirements below.
  Lower it for dealbreakers, missing required skills, or a graduation-date mismatch.
Only state facts the posting or a source you read supports; write "unknown" otherwise.
"""


def system_prompt(profile: CareerProfile) -> list[dict]:
    requirements = (
        f"Graduation: {profile.graduation:%B %Y} (window ±{profile.graduation_window_months} months)\n"
        f"Fields wanted: {', '.join(profile.fields)}\n"
        f"Kinds: {', '.join(profile.kinds)}\n"
        f"Paid only: {profile.paid_only} (min hourly pay {profile.min_hourly_pay or 'any'})\n"
        f"Locations: {', '.join(profile.locations) or 'any'}\n"
        f"Minimum prestige: {profile.min_prestige}\n"
        f"Dealbreakers: {'; '.join(profile.dealbreakers) or 'none'}\n"
    )
    stable = (
        "You research internships and research positions for one student and save "
        "structured results with the save_posting tool. Be accurate over complete: use "
        "null or 'unknown' rather than guessing. Never invent postings, pay, deadlines or "
        "requirements. Use web_search sparingly, only to check prestige, ownership/IP "
        "terms or missing facts. Application process steps use these words: oa, "
        "interview, essays, cover_letter, referral, portfolio, transcript.\n\n"
        f"{RUBRIC}\n## Candidate requirements\n{requirements}\n"
    )
    if len(profile.resumes) == 1:
        stable += f"## Candidate resume\n{profile.resume_text()}"
    else:
        # Several resumes (e.g. swe, mle, qr): score fit against the one the student would
        # send, and record which, so cover letters and the apply assistant use it too.
        stable += (
            "## Candidate resumes\nThe student has one resume per kind of role. Judge fit "
            "against the resume that best matches each posting, and pass its name as "
            "`resume` to save_posting.\n\n" + profile.labeled_resumes()
        )
    return cached_system(stable)


@dataclass
class SaveLog:
    saved: list[int] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def make_save_posting(
    log: SaveLog, job_id: int | None, source: str, resume_names: tuple[str, ...] = ()
):
    """Build the save_posting tool the agent calls; it writes through postings.save()."""

    def save_posting(
        company: str,
        title: str,
        url: str | None = None,
        posting_id: int | None = None,
        kind: str = "internship",
        location: str | None = None,
        deadline: str | None = None,
        fields: list[str] | None = None,
        paid: bool | None = None,
        pay: str | None = None,
        grad_window: str | None = None,
        prestige: int | None = None,
        prestige_reason: str | None = None,
        competitiveness: int | None = None,
        competitiveness_reason: str | None = None,
        ownership: int | None = None,
        ownership_reason: str | None = None,
        skills: list[str] | None = None,
        process: list[str] | None = None,
        fit_score: float | None = None,
        fit_reasons: str | None = None,
        ats: str | None = None,
        resume: str | None = None,
    ) -> str:
        """Save or update one posting with your scores. Call once per posting.

        Args:
            company: Company or lab name.
            title: Position title.
            url: The posting's URL, if it has one.
            posting_id: The id given in the prompt when updating a posting already saved.
            kind: "internship" or "research".
            location: As written, e.g. "Berkeley, CA" or "Remote".
            deadline: "YYYY-MM-DD", or null if not stated.
            fields: From: machine learning, software engineering, biology, biotech, quant.
            paid: true, false, or null if not stated.
            pay: Pay as written, e.g. "$45/hr".
            grad_window: Graduation dates accepted, as written.
            prestige: 1-5 per the rubric.
            prestige_reason: One sentence.
            competitiveness: 1-5 per the rubric.
            competitiveness_reason: One sentence.
            ownership: 1-5 per the rubric.
            ownership_reason: One sentence.
            skills: Skills the application requires or prefers.
            process: Steps, e.g. ["oa", "interview", "essays"].
            fit_score: 0 to 1.
            fit_reasons: Two or three sentences on fit, naming any dealbreaker.
            ats: How you apply: greenhouse, lever, ashby, workday, handshake, email, other.
            resume: Name of the student's resume you judged fit against (if they have several).
        """
        values = {
            k: v
            for k, v in {
                "url": url,
                "kind": kind,
                "location": location,
                "deadline": deadline,
                "fields": fields,
                "paid": paid,
                "pay": pay,
                "grad_window": grad_window,
                "prestige": prestige,
                "prestige_reason": prestige_reason,
                "competitiveness": competitiveness,
                "competitiveness_reason": competitiveness_reason,
                "ownership": ownership,
                "ownership_reason": ownership_reason,
                "skills": skills,
                "process": process,
                "fit_score": fit_score,
                "fit_reasons": fit_reasons,
                "ats": ats,
            }.items()
            if v is not None
        }
        if fit_score is not None and not 0 <= fit_score <= 1:
            return "error: fit_score must be between 0 and 1"
        if resume is not None and resume not in resume_names:
            return f"error: resume must be one of {', '.join(resume_names) or 'none'}"
        try:
            if posting_id is not None:
                values.pop("url", None)  # keep the URL the posting was saved with
                if not postings.update(posting_id, **values):
                    return f"error: no posting {posting_id}"
                pid = posting_id
            else:
                pid = postings.save(company, title, source, found_by_job=job_id, **values)
        except ValueError as e:
            log.errors.append(str(e))
            return f"error: {e}"
        if resume is not None and len(resume_names) > 1:
            postings.set_resume(pid, resume, only_if_unset=True)  # never override your pick
        log.saved.append(pid)
        return f"saved posting {pid}"

    return save_posting


def score_posting(
    job: dict, get_model: Callable[[], AgentModel], profile: CareerProfile, spend: Usage
) -> dict:
    """Score postings: saved ones (posting_ids), or one new one from a URL or pasted text
    (e.g. from Handshake), which is read, scored and saved."""
    params = job["params"]
    if params.get("posting_ids"):
        found = [postings.get(int(i)) for i in params["posting_ids"]]
        missing = [i for i, p in zip(params["posting_ids"], found) if p is None]
        if missing:
            raise ValueError(f"no postings {missing}")
        log = SaveLog()
        stopped = score_saved(found, get_model(), profile, spend, job["id"], log)
        return {"scored": sorted(set(log.saved)), "errors": log.errors, "stopped_early": stopped}
    log = SaveLog()
    if params.get("url"):
        source = "handshake" if "joinhandshake.com" in params["url"] else "web"
        prompt = (
            f"Read this posting with web_fetch and save it with save_posting: {params['url']}\n"
            "If the page needs a login or has no posting on it, don't save anything; say so."
        )
        fetches = 2
    elif params.get("text"):
        source = "handshake" if "handshake" in params["text"].lower() else "web"
        prompt = (
            "The user pasted this posting. Save it with save_posting (no url).\n\n"
            f"<posting>\n{params['text'][:30000]}\n</posting>"
        )
        fetches = 0
    else:
        raise ValueError("score_posting needs posting_ids, a url or text")
    result = get_model().run(
        system_prompt(profile),
        prompt,
        [make_save_posting(log, job["id"], source, tuple(profile.resumes))],
        web_search=3,
        web_fetch=fetches,
    )
    spend.add(result.usage)
    return {"saved": log.saved, "errors": log.errors, "note": result.text[:500]}


def scoring_prompt(chunk: list[dict], notes: str = "") -> str:
    listing = "\n\n".join(
        f"### posting_id {p['id']}: {p['title']} at {p['company']} ({p['location'] or 'location?'})\n"
        f"url: {p['url']}\n{(p['raw_text'] or '')[:2500]}"
        for p in chunk
    )
    return (
        "Score each posting below and call save_posting once per posting with its "
        f"posting_id. Extra notes from the user: {notes or 'none'}.\n\n{listing}"
    )


def score_saved(
    items: list[dict],
    model: AgentModel,
    profile: CareerProfile,
    spend: Usage,
    job_id: int | None,
    log: SaveLog,
    notes: str = "",
) -> bool:
    """Score saved postings, 8 per model call, until the job budget runs out.
    Returns True if it stopped early for the budget."""
    system = system_prompt(profile)
    names = tuple(profile.resumes)
    budget = _budget()
    for start in range(0, len(items), 8):
        if spend.cost_usd >= budget:
            return True
        prompt = scoring_prompt(items[start : start + 8], notes)
        result = model.run(
            system, prompt, [make_save_posting(log, job_id, "web", names)], web_search=4
        )
        spend.add(result.usage)
    return False


def _budget() -> float:
    return float(os.getenv("CAREER_JOB_BUDGET_USD", "3.00"))


def discover_postings(
    job: dict,
    get_model: Callable[[], AgentModel],
    profile: CareerProfile,
    spend: Usage,
    http_client=None,
) -> dict:
    """A brief-driven search. Job boards first (free, no model); then, only as the brief
    allows, AI scores the top ai_score results (-1 = all) and, if search_beyond_boards,
    searches the open web for openings the boards don't list. With both off the model
    is never created, so the search works with no key and costs nothing."""
    params = job["params"]
    criteria = criteria_from(profile, params)
    max_results = int(params.get("max_results", 25))
    ai_score = int(params.get("ai_score", 0))
    beyond = bool(params.get("search_beyond_boards", False))
    found = asyncio.run(
        ingest(profile, criteria, http_client, max_results=max_results, job_id=job["id"])
    )

    log = SaveLog()
    stopped_early = False
    # posting_ids come back best match first, so "top N" means the strongest N.
    to_score = [
        p for i in found["posting_ids"] if (p := postings.get(i)) and p["fit_score"] is None
    ]
    if ai_score >= 0:
        to_score = to_score[:ai_score]
    if to_score:
        stopped_early = score_saved(
            to_score, get_model(), profile, spend, job["id"], log, params.get("notes") or ""
        )

    if beyond:
        if spend.cost_usd >= _budget():
            stopped_early = True
        else:
            kinds = " and ".join(
                {"internship": "internships", "research": "research positions"}[k]
                for k in criteria.kinds
            )
            prompt = (
                f"Search the web for {kinds} for this student that the big job boards miss: "
                "company career pages, undergraduate research programs, REUs, lab openings "
                f"and summer fellowships in {', '.join(criteria.fields)}"
                f"{' for ' + criteria.term if criteria.term else ''}. "
                + (
                    f"Check these first: {', '.join(criteria.include_companies)}. "
                    if criteria.include_companies
                    else ""
                )
                + (
                    f"Skip these: {', '.join(criteria.exclude_companies)}. "
                    if criteria.exclude_companies
                    else ""
                )
                + f"Notes from the user: {params.get('notes') or 'none'}. Prefer official "
                "pages. Save each real, currently open opportunity with save_posting (with its "
                "url); at most 8. Skip anything you can't verify is open."
            )
            result = get_model().run(
                system_prompt(profile),
                prompt,
                [make_save_posting(log, job["id"], "web", tuple(profile.resumes))],
                web_search=8,
                web_fetch=6,
            )
            spend.add(result.usage)

    low = [
        pid
        for pid in set(log.saved)
        if (p := postings.get(pid))
        and p["prestige"] is not None
        and p["prestige"] < int(params.get("min_prestige", profile.min_prestige))
    ]
    logger.info(
        "discover job {}: saved {}, scored {}, used AI: {}",
        job["id"],
        found["saved"],
        len(set(log.saved)),
        bool(to_score or beyond),
    )
    return {
        "from_job_boards": found["saved"],
        "scored": sorted(set(log.saved)),
        "unscored": len(found["posting_ids"]) - len(to_score),
        "used_ai": bool(to_score or beyond),
        "below_min_prestige": sorted(low),
        "dropped": found["dropped"],
        "errors": found["errors"] + log.errors,
        "stopped_early_for_budget": stopped_early,
    }
