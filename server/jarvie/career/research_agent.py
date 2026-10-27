"""The research agent: reads postings, scores them against your profile, finds more.

It runs in the worker, not the voice bot. Everything it learns reaches the database
only through save_posting(), whose arguments are validated like any other write, so
the voice model only ever sees structured, checked results.
"""

import asyncio
import os
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


def score_posting(job: dict, model: AgentModel, profile: CareerProfile, spend: Usage) -> dict:
    """One posting from a URL or pasted text (e.g. from Handshake): read, score, save."""
    params = job["params"]
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
        raise ValueError("score_posting needs a url or text")
    result = model.run(
        system_prompt(profile),
        prompt,
        [make_save_posting(log, job["id"], source, tuple(profile.resumes))],
        web_search=3,
        web_fetch=fetches,
    )
    spend.add(result.usage)
    return {"saved": log.saved, "errors": log.errors, "note": result.text[:500]}


def _budget() -> float:
    return float(os.getenv("CAREER_JOB_BUDGET_USD", "3.00"))


def discover_postings(
    job: dict, model: AgentModel, profile: CareerProfile, spend: Usage, http_client=None
) -> dict:
    """A brief-driven search: job boards first (free), then the agent scores the best few
    and, for research positions, looks for openings the boards don't list."""
    params = job["params"]
    criteria = criteria_from(profile, params)
    max_results = int(params.get("max_results", 25))
    found = asyncio.run(
        ingest(profile, criteria, http_client, max_results=max_results, job_id=job["id"])
    )

    system = system_prompt(profile)
    names = tuple(profile.resumes)
    log = SaveLog()
    budget = _budget()
    stopped_early = False
    to_score = [postings.get(i) for i in found["posting_ids"]]
    to_score = [p for p in to_score if p and p["fit_score"] is None]
    for start in range(0, len(to_score), 8):
        if spend.cost_usd >= budget:
            stopped_early = True
            break
        chunk = to_score[start : start + 8]
        listing = "\n\n".join(
            f"### posting_id {p['id']}: {p['title']} at {p['company']} ({p['location'] or 'location?'})\n"
            f"url: {p['url']}\n{(p['raw_text'] or '')[:2500]}"
            for p in chunk
        )
        prompt = (
            "Score each posting below and call save_posting once per posting with its "
            "posting_id. Extra notes from the user for this search: "
            f"{params.get('notes') or 'none'}.\n\n{listing}"
        )
        result = model.run(
            system, prompt, [make_save_posting(log, job["id"], "web", names)], web_search=4
        )
        spend.add(result.usage)

    if "research" in criteria.kinds and spend.cost_usd < budget and not params.get("skip_web"):
        prompt = (
            "Search the web for research positions for this student that job boards miss: "
            "undergraduate research programs, REUs, lab openings and summer research "
            f"fellowships in {', '.join(criteria.fields)}"
            f"{' for ' + criteria.term if criteria.term else ''}. Prefer official program "
            "or lab pages. Save each real, currently open opportunity with save_posting "
            "(kind research, with its url); at most 8. Skip anything you can't verify is open."
        )
        result = model.run(
            system,
            prompt,
            [make_save_posting(log, job["id"], "web", names)],
            web_search=8,
            web_fetch=6,
        )
        spend.add(result.usage)
    elif "research" in criteria.kinds:
        stopped_early = stopped_early or spend.cost_usd >= budget

    low = [
        pid
        for pid in set(log.saved)
        if (p := postings.get(pid))
        and p["prestige"] is not None
        and p["prestige"] < int(params.get("min_prestige", profile.min_prestige))
    ]
    logger.info(
        "discover job {}: saved {}, scored {}", job["id"], found["saved"], len(set(log.saved))
    )
    return {
        "from_job_boards": found["saved"],
        "scored": sorted(set(log.saved)),
        "below_min_prestige": sorted(low),
        "dropped": found["dropped"],
        "errors": found["errors"] + log.errors,
        "stopped_early_for_budget": stopped_early,
    }
