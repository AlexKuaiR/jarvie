"""Local dashboard for career search and outreach: http://localhost:7861

Run from server/:  .venv/bin/python -m jarvie.dashboard

Server-rendered pages (no JavaScript build); every button is a plain form post that
writes to the same SQLite database the voice bot and worker use. It binds to 127.0.0.1
only, so nothing is reachable from other machines.
"""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from . import briefs, jobs

templates = Jinja2Templates(directory=str(Path(__file__).with_name("dashboard_templates")))

# (path, label) for the nav bar; feature routers add their pages here.
NAV: list[tuple[str, str]] = []


def render(request: Request, name: str, active: str, **context) -> HTMLResponse:
    # Every page warns when jobs are waiting and no worker is running to do them.
    queued = jobs.queued_count()
    stalled = queued if queued and not jobs.worker_running() else 0
    return templates.TemplateResponse(
        request,
        name,
        {
            "nav": NAV,
            "active": active,
            "title": name.removesuffix(".html"),
            "stalled_jobs": stalled,
            "start_worker": jobs.START_WORKER,
            **context,
        },
    )


def back(path: str, error: str | None = None) -> RedirectResponse:
    """Redirect after a form post (303 so the browser follows with GET)."""
    from urllib.parse import quote

    return RedirectResponse(f"{path}?error={quote(error)}" if error else path, status_code=303)


# --- briefs and jobs (shared by career and outreach) --------------------------------


def _hint(q: briefs.Question) -> str:
    if q.words:
        first_word = {}  # one word per value, e.g. none (0) and all (-1)
        for word, value in q.words:
            first_word.setdefault(value, word)
        return " / ".join([*first_word.values(), "a number"])
    if q.choices:
        return " / ".join(q.choices)
    if q.default not in (None, "", []):
        return f"default: {q.default}"
    return ""


def _display(value) -> str:
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    if isinstance(value, bool):
        return "yes" if value else "no"
    return "" if value is None else str(value)


def add_brief_and_job_routes(app: FastAPI) -> None:
    NAV.append(("/briefs", "Searches"))

    @app.get("/briefs", response_class=HTMLResponse)
    def briefs_page(request: Request, error: str | None = None):
        items = []
        for b in briefs.open_briefs():
            b["display"] = {
                k: "all" if k == "ai_score" and v == -1 else _display(v)
                for k, v in b["answers"].items()
            }
            items.append(b)
        questions = {
            kind: [{**q.__dict__, "hint": _hint(q)} for q in qs]
            for kind, qs in briefs.QUESTIONS.items()
        }
        return render(
            request,
            "briefs.html",
            "/briefs",
            open_briefs=items,
            questions=questions,
            kinds=list(briefs.QUESTIONS),
            error=error,
        )

    @app.post("/briefs")
    async def start_brief(request: Request):
        form = await request.form()
        from .job_tools import _brief_prefill

        kind = str(form.get("kind", ""))
        try:
            briefs.start(kind, prefill=_brief_prefill(kind), channel="dashboard")
        except briefs.BriefError as e:
            return back("/briefs", str(e))
        return back("/briefs")

    @app.post("/briefs/{brief_id}/answer")
    async def answer_brief(brief_id: int, request: Request):
        form = await request.form()
        answers = {k: str(v) for k, v in form.items() if str(v).strip()}
        try:
            briefs.answer(brief_id, answers, channel="dashboard")
        except briefs.BriefError as e:
            return back("/briefs", str(e))
        return back("/briefs")

    @app.post("/briefs/{brief_id}/submit")
    def submit_brief(brief_id: int):
        try:
            briefs.submit(brief_id, channel="dashboard")
        except briefs.BriefError as e:
            return back("/briefs", str(e))
        return back("/jobs")

    @app.post("/briefs/{brief_id}/cancel")
    def cancel_brief(brief_id: int):
        try:
            briefs.cancel(brief_id)
        except briefs.BriefError as e:
            return back("/briefs", str(e))
        return back("/briefs")

    @app.get("/jobs", response_class=HTMLResponse)
    def jobs_page(request: Request, error: str | None = None):
        return render(
            request,
            "jobs.html",
            "/jobs",
            items=jobs.recent(50),
            spent=jobs.spent_today(),
            error=error,
        )

    @app.post("/jobs/{job_id}/retry")
    def retry_job(job_id: int):
        if not jobs.retry(job_id):
            return back("/jobs", f"job {job_id} isn't a failed job")
        return back("/jobs")


def create_app() -> FastAPI:
    from .career import web as career_web
    from .outreach import web as outreach_web

    NAV.clear()
    app = FastAPI(title="Jarvie dashboard", docs_url=None, redoc_url=None)
    career_web.add_routes(app)
    add_brief_and_job_routes(app)
    outreach_web.add_routes(app)
    return app


def main() -> None:
    import uvicorn

    uvicorn.run(create_app(), host="127.0.0.1", port=7861, log_level="info")


if __name__ == "__main__":
    # `python -m jarvie.dashboard` runs this file as a second copy of the module named __main__,
    # while the feature packages register handlers/pages on the importable jarvie.dashboard.
    # Running that copy's main() keeps everything in one module.
    from jarvie.dashboard import main as _main

    _main()
