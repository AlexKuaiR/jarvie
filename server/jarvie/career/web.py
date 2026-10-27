"""Dashboard pages for postings and the application pipeline."""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

from ..dashboard import NAV, back, render
from . import postings


def add_routes(app: FastAPI) -> None:
    NAV.extend([("/", "Postings"), ("/pipeline", "Pipeline")])

    @app.get("/", response_class=HTMLResponse)
    def postings_page(
        request: Request,
        kind: str | None = None,
        field: str | None = None,
        status: str | None = None,
        sort: str = "fit",
        error: str | None = None,
    ):
        f = {"kind": kind or None, "field": field or None, "status": status or None, "sort": sort}
        try:
            items = postings.search(f["kind"], f["field"], f["status"], None, sort, limit=500)
        except ValueError as e:
            items, error = [], str(e)
        return render(
            request,
            "postings.html",
            "/",
            items=items,
            f=f,
            kinds=postings.KINDS,
            statuses=postings.STATUSES,
            sorts=list(postings.SORTS),
            error=error,
        )

    @app.get("/postings/{posting_id}", response_class=HTMLResponse)
    def posting_page(request: Request, posting_id: int, error: str | None = None):
        p = postings.get(posting_id)
        if p is None:
            return back("/", f"no posting {posting_id}")
        return render(
            request,
            "posting.html",
            "/",
            p=p,
            history=postings.history(posting_id),
            statuses=postings.STATUSES,
            error=error,
        )

    @app.post("/postings/{posting_id}/status")
    async def set_status(posting_id: int, request: Request):
        form = await request.form()
        try:
            postings.set_status(
                posting_id, str(form.get("status", "")), str(form.get("note", "")) or None
            )
        except ValueError as e:
            return back(f"/postings/{posting_id}", str(e))
        return back(f"/postings/{posting_id}")

    @app.post("/postings/{posting_id}/coffee")
    async def coffee(posting_id: int, request: Request):
        form = await request.form()
        if not postings.set_coffee_chat(posting_id, form.get("had_chat") == "1"):
            return back(f"/postings/{posting_id}", "set an application status first")
        return back(f"/postings/{posting_id}")

    @app.get("/pipeline", response_class=HTMLResponse)
    def pipeline_page(request: Request):
        by_status = {
            s: postings.search(status=s, sort="deadline", limit=500) for s in postings.STATUSES
        }
        return render(
            request,
            "pipeline.html",
            "/pipeline",
            summary=postings.pipeline_summary(),
            statuses=postings.STATUSES,
            by_status=by_status,
        )
