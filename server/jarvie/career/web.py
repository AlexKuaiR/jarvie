"""Dashboard pages for postings and the application pipeline."""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

from .. import jobs
from ..dashboard import NAV, back, render
from . import documents, postings, profile


def add_routes(app: FastAPI) -> None:
    NAV.extend([("/", "Postings"), ("/pipeline", "Pipeline"), ("/documents", "Documents")])

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
        try:
            prof = profile.load()
            resumes, resume_used = list(prof.resumes), prof.resume_for(p).name
        except profile.ProfileError:
            resumes, resume_used = [], None
        return render(
            request,
            "posting.html",
            "/",
            p=p,
            history=postings.history(posting_id),
            docs=documents.for_posting(posting_id),
            statuses=postings.STATUSES,
            resumes=resumes,
            resume_used=resume_used,
            error=error,
        )

    @app.post("/postings/{posting_id}/resume")
    async def set_resume(posting_id: int, request: Request):
        form = await request.form()
        name = str(form.get("resume", ""))
        try:
            name = None if name == "auto" else profile.load().resume(name).name
        except profile.ProfileError as e:
            return back(f"/postings/{posting_id}", str(e))
        postings.set_resume(posting_id, name)
        return back(f"/postings/{posting_id}")

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

    @app.post("/postings/{posting_id}/documents")
    async def write_document(posting_id: int, request: Request):
        form = await request.form()
        kind = str(form.get("kind", "cover_letter"))
        question = str(form.get("question", "")).strip()
        limit = str(form.get("word_limit", "")).strip()
        if kind == "essay" and not question:
            return back(f"/postings/{posting_id}", "an essay needs its prompt")
        if limit and not limit.isdigit():
            return back(f"/postings/{posting_id}", "word limit must be a number")
        jobs.enqueue(
            "write_document",
            {
                "posting_id": posting_id,
                "kind": kind,
                "question": question or None,
                "word_limit": int(limit) if limit else None,
            },
        )
        return back("/jobs")

    @app.get("/documents", response_class=HTMLResponse)
    def documents_page(request: Request):
        return render(request, "documents.html", "/documents", items=documents.for_posting())

    @app.get("/documents/{document_id}", response_class=HTMLResponse)
    def document_page(request: Request, document_id: int):
        d = documents.get(document_id)
        if d is None:
            return back("/documents", f"no document {document_id}")
        return render(request, "document.html", "/documents", d=d, p=postings.get(d["posting_id"]))
