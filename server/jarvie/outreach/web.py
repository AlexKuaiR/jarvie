"""Dashboard pages for contacts and the outreach review queue."""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

from .. import jobs
from ..dashboard import NAV, back, render
from . import contacts, messages, sender


def add_routes(app: FastAPI) -> None:
    NAV.extend([("/contacts", "Contacts"), ("/outreach", "Outreach")])

    @app.get("/contacts", response_class=HTMLResponse)
    def contacts_page(
        request: Request,
        company: str | None = None,
        school: str | None = None,
        error: str | None = None,
    ):
        f = {"company": company or None, "school": school or None}
        items = contacts.search(f["company"], f["school"], limit=500)
        return render(
            request,
            "contacts.html",
            "/contacts",
            items=items,
            f=f,
            schools=contacts.SCHOOLS,
            error=error,
        )

    @app.post("/contacts")
    async def add_contact(request: Request):
        form = {k: str(v).strip() or None for k, v in (await request.form()).items()}
        try:
            contacts.add(
                form.get("name") or "",
                source="user",
                email=form.get("email"),
                company=form.get("company"),
                role=form.get("role"),
                notes=form.get("notes"),
            )
        except ValueError as e:
            return back("/contacts", str(e))
        return back("/contacts")

    @app.post("/contacts/{contact_id}/find-email")
    def find_email(contact_id: int):
        jobs.enqueue("find_email", {"contact_id": contact_id})
        return back("/jobs")

    @app.post("/contacts/{contact_id}/dnc")
    def dnc(contact_id: int):
        contacts.mark_do_not_contact(contact_id)
        return back("/contacts")

    @app.get("/outreach", response_class=HTMLResponse)
    def outreach_page(request: Request, error: str | None = None):
        items = messages.queue()
        counts: dict[str, int] = {}
        for m in items:
            counts[m["status"]] = counts.get(m["status"], 0) + 1
        return render(
            request,
            "outreach.html",
            "/outreach",
            drafts=[m for m in items if m["status"] == "drafted"],
            others=[m for m in reversed(items) if m["status"] != "drafted"],
            counts=counts,
            dry_run=sender.default_mailer().dry_run,
            cap=sender.daily_cap(),
            error=error,
        )

    @app.post("/outreach/{outreach_id}/save")
    async def save_draft(outreach_id: int, request: Request):
        form = await request.form()
        try:
            messages.edit_draft(
                outreach_id, str(form.get("subject", "")), str(form.get("body", ""))
            )
        except messages.OutreachRefused as e:
            return back("/outreach", str(e))
        return back("/outreach")

    @app.post("/outreach/{outreach_id}/status")
    async def set_status(outreach_id: int, request: Request):
        form = await request.form()
        try:
            messages.set_status(outreach_id, str(form.get("status", "")))
        except (messages.OutreachRefused, KeyError) as e:
            return back("/outreach", str(e))
        return back("/outreach")

    @app.post("/outreach/approve-all")
    def approve_all():
        for m in messages.queue("drafted"):
            try:
                messages.set_status(m["id"], "approved")
            except messages.OutreachRefused:
                continue  # e.g. marked do-not-contact since it was drafted
        return back("/outreach")

    @app.post("/outreach/{outreach_id}/coffee")
    def coffee(outreach_id: int):
        m = messages.get(outreach_id)
        if m and m["posting_id"]:
            from ..career import postings

            if not postings.set_coffee_chat(m["posting_id"]):
                return back("/outreach", "set an application status on that posting first")
        return back("/outreach")
