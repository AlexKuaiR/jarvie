"""Dashboard pages for contacts and the outreach review queue."""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

from .. import jobs
from ..dashboard import NAV, back, render
from . import contacts, messages, sender
from .writer import facts_of


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
        drafts = [m for m in items if m["status"] == "drafted"]
        rewrites = [j for j in jobs.recent(100) if j["kind"] == "revise_outreach"]
        for m in drafts:
            # What the rewrite box needs: the person, the versions, and the latest rewrite.
            m["contact"] = contacts.get(m["contact_id"])
            m["facts"] = facts_of(m["contact"])
            m["versions"] = messages.revisions(m["id"])
            mine = [j for j in rewrites if j["params"].get("outreach_id") == m["id"]]
            m["rewrite"] = mine[0] if mine else None  # newest first
        return render(
            request,
            "outreach.html",
            "/outreach",
            drafts=drafts,
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

    @app.post("/outreach/{outreach_id}/revise")
    async def revise(outreach_id: int, request: Request):
        instruction = str((await request.form()).get("instruction", "")).strip()
        m = messages.get(outreach_id)
        if not instruction:
            return back("/outreach", "say what to change")
        if m is None or m["status"] != "drafted":
            return back("/outreach", "only drafts can be rewritten")
        jobs.enqueue("revise_outreach", {"outreach_id": outreach_id, "instruction": instruction})
        return back(f"/outreach#draft-{outreach_id}")

    @app.post("/outreach/{outreach_id}/restore")
    async def restore(outreach_id: int, request: Request):
        form = await request.form()
        try:
            messages.restore(outreach_id, int(form.get("revision_id", 0)))
        except (messages.OutreachRefused, ValueError) as e:
            return back("/outreach", str(e))
        return back(f"/outreach#draft-{outreach_id}")

    @app.post("/contacts/{contact_id}/fact")
    async def add_fact(contact_id: int, request: Request):
        form = await request.form()
        try:
            contacts.add_fact(contact_id, str(form.get("fact", "")))
        except ValueError as e:
            return back("/outreach", str(e))
        return back(str(form.get("next") or "/outreach"))

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
