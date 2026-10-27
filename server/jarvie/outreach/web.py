"""Dashboard pages for contacts (and, from phase 4, the outreach queue)."""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

from .. import jobs
from ..dashboard import NAV, back, render
from . import contacts


def add_routes(app: FastAPI) -> None:
    NAV.append(("/contacts", "Contacts"))

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
