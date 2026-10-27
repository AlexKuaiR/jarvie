"""Phase 3: email finder (Hunter, mocked), people search, standalone lookups, tools, page."""

import httpx
import pytest
from agent_fakes import FakeAgent
from career_fakes import call
from fastapi.testclient import TestClient

from jarvie import briefs, jobs, worker
from jarvie.dashboard import create_app
from jarvie.outreach import contacts
from jarvie.outreach import tools as outreach_tools
from jarvie.outreach.email_finder import (
    EmailFinderError,
    FoundEmail,
    HunterFinder,
    default_finder,
    domain_of,
    split_name,
)

pytestmark = pytest.mark.usefixtures("temp_db")


class FakeFinder:
    def __init__(self, emails: dict | None = None, error: Exception | None = None):
        self.emails = emails or {}  # (first, last) -> email
        self.error = error
        self.lookups: list = []

    def find(self, first, last, domain=None, company=None):
        self.lookups.append((first, last, domain, company))
        if self.error:
            raise self.error
        email = self.emails.get((first, last))
        return FoundEmail(email, 90, "hunter") if email else None

    def verify(self, email):
        return "invalid" if email.startswith("bad") else "valid"


# --- Hunter ---------------------------------------------------------------------


def hunter(status: int = 200, data: dict | None = None, seen: list | None = None) -> HunterFinder:
    def handler(request):
        if seen is not None:
            seen.append(request)
        if status != 200:
            return httpx.Response(status, json={"errors": [{"details": "x"}]})
        if request.url.path.endswith("email-verifier"):
            return httpx.Response(200, json={"data": {"status": "accept_all"}})
        return httpx.Response(200, json={"data": data})

    return HunterFinder("key", httpx.Client(transport=httpx.MockTransport(handler)))


def test_hunter_finds_and_verifies():
    seen: list = []
    finder = hunter(data={"email": "Dana.Lee@Acme.bio", "score": 94}, seen=seen)
    found = finder.find("Dana", "Lee", domain="acme.bio")
    assert (found.email, found.confidence) == ("dana.lee@acme.bio", 94)
    assert finder.verify(found.email) == "accept_all"
    params = seen[0].url.params
    assert (params["first_name"], params["domain"], params["api_key"]) == (
        "Dana",
        "acme.bio",
        "key",
    )


def test_hunter_company_instead_of_domain_and_no_result():
    seen: list = []
    finder = hunter(data={"email": None}, seen=seen)
    assert finder.find("Dana", "Lee", company="Acme Bio") is None
    assert seen[0].url.params["company"] == "Acme Bio"
    with pytest.raises(EmailFinderError, match="domain or company"):
        finder.find("Dana", "Lee")


@pytest.mark.parametrize("status, message", [(401, "HUNTER_API_KEY"), (429, "quota"), (500, "500")])
def test_hunter_errors(status, message):
    with pytest.raises(EmailFinderError, match=message):
        hunter(status=status).find("Dana", "Lee", domain="acme.bio")


def test_default_finder_needs_key(monkeypatch):
    monkeypatch.delenv("HUNTER_API_KEY", raising=False)
    with pytest.raises(EmailFinderError, match="HUNTER_API_KEY"):
        default_finder()


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Dana Lee", ("Dana", "Lee")),
        ("Dr. Dana M. Lee, PhD", ("Dana", "Lee")),
        ("Prof. Ana de la Cruz", ("Ana", "Cruz")),
    ],
)
def test_split_name(name, expected):
    assert split_name(name) == expected


def test_domain_of():
    assert domain_of("https://www.acme.bio/team") == "acme.bio"
    assert domain_of("gene.com") == "gene.com"
    assert domain_of("Acme Bio") is None


# --- find_contacts (agent + finder) ------------------------------------------------


def people_script(prompt, tools):
    save = tools["save_contact"]
    replies = [
        save(
            name="Dana Lee",
            evidence_urls=["https://acme.bio/team"],
            role="University Recruiter",
            company="Acme Bio",
            is_recruiter_or_admissions=True,
            alumni_of=["berkeley"],
            alumni_evidence_url="https://news.berkeley.edu/dana",
            domain="acme.bio",
            features={"school": "UC Berkeley '18", "team": "university recruiting"},
        ),
        save(
            name="Sam Park", evidence_urls=["https://www.linkedin.com/in/sam"], company="Acme Bio"
        ),
        save(
            name="Sam Park",
            evidence_urls=["https://acme.bio/blog"],
            alumni_of=["nnhs"],
            company="Acme Bio",
        ),
        save(
            name="Bo Chen",
            evidence_urls=["https://acme.bio/team"],
            company="Acme Bio",
            domain="acme.bio",
        ),
        save(name="Extra Person", evidence_urls=["https://acme.bio/team"], company="Acme Bio"),
    ]
    return "\n".join(replies)


def run_people_job(finder, max_contacts=2, draft_after=True):
    bid = briefs.start("find_contacts")
    briefs.answer(
        bid,
        {
            "target": "Acme Bio",
            "roles": "recruiter",
            "schools": "berkeley",
            "purpose": "coffee chat",
            "max_contacts": max_contacts,
            "draft_after": draft_after,
        },
    )
    job_id = briefs.submit(bid)
    agent = FakeAgent(people_script)
    worker.run_pending(worker.WorkerContext(agent=agent, email_finder=finder))
    return jobs.get(job_id), agent


def test_people_search_enforces_evidence_and_limit():
    finder = FakeFinder({("Dana", "Lee"): "dana@acme.bio", ("Bo", "Chen"): "bad@acme.bio"})
    job, agent = run_people_job(finder)
    assert job["status"] == "done", job["error"]
    note = job["result"]["note"]
    assert "non-LinkedIn" in note  # LinkedIn-only source refused
    assert "alumni_evidence_url" in note  # alumni claim without proof refused
    assert "limit of 2 reached" in note
    saved = [contacts.get(c) for c in job["result"]["contacts"]]
    assert [c["name"] for c in saved] == ["Dana Lee", "Bo Chen"]
    dana, bo = saved
    assert dana["email"] == "dana@acme.bio" and dana["email_verified"] == "valid"
    assert "https://news.berkeley.edu/dana" in dana["evidence_urls"]
    assert bo["email"] is None  # found address failed verification
    assert "UC Berkeley" in agent.calls[0]["prompt"]
    # draft_after queued a drafting job for the people with emails.
    draft = jobs.get(job["result"]["draft_job"])
    assert draft["kind"] == "draft_outreach" and draft["params"]["contact_ids"] == [dana["id"]]


def test_people_search_still_works_without_hunter_key(monkeypatch):
    monkeypatch.delenv("HUNTER_API_KEY", raising=False)
    job, _ = run_people_job(None, draft_after=False)  # context builds the default finder
    assert job["status"] == "done"
    assert "HUNTER_API_KEY" in job["result"]["email_lookup_error"]
    assert len(job["result"]["contacts"]) == 2
    assert "draft_job" not in job["result"]


def test_finder_outage_stops_lookups_for_rest_of_job():
    finder = FakeFinder(error=EmailFinderError("Hunter rate limit or monthly quota reached"))
    job, _ = run_people_job(finder, max_contacts=5, draft_after=False)
    assert len(finder.lookups) == 1
    assert "quota" in job["result"]["email_lookup_error"]


# --- find_email (standalone) -------------------------------------------------------


def test_find_email_for_a_name_creates_a_user_contact():
    finder = FakeFinder({("Ana", "Ruiz"): "ana@gene.com"})
    job_id = call(outreach_tools.find_email, name="Ana Ruiz", company_or_domain="gene.com")[
        "job_id"
    ]
    worker.run_pending(worker.WorkerContext(email_finder=finder))
    result = jobs.get(job_id)["result"]
    assert result["found"] and result["email"] == "ana@gene.com"
    c = contacts.get(result["contact_id"])
    assert (c["source"], c["company"], c["email_verified"]) == ("user", "gene.com", "valid")
    assert finder.lookups == [("Ana", "Ruiz", "gene.com", None)]


def test_find_email_for_existing_contact_and_failures():
    cid = contacts.add("Lee Wong", source="user", company="Genentech")
    finder = FakeFinder({("Lee", "Wong"): "lwong@gene.com"})
    ok = jobs.enqueue("find_email", {"contact_id": cid})
    missing = jobs.enqueue("find_email", {"contact_id": 999})
    single = jobs.enqueue("find_email", {"name": "Cher", "domain": "x.com"})
    worker.run_pending(worker.WorkerContext(email_finder=finder))
    assert contacts.get(cid)["email"] == "lwong@gene.com"
    assert finder.lookups[0] == ("Lee", "Wong", None, "Genentech")
    assert "no contact 999" in jobs.get(missing)["error"]
    assert "first and last name" in jobs.get(single)["error"]
    assert jobs.get(ok)["status"] == "done"


# --- tools and page --------------------------------------------------------------


def test_contact_tools():
    assert "error" in call(outreach_tools.find_email, name="Ana")
    added = call(outreach_tools.add_contact, name="Prof. Kim", email="kim@berkeley.edu")
    listed = call(outreach_tools.list_contacts, has_email=True)
    assert listed["contacts"][0]["contact_id"] == added["contact_id"]
    assert "email" not in listed["contacts"][0]  # addresses aren't read aloud
    assert call(outreach_tools.do_not_contact, contact_id=added["contact_id"])["updated"]
    assert call(outreach_tools.outreach_queue) == {"counts": {}, "drafts": []}


def test_contacts_page(tmp_path):
    web = TestClient(create_app())
    web.post("/contacts", data={"name": "Prof. Kim", "email": "kim@berkeley.edu", "company": ""})
    cid = contacts.search()[0]["id"]
    page = web.get("/contacts").text
    assert "Prof. Kim" in page and "kim@berkeley.edu" in page
    assert "name" in web.post("/contacts", data={"name": " "}).text  # error shown
    web.post(f"/contacts/{cid}/dnc")
    assert contacts.get(cid)["do_not_contact"] is True
    other = contacts.add("Bo Chen", source="user", company="Acme")
    web.post(f"/contacts/{other}/find-email")
    assert jobs.recent()[0]["params"] == {"contact_id": other}
