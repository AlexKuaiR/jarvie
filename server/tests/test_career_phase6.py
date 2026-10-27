"""Phase 6: apply assistant. Fake browser everywhere; the real one only if installed."""

from pathlib import Path

import pytest
from career_fakes import write_profile

from jarvie import config
from jarvie.career import apply_assist, documents, postings, profile
from jarvie.career.apply_assist import FormField, plan_fill

pytestmark = pytest.mark.usefixtures("temp_db")


@pytest.fixture
def prof(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    path = write_profile(tmp_path)
    resume_pdf = tmp_path / "resume.pdf"
    resume_pdf.write_bytes(b"%PDF-1.4")
    path.write_text(
        path.read_text()
        + f'[applicant]\nname = "Alex Kuai"\nemail = "alex@berkeley.edu"\nphone = "555-0100"\n'
        f'school = "UC Berkeley"\nlinks = ["https://github.com/alex", "https://alex.dev"]\n'
        f'resume_pdf = "{resume_pdf}"\nwork_authorization = "Yes"\n'
    )
    monkeypatch.setattr(config, "CAREER_PROFILE", path)
    return profile.load(path)


FORM = [
    FormField("#fn", "First Name", "text", True),
    FormField("#ln", "Last Name", "text", True),
    FormField("#em", "Email", "email", True),
    FormField("#ph", "Phone", "tel"),
    FormField("#gh", "GitHub URL", "url"),
    FormField("#li", "LinkedIn Profile", "url"),
    FormField("#sc", "School", "select", options=["Stanford", "UC Berkeley"]),
    FormField("#cv", "Resume/CV", "file", True),
    FormField("#cl", "Cover Letter", "file"),
    FormField("#wa", "Are you legally authorized to work in the US?", "text"),
    FormField("#why", "Why do you want to work here?", "textarea", True),
    FormField("#g", "Gender", "select", options=["Male", "Female", "Decline"]),
    FormField("#v", "Veteran status", "text"),
    FormField("#agree", "I agree to the privacy policy", "checkbox", True),
]


class FakeBrowser:
    def __init__(self, fields):
        self._fields = fields
        self.opened, self.filled, self.uploaded, self.shots = None, {}, {}, []
        self.submitted = False  # there's no submit method at all; stays False

    def open(self, url):
        self.opened = url

    def fields(self):
        return self._fields

    def fill(self, f, value):
        self.filled[f.selector] = value

    def upload(self, f, path):
        self.uploaded[f.selector] = Path(path)

    def screenshot(self, path):
        Path(path).write_bytes(b"png")
        self.shots.append(path)


def test_plan_fills_what_it_knows_and_leaves_the_rest(prof):
    plan = plan_fill(FORM, prof.applicant, cover_letter=None)
    filled = {f.selector: v for f, v, _ in plan.fills}
    assert filled == {
        "#fn": "Alex",
        "#ln": "Kuai",
        "#em": "alex@berkeley.edu",
        "#ph": "555-0100",
        "#gh": "https://github.com/alex",
        "#sc": "UC Berkeley",
        "#wa": "Yes",
    }
    assert [f.selector for f, _, _ in plan.uploads] == ["#cv"]
    blank = {f.selector: why for f, why in plan.left_blank}
    assert "demographic" in blank["#g"] and "demographic" in blank["#v"]
    assert "no linkedin" in blank["#li"].lower() or "links" in blank["#li"]
    assert "cover letter" in blank["#cl"]
    assert "#why" in blank and "#agree" in blank


def test_prepare_fills_uploads_screenshots_and_never_submits(prof):
    pid = postings.save(
        company="Acme",
        title="ML Intern",
        source="greenhouse",
        ats="greenhouse",
        url="https://boards.greenhouse.io/acme/jobs/1",
    )
    documents.add_version(pid, "cover_letter", "Dear Acme team, ...")
    browser = FakeBrowser(FORM)
    plan, shot = apply_assist.prepare(pid, browser, prof)
    assert browser.opened == "https://boards.greenhouse.io/acme/jobs/1"
    assert browser.filled["#fn"] == "Alex"
    assert browser.uploaded["#cl"].read_text() == "Dear Acme team, ..."
    assert browser.uploaded["#cv"].name == "resume.pdf"
    assert shot.exists() and browser.submitted is False
    p = postings.get(pid)
    assert p["status"] == "interested"  # not "applied": you submit
    assert "not submitted" in postings.history(pid)[0]["note"]


@pytest.mark.parametrize(
    "ats, message",
    [("workday", "Workday"), ("handshake", "Handshake"), (None, "only greenhouse")],
)
def test_unsupported_sites_are_refused(prof, ats, message):
    pid = postings.save(company="X", title="Intern", source="web", ats=ats, url=f"https://x/{ats}")
    with pytest.raises(ValueError, match=message):
        apply_assist.prepare(pid, FakeBrowser(FORM), prof)


def test_needs_applicant_details(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    bare = profile.load(write_profile(tmp_path))
    pid = postings.save(
        company="Acme", title="Intern", source="greenhouse", ats="greenhouse", url="https://g/1"
    )
    with pytest.raises(ValueError, match="applicant"):
        apply_assist.prepare(pid, FakeBrowser(FORM), bare)


def test_application_urls():
    assert (
        apply_assist.application_url({"url": "https://jobs.lever.co/a/1", "ats": "lever"})
        == "https://jobs.lever.co/a/1/apply"
    )
    assert (
        apply_assist.application_url({"url": "https://jobs.ashbyhq.com/a/1", "ats": "ashby"})
        == "https://jobs.ashbyhq.com/a/1/application"
    )


def _chromium_installed() -> bool:
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            p.chromium.launch(headless=True).close()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _chromium_installed(), reason="run `.venv/bin/playwright install chromium`")
def test_real_browser_reads_and_fills_a_local_form(tmp_path, prof):
    form = tmp_path / "form.html"
    form.write_text(
        '<form><label for="a">First Name</label><input id="a" required>'
        '<label>Email <input type="email" name="e"></label>'
        '<select aria-label="School"><option>Stanford</option><option>UC Berkeley</option></select>'
        '<input type="file" aria-label="Resume"><button type="submit">Submit</button></form>'
    )
    browser = apply_assist.PlaywrightBrowser(headless=True)
    try:
        browser.open(form.as_uri())
        fields = browser.fields()
        assert [f.label for f in fields] == ["First Name", "Email", "School", "Resume"]
        plan = plan_fill(fields, prof.applicant, None)
        for f, v, _ in plan.fills:
            browser.fill(f, v)
        assert browser.page.input_value("#a") == "Alex"
    finally:
        browser.close()
