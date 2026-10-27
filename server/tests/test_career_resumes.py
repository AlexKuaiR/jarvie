"""Several named resumes (swe, mle, qr): loading, picking per posting, and each user."""

from pathlib import Path

import pytest
from agent_fakes import FakeAgent, FakeWriter
from career_fakes import call
from fastapi.testclient import TestClient

from jarvie import config, db, jobs, worker
from jarvie.career import apply_assist, filters, postings, profile, research_agent, schema
from jarvie.career import tools as career_tools
from jarvie.dashboard import create_app
from jarvie.llm import Usage
from jarvie.outreach import contacts, writer
from jarvie.outreach.writer import DraftEmail, Personalization

pytestmark = pytest.mark.usefixtures("temp_db")

RESUMES = {
    "swe": ("Backend services in Go and Python.", ["software engineering"]),
    "mle": ("Trained protein models in PyTorch.", ["machine learning", "biotech"]),
    "qr": ("Stochastic calculus; options pricing project.", ["quant"]),
}


def write_multi(tmp_path, default: str = "swe", extra: str = "") -> Path:
    tables = ""
    for name, (text, fields) in RESUMES.items():
        (tmp_path / f"{name}.md").write_text(f"# Alex K ({name})\n{text}")
        (tmp_path / f"{name}.pdf").write_bytes(b"%PDF-1.4")
        tables += (
            f'[resumes.{name}]\ntext = "{tmp_path}/{name}.md"\npdf = "{tmp_path}/{name}.pdf"\n'
            f"fields = {fields!r}\n".replace("'", '"')
        )
    path = tmp_path / "career_profile.toml"
    path.write_text(
        f'default_resume = "{default}"\ngraduation = "2028-05"\n'
        'fields = ["machine learning", "software engineering", "quant"]\n'
        f"{extra}{tables}"
        '[applicant]\nname = "Alex Kuai"\nemail = "alex@berkeley.edu"\n'
    )
    return path


@pytest.fixture
def prof(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    path = write_multi(tmp_path)
    monkeypatch.setattr(config, "CAREER_PROFILE", path)
    return profile.load(path)


# --- loading ----------------------------------------------------------------------


def test_named_resumes_load(prof, tmp_path):
    assert list(prof.resumes) == ["swe", "mle", "qr"]
    assert prof.default_resume == "swe"
    assert prof.resume("QR").pdf_path == tmp_path / "qr.pdf"  # names match in any case
    assert prof.resume_text("mle").endswith("PyTorch.")
    assert prof.resume("default").name == "swe"
    assert "options pricing" in prof.labeled_resumes()


def test_single_resume_still_works(tmp_path):
    (tmp_path / "r.md").write_text("# Me")
    path = tmp_path / "p.toml"
    path.write_text(f'resume = "{tmp_path}/r.md"\ngraduation = "2028-05"\nfields = ["biology"]\n')
    p = profile.load(path)
    assert list(p.resumes) == ["default"] and p.resume_text() == "# Me"
    assert p.resume_for({"title": "Quant Intern", "fields": ["quant"]}).name == "default"


@pytest.mark.parametrize(
    "text, message",
    [
        ('resume = "r.md"\n[resumes.a]\ntext = "a.md"\n', "not both"),
        ('[resumes.a]\npdf = "a.pdf"\n', "needs text"),
        ('[resumes.a]\ntext = "a.md"\ncolor = "red"\n', "unknown"),
        ('default_resume = "b"\n[resumes.a]\ntext = "a.md"\n', "isn't one of"),
    ],
)
def test_bad_resume_config_is_named(tmp_path, text, message):
    path = tmp_path / "p.toml"
    path.write_text(f'graduation = "2028-05"\nfields = ["x"]\n{text}')
    with pytest.raises(profile.ProfileError, match=message):
        profile.load(path)


def test_missing_resume_file_names_which(prof, tmp_path):
    (tmp_path / "qr.md").unlink()
    with pytest.raises(profile.ProfileError, match="'qr' not found"):
        prof.resume_text("qr")
    with pytest.raises(profile.ProfileError, match="no resume named 'pm'"):
        prof.resume("pm")


# --- picking per posting ----------------------------------------------------------


@pytest.mark.parametrize(
    "title, fields, expected",
    [
        ("Quantitative Research Intern", ["quant", "software engineering"], "qr"),
        ("Machine Learning Engineer Intern", ["machine learning", "software engineering"], "mle"),
        ("Software Engineering Intern", ["software engineering"], "swe"),
        ("Protein Design Intern", ["biotech"], "mle"),
        ("Accounting Intern", [], "swe"),  # nothing matches: the default
    ],
)
def test_resume_picked_by_fields(prof, title, fields, expected):
    assert prof.resume_for({"title": title, "fields": fields}).name == expected


def test_your_pick_wins(prof):
    assert prof.resume_for({"title": "SWE Intern", "fields": [], "resume": "qr"}).name == "qr"


def test_quant_postings_are_tagged():
    assert "quant" in filters.tag_fields("Quant Trader Intern", "")
    assert "quant" in filters.tag_fields("Summer Analyst", "Work on market making strategies.")
    assert "quant" not in filters.tag_fields("Backend Intern", "Quantity of APIs")


def test_old_database_gets_the_resume_column():
    old = schema.CAREER_SCHEMA.replace(
        "    resume TEXT,                    -- resume name picked for it; NULL = best match by fields\n",
        "",
    )
    assert old != schema.CAREER_SCHEMA
    with db.connect() as conn:
        conn.executescript(old)
    pid = postings.save(company="Acme", title="Intern", source="web", url="https://a/1")
    assert postings.set_resume(pid, "qr") and postings.get(pid)["resume"] == "qr"


# --- each place a resume is used --------------------------------------------------


def run(agent, prof, job_kind, **params):
    job_id = jobs.enqueue(job_kind, params)
    worker.run_pending(worker.WorkerContext(agent=agent, profile_loader=lambda: prof))
    return jobs.get(job_id)


def test_scoring_sees_every_resume_and_records_the_one_used(prof):
    replies = []

    def script(prompt, tools):
        save = tools["save_posting"]
        replies.append(save(company="Jane St", title="Quant Intern", resume="pm"))
        replies.append(save(company="Jane St", title="Quant Intern", fit_score=0.8, resume="qr"))

    agent = FakeAgent(script)
    job = run(agent, prof, "score_posting", text="Quant intern at Jane St, paid.")
    assert job["status"] == "done", job["error"]
    system = agent.calls[0]["system"][0]["text"]
    assert all(f"Resume '{n}'" in system for n in RESUMES)
    assert replies[0].startswith("error: resume must be one of")
    [pid] = job["result"]["saved"]
    assert postings.get(pid)["resume"] == "qr"


def test_agent_never_overrides_your_pick(prof):
    pid = postings.save(company="Jane St", title="Quant Intern", source="web", url="https://j/1")
    postings.set_resume(pid, "swe")
    save = research_agent.make_save_posting(research_agent.SaveLog(), None, "web", ("swe", "qr"))
    assert save(company="Jane St", title="Quant Intern", posting_id=pid, resume="qr").startswith(
        "saved"
    )
    assert postings.get(pid)["resume"] == "swe"


def test_cover_letter_uses_the_postings_resume(prof):
    pid = postings.save(
        company="Citadel", title="Quant Research Intern", source="web", fields=["quant"]
    )

    def script(prompt, tools):
        return tools["save_document"](
            text="Dear team", claims_about_student=["Built an options pricing project."]
        )

    agent = FakeAgent(script)
    job = run(agent, prof, "write_document", posting_id=pid, kind="cover_letter")
    assert job["result"]["resume"] == "qr" and job["result"]["flagged"] == []
    system = agent.calls[0]["system"][0]["text"]
    assert "options pricing" in system and "Go and Python" not in system


def test_apply_assistant_uploads_the_postings_pdf(prof, tmp_path):
    from test_career_phase6 import FORM, FakeBrowser

    pid = postings.save(
        company="Acme",
        title="ML Engineer Intern",
        source="greenhouse",
        ats="greenhouse",
        url="https://boards.greenhouse.io/acme/jobs/9",
        fields=["machine learning"],
    )
    browser = FakeBrowser(FORM)
    plan, _ = apply_assist.prepare(pid, browser, prof)
    assert browser.uploaded["#cv"] == tmp_path / "mle.pdf"
    assert plan.uploads[0][2] == "resume 'mle'"


def test_outreach_uses_the_named_resume_or_the_postings(prof):
    cid = contacts.add(
        name="Dana Lee",
        evidence_urls=["https://x/team"],
        source="user",
        company="Citadel",
        email="dana@citadel.com",
        features={"team": "quant research"},
    )
    pid = postings.save(company="Citadel", title="Quant Intern", source="web", fields=["quant"])

    def make(prompt, schema):
        return DraftEmail(
            subject="Hi",
            body="Hi Dana, " + "word " * 100 + "Best, Alex Kuai",
            personalization_used=[Personalization(feature_key="team", how_used="x")],
        )

    for params, expected in [
        ({"resume": "mle"}, "PyTorch"),
        ({"posting_id": pid}, "options pricing"),
        ({}, "Go and Python"),
    ]:
        fake = FakeWriter(make)
        writer.draft_outreach({"id": 1, "params": {"contact_ids": [cid], **params}}, fake, Usage())
        assert expected in fake.calls[0]
        jobs_db_cleanup(cid)


def jobs_db_cleanup(contact_id):
    """Drop the draft so the next loop can draft to the same person again."""
    from jarvie.outreach.schema import connect

    with connect() as conn:
        conn.execute("DELETE FROM outreach WHERE contact_id = ?", (contact_id,))


def test_voice_tool_and_dashboard_set_the_resume(prof):
    pid = postings.save(company="Citadel", title="SWE Intern", source="web", url="https://c/1")
    assert call(career_tools.set_posting_resume, posting_id=pid, resume="QR")["resume"] == "qr"
    assert call(career_tools.get_posting, posting_id=pid)["posting"]["resume_used"] == "qr"
    assert (
        "no resume named"
        in call(career_tools.set_posting_resume, posting_id=pid, resume="pm")["error"]
    )
    web = TestClient(create_app())
    assert "Use this resume" in web.get(f"/postings/{pid}").text
    web.post(f"/postings/{pid}/resume", data={"resume": "auto"})
    assert postings.get(pid)["resume"] is None
    assert call(career_tools.get_posting, posting_id=pid)["posting"]["resume_used"] == "swe"
