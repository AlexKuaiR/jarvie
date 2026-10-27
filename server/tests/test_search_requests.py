"""What a spoken search request turns into, company matching, and the worker heartbeat."""

import threading
import time

import pytest
from career_fakes import call, write_profile
from fastapi.testclient import TestClient

from jarvie import config, jobs, worker
from jarvie.career import filters
from jarvie.career import tools as career_tools
from jarvie.career.filters import Criteria, company_key, judge
from jarvie.career.sources import RawPosting
from jarvie.dashboard import create_app
from jarvie.job_tools import job_status, start_brief, submit_brief

pytestmark = pytest.mark.usefixtures("temp_db")


@pytest.fixture
def profile_path(tmp_path, monkeypatch):
    path = write_profile(tmp_path)
    monkeypatch.setattr(config, "CAREER_PROFILE", path)
    return path


# --- what the user said replaces the profile's defaults ----------------------------


def test_request_answers_replace_profile_defaults(profile_path):
    # "Five paid biotech positions at big companies like Eli Lilly, not Merck or J&J"
    started = call(
        start_brief,
        kind="discover_postings",
        answers={
            "fields": "biotech",
            "max_results": 5,
            "include_companies": ["Eli Lilly"],
            "exclude_companies": "Merck and Johnson & Johnson",
        },
    )
    a = started["answers"]
    assert a["fields"] == ["biotech"]  # not the profile's four fields
    assert a["max_results"] == 5 and a["include_companies"] == ["Eli Lilly"]
    assert a["exclude_companies"] == ["Merck", "Johnson & Johnson"]
    assert a["paid_only"] is True  # still pre-filled from the profile
    assert "term" in [q["key"] for q in started["questions"]]


def test_a_bad_request_answer_keeps_the_brief_open(profile_path):
    started = call(start_brief, kind="discover_postings", answers={"max_results": "lots"})
    assert "number" in started["error"] and started["brief_id"]
    assert started["answers"]["fields"]  # profile defaults are still there


# --- company names match loosely -------------------------------------------------------


@pytest.mark.parametrize(
    "a, b",
    [
        ("Johnson & Johnson", "Johnson and Johnson"),
        ("Eli Lilly and Company", "eli lilly"),
        ("Merck & Co., Inc.", "Merck"),
        ("Genentech, Inc", "GENENTECH"),
    ],
)
def test_company_names_match(a, b):
    assert company_key(a) == company_key(b)


def test_coinbase_is_not_coin():
    assert company_key("Coinbase") != company_key("Coin")


def raw(company):
    return RawPosting(
        source="greenhouse",
        ats="greenhouse",
        company=company,
        title="Biotech Research Intern",
        url=f"https://x/{company}",
        location="Boston, MA",
        text="Protein engineering.",
    )


def test_exclude_and_include_use_loose_matching(profile_path):
    from datetime import date

    c = Criteria(
        graduation=date(2028, 5, 1),
        fields=["biotech"],
        include_companies=["Eli Lilly"],
        exclude_companies=["Johnson and Johnson", "Merck"],
    )
    assert judge(raw("Johnson & Johnson"), c).reason == "excluded company"
    assert judge(raw("Merck & Co."), c).reason == "excluded company"
    lilly, other = raw("Eli Lilly and Company"), raw("Acme Bio")
    assert filters.relevance(lilly, judge(lilly, c), c) > filters.relevance(
        other, judge(other, c), c
    )


# --- the worker heartbeat ----------------------------------------------------------


def test_no_worker_means_a_warning(profile_path):
    started = call(start_brief, kind="discover_postings", answers={"term": "Summer 2027"})
    result = call(submit_brief, brief_id=started["brief_id"])
    assert result["worker_running"] is False and "jarvie.worker" in result["tell_user"]
    assert "jarvie.worker" in call(job_status, job_id=result["job_id"])["tell_user"]
    assert call(job_status)["worker_running"] is False
    letter = call(career_tools.write_cover_letter, posting_id=1)
    assert letter["worker_running"] is False
    assert "worker isn't running" in TestClient(create_app()).get("/jobs").text


def test_running_worker_means_no_warning(profile_path):
    jobs.heartbeat()
    jobs.enqueue("find_email", {"name": "A B", "domain": "x.com"})
    assert jobs.worker_running() and jobs.worker_note() == {}
    assert "worker isn't running" not in TestClient(create_app()).get("/jobs").text
    jobs.heartbeat(alive=False)  # clean shutdown
    assert not jobs.worker_running()


def test_stale_heartbeat_counts_as_stopped(monkeypatch):
    jobs.heartbeat()
    now = time.time()
    monkeypatch.setattr(jobs.time, "time", lambda: now + jobs.HEARTBEAT_STALE + 1)
    assert not jobs.worker_running()  # e.g. the worker crashed or the laptop slept


def test_worker_main_beats_while_running_and_clears_on_exit(monkeypatch):
    seen = []
    stop_after = threading.Event()

    def fake_pending(ctx, stop):
        seen.append(jobs.worker_running())
        return 0

    monkeypatch.setattr(worker, "run_pending", fake_pending)
    monkeypatch.setattr(worker, "run_ticks", lambda ctx: stop_after.set())
    monkeypatch.setattr("sys.argv", ["worker", "--once"])
    monkeypatch.setattr("dotenv.load_dotenv", lambda **kw: None)
    worker.main()
    assert seen == [True]  # alive while it ran
    assert not jobs.worker_running()  # cleared when it stopped
