import pytest

from jarvie import jobs

pytestmark = pytest.mark.usefixtures("temp_db")


def test_job_queue_claims_in_order_once():
    first = jobs.enqueue("score_posting", {"url": "https://x/1"})
    second = jobs.enqueue("find_email", {"name": "Dana Lee", "domain": "acme.bio"})

    claimed = jobs.claim_next()
    assert claimed["id"] == first
    assert claimed["status"] == "running"
    assert claimed["params"] == {"url": "https://x/1"}
    assert jobs.claim_next()["id"] == second
    assert jobs.claim_next() is None


def test_job_finish_fail_and_spend():
    ok = jobs.enqueue("discover_postings")
    bad = jobs.enqueue("draft_outreach", {"contact_ids": [1]})
    jobs.claim_next()
    jobs.claim_next()
    jobs.finish(ok, {"saved": 3}, input_tokens=1000, output_tokens=200, cost_usd=0.25)
    jobs.fail(bad, "rate limited", cost_usd=0.05)

    assert jobs.get(ok)["result"] == {"saved": 3}
    assert jobs.get(ok)["status"] == "done"
    assert jobs.get(bad)["error"] == "rate limited"
    assert jobs.spent_today() == pytest.approx(0.30)
    assert [j["id"] for j in jobs.recent()] == [bad, ok]


def test_outreach_kinds_are_independent_of_career():
    assert jobs.KINDS["find_email"] == "outreach"
    assert jobs.KINDS["draft_outreach"] == "outreach"
    assert jobs.KINDS["discover_postings"] == "career"


def test_unknown_job_kind():
    with pytest.raises(ValueError):
        jobs.enqueue("apply_everywhere")
