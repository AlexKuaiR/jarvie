"""Phase 4: writer + grounding check, batch drafting, approval queue, scheduling, sending."""

import json
import random
from datetime import datetime

import pytest
from agent_fakes import FakeWriter, message, mock_anthropic
from career_fakes import call
from fastapi.testclient import TestClient

from jarvie import jobs, worker
from jarvie.config import TZ
from jarvie.dashboard import create_app
from jarvie.llm import ClaudeWriter, write_many
from jarvie.outreach import contacts, messages, sender, writer
from jarvie.outreach import tools as outreach_tools
from jarvie.outreach.writer import DraftEmail, Personalization, Sender, check_draft

pytestmark = pytest.mark.usefixtures("temp_db")

ME = Sender("Alex Kuai", "UC Berkeley EECS '28. Built a protein structure model in PyTorch.")


def dana(**kw) -> int:
    fields = dict(
        name="Dana Lee",
        evidence_urls=["https://acme.bio/team"],
        company="Acme Bio",
        role="University Recruiter",
        alumni_of=["berkeley"],
        email="dana@acme.bio",
        features={"school": "UC Berkeley '18", "team": "university recruiting"},
    )
    fields.update(kw)
    return contacts.add(**fields)


def good_body(extra: str = "") -> str:
    return (
        "Hi Dana,\n\nI'm a Berkeley EECS student and saw that you're a fellow Cal alum on the "
        "university recruiting team at Acme Bio. I've been building protein structure models "
        "in PyTorch and I'm curious how new grads and interns find their footing on your teams. "
        f"{extra}Would you be open to a 15-minute call sometime in the next few weeks? Any time "
        "that works for you is great, and I'm happy to work around your schedule. Thank you "
        "for considering it, and I hope your recruiting season is going well.\n\nBest,\nAlex"
    )


def draft(body: str | None = None, used=("school",)) -> DraftEmail:
    return DraftEmail(
        subject="Berkeley EECS student, quick question about Acme Bio",
        body=body or good_body(),
        personalization_used=[Personalization(feature_key=k, how_used="...") for k in used],
    )


# --- grounding check ---------------------------------------------------------------


def test_good_draft_passes():
    assert check_draft(draft(), contacts.get(dana()), ME) == []


@pytest.mark.parametrize(
    "bad, problem",
    [
        (draft(body="Hi Dana, coffee? Alex"), "words"),
        (draft(used=("hobbies",)), "isn't stored"),
        (draft(body=good_body("I loved your talk at NeurIPS. ")), "talk"),
        (draft(body=good_body().replace("Alex", "[Your Name]")), "placeholder"),
        (draft(body=good_body().replace("Hi Dana", "Hi there")), "by name"),
    ],
)
def test_bad_drafts_are_caught(bad, problem):
    assert any(problem in p for p in check_draft(bad, contacts.get(dana()), ME))


# --- drafting job ------------------------------------------------------------------


def run_draft_job(make, contact_ids, **params):
    job_id = jobs.enqueue("draft_outreach", {"contact_ids": contact_ids, **params})
    fake = FakeWriter(make)
    worker.run_pending(worker.WorkerContext(writer=fake, extra={"sender": ME}))
    return jobs.get(job_id), fake


def test_drafts_are_checked_queued_and_skips_explained():
    ok = dana()
    blocked = dana(name="Bo Chen", email="bo@acme.bio")
    contacts.mark_do_not_contact(blocked)
    no_email = contacts.add("Ana Ruiz", source="user", company="Acme Bio")
    busy = dana(name="Cy Park", email="cy@acme.bio")
    messages.add_draft(busy, "s", "b")

    job, fake = run_draft_job(lambda p, s: draft(), [ok, blocked, no_email, busy, 999])
    assert job["status"] == "done", job["error"]
    r = job["result"]
    assert len(r["drafted"]) == 1 and messages.get(r["drafted"][0])["contact_id"] == ok
    skipped = {int(k): v for k, v in r["skipped"].items()}
    assert skipped == {
        blocked: "do not contact",
        no_email: "no email address",
        busy: "already has an email in progress",
        999: "no such contact",
    }
    assert "UC Berkeley '18" in fake.calls[0] and "Alex Kuai" in fake.calls[0]
    assert job["cost_usd"] == pytest.approx(0.02)  # the summary, then the email


def test_bad_draft_gets_one_retry_with_the_problems():
    attempts = iter([draft(body=good_body("Loved your podcast! ")), draft()])
    job, fake = run_draft_job(lambda p, s: next(attempts), [dana()])
    assert len(job["result"]["drafted"]) == 1
    assert "podcast" in fake.calls[1] and "fix them" in fake.calls[1]


def test_draft_rejected_after_retry_is_not_queued():
    job, _ = run_draft_job(lambda p, s: draft(used=("hobbies",)), [dana()])
    assert job["result"]["drafted"] == []
    assert "isn't stored" in json.dumps(job["result"]["rejected"])
    assert messages.queue() == []


def test_posting_link_and_purpose_are_used():
    from jarvie.career import postings

    pid = postings.save(company="Acme Bio", title="ML Intern", source="web", url="https://a/1")
    job, fake = run_draft_job(lambda p, s: draft(), [dana()], purpose="referral", posting_id=pid)
    m = messages.get(job["result"]["drafted"][0])
    assert (m["purpose"], m["posting_id"]) == ("referral", pid)
    assert "ML Intern at Acme Bio" in fake.calls[0] and "referral" in fake.calls[0]


def test_missing_sender_name_fails_clearly(tmp_path, monkeypatch):
    from career_fakes import write_profile

    from jarvie import config

    monkeypatch.setattr(config, "CAREER_PROFILE", write_profile(tmp_path))
    job_id = jobs.enqueue("draft_outreach", {"contact_ids": [dana()]})
    worker.run_pending(worker.WorkerContext(writer=FakeWriter(lambda p, s: draft())))
    assert "[applicant]" in jobs.get(job_id)["error"]


# --- batch path (real SDK, mocked HTTP) ----------------------------------------------


def test_write_many_through_batches_api():
    good = json.dumps(draft().model_dump())
    batch = {
        "id": "b1",
        "type": "message_batch",
        "processing_status": "in_progress",
        "request_counts": {
            "processing": 2,
            "succeeded": 0,
            "errored": 0,
            "canceled": 0,
            "expired": 0,
        },
        "created_at": "2026-10-27T00:00:00Z",
        "expires_at": "2026-10-28T00:00:00Z",
        "ended_at": None,
        "archived_at": None,
        "cancel_initiated_at": None,
        "results_url": None,
    }
    ended = {
        **batch,
        "processing_status": "ended",
        "ended_at": "2026-10-27T00:05:00Z",
        "results_url": "https://api.anthropic.com/v1/messages/batches/b1/results",
    }
    lines = [
        {
            "custom_id": "p1",
            "result": {
                "type": "errored",
                "error": {"type": "error", "error": {"type": "api_error", "message": "x"}},
            },
        },
        {
            "custom_id": "p0",
            "result": {
                "type": "succeeded",
                "message": message([{"type": "text", "text": good}], model="claude-sonnet-5-5"),
            },
        },
    ]
    import httpx2

    seen = []

    def handler(request):
        seen.append(request)
        if request.method == "POST":
            return httpx2.Response(200, json=batch)
        if request.url.path.endswith("/results"):
            return httpx2.Response(
                200,
                text="\n".join(json.dumps(x) for x in lines),
                headers={"content-type": "application/binary"},
            )
        return httpx2.Response(200, json=ended)

    import anthropic

    client = anthropic.Anthropic(
        api_key="sk-test",
        max_retries=0,
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    sleeps = []
    out, usage = write_many(
        ClaudeWriter(client=client),
        [{"type": "text", "text": "s"}],
        ["a", "b"],
        DraftEmail,
        sleep=sleeps.append,
    )
    assert out[0][0].subject == draft().subject and out[0][1] is None
    assert out[1] == (None, "batch request errored")
    assert sleeps == [30.0]
    # Half price: 1000 in, 100 out at Sonnet 5.5 rates.
    assert usage.cost_usd == pytest.approx((1000 * 2 + 100 * 10) / 1e6 * 0.5)
    body = json.loads(seen[0].content)
    params = body["requests"][0]["params"]
    assert params["model"] == "claude-sonnet-5-5"
    assert params["output_config"]["format"]["type"] == "json_schema"


def test_claude_writer_parses_structured_output():
    client = mock_anthropic(
        [
            message(
                [{"type": "text", "text": json.dumps(draft().model_dump())}],
                model="claude-sonnet-5-5",
            )
        ]
    )
    out, usage = ClaudeWriter(client=client).write([{"type": "text", "text": "s"}], "p", DraftEmail)
    assert out.body == good_body() and usage.cost_usd > 0


# --- scheduling and sending ----------------------------------------------------------


def approved(n: int) -> list[int]:
    ids = []
    for i in range(n):
        cid = dana(name=f"Person {i}", email=f"p{i}@acme.bio")
        oid = messages.add_draft(cid, f"subject {i}", good_body())
        messages.set_status(oid, "approved")
        ids.append(oid)
    return ids


def at(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=TZ)


def test_schedule_spreads_over_weekday_hours_with_daily_cap(monkeypatch):
    monkeypatch.setenv("CAREER_DAILY_SEND_CAP", "3")
    ids = approved(5)
    sender.schedule_approved(now=at("2026-10-30T16:30"), rng=random.Random(0))  # Friday 4:30pm
    times = [datetime.fromisoformat(messages.get(i)["scheduled_at"]) for i in ids]
    assert times[0] == at("2026-10-30T16:30")
    assert all(t.weekday() < 5 and 9 <= t.hour < 17 for t in times)
    assert times == sorted(times)
    gaps = [(b - a).total_seconds() / 60 for a, b in zip(times, times[1:]) if a.date() == b.date()]
    assert all(20 <= g <= 60 for g in gaps)
    per_day: dict = {}
    for t in times:
        per_day[t.date()] = per_day.get(t.date(), 0) + 1
    assert max(per_day.values()) <= 3
    assert times[1].date().isoformat() == "2026-11-02"  # Monday, after the weekend


def test_dry_run_send_and_stop_rules(tmp_path):
    first, second, third = approved(3)
    sender.schedule_approved(now=at("2026-10-28T10:00"), rng=random.Random(1))
    contacts.mark_do_not_contact(messages.get(third)["contact_id"])  # stops it while scheduled
    mailer = sender.DryRunMailer(tmp_path / "outbox.jsonl")
    sent = sender.send_due(mailer, now=at("2026-10-28T23:00"))
    assert sent == [first, second]
    assert messages.get(first)["gmail_message_id"].startswith("dryrun-")
    assert messages.get(third)["status"] == "stopped"
    lines = [json.loads(x) for x in (tmp_path / "outbox.jsonl").read_text().splitlines()]
    assert [x["to"] for x in lines] == ["p0@acme.bio", "p1@acme.bio"]
    assert sender.check_replies(mailer) == {}  # dry-run threads are never checked


def test_nothing_sends_before_its_time_or_without_approval(tmp_path):
    [oid] = approved(1)
    cid = dana(name="Unapproved", email="u@acme.bio")
    messages.add_draft(cid, "s", good_body())
    sender.schedule_approved(now=at("2026-10-28T10:00"))
    assert sender.send_due(sender.DryRunMailer(tmp_path / "o"), now=at("2026-10-28T09:00")) == []
    assert [m["status"] for m in messages.queue()] == ["scheduled", "drafted"]


class FakeGmail:
    dry_run = False

    def __init__(self, states=None, fail=False):
        self.sent, self.states, self.fail = [], states or {}, fail

    def send(self, to, subject, body):
        if self.fail:
            raise RuntimeError("gmail down")
        self.sent.append(to)
        return f"m{len(self.sent)}", f"t{len(self.sent)}"

    def thread_state(self, thread_id):
        return self.states.get(thread_id)


def test_replies_and_bounces_are_recorded_and_failures_retry():
    a, b = approved(2)
    sender.schedule_approved(now=at("2026-10-28T10:00"))
    assert sender.send_due(FakeGmail(fail=True), now=at("2026-10-28T23:00")) == []
    assert messages.get(a)["status"] == "scheduled"  # retried next tick
    gmail = FakeGmail(states={"t1": "replied", "t2": "bounced"})
    sender.send_due(gmail, now=at("2026-10-28T23:00"))
    assert sender.check_replies(gmail) == {a: "replied", b: "bounced"}


def test_classify_thread():
    def thread(*froms):
        return {
            "messages": [{"payload": {"headers": [{"name": "From", "value": f}]}} for f in froms]
        }

    assert sender.classify_thread(thread("me@x.com"), "me@x.com") is None
    assert (
        sender.classify_thread(thread("me@x.com", "Dana <dana@acme.bio>"), "me@x.com") == "replied"
    )
    assert (
        sender.classify_thread(
            thread("me@x.com", "Mail Delivery <MAILER-DAEMON@google.com>"), "me@x.com"
        )
        == "bounced"
    )


def test_default_mailer_is_dry_run_unless_enabled(monkeypatch):
    monkeypatch.delenv("CAREER_SEND_ENABLED", raising=False)
    assert sender.default_mailer().dry_run is True
    monkeypatch.setenv("CAREER_SEND_ENABLED", "1")
    assert isinstance(sender.default_mailer(), sender.GmailMailer)


def test_worker_tick_schedules_and_sends():
    approved(1)
    gmail = FakeGmail()
    ctx = worker.WorkerContext(mailer=gmail)
    worker.run_ticks(ctx)  # schedules now (if inside the window) and sends what's due
    statuses = {m["status"] for m in messages.queue()}
    assert statuses <= {"scheduled", "sent"}


# --- tools and dashboard -------------------------------------------------------------


def test_draft_emails_tool():
    cid = dana()
    r = call(outreach_tools.draft_emails, contact_ids=[cid], purpose="advice")
    assert jobs.get(r["job_id"])["params"]["purpose"] == "advice"
    assert "error" in call(outreach_tools.draft_emails, contact_ids=[cid], purpose="spam")


def test_outreach_page_edit_approve_reject(monkeypatch):
    monkeypatch.delenv("CAREER_SEND_ENABLED", raising=False)
    web = TestClient(create_app())
    a = messages.add_draft(dana(), "Original", good_body())
    b = messages.add_draft(dana(name="Bo Chen", email="bo@acme.bio"), "Other", good_body())
    page = web.get("/outreach").text
    assert "DRY RUN" in page and "Original" in page
    web.post(f"/outreach/{a}/save", data={"subject": "Edited", "body": good_body()})
    assert messages.get(a)["subject"] == "Edited"
    web.post(f"/outreach/{b}/status", data={"status": "stopped"})
    web.post("/outreach/approve-all")
    assert (messages.get(a)["status"], messages.get(b)["status"]) == ("approved", "stopped")
    r = web.post(f"/outreach/{a}/status", data={"status": "sent"})
    assert "go from approved to sent" in r.text  # refused: only the sender sends
