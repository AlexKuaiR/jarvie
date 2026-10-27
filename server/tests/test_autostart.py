"""The worker starts itself when a job is queued and stops after it's been idle."""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from jarvie import autostart, jobs, worker

pytestmark = pytest.mark.usefixtures("temp_db")


@pytest.fixture
def launches(monkeypatch):
    """Turn auto-start on, with a fake launcher that records instead of starting."""
    started: list[list[str]] = []
    monkeypatch.setenv("JARVIE_WORKER_AUTOSTART", "1")
    monkeypatch.setattr(autostart, "LAUNCHER", lambda cmd, log: started.append(cmd))
    return started


def test_queuing_a_job_starts_one_worker(launches):
    jobs.enqueue("find_email", {"name": "A B", "domain": "x.com"})
    jobs.enqueue("find_email", {"name": "C D", "domain": "x.com"})  # one is already starting
    assert len(launches) == 1
    cmd = launches[0]
    assert cmd[1:4] == ["-m", "jarvie.worker", "--idle-exit"] and cmd[4] == "60"
    assert jobs.worker_note() == {}  # the bot doesn't warn while it's starting


def test_a_live_worker_means_no_new_one(launches):
    jobs.heartbeat()
    jobs.enqueue("find_email", {"name": "A B", "domain": "x.com"})
    assert launches == []


def test_a_dead_worker_is_replaced(launches, monkeypatch):
    jobs.heartbeat()
    later = time.time() + jobs.HEARTBEAT_STALE + 1  # e.g. it crashed half a minute ago
    monkeypatch.setattr(jobs.time, "time", lambda: later)
    jobs.enqueue("find_email", {"name": "A B", "domain": "x.com"})
    assert len(launches) == 1


def test_autostart_can_be_turned_off(launches, monkeypatch):
    monkeypatch.setenv("JARVIE_WORKER_AUTOSTART", "0")
    jobs.enqueue("find_email", {"name": "A B", "domain": "x.com"})
    assert launches == []
    assert "start it:" in jobs.worker_note()["tell_user"]


def test_a_failed_launch_lets_the_next_job_try_again(launches, monkeypatch):
    def broken(cmd, log):
        raise OSError("no such file")

    monkeypatch.setattr(autostart, "LAUNCHER", broken)
    jobs.enqueue("find_email", {"name": "A B", "domain": "x.com"})
    assert not jobs.worker_running()  # the claim was released
    monkeypatch.setattr(autostart, "LAUNCHER", lambda cmd, log: launches.append(cmd))
    jobs.enqueue("find_email", {"name": "C D", "domain": "x.com"})
    assert len(launches) == 1


def test_idle_seconds_setting(launches, monkeypatch):
    monkeypatch.setenv("JARVIE_WORKER_IDLE_SECONDS", "30")
    jobs.enqueue("find_email", {"name": "A B", "domain": "x.com"})
    assert launches[0][-1] == "30"


# --- idle exit ---------------------------------------------------------------------


@pytest.fixture
def quiet_worker(monkeypatch):
    monkeypatch.setattr(worker, "run_pending", lambda ctx, stop: 0)
    monkeypatch.setattr(worker, "run_ticks", lambda ctx: None)
    monkeypatch.setattr("dotenv.load_dotenv", lambda **kw: None)


def test_idle_worker_exits_and_clears_its_heartbeat(quiet_worker):
    started = time.time()
    worker.main(["--idle-exit", "0.2", "--poll", "0.05"])
    assert 0.2 <= time.time() - started < 3
    assert not jobs.worker_running()


def test_waiting_work_keeps_it_running(quiet_worker, monkeypatch):
    answers = [True] * 4  # busy for the first four checks, idle after
    seen = []

    def busy():
        seen.append(time.time())
        return answers.pop(0) if answers else False

    monkeypatch.setattr(worker, "busy", busy)
    worker.main(["--idle-exit", "0.2", "--poll", "0.05"])
    assert not answers  # it kept checking through every busy answer
    last_busy = seen[3]
    assert seen[-1] - last_busy >= 0.2  # and only left a full idle period after the last


def test_queued_jobs_and_scheduled_emails_count_as_busy():
    from jarvie.outreach import contacts, messages

    assert not worker.busy()
    job = jobs.enqueue("find_email", {"name": "A B", "domain": "x.com"})
    assert worker.busy()
    jobs.claim(job)
    jobs.finish(job, {})
    assert not worker.busy()
    cid = contacts.add("Dana Lee", ["https://x/team"], source="user", email="d@x.com")
    mid = messages.add_draft(cid, "Hi", "Hello " * 30, "coffee_chat")
    assert not worker.busy()  # a draft waits for you, not the worker
    messages.set_status(mid, "approved")
    assert worker.busy()  # approved emails still need sending


def test_a_job_queued_while_going_idle_keeps_it_running(quiet_worker, monkeypatch):
    # The worker marks itself gone, then looks again: the job queued in between wins.
    calls = {"n": 0}

    def busy():
        calls["n"] += 1
        return calls["n"] == 2  # idle at first, then the second look finds a job

    monkeypatch.setattr(worker, "busy", busy)
    worker.main(["--idle-exit", "0.1", "--poll", "0.05"])
    assert calls["n"] >= 3  # it went back to work instead of exiting at the second look


# --- for real: a subprocess worker starts, runs the job, and exits ------------------

SERVER = Path(__file__).resolve().parents[1]


def test_real_autostarted_worker_runs_the_job_then_exits(tmp_path):
    env = {
        **os.environ,
        "HOME": str(tmp_path),  # a throwaway ~/jarvie-data
        "JARVIE_WORKER_AUTOSTART": "1",
        "JARVIE_WORKER_IDLE_SECONDS": "1",
    }

    def py(code: str) -> str:
        out = subprocess.run(
            [sys.executable, "-c", code], cwd=SERVER, env=env, capture_output=True, text=True
        )
        assert out.returncode == 0, out.stderr
        return out.stdout.strip()

    job_id = py(
        "from jarvie import jobs; print(jobs.enqueue('discover_postings', {'ai_score': 0}))"
    )
    status = ""
    for _ in range(100):  # up to ~20s
        status = py(
            f"from jarvie import jobs; j = jobs.get({job_id}); "
            "print(j['status'], jobs.worker_running(), sep='|')"
        )
        if status == "failed|False":  # no profile in the temp home, so the search fails
            break
        time.sleep(0.2)
    assert status == "failed|False", status
    error = py(f"from jarvie import jobs; print(jobs.get({job_id})['error'])")
    assert "no career profile" in error  # it ran the job (not "no handler")
    log = (tmp_path / "jarvie-data" / "logs" / "worker.log").read_text()
    assert "worker started" in log and "exiting" in log
