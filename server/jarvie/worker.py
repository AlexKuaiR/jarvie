"""The background worker: runs queued jobs (searches, lookups, drafting) and sends email.

Run from server/, next to the bot and the dashboard:
    .venv/bin/python -m jarvie.worker            # keep running, poll every few seconds
    .venv/bin/python -m jarvie.worker --once     # run what's queued, then exit

It's a separate process so slow model calls never stall the voice pipeline; the bot
and dashboard just add rows to the jobs table and read results back.
"""

import argparse
import os
import signal
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

from loguru import logger

from . import jobs
from .llm import AgentModel, ClaudeAgent, ClaudeWriter, LLMError, Usage, WriterModel


@dataclass
class WorkerContext:
    """What handlers need, built lazily so a missing key only fails the jobs that use it.

    Tests pass fakes for each part instead of the real Claude/Hunter/Gmail clients.
    """

    agent: AgentModel | None = None
    writer: WriterModel | None = None
    email_finder: object | None = None
    mailer: object | None = None
    profile_loader: Callable | None = None
    extra: dict = field(default_factory=dict)

    def get_agent(self) -> AgentModel:
        if self.agent is None:
            self.agent = ClaudeAgent()
        return self.agent

    def get_writer(self) -> WriterModel:
        if self.writer is None:
            self.writer = ClaudeWriter()
        return self.writer

    def get_profile(self):
        if self.profile_loader is not None:
            return self.profile_loader()
        from .career import profile

        return profile.load()

    def get_email_finder(self):
        if self.email_finder is None:
            from .outreach.email_finder import default_finder

            self.email_finder = default_finder()
        return self.email_finder

    def get_mailer(self):
        if self.mailer is None:
            from .outreach.sender import default_mailer

            self.mailer = default_mailer()
        return self.mailer


Handler = Callable[[dict, WorkerContext, Usage], dict]
HANDLERS: dict[str, Handler] = {}
# Things to do every loop besides jobs, e.g. send scheduled email. Each takes the context.
TICKS: list[Callable[[WorkerContext], None]] = []


def handler(kind: str):
    def register(fn: Handler) -> Handler:
        HANDLERS[kind] = fn
        return fn

    return register


def _register_all() -> None:
    """Import the feature modules that register handlers and ticks (once)."""
    from .career import worker_handlers  # noqa: F401
    from .outreach import worker_handlers as _o  # noqa: F401


def daily_budget() -> float:
    return float(os.getenv("JARVIE_DAILY_BUDGET_USD", "10.00"))


def run_one(job: dict, ctx: WorkerContext) -> None:
    """Run one claimed job and record its result or error, with what it cost."""
    spend = Usage()
    fn = HANDLERS.get(job["kind"])
    log = logger.bind(job_id=job["id"], kind=job["kind"])
    if fn is None:
        jobs.fail(job["id"], f"no handler for {job['kind']}")
        return
    log.info("job started")
    try:
        result = fn(job, ctx, spend)
    except (LLMError, ValueError, LookupError, OSError) as e:
        # Expected failures (missing key, bad input, network): record and move on.
        log.warning("job failed: {}", e)
        jobs.fail(job["id"], str(e), cost_usd=spend.cost_usd)
        return
    except Exception as e:  # a bug in a handler must not kill the worker loop
        log.exception("job crashed")
        jobs.fail(job["id"], f"internal error: {e!r}", cost_usd=spend.cost_usd)
        return
    jobs.finish(job["id"], result, spend.input_tokens, spend.output_tokens, spend.cost_usd)
    log.info("job done, ${:.3f}", spend.cost_usd)


def run_pending(ctx: WorkerContext, stop: threading.Event | None = None) -> int:
    """Run queued jobs until the queue is empty (or the daily budget is spent)."""
    _register_all()
    done = 0
    while stop is None or not stop.is_set():
        if jobs.spent_today() >= daily_budget():
            logger.warning("daily budget ${:.2f} reached; leaving jobs queued", daily_budget())
            break
        job = jobs.claim_next()
        if job is None:
            break
        run_one(job, ctx)
        done += 1
    return done


def run_ticks(ctx: WorkerContext) -> None:
    _register_all()
    for tick in TICKS:
        try:
            tick(ctx)
        except Exception:  # one failing tick (e.g. Gmail down) shouldn't stop the others
            logger.exception("tick {} failed", tick.__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Jarvie's background jobs.")
    parser.add_argument("--once", action="store_true", help="run queued jobs, then exit")
    parser.add_argument("--poll", type=float, default=5.0, help="seconds between checks")
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv(override=True)  # same keys as the bot (server/.env)
    ctx = WorkerContext()
    stop = threading.Event()
    # Ctrl-C / kill finish the current job, then exit cleanly.
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    logger.info("worker started (daily budget ${:.2f})", daily_budget())
    while not stop.is_set():
        run_pending(ctx, stop)
        run_ticks(ctx)
        if args.once:
            break
        stop.wait(args.poll)
    logger.info("worker stopped")


if __name__ == "__main__":
    main()
