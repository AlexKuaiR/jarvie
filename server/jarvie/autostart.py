"""Start the background worker on demand, so you don't have to.

When a job is queued and no worker is alive, a worker is started in the background
with --idle-exit, so it stops by itself once the queue has been empty for a while
(JARVIE_WORKER_IDLE_SECONDS, default 60). It's its own process, detached from the bot,
so restarting the bot doesn't cut off a running search. Output goes to
~/jarvie-data/logs/worker.log. Set JARVIE_WORKER_AUTOSTART=0 to turn this off and run
`python -m jarvie.worker` yourself.
"""

import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from loguru import logger

from . import config, jobs

SERVER_DIR = Path(__file__).resolve().parents[1]  # where `python -m jarvie.worker` runs


def enabled() -> bool:
    return os.getenv("JARVIE_WORKER_AUTOSTART", "1") != "0"


def idle_seconds() -> float:
    return float(os.getenv("JARVIE_WORKER_IDLE_SECONDS", "60"))


def log_path() -> Path:
    path = config.DATA_DIR / "logs" / "worker.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def worker_command() -> list[str]:
    return [sys.executable, "-m", "jarvie.worker", "--idle-exit", f"{idle_seconds():g}"]


def _popen(cmd: list[str], log: Path) -> None:
    with log.open("a") as out:
        # start_new_session detaches it from the bot's terminal and Ctrl-C.
        subprocess.Popen(
            cmd,
            cwd=SERVER_DIR,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


LAUNCHER: Callable[[list[str], Path], None] = _popen  # tests swap in a fake


def ensure_worker() -> bool:
    """Start a worker if none is alive. True if this call started one."""
    if not enabled() or not jobs.claim_start():
        return False
    try:
        LAUNCHER(worker_command(), log_path())
    except OSError as e:
        logger.warning("couldn't start the worker: {}", e)
        jobs.heartbeat(alive=False)  # let the next job try again
        return False
    logger.info("started the background worker (log: {})", log_path())
    return True
