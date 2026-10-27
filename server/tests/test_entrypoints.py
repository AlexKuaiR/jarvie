"""The real `python -m` commands, run as subprocesses (the way you start them)."""

import os
import subprocess
import sys
from pathlib import Path

SERVER = Path(__file__).resolve().parents[1]


def run_module(module: str, home: Path, *args: str) -> subprocess.CompletedProcess:
    # HOME points ~/jarvie-data at a temp folder, so this never touches your real data.
    env = {**os.environ, "HOME": str(home), "ANTHROPIC_API_KEY": "", "HUNTER_API_KEY": ""}
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=SERVER,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_worker_module_has_its_handlers(tmp_path):
    # Queue a search in the temp home's database, then run the real worker command once.
    code = "from jarvie import jobs; print(jobs.enqueue('discover_postings', {'ai_score': 0}))"
    env = {**os.environ, "HOME": str(tmp_path)}
    job_id = subprocess.run(
        [sys.executable, "-c", code], cwd=SERVER, env=env, capture_output=True, text=True
    ).stdout.strip()
    proc = run_module("jarvie.worker", tmp_path, "--once")
    assert proc.returncode == 0, proc.stderr
    check = f"from jarvie import jobs; j = jobs.get({job_id}); print(j['status'], '|', j['error'])"
    status = subprocess.run(
        [sys.executable, "-c", check], cwd=SERVER, env=env, capture_output=True, text=True
    ).stdout
    # No profile in the temp home, so the search fails, but for that reason, not
    # "no handler for discover_postings".
    assert "no handler" not in status
    assert "no career profile" in status
